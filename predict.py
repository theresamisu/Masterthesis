"""
creates segmentation map for whole tile BB 2020 from given checkpoint
python predict.py --config /path/to/config --checkpoint /path/to/checkpoint --gpu x 
"""
import os
from utils.gpu_mars_dict import gpu_mapping
import argparse
# Handle cli arguments passed by the user
parser = argparse.ArgumentParser(description='Training script to initialize a training run on the BEN data set using a '
                                             'ViT architecture.')
parser.add_argument('--config', type=str,  default=None,
                    help='Path to the config.yaml file for the checkpoint')
parser.add_argument('--checkpoint', type=str, default=None,
                        help="path to checkpoint.ckpt")
parser.add_argument('--gpu', type=str, default=None,
                    help='Specifies the GPU on which this run should be performed. If not provided training is '
                         'performed on the CPU')
parser.add_argument('--backbone', type=str, default=None,
                    help='Specifies backbone model for the fusion (C3D, tempcnn, U-TAE, TSViT)')
parser.add_argument('--batchsize', type=int, default=8,
                    help='Batchsize to use')
parser.add_argument('--factor', type=int, default=1,
                    help='1/factor of dataset is used')
args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]= gpu_mapping[args["gpu"]]
from model_utils import load_model
from model_handler import ModelHandler
from config import Config
from transforms import TransformHandler

import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

from pytorch_lightning import Trainer

import numpy as np
from pathlib import Path
from utils.training_utils import get_dataloader
from eoekoland_dataset import CT_CLASSES_condensed

def plot_and_save(imgs, fpath):
    if not isinstance(imgs, list):
        plt.clf()
        imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), figsize=(12,10), squeeze=False)
        for i, img in enumerate(imgs):
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            #axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
        plt.savefig(fpath)

pl.seed_everything(35)

# Initialize the config handling object from either the default path or the path specified by the user.
config_path = args["config"]
config = Config.parse_yaml_raw(Path(config_path), args)
# adapt batch size and worker count to make prediction faster
config.batch_size = 34
config.num_workers = 16
config.data_path = "/mnt/storagecube/data/projects/eoekoland/data/CT_CODE_condensed_patches"
print("configuration:", config)

gpu_training = False
if args["gpu"] is not None and torch.cuda.is_available():
    gpu_training = True

transform_handler = TransformHandler(scale=config.patchsize, flip=True, normalize=True)
selected_time_points = (np.arange(37)*10) # 37 time points in a 10 day interval to match the Sentinel observations
model = load_model(config, timesteps=len(selected_time_points), img_size=config.patchsize) 
model_handler = ModelHandler.load_from_checkpoint(args["checkpoint"], model=model, modalities=config.modalities)

input_path = config.data_path
dataloader = get_dataloader(input_path, config, config.batch_size, transform_handler, selected_time_points, "all", factor=5)

# patches for whole tile will be accumulated in these maps
classification_map = torch.zeros((100*config.patchsize, 100*config.patchsize))
# winter cereal contains output of TSViT for this class (high values -> high probability)
winter_cereal_map = torch.zeros((100*config.patchsize, 100*config.patchsize))
# per pixel green if correctly classified, red if not.
diff_map = torch.zeros((100*config.patchsize,100*config.patchsize))


trainer = Trainer(accelerator="gpu" if gpu_training else "cpu",
                  devices=[0],
                  precision=16)
predictions = trainer.predict(model_handler, dataloader) # B x (2 x #patches x H x W)

# split output of predict method into three lists of patches
pred, target, winter_cereal = [], [], []
for b in range(len(predictions)): # num batches
     pred += [predictions[b][0]]
     target += [predictions[b][1]]
     winter_cereal += [predictions[b][2][:,14]]

pred = torch.concat(pred, dim=0)
winter_cereal = torch.concat(winter_cereal, dim=0)
target = torch.concat(target, dim=0)
print(target.shape, pred.shape, winter_cereal.shape)

# assemble patches at correct position in output map for tile (100x100 patches)
for row in range(100):
     #print(row*100+10)
     for col in range(100):
          classification_map[row*config.patchsize:(row+1)*config.patchsize, col*config.patchsize:(col+1)*config.patchsize] = pred[row*100+col]
          winter_cereal_map[row*config.patchsize:(row+1)*config.patchsize, col*config.patchsize:(col+1)*config.patchsize] = winter_cereal[row*100+col]
          diff_map[row*config.patchsize:(row+1)*config.patchsize, col*config.patchsize:(col+1)*config.patchsize] = (pred[row*100+col] == target[row*100+col])


fig, ax = plt.subplots(1,1,figsize=(12,10), squeeze=False)
pos = ax[0,0].imshow(winter_cereal_map, cmap="gray")
fig.colorbar(pos, ax=ax, shrink=0.7)
ax[0,0].set_title("Winter Cereal Classification "+config.backbone+" "+config.modalities)
path = Path(args["config"]).parents[0]
plt.savefig(path / f"winter_cereal_b{config.batch_size}.pdf")

# first downscaled then interpolated -> small details lost but sharper
my_cmap = ListedColormap(sns.color_palette("tab20", n_colors=len(CT_CLASSES_condensed)).as_hex()[:int(torch.max(classification_map)+1)])
fig, ax = plt.subplots(1,1,figsize=(12,10), squeeze=False)
ax[0,0].imshow(classification_map, cmap=my_cmap, interpolation="nearest")
ax[0,0].set_title("Classification Map "+config.backbone+" "+config.modalities)
plt.savefig(path / "prediction.pdf")

my_cmap_bin = ListedColormap(["red", "green"])
fig, ax = plt.subplots(1,1,figsize=(12,10), squeeze=False)
ax[0,0].imshow(diff_map, cmap=my_cmap_bin, interpolation="nearest")
ax[0,0].set_title("Classification Error "+config.backbone+" "+config.modalities)
plt.savefig(path / "diffmap.pdf")

# first interpolated then downscaled -> more details but a bit blurry
segmentation_palette = sns.color_palette("tab20").as_hex()
one_hot = torch.nn.functional.one_hot(classification_map.to(torch.int64), config.num_classes).to(torch.bool)
base_map = torch.ones((3,100*config.patchsize,100*config.patchsize)) # attention map 
plot_and_save(draw_segmentation_masks(image=torch.Tensor(255*base_map).to(torch.uint8), 
     masks=one_hot.swapaxes(0, 2).swapaxes(1, 2), 
     alpha=1.0, 
     colors=segmentation_palette
     ),
     path / 'prediction.png'
)

base_map = torch.ones((3,100*config.patchsize,100*config.patchsize)) # attention map 
one_hot = torch.nn.functional.one_hot(diff_map.to(torch.int64), 2).to(torch.bool)
plot_and_save(draw_segmentation_masks(image=torch.Tensor(255*base_map).to(torch.uint8), 
     masks=one_hot.swapaxes(0, 2).swapaxes(1, 2), 
     alpha=1.0, 
     colors=["red", "green"]
     ),
     path / 'diffmap.png'
)


# this plots winter cereal predictions for a few sample patches
for row in range(1):
    for col in range(0,36,1):
          patch = winter_cereal[row*100+col]
          classification = classification_map[row*100+col]
          print(row*100+col, "max", np.max(patch.numpy().flatten()), "min", np.min(patch.numpy().flatten()), np.mean(patch.numpy().flatten()))
          fig, ax = plt.subplots(1,1,figsize=(12,10), squeeze=False)
          ax[0,0].imshow(patch, cmap="gray",vmin=np.min(patch.numpy().flatten()), vmax=np.max(patch.numpy().flatten()))
          fig.colorbar(pos, ax=ax, shrink=0.7)
          ax[0,0].set_title("Winter Cereal "+config.backbone+" "+config.modalities+" id="+str(row*100+col))
          plt.savefig(f"wintercereal/winter_cereal_{row*100+col}_b{config.batch_size}.png")

          fig, ax = plt.subplots(1,1,figsize=(12,10), squeeze=False)
          ax[0,0].imshow(patch, cmap="gray",vmin=np.min(patch.numpy().flatten()), vmax=np.max(patch.numpy().flatten()))
          fig.colorbar(pos, ax=ax, shrink=0.7)
          ax[0,0].set_title("Winter Cereal "+config.backbone+" "+config.modalities+" id="+str(row*100+col))
          plt.savefig(f"wintercereal/classification_{row*100+col}_b{config.batch_size}.png")