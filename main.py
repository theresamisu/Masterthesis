import os
if os.path.isdir("/mnt/storagecube/data/projects/eoekoland/data/CT_CODE_condensed_patches"):
     from utils.gpu_mars_dict import gpu_mapping
else:
     from utils.gpu_erde_dict import gpu_mapping
import argparse
# Handle cli arguments passed by the user
parser = argparse.ArgumentParser(description='Training script to initialize a training run on the BEN data set using a '
                                             'ViT architecture.')
parser.add_argument('--config', type=str,  default="config.yaml",
                    help='Path to a config.yaml file specifying hyper-parameters for a training run. If not provided '
                         'will assume that a "config.yaml" file is located in default working directory')
parser.add_argument('--checkpoint', type=str, default=None,
                        help="path to checkpoint.ckpt")
parser.add_argument('--gpu', type=str, default=None,
                    help='Specifies the GPU on which this run should be performed. If not provided training is '
                         'performed on the CPU')
parser.add_argument('--backbone', type=str, default=None,
                    help='Specifies model')
parser.add_argument('--batchsize', type=int, default=None,
                    help='Batchsize to use for training')
parser.add_argument('--factor', type=int, default=None,
                    help='1/factor of dataset is used for training and validation')
args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]= gpu_mapping[args["gpu"]]

from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import time

import numpy as np

from os.path import join

from transforms import TransformHandler
from utils.training_utils import store_result, get_dataloader

from model_utils import load_model
from model_handler import ModelHandler
from config import Config
from eoekoland_dataset import TRAIN_WEIGHTS

pl.seed_everything(35, workers=True)

# Initialize the config handling object from either the default path or the path specified by the user.
config_path = args["config"]
config = Config.parse_yaml_raw(Path(config_path), args)


# use gradient accumulation to fit on GPU
if not config.backbone.startswith("U-TAE") and config.backbone != "tempcnn": # tsvit
     config.gradient_accumulation = config.batch_size # per default grad.accumulation is 1
     batch_size = 1
else:
     batch_size = config.batch_size

print("configuration:", config)

gpu_training = False
print("torch.cuda.is_available()", torch.cuda.is_available())
if args["gpu"] is not None and torch.cuda.is_available():
    gpu_training = True

input_path = config.data_path
transform_handler = TransformHandler(scale=config.patchsize, flip=True, normalize=True)
selected_time_points = (np.arange(37)*10) # 37 time points in a 10 day interval to match the Sentinel observations

factor=config.factor
traindataloader = get_dataloader(input_path, config, batch_size, transform_handler, selected_time_points, "train", factor=factor)
valdataloader = get_dataloader(input_path, config, config.batch_size, transform_handler, selected_time_points, "val", factor=factor)


model = load_model(config, timesteps=len(selected_time_points), img_size=config.patchsize) # dataset.planet_patchsize if config.modalities=="P" else dataset.sentinel_patchsize need image resolution for TSViT. Is this too complicated?)
model_handler = ModelHandler(model=model,
                            epochs=config.max_epochs,
                            modalities=config.modalities,
                            lr=config.lr,
                            class_weights=TRAIN_WEIGHTS,
                            loss=config.loss)

logger = TensorBoardLogger(save_dir="./log_eoeko", name=config.run_name+"_"+config.backbone)
logger.log_hyperparams(args)
config.save_dict(join(logger.log_dir, "config.yaml"))
checkpoint_callback = ModelCheckpoint(monitor="val/loss")

trainer = Trainer(max_epochs=config.max_epochs,
                accelerator="gpu" if gpu_training else "cpu",
                devices=[0] if gpu_training else 0,
                auto_select_gpus=False,
                logger=logger,
                accumulate_grad_batches=config.gradient_accumulation,
                enable_progress_bar=True,
                precision=16, 
                callbacks=[checkpoint_callback])

start = time.time()
# train + val
trainer.fit(model_handler, traindataloader, valdataloader, ckpt_path=args["checkpoint"])
duration = time.time() - start
print("training duration:", duration)

# testing
testdataloader = get_dataloader(input_path, config, config.batch_size, transform_handler, selected_time_points, "test")
start = time.time()
test_res = trainer.test(ckpt_path="best", dataloaders=testdataloader)
test_time = (time.time() - start)
batch_inference_time = test_time/len(testdataloader)
print("batch inference time (s) =", batch_inference_time)

num_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_parameters = sum([np.prod(p.size()) for p in num_parameters])
print("parameters =", num_parameters)

# evaluate validation set on best epoch
val_res = trainer.validate(ckpt_path="best", dataloaders=valdataloader)

# store test and validation results on best epoch
store_result(test_res[0], num_parameters, batch_inference_time, logger.log_dir, val=None if val_res is None else val_res[0])

# usage: 
# [optional]
# python main.py --config config/planet.yaml --gpu 0 [--backbone TSViT] [--batchsize 8]
