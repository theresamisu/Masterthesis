
import os
from utils.gpu_mars_dict import gpu_mapping
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
                    help='1/factor of dataset is used')
args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]= gpu_mapping[args["gpu"]]
from model_utils import load_model
from model_handler import ModelHandler
from config import Config
from transforms import TransformHandler

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import numpy as np
import time
from os.path import join
from pathlib import Path
from utils.training_utils import store_result, get_dataloader

pl.seed_everything(35)

# Initialize the config handling object from either the default path or the path specified by the user.
config_path = args["config"]
config = Config.parse_yaml_raw(Path(config_path), args)

print("configuration:", config)
input_path = config.data_path

gpu_training = False
if args["gpu"] is not None and torch.cuda.is_available():
    gpu_training = True

logger = TensorBoardLogger(save_dir="./log_eoeko/eval", name=config.run_name+"_"+config.backbone) # Set up tensorboard logger
logger.log_hyperparams(args) # Log hyperparameters to tensorboard    
config.save_dict(join(logger.log_dir, "config.yaml"))
print("version", logger.version)

transform_handler = TransformHandler(scale=config.patchsize, flip=True, normalize=True)
selected_time_points = (np.arange(37)*10) # 37 time points in a 10 day interval to match the Sentinel observations

model = load_model(config, timesteps=len(selected_time_points), img_size=config.patchsize) # dataset.planet_patchsize if config.modalities=="P" else dataset.sentinel_patchsize need image resolution for TSViT. Is this too complicated?)
model_handler = ModelHandler.load_from_checkpoint(args["checkpoint"], model=model, modalities=config.modalities)
trainer = Trainer(max_epochs=1,
                  accelerator="gpu" if gpu_training else "cpu",
                  devices=[0] if gpu_training else 0, #[args.gpu] if torch.cuda.is_available() and gpu_training else 0,
                  logger=logger,
                  precision=16) # if args["half_precision"] else None,
 
 # evaluate on validation set              
valdataloader = get_dataloader(input_path, config, 1, transform_handler, selected_time_points, "val", factor=config.factor)
val_res = trainer.validate(model_handler, dataloaders=valdataloader)

# testing
testdataloader = get_dataloader(input_path, config, 8, transform_handler, selected_time_points, "test", factor=1)
start = time.time()
test_res = trainer.test(model_handler, dataloaders=testdataloader)
test_time = (time.time() - start)
batch_inference_time = test_time/len(testdataloader)
print("batch inference time (s) =", batch_inference_time)

num_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_parameters = sum([np.prod(p.size()) for p in num_parameters])
print("parameters =", num_parameters)

store_result(test_res[0], num_parameters, batch_inference_time, logger.log_dir, val=val_res[0] if val_res is not None else None)
