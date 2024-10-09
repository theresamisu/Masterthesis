import torch
import os
import geopandas as gpd
#import rasterio as rio
import pandas as pd
#from rasterio import features
import numpy as np
import zipfile
from os.path import join
from torch.utils.data import DataLoader, Subset
#import pickle
#from tqdm import tqdm
import pytorch_lightning as pl
from transforms import filter_and_pad_timepoints
from torch.utils.data import DataLoader

CT_CLASSES_condensed = ['Background', 'Fallow land', 'Grassland', 'Hops', 'Legumes',
       'Maize', 'Orchards', 'Other agricultural areas', 'Potato',
       'Rapeseed', 'Spring cereals', 'Sugar beet', 'Sunflowers',
       'Vegetables', 'Winter cereals']

# can be used for weighted loss 
TRAIN_COUNTS = np.array([2.4072665e+07, 1.9256040e+06, 1.5372557e+07, 7.8515440e+06, 2.0419100e+06,
 1.5890524e+07, 1.9383000e+04, 2.2881500e+05, 1.8305480e+06, 7.3720480e+06,
 2.7036240e+06, 1.0434100e+05, 5.5536400e+05, 2.7660700e+05, 3.4832866e+07])
TRAIN_WEIGHTS = 1 - (TRAIN_COUNTS / np.sum(TRAIN_COUNTS))

class EOekolandDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, selected_time_points, patchsize=80, transform=None, mode="train"):
        """
        @param input_dir: path to datafolder
        @param selected_time_points: selected days of year (subset of [1,365])
        @param transform: function that is applied to each item in get_item function. (e.g. normalize, data augmentation, resize)
        @param mode: train/val/test
        """
        
        self.input_dir = input_dir
        self.selected_time_points = selected_time_points
        
        # existing patches will be rescaled to this patch size and (randomly) cropped to this imagesize in get_item
        self.patchsize = patchsize # this is usually the larger patch (e.g. planet patch size=200) so that the smaller patches will be upscaled
        #self.imagesize = imagesize # patches will be cropped to this after resizing to patchsize
        #print(f"rescale patches to {patchsize} and crop to {imagesize}")

        self.planet_dir = join(input_dir, "Planet")
        self.s1_dir = join(input_dir, "S1")
        self.s2_dir = join(input_dir, "S2")

        # if they exists it is faster to just read the list of file names from (one of) the csv file(s)
        if mode=="train":
            self.files = pd.read_csv(join(self.input_dir, "train_bg_shuffled.csv"))
        elif mode=="val":
            self.files = pd.read_csv(join(self.input_dir, "val_bg_shuffled.csv"))    
        elif mode=="test":
            self.files =  pd.read_csv(join(self.input_dir, "test.csv"))
        else: # all samples for e.g. predict.py
            self.files =  pd.read_csv(join(self.input_dir, "samples.csv"))
        self.data_transform = transform
        

    def get_planet_data(self, npyfile):
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                labels = object["label"].astype(np.int64)
                year = object["year"].astype(np.int64)
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                raise
        else:
            # need to use "setup" method to get the npy files for each patch
            print("ERROR: {} is missing...".format(npyfile))
            raise

        if self.selected_time_points is not None:
            # filter only the observation days that are given by "selected_time_points" (e.g. 37 days with a 10 day interval to match the Sentinel temporal resolution)
            timestamps = pd.DataFrame({"timestamp": np.arange(365)}) #[1,365]
            image_stack = filter_and_pad_timepoints(image_stack, timestamps, self.selected_time_points)
            
        return image_stack, labels, year

    def get_sentinel_data(self, input_dir, filename, means=None, std=None):
        npyfile=os.path.join(input_dir, filename)
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                labels = object["label"].astype(np.int64) #CT_CLASSES_2
                year = object["year"].astype(np.int64)
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                raise
        else:
            print("ERROR: {} is missing...".format(npyfile))
            raise
        
        return image_stack, labels, year


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files.iloc[idx].image_file

        # load three modalities, both sentinels have the same resolution -> the same label map
        planet_image_stack, planet_labels, _ = self.get_planet_data(join(self.planet_dir, filename))
        s1_image_stack, _, _ = self.get_sentinel_data(self.s1_dir, filename)
        s2_image_stack, sentinel_labels, year = self.get_sentinel_data(self.s2_dir, filename)
         
        sample = dict(s1_bands=s1_image_stack, s2_bands=s2_image_stack, p_bands=planet_image_stack, p_labels=planet_labels, s_labels=sentinel_labels, timepoints=torch.as_tensor(self.selected_time_points), index=idx, name=filename, year=year)
        if self.data_transform is not None:
            sample = self.data_transform(sample)
        return sample
    

if __name__ == '__main__':
    pl.seed_everything(35)
    input_path = "/media/storagecube/data/projects/eoekoland/CT_CODE_condensed_2" 
        
    selected_time_points = np.arange(0,364,10)
    print(len(selected_time_points))
    traindataset = EOekolandDataset(input_path, selected_time_points=selected_time_points, mode="train", transform=None) #, selected_time_points=common_timestamps)
    valdataset = EOekolandDataset(input_path, selected_time_points=selected_time_points, mode="val", transform=None) #, selected_time_points=common_timestamps)
    testdataset = EOekolandDataset(input_path, selected_time_points=selected_time_points, mode="test", transform=None) #, selected_time_points=common_timestamps)

    print("sentinel 1", traindataset[0]["s1_bands"].shape)
    print("sentinel 2", traindataset[0]["s2_bands"].shape)
    print("planet", traindataset[0]["p_bands"].shape)
    print("planet labels", traindataset[0]["p_labels"].shape)
    print("sentinel labels", traindataset[0]["s_labels"].shape)
    print("training: ", len(traindataset), ", validation: ", len(valdataset), ", testing: ", len(testdataset))
    
    # class counts
    dataset_name = ["train", "val", "test"]
    for idx, dataset in enumerate([traindataset, valdataset, testdataset]):
        class_counts = np.zeros(len(CT_CLASSES_condensed))
        for sample in dataset:
            p = sample["p_labels"]
            classes, count = np.unique(p, return_counts=True)
            class_counts[classes] += count #print(classes, count)
        print(dataset_name[idx], class_counts)

    
    # calculate means and std of training set
    X = traindataset[0]
    print(X["s1_bands"].shape)
    print("single time series image mean, std", X["s2_bands"].mean(axis=(1,2,3)), X["s2_bands"].std(axis=(1,2,3)))
    
    x = np.zeros((2+10+4))
    y = np.zeros((2+10+4))
    batch_size = 10
    nans_s1 = 0
    infs_s1 = 0
    traindataloader = DataLoader(traindataset, batch_size=batch_size, num_workers=1)
    for i, sample in enumerate(traindataloader):
        s2 = sample["s2_bands"]
        s1 = sample["s1_bands"]
        p = sample["p_bands"]
        
        if np.isnan(s1).any():
            nans_s1 += len(np.where(np.isnan(s1))[0])
        if np.isinf(s1).any():
            infs_s1 += len(np.where(np.isinf(s1))[0])
        
        x[:2] += np.nansum(s1, axis=(0,1,3,4))
        y[:2] += np.nansum((s1**2), axis=(0,1,3,4))

        x[2:12] += np.nansum(s2, axis=(0,1,3,4))
        y[2:12] += np.nansum((s2**2), axis=(0,1,3,4))
        
        x[12:] += np.nansum(p, axis=(0,1,3,4))
        y[12:] += np.nansum((p**2), axis=(0,1,3,4))

        #print("s2 mean, std:", s2.mean(axis=(0,2,3,4)), s2.std(axis=(0,2,3,4)))
        #print("s1 mean, std:", s1.mean(axis=(0,2,3,4)), s1.std(axis=(0,2,3,4)))
        #print("p mean, std:", p.mean(axis=(0,2,3,4)), p.std(axis=(0,2,3,4)))

    print("----")
    print(np.isnan(x).any(), np.isnan(y).any())
    # pixels per channel = len(traindataset) * num_timesteps * height * width
    N_s1 = len(traindataset) * X["s1_bands"].shape[0] * X["s1_bands"].shape[2] * X["s1_bands"].shape[3] - nans_s1
    N_s2 = len(traindataset) * X["s2_bands"].shape[0] * X["s2_bands"].shape[2] * X["s2_bands"].shape[3]
    N_p = len(traindataset) * X["p_bands"].shape[0] * X["p_bands"].shape[2] * X["p_bands"].shape[3]
    
    print(f"N={N_s1} found {nans_s1} nans and {infs_s1} infs")
    print("N=", N_s2, N_s1, N_p)
    
    means = x[:2] / N_s1
    std = np.sqrt(y[:2]/N_s1 - (x[:2]/N_s1)**2)
    print("s1: means, std", means, std)
    
    means = x[2:12] / N_s2
    std = np.sqrt(y[2:12]/N_s2 - (x[2:12]/N_s2)**2)
    print("s2: means, std", means, std) 
    
    means = x[12:] / N_p
    std = np.sqrt(y[12:]/N_p - (x[12:]/N_p)**2)
    print("planet: means, std", means, std)

    print("----\n", x, y)
