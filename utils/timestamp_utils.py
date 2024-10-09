import pandas as pd
from os.path import join
from config import Config
import numpy as np

def convert_timestamp(t):
    """
    timestamps for S1 and S2 are given in DD MM YYYY format
    this function converts a given timestamp t to day of year [1,365]
    """
    # error in isocalendar function 31.01.2018 maps to (2019, 1, 1) = first day of first week of 2019
    # spans only 52 weeks
    if t.day==31 and t.month==12:
        return 365
    elif t.day==30 and t.month==12:
        return 364
        
    week = t.isocalendar()[1]
    day = t.isocalendar()[2]
    if t.year==2019:
        day_of_year = (week-1)*7+day - 1
    else:
        day_of_year = (week-1)*7+day
    return day_of_year

def convert_timestamp_list(t_list):
    """
    t_list = list of datetime.datetime entries 
    """
    return [convert_timestamp(t) for t in t_list]

def get_time_points(data_path, modalities, mode="train", s1="all"):
    """
    returns list of days of year (subset of [1,365]) that will be used for training / testing
    """
    #modalities = args["modalities"]

    # multiple modalities
    if modalities == "S":
        time_points = pd.read_csv(join(data_path, "S2", mode, "timestamps.csv")).timestamp.values
    # single modalities:
    elif modalities == "S2":
        time_points = pd.read_csv(join(data_path, "S2", mode, "timestamps.csv")).timestamp.values
        # test
        if mode=="test":
            # monthly
            #idx = np.array([7, 18, 31, 44, 52, 60, 81, 93, 104, 110, 131, 135])
            # weekly image with lowest mean cloud probability in this week
            idx = np.array([0, 4, 7, 8, 11, 15, 18, 19, 23, 25, 28, 31, 35, 36, 39, 43, 44, 49, 52, 54, 56, 59, 60, 63, 66, 69, 71, 75, 78, 81, 84, 87, 88, 93, 96, 98, 100, 104, 106, 110, 111, 113, 118, 120, 124, 125, 129, 131, 133, 135, 139, 142])
        # train
        else:
            # weekly image with lowest mean cloud probability in this week
            idx = np.array([0, 3, 7, 8, 12, 14, 17, 21, 24, 26, 28, 30, 33, 38, 39, 43, 44, 49, 50, 53, 56, 58, 61, 63, 66, 70, 72, 76, 77, 80, 83, 86, 90, 92, 96, 97, 99, 103, 107, 110, 112, 113, 116, 120, 124, 125, 126, 130, 133, 135, 138, 141])
            # idx = np.array([3,17,30,47,50,61,83,86,97,113,130,133]) # montly with lowest cloud cover
        #print(time_points)
        #idx = np.arange(0,len(time_points), 37) # choose 37 time points evenly distributed
        time_points = time_points[idx]
        #print(time_points)
    elif modalities == "S1":
        time_points = pd.read_csv(join(data_path, "S1", mode, s1, "timestamps.csv")).timestamp.values
    elif modalities == "P" or modalities=="ALL":
        time_points = list(np.arange(1,365,10)) # list(4 + np.arange(73) * 5) # s1a_s2_p5
        #time_points = list(np.arange(1,365,1))
    elif modalities == "P5": # planet 5 subset has an image every five days starting at 3.1.
        time_points = list(np.arange(1,365,5)+3)
    return time_points

def get_common_timestamps(timesteps_list):
    if len(timesteps_list) == 0:
        raise ValueError("no list was given")
    if len(timesteps_list) == 1:
        return timesteps_list[0]

    common_timepoints = timesteps_list[0]
    for t_list in timesteps_list[1:]:
        common_timepoints = list(set(t_list).intersection(common_timepoints))

    return common_timepoints

def compute_common_timestamps(args, mode="train"):
    input_path = args["data_path"]
    modalities = args["modalities"]
    
    if modalities == "S1" or modalities == "P" or modalities == "P5" or modalities == "S2":
        return None
    
    s1_timestamps = pd.read_csv(join(input_path, "S1", mode, args["s1"], "timestamps.csv"))["timestamp"].values
    s2_timestamps = pd.read_csv(join(input_path, "S2", mode, "timestamps.csv"))["timestamp"].values
    planet5_timestamps = list(np.arange(73)*5+4) # should be +3 actually but then there is no overlap at all with sentinel 2 and 1 timestamps
    planet_timestamps = list(np.arange(365)+1) 
    
    if modalities == "S":
        common_timestamps = get_common_timestamps([s1_timestamps, s2_timestamps])
    elif modalities == "PS2":
        common_timestamps = get_common_timestamps([planet_timestamps, s2_timestamps])
    elif modalities == "ALL":
        common_timestamps = get_common_timestamps([planet_timestamps, s2_timestamps, s1_timestamps])
    elif modalities == "ALL5":
        common_timestamps = get_common_timestamps([planet5_timestamps, s2_timestamps, s1_timestamps])
    return common_timestamps