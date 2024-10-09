
from eoekoland_dataset import EOekolandDataset
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from os.path import join

def get_dataloader(input_path, config, batch_size, transform_handler, selected_time_points, subset="train", factor=1):
    if subset=="train":
        transform = transform_handler.train_transform()
        shuffle = True
    else: #val test
        transform = transform_handler.test_transform()
        shuffle = False

    workers=config.num_workers

    dataset = EOekolandDataset(input_path, selected_time_points=selected_time_points, patchsize=config.patchsize, mode=subset, transform=transform) #, selected_time_points=common_timestamps)
    
    dataset = Subset(dataset, np.arange(int(len(dataset)/factor)))
    print(subset, ":", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return dataloader

def update_df(val, test, parameters: dict, d: pd.DataFrame):
    test_frame = pd.DataFrame(pd.Series(val), columns=["Value"], index=None).T
    val_frame = pd.DataFrame(pd.Series(test), columns=["Value"], index=None).T
    val_frame.columns = [c.replace("test", "val") for c in val_frame.columns]
    for key in parameters.keys():
        val_frame[key] = parameters[key]
    df = pd.concat([test_frame, val_frame], axis=1)
    d = pd.concat([d, df])
    return d

def store_result(test, parameters, inference_time, path, val=None):
    df_val = pd.DataFrame()
    if val is not None:
        frame = {'Metric': [k.replace("test", "val") for k in val.keys()],
             'Value': [round(100*item, 2) for item in val.values()]} # 'Value': ["%.2f"%(100*item) for item in dic.values()]}
        df_val = pd.DataFrame(frame) 
    frame = {'Metric': test.keys(),
         'Value': [round(100*item, 2) for item in test.values()]} # convert to % and two digits after decimal
    df_test = pd.DataFrame(frame)
    df = pd.concat([df_val, df_test])
    row = {"Metric": "parameters", "Value": parameters}
    df = df.append(row, ignore_index=True)
    row = {"Metric": "inference_time", "Value": inference_time}
    df = df.append(row, ignore_index=True)
    df.to_csv(join(path, "metric_results.csv"))
