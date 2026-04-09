import re
import os
import ast
import pickle
import random
from os import listdir
from os.path import join
import torch
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from datetime import datetime, timedelta
from torchvision.transforms import Normalize, Compose
from concurrent.futures import ProcessPoolExecutor, as_completed

COMPLETE_VARIABLE_MAP = {
    'u10': 'u10', 
    'v10': 'v10', 
    't2m': 't2m', 
    'mslma': 'msl', 
    'gh': 'z',
    'q': 'q',  
    't': 't',  
    'u': 'u',
    'v': 'v' 
}

CROPPED_VARIABLES = ['u10', 'v10', 't2m', 'mslma']
for plevel in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
    CROPPED_VARIABLES.extend([f"{s}_{plevel}" for s in ['gh', 'q', 't', 'u', 'v']])
CROPPED_VARIABLE_MAP = COMPLETE_VARIABLE_MAP

def surface_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        surface_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        surface_std = pickle.load(f)

    mean_seq, std_seq, channel_seq = [], [], []
    variables = sorted(list(surface_mean.keys()))
    for v in variables:
        channel_seq.append(v)
        mean_seq.append(surface_mean[v])
        std_seq.append(surface_std[v])

    return Normalize(mean_seq, std_seq), channel_seq

def upper_air_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f) # key 1: pressure level; key 2: variable

    with open(std_path, "rb") as f:
        upper_air_std = pickle.load(f)

    pLevels = sorted(list(upper_air_mean.keys()))
    variables = sorted(list(list(upper_air_mean.values())[0].keys()))
    normalize = {}
    for pl in pLevels:
        mean_seq, std_seq = [], []
        for v in variables:
            mean_seq.append(upper_air_mean[pl][v])
            std_seq.append(upper_air_std[pl][v])

        normalize[pl] = Normalize(mean_seq, std_seq)

    return normalize, variables, pLevels

def downscale_extreme_data(data_dir, year, down_size):
    
    save_path = os.path.join(data_dir, f'{year}.csv')
    df = pd.read_csv(save_path)
    
    def resize_bbox(bbox_str):
        assert not pd.isna(bbox_str)
        ymin, xmin, ymax, xmax = map(int, bbox_str.split('_'))
        ymin_new = round(ymin / down_size)
        xmin_new = round(xmin / down_size)
        ymax_new = round(ymax / down_size)
        xmax_new = round(xmax / down_size)
        return f"{ymin_new}_{xmin_new}_{ymax_new}_{xmax_new}"
    df['bounding_box'] = df['bounding_box'].apply(resize_bbox)
    
    # downscale_dir = data_dir.replace("merged", f"merged_downscale_{down_size}")
    downscale_dir = f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_{down_size}/HRRR/extreme/merged"
    os.makedirs(downscale_dir, exist_ok=True)
    df.to_csv(os.path.join(downscale_dir, f'{year}.csv'), index=False)

def resize_grid_skimage(data, output_shape):
    
    assert data.ndim == 3
    C = data.shape[0]
    out = np.zeros((C, output_shape[0], output_shape[1]), dtype=np.float32)
    for c in range(C):
        out[c] = resize(data[c], output_shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
        
    return out

def downscale_preloaded_time_data(time, data_dir, surface_variables, upper_air_variables, upper_air_pLevels, down_size, NWP=False):
    
    time_str = datetime.strftime(time, '%Y%m%d%H')
    # downscale_dir = data_dir.replace("hrrr", f"preload_downscale_{down_size}")
    downscale_dir = f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_{down_size}/HRRR/raw/preload"
    if NWP:
        downscale_dir = downscale_dir.replace("/raw/", "/NWP/WRF-ARW/")
    os.makedirs(downscale_dir, exist_ok=True)
    downscale_path = os.path.join(downscale_dir, f"{time_str}.pkl")
    if os.path.exists(downscale_path):
        return
    
    preload_dir = data_dir
    if NWP:
        preload_dir = preload_dir.replace("/raw/", "/NWP/WRF-ARW/")
    preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
    if NWP:
        assert "raw" not in downscale_path and "raw" not in preload_dir
    try:
        with open(preload_path, 'rb') as f:
            data = pickle.load(f)
    except:
        if NWP:
            print(time_str)
            return
        else:
            raise ValueError
    surface_data_dict = data['surface_data_dict']
    upper_air_data_dict = data['upper_air_data_dict']
    surface_data = np.stack([surface_data_dict[v] for v in surface_variables], axis=0).astype(np.float32)
    upper_air_data = [np.stack([upper_air_data_dict[pl][v] for v in upper_air_variables], axis=0).astype(np.float32) for pl in upper_air_pLevels]
    upper_air_data = np.stack(upper_air_data, axis=1)
    Cs, H, W = surface_data.shape
    Cu, P, _, _ = upper_air_data.shape
    # print(surface_data.shape, upper_air_data.shape, surface_data.min(), surface_data.mean(), surface_data.max())
    
    target_shape = (round(H / down_size), round(W / down_size))
    surface_data = resize_grid_skimage(surface_data, target_shape)
    upper_air_data = resize_grid_skimage(upper_air_data.reshape((-1, H, W)), target_shape).reshape((Cu, P, target_shape[0], target_shape[1]))
    # print(surface_data.shape, upper_air_data.shape, surface_data.min(), surface_data.mean(), surface_data.max())
    
    with open(downscale_path, 'wb') as f:
        pickle.dump({
            'surface_data_dict': {v: surface_data[idx] for idx, v in enumerate(surface_variables)}, 
            'upper_air_data_dict': {pl: {v: upper_air_data[idx_v, idx_p] for idx_v, v in enumerate(upper_air_variables)} 
                                    for idx_p, pl in enumerate(upper_air_pLevels)}
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == "__main__":

    for down_size in [2]:
        data_dir = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw/preload"
        # time_span = ["2019010100", "2024123123"]
        time_span = ["2024010100", "2024123123"]
        time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=time_span[0], end=time_span[1], freq='h'))
        
        mean_std_dir = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw/mean_std/2024010100-2024053123"
        _, surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                        join(mean_std_dir, "surface_std.pkl"))
        _, upper_air_variables, upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                        join(mean_std_dir, "upper_air_std.pkl"))
        
        def process_time(time):
            downscale_preloaded_time_data(
                time, data_dir, surface_variables, upper_air_variables, upper_air_pLevels, down_size=down_size, NWP=True
            )
        batch_size = 30
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            for i in tqdm(range(0, len(time_list), batch_size)):
                batch_times = time_list[i:i+batch_size]
                futures = [executor.submit(process_time, time) for time in batch_times]
                for future in as_completed(futures):
                    future.result()
        
        # for time in tqdm(time_list):
        #     downscale_preloaded_time_data(time, data_dir, surface_variables, upper_air_variables, upper_air_pLevels, down_size=down_size)
        #     # break
        
        # data_dir = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/extreme/merged"
        # for year in tqdm(range(2019, 2024 + 1)):
        #     downscale_extreme_data(data_dir, year, down_size=down_size)