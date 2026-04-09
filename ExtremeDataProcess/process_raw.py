import os
import json
import zarr
import warnings
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from herbie import FastHerbie, Herbie
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logging.getLogger('cfgrib').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


# variable initialization
VARIABLE_TO_SEARCHSTRING_MAP = {
    'u10': 'UGRD:10 m above', 
    'v10': 'VGRD:10 m above', 
    't2m': 'TMP:2 m above', 
    'mslma': 'MSLMA:mean sea level', 
    'gh': 'HGT',
    'q': 'SPFH',  
    't': 'TMP',  
    'u': 'UGRD',
    'v': 'VGRD'
}
FULL_SEARCHSTRING_LIST = ['UGRD:10 m above', 'VGRD:10 m above', 'TMP:2 m above', 'MSLMA:mean sea level']
for plevel in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
    FULL_SEARCHSTRING_LIST.extend([f"{s}:{plevel} mb" for s in ['HGT', 'SPFH', 'TMP', 'UGRD', 'VGRD']])
GRIB_VARIABLE_LIST = ['u10', 'v10', 't2m', 'mslma']
for plevel in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
    GRIB_VARIABLE_LIST.extend([f"{s}_{plevel}" for s in ['gh', 'q', 't', 'u', 'v']])


def map_var(pair):
    
    var_name, plevel = pair
    if plevel is None:
        searchString = VARIABLE_TO_SEARCHSTRING_MAP[var_name]
    else:
        searchString = f"{VARIABLE_TO_SEARCHSTRING_MAP[var_name]}:{plevel} mb"
        
    return searchString

def crop_images(data, horizon, time_list, process_dir, win_size=(320, 320), slide_step=150):

    T, V, H, W = data.shape
    wh, ww = win_size
    h_steps = list(range(0, H - wh + 1, slide_step))
    w_steps = list(range(0, W - ww + 1, slide_step))
    N = len(h_steps) * len(w_steps)
    crops = []
    for hi in h_steps:
        for wi in w_steps:
            crop = data[..., hi:hi+wh, wi:wi+ww]  # [T, V, wh, ww]
            crops.append(crop)
    crops = np.stack(crops, axis=1)  # [T, N, V, wh, ww]
    assert crops.shape == (T, N, V, wh, ww)
    
    horizon_dir = os.path.join(process_dir, f'horizon_{horizon}')
    os.makedirs(horizon_dir, exist_ok=True)
    # print(crops.shape)
    for time_idx, time in enumerate(time_list):
        # time_save_name = f"{time.strftime('%Y%m%d%H')}.npz"
        # save_path = os.path.join(horizon_dir, time_save_name)
        # np.savez(save_path, crops=crops[time_idx, ...])
        time_save_name = f"{time.strftime('%Y%m%d%H')}.zarr"
        save_path = os.path.join(horizon_dir, time_save_name)
        zarr.save(save_path, crops[time_idx, ...])

def process_local_files(time_list, horizon, raw_dir, process_dir, max_workers=96, use_multi=False):
    
    horizon_dir = os.path.join(process_dir, f'horizon_{horizon}')
    if os.path.exists(horizon_dir):
        processed_time_str_list = []
        for f in os.listdir(horizon_dir):
            time_str = f.strip('.zarr')
            processed_time_str_list.append(time_str)
        processed_time_str_set = set(processed_time_str_list)
        time_list = [t for t in time_list if datetime.strftime(t, '%Y%m%d%H') not in processed_time_str_set]
    
    def sub_func(time):
        
        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = os.path.join(raw_dir, time_str[:8])
        var_path_list = [f for f in os.listdir(time_dir) if f.endswith(f"f0{horizon}.grib2") and f".t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        
        var_data_dict = {}
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), indexpath='')
            var_name = list(d.data_vars)[0]
            var_data = d[var_name].values
            assert var_data.shape == (1059, 1799)
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
                var_name = f"{var_name}_{plevel}"
            else:
                plevel = None
            var_data_dict[var_name] = var_data
            d.close()
        
        data = np.zeros((69, 1059, 1799))
        for var_idx, var_name in enumerate(GRIB_VARIABLE_LIST):
            data[var_idx] = var_data_dict[var_name]
        
        return data
    
    if use_multi:
        batch_size = 24
        num_total = len(time_list)
        num_batches = num_total // batch_size if num_total % batch_size == 0 else num_total // batch_size + 1
        for batch_idx in tqdm(range(num_batches)):  
            batch_time_list = time_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            data = np.zeros([len(batch_time_list), 69, 1059, 1799])
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_time = {executor.submit(sub_func, time): time_idx for time_idx, time in enumerate(batch_time_list)}
                for future in tqdm(as_completed(future_to_time)):
                    time_idx = future_to_time[future]
                    df = future.result()
                    data[time_idx] = df
            crop_images(data, horizon, batch_time_list, process_dir)
    else:
        for time in tqdm(time_list):
            df = sub_func(time)
            data = df[np.newaxis, :]
            crop_images(data, horizon, [time], process_dir)

YEAR = 2024
BEGIN_DATETIME = datetime.strptime(f"2024010100", '%Y%m%d%H')
END_DATETIME = datetime.strptime(f"2024063023", '%Y%m%d%H')
time_list = list(pd.date_range(start=BEGIN_DATETIME, end=END_DATETIME, freq='h'))
horizon = 0
raw_dir = '/data/nihang/weather_data/HRRR/raw/hrrr'
process_dir = '/data/nihang/weather_data/HRRR/cropped_raw'
os.makedirs(process_dir, exist_ok=True)
print(time_list[0], time_list[-1], len(time_list), horizon)

# process files
process_local_files(time_list, horizon, raw_dir, process_dir)