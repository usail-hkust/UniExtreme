import argparse
import os
import pickle
from torch import nn
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

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

def safe_load(fname, default=None):
    
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return default

def cal_mean_std_pool_from_preload(data_dir, mean_std_dir_ori, time_list):
    
    preload_dir = data_dir.replace("hrrr", "preload")
    os.makedirs(mean_std_dir_ori, exist_ok=True)
    surface_vars = ['u10', 'v10', 't2m', 'msl']
    upper_air_vars = ['z', 'q', 't', 'u', 'v']
    plevels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    mean_std_dir = mean_std_dir_ori

    done_mean = safe_load(os.path.join(mean_std_dir, 'done_mean.pkl'), set())
    done_std = safe_load(os.path.join(mean_std_dir, 'done_std.pkl'), set())
    surface_mean_dict = safe_load(os.path.join(mean_std_dir, 'surface_mean_tmp.pkl'),
        {v: {"sum": 0., "num": 0} for v in surface_vars})
    surface_std_dict = safe_load(os.path.join(mean_std_dir, 'surface_std_tmp.pkl'),
        {v: {"sum": 0., "num": 0} for v in surface_vars})
    upper_air_mean_dict = safe_load(os.path.join(mean_std_dir, 'upper_air_mean_tmp.pkl'),
        {plevel: {v: {"sum": 0., "num": 0} for v in upper_air_vars} for plevel in plevels})
    upper_air_std_dict = safe_load(os.path.join(mean_std_dir, 'upper_air_std_tmp.pkl'),
        {plevel: {v: {"sum": 0., "num": 0} for v in upper_air_vars} for plevel in plevels})

    if not os.path.exists(os.path.join(mean_std_dir, 'surface_mean.pkl')):
        print("Calculating Mean...")
        for time_idx, time in enumerate(tqdm(time_list)):
            time_str = datetime.strftime(time, '%Y%m%d%H')
            if time_str in done_mean: continue  # 已做过跳过
            preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
            with open(preload_path, 'rb') as f:
                data = pickle.load(f)
            surface_data_dict = data['surface_data_dict']
            upper_air_data_dict = data['upper_air_data_dict']
            
            for var_name, data in surface_data_dict.items():
                surface_mean_dict[var_name]["sum"] += np.nansum(data).item()
                surface_mean_dict[var_name]["num"] += (~np.isnan(data)).sum().item()
            for plevel, p_data in upper_air_data_dict.items():
                for var_name, data in p_data.items():
                    upper_air_mean_dict[plevel][var_name]["sum"] += np.nansum(data).item()
                    upper_air_mean_dict[plevel][var_name]["num"] += (~np.isnan(data)).sum().item()

            done_mean.add(time_str)
            if time_idx % 50 != 0:
                continue
            with open(os.path.join(mean_std_dir, 'surface_mean_tmp.pkl'), 'wb') as f:
                pickle.dump(surface_mean_dict, f)
            with open(os.path.join(mean_std_dir, 'upper_air_mean_tmp.pkl'), 'wb') as f:
                pickle.dump(upper_air_mean_dict, f)
            with open(os.path.join(mean_std_dir, 'done_mean.pkl'), 'wb') as f:
                pickle.dump(done_mean, f)

        surface_mean_final = {k: v["sum"] / v["num"] for k, v in surface_mean_dict.items()}
        upper_air_mean_final = {p: {k: v["sum"] / v["num"] for k, v in d.items()} for p, d in upper_air_mean_dict.items()}
        with open(os.path.join(mean_std_dir, 'surface_mean_dict.pkl'), 'wb') as f:
            pickle.dump(surface_mean_dict, f)
        with open(os.path.join(mean_std_dir, 'upper_air_mean_dict.pkl'), 'wb') as f:
            pickle.dump(upper_air_mean_dict, f)
        with open(os.path.join(mean_std_dir, 'surface_mean.pkl'), 'wb') as f:
            pickle.dump(surface_mean_final, f)
        with open(os.path.join(mean_std_dir, 'upper_air_mean.pkl'), 'wb') as f:
            pickle.dump(upper_air_mean_final, f)

    if not os.path.exists(os.path.join(mean_std_dir, 'surface_std.pkl')):
        print("Calculating Std...")
        surface_mean_final = safe_load(os.path.join(mean_std_dir, 'surface_mean.pkl'))
        upper_air_mean_final = safe_load(os.path.join(mean_std_dir, 'upper_air_mean.pkl'))

        for time_idx, time in enumerate(tqdm(time_list)):
            time_str = datetime.strftime(time, '%Y%m%d%H')
            if time_str in done_std: continue  # 已做过跳过
            preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
            with open(preload_path, 'rb') as f:
                data = pickle.load(f)
            surface_data_dict = data['surface_data_dict']
            upper_air_data_dict = data['upper_air_data_dict']
            
            for var_name, data in surface_data_dict.items():
                mean = surface_mean_final[var_name]
                surface_std_dict[var_name]["sum"] += np.nansum((data - mean) ** 2).item()
                surface_std_dict[var_name]["num"] += (~np.isnan(data)).sum().item()
            for plevel, p_data in upper_air_data_dict.items():
                for var_name, data in p_data.items():
                    mean = upper_air_mean_final[plevel][var_name]
                    upper_air_std_dict[plevel][var_name]["sum"] += np.nansum((data - mean) ** 2).item()
                    upper_air_std_dict[plevel][var_name]["num"] += (~np.isnan(data)).sum().item()

            done_std.add(time_str)
            if time_idx % 50 != 0:
                continue
            with open(os.path.join(mean_std_dir, 'surface_std_tmp.pkl'), 'wb') as f:
                pickle.dump(surface_std_dict, f)
            with open(os.path.join(mean_std_dir, 'upper_air_std_tmp.pkl'), 'wb') as f:
                pickle.dump(upper_air_std_dict, f)
            with open(os.path.join(mean_std_dir, 'done_std.pkl'), 'wb') as f:
                pickle.dump(done_std, f)

        surface_std_final = {k: np.sqrt(v["sum"] / v["num"]) for k, v in surface_std_dict.items()}
        upper_air_std_final = {p: {k: np.sqrt(v["sum"] / v["num"]) for k, v in d.items()} for p, d in upper_air_std_dict.items()}
        with open(os.path.join(mean_std_dir, 'surface_std_dict.pkl'), 'wb') as f:
            pickle.dump(surface_std_dict, f)
        with open(os.path.join(mean_std_dir, 'upper_air_std_dict.pkl'), 'wb') as f:
            pickle.dump(upper_air_std_dict, f)
        with open(os.path.join(mean_std_dir, 'surface_std.pkl'), 'wb') as f:
            pickle.dump(surface_std_final, f)
        with open(os.path.join(mean_std_dir, 'upper_air_std.pkl'), 'wb') as f:
            pickle.dump(upper_air_std_final, f)
            
        print("All Done!")


if __name__ == "__main__":
    
    for down_size in [2]:
        time_span = ["2019010100", "2022123123"]
        data_dir = f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_{down_size}/HRRR/raw/preload"
        mean_std_dir = os.path.join(data_dir.replace("preload", "mean_std"), f"{time_span[0]}-{time_span[1]}")
        time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=time_span[0], end=time_span[1], freq='h'))
        cal_mean_std_pool_from_preload(data_dir, mean_std_dir, time_list)