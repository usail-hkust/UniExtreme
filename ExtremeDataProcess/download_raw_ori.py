import os
import json
import time
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
single_vars = ["msl", "2t", "10u", "10v"]
atmos_vars = ["hgtn", "u", "v", "t", "q"]
atmos_levels = [5000., 10000., 15000., 20000., 
                25000., 30000., 40000., 50000., 
                60000., 70000., 85000., 92500., 
                100000.]
var_name = single_vars + [f"{v}_{int(p/100)}" for v in atmos_vars for p in atmos_levels]
var_name = np.array(var_name)
herbie_maps = {'hgtn': 'HGT',
    'u': 'UGRD',
    'v': 'VGRD',
    't': 'TMP',
    'q': 'SPFH',
    'msl': 'MSLMA:mean sea level',
    '2t': 'TMP:2 m above',
    '10u': 'UGRD:10 m above',
    '10v': 'VGRD:10 m above'}
new_var_names = []
for var in var_name:
    if var in ['msl','2t','10u','10v']:
        new_var_names.append(herbie_maps[var])
    else:
        var,level_ = var.split('_')
        searchString = f"{herbie_maps[var]}:{level_} mb"
        new_var_names.append(searchString)
        
def download_variable(H, searchString, raw_dir, max_workers=96, if_check=True):
    
    path_list = H.download(search=searchString, save_dir=raw_dir, verbose=False)

def multi_core_download(H, var_name, raw_dir, max_workers=96, use_multi=True):
    
    if use_multi:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_variable = {executor.submit(download_variable, H, var, raw_dir, max_workers): var for var in var_name}
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                # try:
                future.result()
                # except Exception as exc:
                #     print(f'{variable} generated an exception: {exc}')
    else:
        for var in var_name:
            download_variable(H, var, raw_dir, max_workers)

def load_files_herbie(time_list, horizon, raw_dir, max_workers=96):
    
    def sub_func(time):
        
        H = Herbie(time, model="hrrr", fxx=horizon, product='prs', 
                        #    priority='google', 
                           priority='aws', 
                        #    priority='azure', 
                           max_threads=max_workers, verbose=False, save_dir=raw_dir)
        multi_core_download(H, new_var_names, raw_dir, max_workers=max_workers)
    
    batch_size = 96
    num_total = len(time_list)
    num_batches = num_total // batch_size if num_total % batch_size == 0 else num_total // batch_size + 1
    for batch_idx in tqdm(range(num_batches)):
        batch_time_list = time_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_time = {executor.submit(sub_func, time): time for time in batch_time_list}
            for future in tqdm(as_completed(future_to_time)):
                time = future_to_time[future]
                future.result()
        # if batch_idx % 5 == 0:
        #     time.sleep(6)


YEAR = 2020
BEGIN_DATETIME = datetime.strptime(f"{YEAR}122400", '%Y%m%d%H')
END_DATETIME = datetime.strptime(f"{YEAR}123123", '%Y%m%d%H')
# END_DATETIME = datetime.strptime(f"{YEAR}070101", '%Y%m%d%H')
time_list = list(pd.date_range(start=BEGIN_DATETIME, end=END_DATETIME, freq='h'))
# horizon = 1
# raw_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/NWP/WRF-ARW'
horizon = 0
raw_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw'
print(time_list[0], time_list[-1], len(time_list), horizon)
load_files_herbie(time_list, horizon, raw_dir)