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
        
def map_var(pair):
    
    var_name, plevel = pair
    if plevel is None:
        searchString = VARIABLE_TO_SEARCHSTRING_MAP[var_name]
    else:
        searchString = f"{VARIABLE_TO_SEARCHSTRING_MAP[var_name]}:{plevel} mb"
        
    return searchString

def check_local_files(time_list, horizon, raw_dir, check_dir, max_workers=96, use_multi=True):
    
    fail_var_dict = {}
    checked_time_str_list = []
    for f in os.listdir(check_dir):
        time_str, len_fail = f.strip('.json').split('_')
        checked_time_str_list.append(time_str)
        if eval(len_fail) > 0:
            with open(os.path.join(check_dir, f), 'r') as fn:
                fail_var_list = json.load(fn)
            fail_var_dict[time_str] = fail_var_list
    checked_time_str_set= set(checked_time_str_list)
    time_list = [t for t in time_list if datetime.strftime(t, '%Y%m%d%H') not in checked_time_str_set]
    
    def sub_func(time):
        
        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = os.path.join(raw_dir, time_str[:8])
        var_path_list = [f for f in os.listdir(time_dir) if f.endswith(f"f0{horizon}.grib2") and f".t{time_str[-2:]}z." in f]
        assert len(var_path_list) <= 69
        
        success_var_list = []
        for f in var_path_list:
            d = None
            try:
                d = xr.open_dataset(os.path.join(time_dir, f), indexpath='')
                var_name = list(d.data_vars)[0]
                assert d[var_name].values.shape == (1059, 1799)
                if 'isobaricInhPa' in d[var_name].coords:
                    plevel = int(d[var_name].coords['isobaricInhPa'])
                else:
                    plevel = None
                success_var_list.append((var_name, plevel))
                d.close()
            except:
                if d is not None:
                    d.close()
                os.remove(os.path.join(time_dir, f))
                if os.path.exists(f"{os.path.join(time_dir, f)}.5b7b6.idx"):
                    os.remove(f"{os.path.join(time_dir, f)}.5b7b6.idx")
        
        if len(success_var_list) < 69:
            fail_var_list = list(set(FULL_SEARCHSTRING_LIST) - {map_var(pair) for pair in success_var_list})
            print(time_str, len(fail_var_list))
        else:
            fail_var_list = []
        with open(os.path.join(check_dir, f"{time_str}_{len(fail_var_list)}.json"), 'w') as f:
            json.dump(fail_var_list, f)
            
        return fail_var_list
    
    if use_multi:
        batch_size = 24
        num_total = len(time_list)
        num_batches = num_total // batch_size if num_total % batch_size == 0 else num_total // batch_size + 1
        for batch_idx in tqdm(range(num_batches)):        
            batch_time_list = time_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_time = {executor.submit(sub_func, time): time for time in batch_time_list}
                for future in tqdm(as_completed(future_to_time)):
                    time = future_to_time[future]
                    time_str = datetime.strftime(time, '%Y%m%d%H')
                    fail_var_list = future.result()
                    if len(fail_var_list) > 0:
                        fail_var_dict[time_str] = fail_var_list
    else:
        for time in tqdm(time_list):
            fail_var_list = sub_func(time)
            if len(fail_var_list) > 0:
                fail_var_dict[time_str] = fail_var_list
    
    print("Total Failed Timestamps:", len(fail_var_dict))

def download_variable(H, searchString, raw_dir, max_workers=96, if_check=True):
    
    H.download(search=searchString, save_dir=raw_dir, verbose=False, overwrite=True)

def multi_core_download(H, var_name, raw_dir, max_workers=96, use_multi=False):
    
    if use_multi:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_variable = {executor.submit(download_variable, H, var, raw_dir, max_workers): var for var in var_name}
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                future.result()
    else:
        for var in var_name:
            download_variable(H, var, raw_dir, max_workers)

def reload_files(horizon, raw_dir, check_dir, max_workers=96, use_multi=True):
    
    fail_var_dict = {}
    checked_time_str_list = []
    for f in os.listdir(check_dir):
        time_str, len_fail = f.strip('.json').split('_')
        checked_time_str_list.append(time_str)
        if eval(len_fail) > 0:
            with open(os.path.join(check_dir, f), 'r') as fn:
                fail_var_list = json.load(fn)
            fail_var_dict[time_str] = {
                "var_list": fail_var_list, 
                "fname": os.path.join(check_dir, f)
            }
    print("Total Time: ", len(fail_var_dict))
    
    def sub_func(time_str):
        
        time = datetime.strptime(time_str, '%Y%m%d%H')
        var_list, fname = fail_var_dict[time_str]["var_list"], fail_var_dict[time_str]["fname"]
        H = Herbie(time, model="hrrr", fxx=horizon, product='prs', 
                    priority='aws', 
                    max_threads=max_workers, verbose=False, save_dir=raw_dir)
        multi_core_download(H, var_list, raw_dir, max_workers=max_workers)
        os.remove(fname)        
    
    if use_multi:
        batch_size = 24
        time_str_list = list(fail_var_dict.keys())
        num_total = len(time_str_list)
        num_batches = num_total // batch_size if num_total % batch_size == 0 else num_total // batch_size + 1
        for batch_idx in tqdm(range(num_batches)):        
            batch_time_str_list = time_str_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_time = {executor.submit(sub_func, time_str): time_str for time_str in batch_time_str_list}
                for future in tqdm(as_completed(future_to_time)):
                    time_str = future_to_time[future]
                    future.result()
    else:
        for time_str in tqdm(fail_var_dict):
            sub_func(time_str)
            # break


YEAR = 2020
BEGIN_DATETIME = datetime.strptime(f"{YEAR}010100", '%Y%m%d%H')
END_DATETIME = datetime.strptime(f"{YEAR}063023", '%Y%m%d%H')
time_list = list(pd.date_range(start=BEGIN_DATETIME, end=END_DATETIME, freq='h'))
horizon = 0
raw_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw/hrrr'
check_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/check'
# horizon = 1
# raw_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/NWP/WRF-ARW/hrrr/'
# check_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/check_nwp'
os.makedirs(check_dir, exist_ok=True)
print(time_list[0], time_list[-1], len(time_list), horizon)

# check files
check_local_files(time_list, horizon, raw_dir, check_dir)

# reload files
raw_dir = '/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw'
reload_files(horizon, raw_dir, check_dir)