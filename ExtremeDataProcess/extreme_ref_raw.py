import os
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from herbie import FastHerbie
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_date_range(dates, pres=2, after=2):
    results = []
    for d in dates:
        results.append(pd.date_range(start=d - timedelta(hours=pres), 
                                     end=d + timedelta(hours=after), 
                                     freq='H'))
    return results

def load_variable(H_pre, searchString, raw_dir):
    
    try:
        df = H_pre.xarray(searchString, save_dir=raw_dir, remove_grib=False).to_array().values[0]
    except:
        H_pre.download(search=searchString, save_dir=raw_dir, verbose=False, overwrite=True)
        df = H_pre.xarray(searchString, save_dir=raw_dir).to_array().values[0]
    
    return df

def multi_core(H_pre, var_name, raw_dir, max_workers=80, use_multi=True):
    
    results = {}
    if use_multi:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_variable = {executor.submit(load_variable, H_pre, var, raw_dir): var for var in var_name}
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                # try:
                data = future.result()
                results[variable] = data
                # except Exception as exc:
                #     print(f'{variable} generated an exception: {exc}')
    else:
        results = {var: load_variable(H_pre, var, raw_dir) for var in var_name}
    return results

def download_variable(H_pre, searchString, raw_dir):
    
    H_pre.download(search=searchString, save_dir=raw_dir, verbose=False)

def download(H_pre, var_name, raw_dir, max_workers=80, use_multi=True):
    
    if use_multi:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_variable = {executor.submit(download_variable, H_pre, var, raw_dir): var for var in var_name}
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                # try:
                future.result()
                # except Exception as exc:
                #     print(f'{variable} generated an exception: {exc}')
    else:
        for var in var_name:
            download_variable(H_pre, var, raw_dir)

def load_files_herbie(filenames, raw_dir):
    
    # multi-core loading
    filenames = [f.strftime("%Y%m%d%H")+'.npy' for f in filenames]
    files = {}
    if len(filenames)==0:
        return files
    print(filenames)
    for f in tqdm(filenames):
        date_str = f.split('.')[0]
        # print(date_str)
        files[date_str] = np.zeros([1,69,1059,1799])
        date = pd.to_datetime(date_str, format='%Y%m%d%H')
        H_pre = FastHerbie([date], model="hrrr", fxx=[0], product='prs', max_threads=96, verbose=True, save_dir=raw_dir)
        download(H_pre, new_var_names, raw_dir)
        pre_array = multi_core(H_pre, new_var_names, raw_dir)
        for i, var in enumerate(new_var_names):
            files[date_str][0, i] = pre_array[var]
    return files

def crop_images(image, x_min, x_max, y_min, y_max):
    
    cropped_images = []
    masks = []
    height, width = 1059, 1799
    # if the bounding box size is larger than 320, generate multiple crops to cover the box
    # then mask the bounding boxs for loss calculation specific for extreme regions
    for y in range(y_min, y_max, 320):
        for x in range(x_min, x_max, 320):
            left, upper = x, y
            mask = np.zeros((1, 320, 320))
            if y+320>height and x+320>width:
                right, lower = x_max, y_max     # right lower left upper are area need to crop
                crop = image[...,lower-320:lower,right-320:right]
                mask[...,320-(lower-upper):,320-(right-left):] = 1
            elif y+320>height and x+320<=width:
                right, lower = min(x+320, x_max), min(y+320,y_max) 
                crop = image[...,lower-320:lower,left:left+320]
                mask[...,320-(lower-upper):,:right-left] = 1
            elif x+320>width:
                right, lower = min(x+320, x_max), min(y+320,y_max)
                crop = image[...,upper:upper+320,right-320:right]
                mask[...,:lower-upper,320-(right-left):] = 1 
            else:
                right, lower = min(x + 320, width), min(y + 320, height)
                crop = image[...,upper:lower, left:right]
                mask[...,:y_max-upper, :x_max-left] = 1
            # Determine if padding is needed (at the edges of the original image)
            pad_height = 320 - crop.shape[-2]
            pad_width = 320 - crop.shape[-1]
            # Create a mask for the valid area within the specified range
            # Apply padding if necessary
            if pad_height > 0 or pad_width > 0:
                print('should not have pad')
                print(crop.shape)
                assert False
            cropped_images.append(crop)
            masks.append(mask)
    return cropped_images, masks
        
def get_cropped_images(files, names, x_min, x_max, y_min, y_max):
    
    images = np.concatenate([files[i.strftime("%Y%m%d%H")] for i in names], axis=0)
    cropped_images, masks = crop_images(images,x_min, x_max, y_min, y_max)
    return cropped_images,masks
            
def align_raw_data(save_dir, raw_dir, YEAR, BEGIN_DATETIME, END_DATETIME, pres, after):
    
    df = pd.read_csv(f"/home/nihang/ExtremeWeatherForecast/HR-Extreme/index_files/data_all202007_info_new.csv", parse_dates=['begin_time','end_time'])
    df = df[(df.begin_time>="20200701")&(df.begin_time<"20201231")]
    # df = pd.read_csv(f'/data/nihang/weather_data/HRRR/extreme/extreme_data_{YEAR}.csv', parse_dates=['begin_time','end_time'])
    # df = df[(df.begin_time>=BEGIN_DATETIME)&(df.begin_time<END_DATETIME)]
    exist_files = os.listdir(save_dir)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        # prepare history-horizon time ranges, up to 24 hours/timesteps
        event_span = pd.date_range(start=row['begin_time'],end=row['end_time'], freq='h') # (duration + 1)
        datetimes_all = get_date_range(event_span,pres,after) # (duration + 1, pres + after + 1)
        gap = 24 # each time process 24 hours at most
        
        for i in range(0, len(datetimes_all), gap):
            datetimes = datetimes_all[i:i+gap]
            x_min, y_min, x_max, y_max = [int(i) for i in row['bounding_box'].split('_')]
            unique_datetimes = []
            for i in datetimes:
                for j in i:
                    if j not in unique_datetimes:
                        unique_datetimes.append(j)
            print('len of unique datetimes:', len(unique_datetimes))
            
            need_save = False
            for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
                save_name =  '_'.join([timestamp[pres].strftime("%Y%m%d%H"), row['type'], row['bounding_box']]) + '.npz'
                if save_name not in exist_files:
                    need_save = True
                    break
            
            # load raw data referenced by extreme events
            if need_save:
                
                # load raw data of specific timestamps and variables 
                imgs = load_files_herbie(unique_datetimes, raw_dir=raw_dir)
                
                for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
                    save_name =  '_'.join([timestamp[pres].strftime("%Y%m%d%H"), row['type'], row['bounding_box']]) + '.npz'
                    
                    # get and dave data crops & masks according to bounding boxes
                    cropped_images, masks = get_cropped_images(imgs, timestamp, x_min, x_max, y_min, y_max)
                    cropped_images = np.stack(cropped_images)
                    masks = np.concatenate(masks,axis=0)
                    np.savez(os.path.join(save_dir,save_name), 
                             inputs= cropped_images[:,:pres], 
                             targets=cropped_images[:,pres:],
                             masks=masks)
            
YEAR = 2024
BEGIN_DATETIME = f"{YEAR}0701"
END_DATETIME = f"{YEAR}1231"

save_dir = f'/data/nihang/weather_data/HRRR/extreme/reference_raw_data_{YEAR}'
raw_dir = '/data/nihang/weather_data/HRRR/raw/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
align_raw_data(save_dir, raw_dir, YEAR, BEGIN_DATETIME, END_DATETIME, pres=2, after=0)