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
import requests
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

# 加载下载状态
def load_status(status_file='download_status.json'):
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}

# 保存下载状态
def save_status(status, status_file='download_status.json'):
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)

# 检查文件是否已下载
def is_file_downloaded(time, var, raw_dir):
    H = Herbie(time, model="hrrr", fxx=0, product='prs', priority='aws', verbose=False, save_dir=raw_dir)
    try:
        file_path = H.get_localFilePath(search=var)
        return os.path.exists(file_path)
    except:
        return False

def download_variable(H, searchString, raw_dir, max_workers=96, if_check=True, max_retries=3, retry_delay=5):
    retry_count = 0
    while retry_count < max_retries:
        try:
            path_list = H.download(search=searchString, save_dir=raw_dir, verbose=False)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            print(f"下载 {searchString} 失败: {e}，正在重试 ({retry_count+1}/{max_retries})...")
            retry_count += 1
            time.sleep(retry_delay)
        except Exception as e:
            print(f"下载 {searchString} 失败: {e}")
            return False
    print(f"下载 {searchString} 多次尝试失败，跳过此变量")
    return False

def multi_core_download(H, var_name, raw_dir, max_workers=96, use_multi=True):
    failed_vars = []
    
    if use_multi:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_variable = {executor.submit(download_variable, H, var, raw_dir, max_workers): var for var in var_name}
            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                try:
                    success = future.result()
                    if not success:
                        failed_vars.append(variable)
                except Exception as exc:
                    print(f'{variable} 生成异常: {exc}')
                    failed_vars.append(variable)
    else:
        for var in var_name:
            success = download_variable(H, var, raw_dir, max_workers)
            if not success:
                failed_vars.append(var)
    
    return failed_vars

def load_files_herbie(time_list, horizon, raw_dir, status_file='download_status.json', max_workers=96):
    # 加载已有的下载状态
    status = load_status(status_file)
    
    def sub_func(time):
        time_str = time.strftime('%Y%m%d%H')
        # 检查该时间点是否已经下载完成
        if time_str in status and status[time_str] == 'completed':
            print(f"时间点 {time_str} 已经下载完成，跳过")
            return True
        
        H = Herbie(time, model="hrrr", fxx=horizon, product='prs', 
                           priority='aws', 
                           max_threads=max_workers, verbose=False, save_dir=raw_dir)
        
        # 下载所有变量
        failed_vars = multi_core_download(H, new_var_names, raw_dir, max_workers=max_workers)
        
        # 如果所有变量都下载成功，标记该时间点为完成
        if not failed_vars:
            status[time_str] = 'completed'
            save_status(status, status_file)
            return True
        else:
            # 记录失败的变量
            status[time_str] = {'status': 'partial', 'failed_vars': failed_vars}
            save_status(status, status_file)
            return False
    
    batch_size = 96
    num_total = len(time_list)
    num_batches = num_total // batch_size if num_total % batch_size == 0 else num_total // batch_size + 1
    
    for batch_idx in tqdm(range(num_batches)):
        batch_time_list = time_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        
        # 过滤掉已经完成的时间点
        batch_time_list = [t for t in batch_time_list if t.strftime('%Y%m%d%H') not in status or status[t.strftime('%Y%m%d%H')] != 'completed']
        
        if not batch_time_list:
            print(f"批次 {batch_idx+1}/{num_batches} 所有时间点已完成，跳过")
            continue
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_time = {executor.submit(sub_func, time): time for time in batch_time_list}
            for future in tqdm(as_completed(future_to_time)):
                time = future_to_time[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"处理时间点 {time.strftime('%Y%m%d%H')} 时出错: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HRRR raw data')
    parser.add_argument('--year', type=int, default=2020, help='Year to download')
    parser.add_argument('--begin_date', type=str, default='0101', help='Begin date in MMDD format')
    parser.add_argument('--end_date', type=str, default='1231', help='End date in MMDD format')
    parser.add_argument('--horizon', type=int, default=0, help='Forecast horizon')
    parser.add_argument('--raw_dir', type=str, default='/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw', help='Raw data directory')
    parser.add_argument('--status_file', type=str, default='download_status.json', help='Status file path')
    parser.add_argument('--max_workers', type=int, default=96, help='Maximum number of workers')
    
    args = parser.parse_args()
    
    YEAR = args.year
    BEGIN_DATETIME = datetime.strptime(f"{YEAR}{args.begin_date}00", '%Y%m%d%H')
    END_DATETIME = datetime.strptime(f"{YEAR}{args.end_date}23", '%Y%m%d%H')
    time_list = list(pd.date_range(start=BEGIN_DATETIME, end=END_DATETIME, freq='h'))
    horizon = args.horizon
    raw_dir = args.raw_dir
    status_file = args.status_file
    max_workers = args.max_workers
    
    print(f"开始时间: {time_list[0]}")
    print(f"结束时间: {time_list[-1]}")
    print(f"总时间点数: {len(time_list)}")
    print(f"预报时效: {horizon}")
    print(f"数据目录: {raw_dir}")
    print(f"状态文件: {status_file}")
    print(f"最大工作线程: {max_workers}")
    
    load_files_herbie(time_list, horizon, raw_dir, status_file, max_workers)