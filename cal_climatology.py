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
import calendar

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

def get_hoy_from_datetime(dt):
    """Get hour of year (hoy) string in MMDDHH format from datetime object."""
    return dt.strftime("%m%d%H")

def get_sliding_window_times(target_time, K_days=2, K_hours=2):
    """Get all times within the sliding window around target_time, handling Feb 29th specially."""
    window_times = []
    
    # Special handling for Feb 29th (0229)
    if isinstance(target_time, tuple):
        year, month, day, hour = target_time
        if calendar.isleap(year):
            target_time = datetime(year, month, day, hour)
            assert target_time.strftime("%m%d") == "0229"
            for day_offset in range(-K_days, K_days + 1):
                for hour_offset in range(-K_hours, K_hours + 1):
                    window_time = target_time + timedelta(days=day_offset, hours=hour_offset)
                    window_times.append(window_time)
            # print("###", year, window_times)
        else:
            for day_offset in range(-K_days, K_days + 1):
                if day_offset == 0:
                    continue
                target_time = datetime(2020, month, day, hour) + timedelta(days=day_offset)
                target_time = target_time.replace(year=year)
                for hour_offset in range(-K_hours, K_hours + 1):
                    window_time = target_time + timedelta(hours=hour_offset)
                    window_times.append(window_time)
            # print("###", year, window_times)
    else:
        # Normal case
        for day_offset in range(-K_days, K_days + 1):
            for hour_offset in range(-K_hours, K_hours + 1):
                window_time = target_time + timedelta(days=day_offset, hours=hour_offset)
                window_times.append(window_time)
    
    return window_times

def calculate_climatology(data_dir, climatology_dir, time_list, K_days=0, K_hours=0):
    """Calculate pixel-specific, variable-specific, hoy-specific climatology."""
    preload_dir = data_dir.replace("hrrr", "preload")
    os.makedirs(climatology_dir, exist_ok=True)
    
    all_hoys = sorted(list({get_hoy_from_datetime(t) for t in time_list}))

    for hoy in tqdm(all_hoys, desc="Processing hoys"):
        
        # Find all target times that match this hoy across years
        target_times = []
        for year in range(2019, 2023):
            try:
                # Parse the hoy string to create datetime
                month = int(hoy[:2])
                day = int(hoy[2:4])
                hour = int(hoy[4:6])
                
                # Special handling for Feb 29th in non-leap years
                if month == 2 and day == 29:
                    target_times.append((year, month, day, hour))
                    continue  # Skip Feb 29th for non-leap years
                    
                dt = datetime(year, month, day, hour)
                target_times.append(dt)
            except:
                continue
        
        # For each target time, get all times in its sliding window
        all_window_times = []
        for target_time in target_times:
            window_times = get_sliding_window_times(target_time, K_days, K_hours)
            all_window_times.extend(window_times)
        
        # Filter to only include times that exist in our dataset
        valid_window_times = [t for t in all_window_times if t in time_list]
        
        assert valid_window_times
            
        # Initialize climatology accumulators
        surface_climatology = None
        upper_air_climatology = None
        count = 0
        
        # Process each valid time in the window
        for window_time in valid_window_times:
            time_str = datetime.strftime(window_time, '%Y%m%d%H')
            preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
                
            try:
                with open(preload_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Initialize accumulators on first iteration
                if surface_climatology is None:
                    surface_climatology = {
                        var: np.zeros_like(arr, dtype=np.float64) 
                        for var, arr in data['surface_data_dict'].items()
                    }
                    upper_air_climatology = {
                        plev: {
                            var: np.zeros_like(arr, dtype=np.float64)
                            for var, arr in p_data.items()
                        }
                        for plev, p_data in data['upper_air_data_dict'].items()
                    }
                
                # Accumulate surface data
                for var, arr in data['surface_data_dict'].items():
                    surface_climatology[var] += np.where(np.isnan(arr), 0, arr)
                
                # Accumulate upper air data
                for plev, p_data in data['upper_air_data_dict'].items():
                    for var, arr in p_data.items():
                        upper_air_climatology[plev][var] += np.where(np.isnan(arr), 0, arr)
                
                count += 1
            except Exception as e:
                print(f"Error processing {preload_path}: {str(e)}")
                raise ValueError
                continue
        
        assert count > 0
            
        # Calculate averages
        surface_climatology_avg = {
            var: arr / count for var, arr in surface_climatology.items()
        }
        upper_air_climatology_avg = {
            plev: {
                var: arr / count for var, arr in p_data.items()
            }
            for plev, p_data in upper_air_climatology.items()
        }
        
        # Save climatology for this hoy
        output_path = os.path.join(climatology_dir, f"{hoy}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'surface_climatology': surface_climatology_avg,
                'upper_air_climatology': upper_air_climatology_avg,
                'num_samples': count
            }, f)

if __name__ == "__main__":

    for down_size in [2]:
        time_span = ["2019010100", "2022123123"]
        data_dir = f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_{down_size}/HRRR/raw/preload"
        K_days, K_hours = 1, 1
        climatology_dir = os.path.join(data_dir.replace("preload", "climatology"), f"{time_span[0]}-{time_span[1]}_{K_days}_{K_hours}")
        
        time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=time_span[0], end=time_span[1], freq='h'))
        
        # Calculate climatology with sliding window (K_days=2, K_hours=2)
        calculate_climatology(data_dir, climatology_dir, time_list, K_days=K_days, K_hours=K_hours)