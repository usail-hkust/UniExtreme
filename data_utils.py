from os import listdir
from os.path import join
import re
import os
import ast
import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Literal
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import pad
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose
import torch
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import logging
logging.getLogger("cfgrib").setLevel(logging.ERROR)


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

class BatchNormalize:
    
    def __init__(self, mean, std):

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        
        self.std[self.std == 0] = 1e-20

    def __call__(self, tensor):

        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)

def surface_inv_transform(mean_path, std_path):
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

    invTrans = Compose([
        BatchNormalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
        BatchNormalize([-x for x in mean_seq], [1.] * len(std_seq))
    ])
    return invTrans, channel_seq

def upper_air_inv_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f)

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

        invTrans = Compose([
            BatchNormalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
            BatchNormalize([-x for x in mean_seq], [1.] * len(std_seq))
        ])
        normalize[pl] = invTrans

    return normalize, variables, pLevels

def cal_mean_std(data_dir, mean_std_dir, time_list):

    os.makedirs(mean_std_dir, exist_ok=True)
    surface_mean_dict = {'u10': {"sum": 0., "num": 0}, 'v10': {"sum": 0., "num": 0}, 't2m': {"sum": 0., "num": 0}, 'msl': {"sum": 0., "num": 0}}
    surface_std_dict = {'u10': {"sum": 0., "num": 0}, 'v10': {"sum": 0., "num": 0}, 't2m': {"sum": 0., "num": 0}, 'msl': {"sum": 0., "num": 0}}
    upper_air_mean_dict = {plevel: {'z': {"sum": 0., "num": 0}, 'q': {"sum": 0., "num": 0}, 't': {"sum": 0., "num": 0}, 'u': {"sum": 0., "num": 0}, 'v': {"sum": 0., "num": 0}} 
                            for plevel in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]}
    upper_air_std_dict = {plevel: {'z': {"sum": 0., "num": 0}, 'q': {"sum": 0., "num": 0}, 't': {"sum": 0., "num": 0}, 'u': {"sum": 0., "num": 0}, 'v': {"sum": 0., "num": 0}} 
                            for plevel in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]}
    
    print("Calculating Mean...")
    for time in tqdm(time_list):
        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = join(data_dir, time_str[:8])
        var_path_list = [f for f in listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
            var_name = list(d.data_vars)[0]
            data = d[var_name].values
            assert data.shape == (530, 900)
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
            else:
                plevel = None
            d.close()
            
            var_name = COMPLETE_VARIABLE_MAP[var_name]
            if plevel is None:
                surface_mean_dict[var_name]["sum"] += np.nansum(data)
                surface_mean_dict[var_name]["num"] += (~np.isnan(data)).sum()
            else:
                upper_air_mean_dict[plevel][var_name]["sum"] += np.nansum(data)
                upper_air_mean_dict[plevel][var_name]["num"] += (~np.isnan(data)).sum()
    surface_mean_dict = {var_name: vs["sum"] / vs["num"] for var_name, vs in surface_mean_dict.items()}
    upper_air_mean_dict = {plevel: {var_name: vs["sum"] / vs["num"] for var_name, vs in vs_l.items()} for plevel, vs_l in upper_air_mean_dict.items()}
    with open(join(mean_std_dir, 'surface_mean.pkl'), 'wb') as f:
        pickle.dump(surface_mean_dict, f)
    with open(join(mean_std_dir, 'upper_air_mean.pkl'), 'wb') as f:
        pickle.dump(upper_air_mean_dict, f)
    
    print("Calculating Std...")
    for time in tqdm(time_list):
        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = join(data_dir, time_str[:8])
        var_path_list = [f for f in listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
            var_name = list(d.data_vars)[0]
            data = d[var_name].values
            assert data.shape == (530, 900)
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
            else:
                plevel = None
            d.close()
            
            var_name = COMPLETE_VARIABLE_MAP[var_name]
            if plevel is None:
                mean = surface_mean_dict[var_name]
                surface_std_dict[var_name]["sum"] += np.nansum((data - mean) ** 2)
                surface_std_dict[var_name]["num"] += (~np.isnan(data)).sum()
            else:
                mean = upper_air_mean_dict[plevel][var_name]
                upper_air_std_dict[plevel][var_name]["sum"] += np.nansum((data - mean) ** 2)
                upper_air_std_dict[plevel][var_name]["num"] += (~np.isnan(data)).sum()
    surface_std_dict = {var_name: np.sqrt(vs["sum"] / vs["num"]) for var_name, vs in surface_std_dict.items()}
    upper_air_std_dict = {plevel: {var_name: np.sqrt(vs["sum"] / vs["num"]) for var_name, vs in vs_l.items()} for plevel, vs_l in upper_air_std_dict.items()}
    with open(join(mean_std_dir, 'surface_std.pkl'), 'wb') as f:
        pickle.dump(surface_std_dict, f)
    with open(join(mean_std_dir, 'upper_air_std.pkl'), 'wb') as f:
        pickle.dump(upper_air_std_dict, f)
   
def get_extreme_data(data_dir, start_year, end_year):
    
    df_list = []
    for YEAR in range(start_year, end_year + 1):
        save_path = os.path.join(data_dir, f'{YEAR}.csv')
        df = pd.read_csv(save_path, parse_dates=['begin_time', 'end_time'])
        df['span'] = pd.to_timedelta(df['span'])
        df_list.append(df)
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.sort_values(by='begin_time').reset_index(drop=True)
    
    return df

def _get_preloaded_time_data(time, data_dir, surface_variables, upper_air_variables, upper_air_pLevels, 
                             surface_transform, upper_air_transform, use_trans=True):
    
    time_str = datetime.strftime(time, '%Y%m%d%H')

    preload_dir = data_dir.replace("hrrr", "preload")
    os.makedirs(preload_dir, exist_ok=True)
    preload_path = os.path.join(preload_dir, f"{time_str}.pkl")

    try: 
        assert os.path.exists(preload_path)
        with open(preload_path, 'rb') as f:
            data = pickle.load(f)
        surface_data_dict = data['surface_data_dict']
        upper_air_data_dict = data['upper_air_data_dict']
    except:
        raise ValueError
        try:
            time_dir = os.path.join(data_dir, time_str[:8])
            var_path_list = [f for f in os.listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
            assert len(var_path_list) == 69

            surface_data_dict = {}
            upper_air_data_dict = {e: {} for e in upper_air_pLevels}

            for f in var_path_list:
                d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
                var_name = list(d.data_vars)[0]
                if 'isobaricInhPa' in d[var_name].coords:
                    plevel = int(d[var_name].coords['isobaricInhPa'])
                    upper_air_data_dict[plevel][COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
                else:
                    surface_data_dict[COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
                d.close()

            with open(preload_path, 'wb') as f:
                pickle.dump({
                    'surface_data_dict': surface_data_dict,
                    'upper_air_data_dict': upper_air_data_dict
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print(time)
            return torch.zeros(4, 1059, 1799), torch.zeros(5, 13, 1059, 1799)
        
    surface_data = np.stack([surface_data_dict[v] for v in surface_variables], axis=0)
    surface_data = torch.from_numpy(surface_data.astype(np.float32))
    if use_trans:
        surface_data = surface_transform(surface_data)

    if use_trans:
        upper_air_data = [upper_air_transform[pl](
            torch.from_numpy(np.stack(
                [upper_air_data_dict[pl][v] for v in upper_air_variables], axis=0).astype(np.float32)))
                            for pl in upper_air_pLevels]
    else:
        upper_air_data = [torch.from_numpy(np.stack(
            [upper_air_data_dict[pl][v] for v in upper_air_variables], axis=0).astype(np.float32))
                            for pl in upper_air_pLevels]
    upper_air_data = torch.stack(upper_air_data, dim=1)

    return surface_data, upper_air_data

class NOAADataComplete(Dataset):
    
    def __init__(self, data_dir, time_span, flag, train_mean_std_dir=None, debug_flag=False, preload_flag=False):
        
        super().__init__()
        self.data_dir = data_dir
        self.flag = flag
        self.preload_flag = preload_flag
        mean_std_dir = join(data_dir.replace("hrrr", "mean_std"), f"{time_span[0]}-{time_span[1]}")
        # mean_std_dir = join(data_dir.replace("hrrr", "mean_std"), f"debug")
        time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=time_span[0], end=time_span[1], freq='h'))
        self.time_list = time_list
        
        if debug_flag:
            mean_std_dir = join(data_dir.replace("hrrr", "mean_std"), "debug")
            self.mean_std_dir = mean_std_dir
            raise ValueError
        else:
            if flag == "train":
                if not os.path.exists(join(mean_std_dir, "surface_std.pkl")):
                    raise ValueError
                self.mean_std_dir = mean_std_dir
            else:
                mean_std_dir = train_mean_std_dir
                assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))

    def __getitem__(self, index):
        
        surface_t, upper_air_t = _get_preloaded_time_data(self.time_list[index], self.data_dir, 
                                                              self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                              self.surface_transform, self.upper_air_transform)
        if self.preload_flag:
            return surface_t, upper_air_t
        surface_t_1, upper_air_t_1 = _get_preloaded_time_data(self.time_list[index + 1], self.data_dir, 
                                                              self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                              self.surface_transform, self.upper_air_transform, 
                                                              use_trans=True)
        if self.flag == "train":
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
                eval(datetime.strftime(self.time_list[index], '%Y%m%d%H')), eval(datetime.strftime(self.time_list[index + 1], '%Y%m%d%H'))])
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
            eval(datetime.strftime(self.time_list[index], '%Y%m%d%H')), eval(datetime.strftime(self.time_list[index + 1], '%Y%m%d%H'))])

    def _get_time_data(self, time, use_trans=True):

        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = join(self.data_dir, time_str[:8])
        var_path_list = [f for f in listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        
        surface_data_dict = {}
        upper_air_data_dict = {e: {} for e in self.upper_air_pLevels}
        
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
            var_name = list(d.data_vars)[0]
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
                upper_air_data_dict[plevel][COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            else:
                surface_data_dict[COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            d.close()
        
        surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
        surface_data = torch.from_numpy(surface_data.astype(np.float32))
        if use_trans:
            surface_data = self.surface_transform(surface_data)
        
        if use_trans:
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))) 
                            for pl in self.upper_air_pLevels]
        else:
            upper_air_data = [torch.from_numpy(np.stack(
                [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))
                            for pl in self.upper_air_pLevels]
        upper_air_data = torch.stack(upper_air_data, dim=1)
        
        return surface_data, upper_air_data

    def __len__(self):
        return len(self.time_list) - 1 if not self.preload_flag else len(self.time_list)

    def get_lat_lon(self):
        
        data = np.load("/home/nihang/ExtremeWeatherForecast/HR-Extreme/index_files/latlon_grid_hrrr.npy") # (530, 900, 2)
        
        return data

class NOAAExtremeDataComplete(Dataset):
    
    def __init__(self, data_dir, time_span, flag, history=0, horizon=1, stride_mode="begin_end", early_stride=0, use_merge=True, 
                 train_mean_std_dir=None, debug_flag=False, only_begin=False, use_all=False):
        
        super().__init__()
        self.data_dir = data_dir
        self.flag = flag
        self.history = history
        self.horizon = horizon
        assert stride_mode in ["begin", "begin_end", "begin_span"]
        self.stride_mode = stride_mode
        self.early_stride = early_stride
        self.use_merge = use_merge
        mean_std_dir = train_mean_std_dir
        assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))
        self.mean_std_dir = mean_std_dir
        self.time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=self.time_span[0], end=self.time_span[1], freq='h'))
        self.time_list = time_list
        self.only_begin = only_begin
        self.use_all = use_all

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))
        
        extreme_data = get_extreme_data(data_dir.replace("raw/hrrr", "extreme/merged"), eval(time_span[0][:4]), eval(time_span[-1][:4]))
        self._init_extreme_indexes(extreme_data)
        
    def _init_extreme_indexes(self, df):
        
        sample_dict_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row["end_time"] + pd.Timedelta(hours=1) > pd.to_datetime("2024123123", format='%Y%m%d%H'):
                continue
            if row["begin_time"] < self.time_list[0] or row["end_time"] > self.time_list[-1]:
                continue
            sample_dict = { 
                "begin_time": row["begin_time"], 
                "end_time": row["end_time"], 
                "span": row["span"], 
                "types": row["type"].split('+'), 
                "bbox": row["bounding_box"].split('_')
            }
            if self.stride_mode == "begin":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'], freq='h')
            elif self.stride_mode == "begin_end":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['end_time'], freq='h')
            elif self.stride_mode == "begin_span":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'] + row["span"], freq='h')               
            else:
                raise ValueError
            stride_list = list(range(-self.early_stride, len(span) - self.early_stride))
            
            time_instances = []
            if self.only_begin:
                t = row['begin_time']
                time_instances.append(pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            else:
                for t in span:
                    time_instances.append(
                        pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            sample_dict["time_instances"] = time_instances
            sample_dict["stride_list"] = stride_list
            assert len(time_instances) == len(stride_list)
            sample_dict_list.append(sample_dict)
        print("# Extreme Events: ", len(sample_dict_list))
        
        extreme_instances = []
        for d in sample_dict_list:
            for idx, tt in enumerate(d["time_instances"]):
                extreme_instances.append({
                    "times": tt, 
                    "stride": d["stride_list"][idx], 
                    "begin_time": d["begin_time"], 
                    "end_time": d["end_time"], 
                    "span": d["span"], 
                    "types": d["types"], 
                    "bbox": d["bbox"]
                })
        print("# Extreme Instances: ", len(extreme_instances))
        self.extreme_instances = extreme_instances
        
        merged_dict = {}
        for instance in self.extreme_instances:
            start_time = instance["times"][0]
            if start_time not in merged_dict:
                merged_dict[start_time] = {
                    "times": instance["times"], 
                    "strides": [],
                    "begin_times": [],
                    "end_times": [],
                    "spans": [],
                    "types_list": [],
                    "bboxes": [],
                }
            merged_dict[start_time]["strides"].append(instance["stride"])
            merged_dict[start_time]["begin_times"].append(instance["begin_time"])
            merged_dict[start_time]["end_times"].append(instance["end_time"])
            merged_dict[start_time]["spans"].append(instance["span"])
            merged_dict[start_time]["types_list"].append(instance["types"])
            merged_dict[start_time]["bboxes"].append(instance["bbox"])
        self.merged_extreme_instances = [
            {
                "times": details["times"],
                "strides": details["strides"],
                "begin_times": details["begin_times"],
                "end_times": details["end_times"],
                "spans": details["spans"],
                "types_list": details["types_list"],
                "bboxes": details["bboxes"],
            }
            for start_time, details in merged_dict.items()
        ]
        print("# Merged Extreme Instances: ", len(self.merged_extreme_instances))
        
        counts_path = os.path.join(self.data_dir.replace("raw/hrrr", "extreme/counts"), f"{self.time_span}")
        if os.path.exists(counts_path):
            data = torch.load(counts_path, weights_only=False)
            class_counts, event_counts = data["class_counts"], data["event_counts"]
        else:
            class_counts = {}
            event_counts = 0
            normal_counts = 0
            for inst in tqdm(self.merged_extreme_instances):
                ext_bbox = torch.zeros((530, 900))
                for idx in range(len(inst["bboxes"])):
                    bbox = inst["bboxes"][idx]
                    event_types = inst["types_list"][idx]
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    ext_bbox[x_min: x_max, y_min: y_max] = 1.
                    bbox_area = (y_max - y_min) * (x_max - x_min)
                    event_counts += bbox_area
                    for event_type in event_types:
                        if event_type not in class_counts:
                            class_counts[event_type] = 0
                        class_counts[event_type] += bbox_area
                normal_counts += ext_bbox.numel() - ext_bbox.sum()
            class_counts['normal'] = normal_counts
            torch.save(
                {"class_counts": class_counts, "event_counts": event_counts}, counts_path)
        self.class_counts = class_counts
        self.event_counts = event_counts
        
        if self.use_all:
            self.extreme_start_time_list = [instance["times"][0] for instance in self.merged_extreme_instances]
            self.all_start_time_list = self.time_list[:-1]
            self.normal_start_time_list = list(set(self.all_start_time_list) - set(self.extreme_start_time_list))

    def __getitem__(self, index):
        
        if self.use_all:
            pass
        else:
            instance = self.merged_extreme_instances[index] if self.use_merge else self.extreme_instances[index]
            assert len(instance["times"]) == 2
            surface_t, upper_air_t = _get_preloaded_time_data(instance["times"][0], self.data_dir, 
                                                                self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                                self.surface_transform, self.upper_air_transform)
            surface_t_1, upper_air_t_1 = _get_preloaded_time_data(instance["times"][1], self.data_dir, 
                                                                self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                                self.surface_transform, self.upper_air_transform, 
                                                                use_trans=True)
            target_bbox = torch.zeros((530, 900))
            if self.use_merge:
                for bbox in instance["bboxes"]:
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    target_bbox[x_min: x_max, y_min: y_max] = 1.
                types = instance["types_list"]
                stride = instance["strides"]
                bbox = instance["bboxes"]
            else:
                y_min, x_min, y_max, x_max = [eval(e) for e in instance["bbox"]]
                target_bbox[x_min: x_max, y_min: y_max] = 1.
                types = instance["types"]
                stride = instance["stride"]
                bbox = instance["bbox"]
            
            if self.flag == "train":
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
                eval(datetime.strftime(instance["times"][0], '%Y%m%d%H')), eval(datetime.strftime(instance["times"][1], '%Y%m%d%H'))])
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
                eval(datetime.strftime(instance["times"][0], '%Y%m%d%H')), eval(datetime.strftime(instance["times"][1], '%Y%m%d%H'))])

    def _get_time_data(self, time, use_trans=True):

        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = join(self.data_dir, time_str[:8])
        var_path_list = [f for f in listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        
        surface_data_dict = {}
        upper_air_data_dict = {e: {} for e in self.upper_air_pLevels}
        
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
            var_name = list(d.data_vars)[0]
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
                upper_air_data_dict[plevel][COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            else:
                surface_data_dict[COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            d.close()
        
        surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
        surface_data = torch.from_numpy(surface_data.astype(np.float32))
        if use_trans:
            surface_data = self.surface_transform(surface_data)
        
        if use_trans:
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))) 
                            for pl in self.upper_air_pLevels]
        else:
            upper_air_data = [torch.from_numpy(np.stack(
                [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))
                            for pl in self.upper_air_pLevels]
        upper_air_data = torch.stack(upper_air_data, dim=1)
        
        return surface_data, upper_air_data

    def __len__(self):
        
        return (len(self.merged_extreme_instances) + len(self.normal_start_time_list) if self.use_all else len(self.merged_extreme_instances)) \
            if self.use_merge else len(self.extreme_instances)

    def get_lat_lon(self):
        
        data = np.load("/home/nihang/ExtremeWeatherForecast/HR-Extreme/index_files/latlon_grid_hrrr.npy") # (530, 900, 2)
        
        return data

def extreme_collate(batch):
    
    collated = []
    for i in range(len(batch[0])):
        field_samples = [sample[i] for sample in batch]
        if isinstance(field_samples[0], torch.Tensor):
            shapes = [s.shape for s in field_samples]
            assert all(s == shapes[0] for s in shapes)
            collated_field = torch.stack(field_samples, dim=0)
        else:
            collated_field = field_samples
        collated.append(collated_field)
    
    return tuple(collated)

class NOAACompletePlusExtremeDataComplete(Dataset):
    
    def __init__(self, data_dir, time_span, flag, history=0, horizon=1, stride_mode="begin_end", early_stride=0, use_merge=True, 
                 train_mean_std_dir=None, CLIMATOLOGY_DIR=None, debug_flag=False, only_begin=False, use_all=False):
        
        super().__init__()
        self.data_dir = data_dir
        self.flag = flag
        self.history = history
        self.horizon = horizon
        assert stride_mode in ["begin", "begin_end", "begin_span"]
        self.stride_mode = stride_mode
        self.early_stride = early_stride
        self.use_merge = use_merge
        mean_std_dir = train_mean_std_dir
        assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))
        self.mean_std_dir = mean_std_dir
        self.time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=self.time_span[0], end=self.time_span[1], freq='h'))
        self.time_list = time_list
        self.only_begin = only_begin
        self.use_all = use_all
        self.CLIMATOLOGY_DIR = CLIMATOLOGY_DIR

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))
        
        extreme_data = get_extreme_data(data_dir.replace("raw/hrrr", "extreme/merged"), eval(time_span[0][:4]), eval(time_span[-1][:4]))
        self._init_extreme_indexes(extreme_data)
        
    def _init_extreme_indexes(self, df):
        
        sample_dict_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row["end_time"] + pd.Timedelta(hours=1) > pd.to_datetime("2024123123", format='%Y%m%d%H'):
                continue
            if row["begin_time"] < self.time_list[0] or row["end_time"] > self.time_list[-1]:
                continue
            sample_dict = { 
                "begin_time": row["begin_time"], 
                "end_time": row["end_time"], 
                "span": row["span"], 
                "types": row["type"].split('+'), 
                "bbox": row["bounding_box"].split('_')
            }
            if self.stride_mode == "begin":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'], freq='h')
            elif self.stride_mode == "begin_end":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['end_time'], freq='h')
            elif self.stride_mode == "begin_span":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'] + row["span"], freq='h')               
            else:
                raise ValueError
            stride_list = list(range(-self.early_stride, len(span) - self.early_stride))
            
            time_instances = []
            if self.only_begin:
                t = row['begin_time']
                time_instances.append(pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            else:
                for t in span:
                    time_instances.append(
                        pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            sample_dict["time_instances"] = time_instances
            sample_dict["stride_list"] = stride_list
            assert len(time_instances) == len(stride_list)
            sample_dict_list.append(sample_dict)
        print("# Extreme Events: ", len(sample_dict_list))
        
        extreme_instances = []
        for d in sample_dict_list:
            for idx, tt in enumerate(d["time_instances"]):
                extreme_instances.append({
                    "times": tt, 
                    "stride": d["stride_list"][idx], 
                    "begin_time": d["begin_time"], 
                    "end_time": d["end_time"], 
                    "span": d["span"], 
                    "types": d["types"], 
                    "bbox": d["bbox"]
                })
        print("# Extreme Instances: ", len(extreme_instances))
        self.extreme_instances = extreme_instances
        
        merged_dict = {}
        for instance in self.extreme_instances:
            start_time = instance["times"][0]
            if start_time not in merged_dict:
                merged_dict[start_time] = {
                    "times": instance["times"], 
                    "strides": [],
                    "begin_times": [],
                    "end_times": [],
                    "spans": [],
                    "types_list": [],
                    "bboxes": [],
                }
            merged_dict[start_time]["strides"].append(instance["stride"])
            merged_dict[start_time]["begin_times"].append(instance["begin_time"])
            merged_dict[start_time]["end_times"].append(instance["end_time"])
            merged_dict[start_time]["spans"].append(instance["span"])
            merged_dict[start_time]["types_list"].append(instance["types"])
            merged_dict[start_time]["bboxes"].append(instance["bbox"])
        self.merged_extreme_instances = [
            {
                "times": details["times"],
                "strides": details["strides"],
                "begin_times": details["begin_times"],
                "end_times": details["end_times"],
                "spans": details["spans"],
                "types_list": details["types_list"],
                "bboxes": details["bboxes"],
            }
            for start_time, details in merged_dict.items()
        ]
        print("# Merged Extreme Instances: ", len(self.merged_extreme_instances))
        
        counts_path = os.path.join(self.data_dir.replace("raw/hrrr", "extreme/counts"), f"{self.time_span}")
        if os.path.exists(counts_path):
            data = torch.load(counts_path, weights_only=False)
            class_counts, event_counts = data["class_counts"], data["event_counts"]
        else:
            class_counts = {}
            event_counts = 0
            normal_counts = 0
            for inst in tqdm(self.merged_extreme_instances):
                ext_bbox = torch.zeros((530, 900))
                for idx in range(len(inst["bboxes"])):
                    bbox = inst["bboxes"][idx]
                    event_types = inst["types_list"][idx]
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    ext_bbox[x_min: x_max, y_min: y_max] = 1.
                    bbox_area = (y_max - y_min) * (x_max - x_min)
                    event_counts += bbox_area
                    for event_type in event_types:
                        if event_type not in class_counts:
                            class_counts[event_type] = 0
                        class_counts[event_type] += bbox_area
                normal_counts += ext_bbox.numel() - ext_bbox.sum()
            class_counts['normal'] = normal_counts
            torch.save(
                {"class_counts": class_counts, "event_counts": event_counts}, counts_path)
        self.class_counts = class_counts
        self.event_counts = event_counts
        
        self.merged_extreme_instances_times = [d["times"][0] for d in self.merged_extreme_instances]
        assert len(self.merged_extreme_instances_times) == len(set(self.merged_extreme_instances_times))
        # print(self.merged_extreme_instances_times)
        # exit(-1)

    def __getitem__(self, index):
        
        index_time, index_time_next = self.time_list[index], self.time_list[index + 1]
        try:
            extreme_index = self.merged_extreme_instances_times.index(index_time)
            # print(extreme_index)
        except:
            extreme_index = None
        
        surface_t, upper_air_t = _get_preloaded_time_data(index_time, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform)
        surface_t_1, upper_air_t_1 = _get_preloaded_time_data(index_time_next, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform, 
                                                            use_trans=True)
        target_bbox = torch.zeros((530, 900))
        if extreme_index is None:
            bbox, types, stride = None, None, None
        else:
            instance = self.merged_extreme_instances[extreme_index]
            assert self.use_merge
            for bbox in instance["bboxes"]:
                y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                target_bbox[x_min: x_max, y_min: y_max] = 1.
            types = instance["types_list"]
            stride = instance["strides"]
            bbox = instance["bboxes"]
            
        if self.flag != "train" and self.CLIMATOLOGY_DIR is not None:
            hour_of_year = datetime.strftime(index_time_next, '%Y%m%d%H')[4:]
            clim_path = os.path.join(self.CLIMATOLOGY_DIR, f"{hour_of_year}.pkl")
            with open(clim_path, 'rb') as f:
                climatology = pickle.load(f)
            surface_data_dict, upper_air_data_dict = climatology['surface_climatology'], climatology['upper_air_climatology']
            surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
            surface_data = torch.from_numpy(surface_data.astype(np.float32))
            surface_data = self.surface_transform(surface_data)
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32)))
                                for pl in self.upper_air_pLevels]
            upper_air_data = torch.stack(upper_air_data, dim=1)
            climatology = torch.concat([surface_data, upper_air_data.flatten(0, 1)], dim=0)
        else:
            climatology = None
        
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
            eval(datetime.strftime(index_time, '%Y%m%d%H')), eval(datetime.strftime(index_time_next, '%Y%m%d%H'))]), climatology

    def __len__(self):
        
        return len(self.time_list) - 1

class NOAACompletePlusExtremeDataCompleteAutoReg(Dataset):
    
    def __init__(self, data_dir, time_span, flag, history=0, horizon=1, stride_mode="begin_end", early_stride=0, use_merge=True, 
                 train_mean_std_dir=None, CLIMATOLOGY_DIR=None, debug_flag=False, only_begin=False, use_all=False, autoreg_horizon=1):
        
        super().__init__()
        self.data_dir = data_dir
        self.flag = flag
        self.history = history
        self.horizon = horizon
        self.autoreg_horizon = autoreg_horizon
        assert stride_mode in ["begin", "begin_end", "begin_span"]
        self.stride_mode = stride_mode
        self.early_stride = early_stride
        self.use_merge = use_merge
        mean_std_dir = train_mean_std_dir
        assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))
        self.mean_std_dir = mean_std_dir
        self.time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=self.time_span[0], end=self.time_span[1], freq='h'))
        self.time_list = time_list
        self.only_begin = only_begin
        self.use_all = use_all
        self.CLIMATOLOGY_DIR = CLIMATOLOGY_DIR

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))
        
        extreme_data = get_extreme_data(data_dir.replace("raw/hrrr", "extreme/merged"), eval(time_span[0][:4]), eval(time_span[-1][:4]))
        self._init_extreme_indexes(extreme_data)
        
    def _init_extreme_indexes(self, df):
        
        sample_dict_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row["end_time"] + pd.Timedelta(hours=1) > pd.to_datetime("2024123123", format='%Y%m%d%H'):
                continue
            if row["begin_time"] < self.time_list[0] or row["end_time"] > self.time_list[-1]:
                continue
            sample_dict = { 
                "begin_time": row["begin_time"], 
                "end_time": row["end_time"], 
                "span": row["span"], 
                "types": row["type"].split('+'), 
                "bbox": row["bounding_box"].split('_')
            }
            if self.stride_mode == "begin":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'], freq='h')
            elif self.stride_mode == "begin_end":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['end_time'], freq='h')
            elif self.stride_mode == "begin_span":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'] + row["span"], freq='h')               
            else:
                raise ValueError
            stride_list = list(range(-self.early_stride, len(span) - self.early_stride))
            
            time_instances = []
            if self.only_begin:
                t = row['begin_time']
                time_instances.append(pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            else:
                for t in span:
                    time_instances.append(
                        pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            sample_dict["time_instances"] = time_instances
            sample_dict["stride_list"] = stride_list
            assert len(time_instances) == len(stride_list)
            sample_dict_list.append(sample_dict)
        print("# Extreme Events: ", len(sample_dict_list))
        
        extreme_instances = []
        for d in sample_dict_list:
            for idx, tt in enumerate(d["time_instances"]):
                extreme_instances.append({
                    "times": tt, 
                    "stride": d["stride_list"][idx], 
                    "begin_time": d["begin_time"], 
                    "end_time": d["end_time"], 
                    "span": d["span"], 
                    "types": d["types"], 
                    "bbox": d["bbox"]
                })
        print("# Extreme Instances: ", len(extreme_instances))
        self.extreme_instances = extreme_instances
        
        merged_dict = {}
        for instance in self.extreme_instances:
            start_time = instance["times"][0]
            if start_time not in merged_dict:
                merged_dict[start_time] = {
                    "times": instance["times"], 
                    "strides": [],
                    "begin_times": [],
                    "end_times": [],
                    "spans": [],
                    "types_list": [],
                    "bboxes": [],
                }
            merged_dict[start_time]["strides"].append(instance["stride"])
            merged_dict[start_time]["begin_times"].append(instance["begin_time"])
            merged_dict[start_time]["end_times"].append(instance["end_time"])
            merged_dict[start_time]["spans"].append(instance["span"])
            merged_dict[start_time]["types_list"].append(instance["types"])
            merged_dict[start_time]["bboxes"].append(instance["bbox"])
        self.merged_extreme_instances = [
            {
                "times": details["times"],
                "strides": details["strides"],
                "begin_times": details["begin_times"],
                "end_times": details["end_times"],
                "spans": details["spans"],
                "types_list": details["types_list"],
                "bboxes": details["bboxes"],
            }
            for start_time, details in merged_dict.items()
        ]
        print("# Merged Extreme Instances: ", len(self.merged_extreme_instances))
        
        counts_path = os.path.join(self.data_dir.replace("raw/hrrr", "extreme/counts"), f"{self.time_span}")
        if os.path.exists(counts_path):
            data = torch.load(counts_path, weights_only=False)
            class_counts, event_counts = data["class_counts"], data["event_counts"]
        else:
            class_counts = {}
            event_counts = 0
            normal_counts = 0
            for inst in tqdm(self.merged_extreme_instances):
                ext_bbox = torch.zeros((530, 900))
                for idx in range(len(inst["bboxes"])):
                    bbox = inst["bboxes"][idx]
                    event_types = inst["types_list"][idx]
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    ext_bbox[x_min: x_max, y_min: y_max] = 1.
                    bbox_area = (y_max - y_min) * (x_max - x_min)
                    event_counts += bbox_area
                    for event_type in event_types:
                        if event_type not in class_counts:
                            class_counts[event_type] = 0
                        class_counts[event_type] += bbox_area
                normal_counts += ext_bbox.numel() - ext_bbox.sum()
            class_counts['normal'] = normal_counts
            torch.save(
                {"class_counts": class_counts, "event_counts": event_counts}, counts_path)
        self.class_counts = class_counts
        self.event_counts = event_counts
        
        self.merged_extreme_instances_times = [d["times"][0] for d in self.merged_extreme_instances]
        assert len(self.merged_extreme_instances_times) == len(set(self.merged_extreme_instances_times))
        # print(self.merged_extreme_instances_times)
        # exit(-1)

    def __getitem__(self, index):
        
        index_time, index_time_next = self.time_list[index], self.time_list[index + self.autoreg_horizon]
        try:
            extreme_index = self.merged_extreme_instances_times.index(self.time_list[index + self.autoreg_horizon - 1])
            # print(extreme_index)
        except:
            extreme_index = None
        
        surface_t, upper_air_t = _get_preloaded_time_data(index_time, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform)
        surface_t_1, upper_air_t_1 = _get_preloaded_time_data(index_time_next, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform, 
                                                            use_trans=True)
        target_bbox = torch.zeros((530, 900))
        if extreme_index is None:
            bbox, types, stride = None, None, None
        else:
            instance = self.merged_extreme_instances[extreme_index]
            assert self.use_merge
            for bbox in instance["bboxes"]:
                y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                target_bbox[x_min: x_max, y_min: y_max] = 1.
            types = instance["types_list"]
            stride = instance["strides"]
            bbox = instance["bboxes"]
            
        if self.flag != "train" and self.CLIMATOLOGY_DIR is not None:
            hour_of_year = datetime.strftime(index_time_next, '%Y%m%d%H')[4:]
            clim_path = os.path.join(self.CLIMATOLOGY_DIR, f"{hour_of_year}.pkl")
            with open(clim_path, 'rb') as f:
                climatology = pickle.load(f)
            surface_data_dict, upper_air_data_dict = climatology['surface_climatology'], climatology['upper_air_climatology']
            surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
            surface_data = torch.from_numpy(surface_data.astype(np.float32))
            surface_data = self.surface_transform(surface_data)
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32)))
                                for pl in self.upper_air_pLevels]
            upper_air_data = torch.stack(upper_air_data, dim=1)
            climatology = torch.concat([surface_data, upper_air_data.flatten(0, 1)], dim=0)
        else:
            climatology = None
        
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
            eval(datetime.strftime(index_time, '%Y%m%d%H')), eval(datetime.strftime(index_time_next, '%Y%m%d%H'))]), climatology

    def __len__(self):
        
        return len(self.time_list) - self.autoreg_horizon


class NOAACompletePlusExtremeDataCompleteTestNWP(Dataset):
    
    def __init__(self, data_dir, time_span, flag, history=0, horizon=1, stride_mode="begin_end", early_stride=0, use_merge=True, 
                 train_mean_std_dir=None, CLIMATOLOGY_DIR=None, debug_flag=False, only_begin=False, use_all=False):
        
        super().__init__()
        self.data_dir = data_dir
        self.nwp_data_dir = self.data_dir.replace("/raw/", "/NWP/WRF-ARW/")
        self.flag = flag
        self.history = history
        self.horizon = horizon
        assert stride_mode in ["begin", "begin_end", "begin_span"]
        self.stride_mode = stride_mode
        self.early_stride = early_stride
        self.use_merge = use_merge
        mean_std_dir = train_mean_std_dir
        assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))
        self.mean_std_dir = mean_std_dir
        self.time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=self.time_span[0], end=self.time_span[1], freq='h'))
        self.time_list = time_list
        self.only_begin = only_begin
        self.use_all = use_all
        self.CLIMATOLOGY_DIR = CLIMATOLOGY_DIR
        
        def check_nwp_path(time):
            time_str = datetime.strftime(time, '%Y%m%d%H')
            preload_dir = self.nwp_data_dir.replace("hrrr", "preload")
            preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
            return os.path.exists(preload_path)
        # print(len(self.time_list))
        self.time_list = [t for t in self.time_list if check_nwp_path(t) and check_nwp_path(t + timedelta(hours=1))]
        # print(len(self.time_list))
        # exit(-1)

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))
        
        extreme_data = get_extreme_data(data_dir.replace("raw/hrrr", "extreme/merged"), eval(time_span[0][:4]), eval(time_span[-1][:4]))
        self._init_extreme_indexes(extreme_data)
        
    def _init_extreme_indexes(self, df):
        
        sample_dict_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row["end_time"] + pd.Timedelta(hours=1) > pd.to_datetime("2024123123", format='%Y%m%d%H'):
                continue
            if row["begin_time"] < self.time_list[0] or row["end_time"] > self.time_list[-1]:
                continue
            sample_dict = { 
                "begin_time": row["begin_time"], 
                "end_time": row["end_time"], 
                "span": row["span"], 
                "types": row["type"].split('+'), 
                "bbox": row["bounding_box"].split('_')
            }
            if self.stride_mode == "begin":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'], freq='h')
            elif self.stride_mode == "begin_end":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['end_time'], freq='h')
            elif self.stride_mode == "begin_span":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'] + row["span"], freq='h')               
            else:
                raise ValueError
            stride_list = list(range(-self.early_stride, len(span) - self.early_stride))
            
            time_instances = []
            if self.only_begin:
                t = row['begin_time']
                time_instances.append(pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            else:
                for t in span:
                    time_instances.append(
                        pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            sample_dict["time_instances"] = time_instances
            sample_dict["stride_list"] = stride_list
            assert len(time_instances) == len(stride_list)
            sample_dict_list.append(sample_dict)
        print("# Extreme Events: ", len(sample_dict_list))
        
        extreme_instances = []
        for d in sample_dict_list:
            for idx, tt in enumerate(d["time_instances"]):
                extreme_instances.append({
                    "times": tt, 
                    "stride": d["stride_list"][idx], 
                    "begin_time": d["begin_time"], 
                    "end_time": d["end_time"], 
                    "span": d["span"], 
                    "types": d["types"], 
                    "bbox": d["bbox"]
                })
        print("# Extreme Instances: ", len(extreme_instances))
        self.extreme_instances = extreme_instances
        
        merged_dict = {}
        for instance in self.extreme_instances:
            start_time = instance["times"][0]
            if start_time not in merged_dict:
                merged_dict[start_time] = {
                    "times": instance["times"], 
                    "strides": [],
                    "begin_times": [],
                    "end_times": [],
                    "spans": [],
                    "types_list": [],
                    "bboxes": [],
                }
            merged_dict[start_time]["strides"].append(instance["stride"])
            merged_dict[start_time]["begin_times"].append(instance["begin_time"])
            merged_dict[start_time]["end_times"].append(instance["end_time"])
            merged_dict[start_time]["spans"].append(instance["span"])
            merged_dict[start_time]["types_list"].append(instance["types"])
            merged_dict[start_time]["bboxes"].append(instance["bbox"])
        self.merged_extreme_instances = [
            {
                "times": details["times"],
                "strides": details["strides"],
                "begin_times": details["begin_times"],
                "end_times": details["end_times"],
                "spans": details["spans"],
                "types_list": details["types_list"],
                "bboxes": details["bboxes"],
            }
            for start_time, details in merged_dict.items()
        ]
        print("# Merged Extreme Instances: ", len(self.merged_extreme_instances))
        
        counts_path = os.path.join(self.data_dir.replace("raw/hrrr", "extreme/counts"), f"{self.time_span}")
        if os.path.exists(counts_path):
            data = torch.load(counts_path, weights_only=False)
            class_counts, event_counts = data["class_counts"], data["event_counts"]
        else:
            class_counts = {}
            event_counts = 0
            normal_counts = 0
            for inst in tqdm(self.merged_extreme_instances):
                ext_bbox = torch.zeros((530, 900))
                for idx in range(len(inst["bboxes"])):
                    bbox = inst["bboxes"][idx]
                    event_types = inst["types_list"][idx]
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    ext_bbox[x_min: x_max, y_min: y_max] = 1.
                    bbox_area = (y_max - y_min) * (x_max - x_min)
                    event_counts += bbox_area
                    for event_type in event_types:
                        if event_type not in class_counts:
                            class_counts[event_type] = 0
                        class_counts[event_type] += bbox_area
                normal_counts += ext_bbox.numel() - ext_bbox.sum()
            class_counts['normal'] = normal_counts
            torch.save(
                {"class_counts": class_counts, "event_counts": event_counts}, counts_path)
        self.class_counts = class_counts
        self.event_counts = event_counts
        
        self.merged_extreme_instances_times = [d["times"][0] for d in self.merged_extreme_instances]
        assert len(self.merged_extreme_instances_times) == len(set(self.merged_extreme_instances_times))
        # print(self.merged_extreme_instances_times)
        # exit(-1)

    def __getitem__(self, index):
        
        # index_time, index_time_next = self.time_list[index], self.time_list[index + 1]
        index_time, index_time_next = self.time_list[index], self.time_list[index] + timedelta(hours=1)
        try:
            extreme_index = self.merged_extreme_instances_times.index(index_time)
            # print(extreme_index)
        except:
            extreme_index = None
        
        surface_t, upper_air_t = _get_preloaded_time_data(index_time, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform)
        surface_t_1, upper_air_t_1 = _get_preloaded_time_data(index_time_next, self.data_dir, 
                                                            self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                            self.surface_transform, self.upper_air_transform, 
                                                            use_trans=True)
        target_bbox = torch.zeros((530, 900))
        if extreme_index is None:
            bbox, types, stride = None, None, None
        else:
            instance = self.merged_extreme_instances[extreme_index]
            assert self.use_merge
            for bbox in instance["bboxes"]:
                y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                target_bbox[x_min: x_max, y_min: y_max] = 1.
            types = instance["types_list"]
            stride = instance["strides"]
            bbox = instance["bboxes"]
            
        if self.flag != "train" and self.CLIMATOLOGY_DIR is not None:
            hour_of_year = datetime.strftime(index_time_next, '%Y%m%d%H')[4:]
            clim_path = os.path.join(self.CLIMATOLOGY_DIR, f"{hour_of_year}.pkl")
            with open(clim_path, 'rb') as f:
                climatology = pickle.load(f)
            surface_data_dict, upper_air_data_dict = climatology['surface_climatology'], climatology['upper_air_climatology']
            surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
            surface_data = torch.from_numpy(surface_data.astype(np.float32))
            surface_data = self.surface_transform(surface_data)
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32)))
                                for pl in self.upper_air_pLevels]
            upper_air_data = torch.stack(upper_air_data, dim=1)
            climatology = torch.concat([surface_data, upper_air_data.flatten(0, 1)], dim=0)
        else:
            climatology = None
        
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
            eval(datetime.strftime(index_time, '%Y%m%d%H')), eval(datetime.strftime(index_time_next, '%Y%m%d%H'))]), climatology

    def __len__(self):
        
        return len(self.time_list) - 1


class NOAAExtremeDataTestNWP(Dataset):
    
    def __init__(self, data_dir, time_span, flag, history=0, horizon=1, stride_mode="begin_end", early_stride=0, use_merge=True, 
                 train_mean_std_dir=None, debug_flag=False, only_begin=False, use_all=False, sample_miss=False):
        
        super().__init__()
        self.data_dir = data_dir
        self.nwp_data_dir = self.data_dir.replace("/raw/", "/NWP/WRF-ARW/")
        self.flag = flag
        self.history = history
        self.horizon = horizon
        self.sample_miss = sample_miss
        assert stride_mode in ["begin", "begin_end", "begin_span"]
        self.stride_mode = stride_mode
        self.early_stride = early_stride
        self.use_merge = use_merge
        mean_std_dir = train_mean_std_dir
        assert mean_std_dir is not None and os.path.exists(join(mean_std_dir, "surface_std.pkl"))
        self.mean_std_dir = mean_std_dir
        self.time_span = [datetime.strptime(t, '%Y%m%d%H') for t in time_span]
        time_list = list(pd.date_range(start=self.time_span[0], end=self.time_span[1], freq='h'))
        self.time_list = time_list
        self.only_begin = only_begin
        self.use_all = use_all

        self.surface_transform, self.surface_variables = surface_transform(join(mean_std_dir, "surface_mean.pkl"), 
                                                                           join(mean_std_dir, "surface_std.pkl"))
        self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(mean_std_dir, "upper_air_mean.pkl"), 
                                                                                                         join(mean_std_dir, "upper_air_std.pkl"))
        
        extreme_data = get_extreme_data(data_dir.replace("raw/hrrr", "extreme/merged"), eval(time_span[0][:4]), eval(time_span[-1][:4]))
        self._init_extreme_indexes(extreme_data)
        
    def _init_extreme_indexes(self, df):
        
        sample_dict_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row["end_time"] + pd.Timedelta(hours=1) > pd.to_datetime("2024123123", format='%Y%m%d%H'):
                continue
            if row["begin_time"] < self.time_list[0] or row["end_time"] > self.time_list[-1]:
                continue
            sample_dict = { 
                "begin_time": row["begin_time"], 
                "end_time": row["end_time"], 
                "span": row["span"], 
                "types": row["type"].split('+'), 
                "bbox": row["bounding_box"].split('_')
            }
            if self.stride_mode == "begin":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'], freq='h')
            elif self.stride_mode == "begin_end":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['end_time'], freq='h')
            elif self.stride_mode == "begin_span":
                span = pd.date_range(start=row['begin_time'] - timedelta(hours=self.early_stride), 
                                     end=row['begin_time'] + row["span"], freq='h')               
            else:
                raise ValueError
            stride_list = list(range(-self.early_stride, len(span) - self.early_stride))
            
            time_instances = []
            if self.only_begin:
                t = row['begin_time']
                time_instances.append(pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            else:
                for t in span:
                    time_instances.append(
                        pd.date_range(start=t - timedelta(hours=self.history), 
                                    end=t + timedelta(hours=self.horizon), freq='h'))
            sample_dict["time_instances"] = time_instances
            sample_dict["stride_list"] = stride_list
            assert len(time_instances) == len(stride_list)
            sample_dict_list.append(sample_dict)
        print("# Extreme Events: ", len(sample_dict_list))
        
        extreme_instances = []
        for d in sample_dict_list:
            for idx, tt in enumerate(d["time_instances"]):
                extreme_instances.append({
                    "times": tt, 
                    "stride": d["stride_list"][idx], 
                    "begin_time": d["begin_time"], 
                    "end_time": d["end_time"], 
                    "span": d["span"], 
                    "types": d["types"], 
                    "bbox": d["bbox"]
                })
        print("# Extreme Instances: ", len(extreme_instances))
        self.extreme_instances = extreme_instances
        
        merged_dict = {}
        for instance in self.extreme_instances:
            start_time = instance["times"][0]
            if start_time not in merged_dict:
                merged_dict[start_time] = {
                    "times": instance["times"], 
                    "strides": [],
                    "begin_times": [],
                    "end_times": [],
                    "spans": [],
                    "types_list": [],
                    "bboxes": [],
                }
            merged_dict[start_time]["strides"].append(instance["stride"])
            merged_dict[start_time]["begin_times"].append(instance["begin_time"])
            merged_dict[start_time]["end_times"].append(instance["end_time"])
            merged_dict[start_time]["spans"].append(instance["span"])
            merged_dict[start_time]["types_list"].append(instance["types"])
            merged_dict[start_time]["bboxes"].append(instance["bbox"])
        self.merged_extreme_instances = [
            {
                "times": details["times"],
                "strides": details["strides"],
                "begin_times": details["begin_times"],
                "end_times": details["end_times"],
                "spans": details["spans"],
                "types_list": details["types_list"],
                "bboxes": details["bboxes"],
            }
            for start_time, details in merged_dict.items()
        ]
        print("# Merged Extreme Instances: ", len(self.merged_extreme_instances))
        
        class_counts = {}
        event_counts = 0
        for inst in self.merged_extreme_instances:
            for idx in range(len(inst["bboxes"])):
                bbox = inst["bboxes"][idx]
                event_types = inst["types_list"][idx]
                y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                bbox_area = (y_max - y_min) * (x_max - x_min)
                event_counts += bbox_area
                for event_type in event_types:
                    if event_type not in class_counts:
                        class_counts[event_type] = 0
                    class_counts[event_type] += bbox_area
        self.class_counts = class_counts
        self.event_counts = event_counts
        
        original_num = len(self.merged_extreme_instances)
        def check_nwp_path(inst):
            time = inst["times"][0]
            time_str = datetime.strftime(time, '%Y%m%d%H')
            preload_dir = self.nwp_data_dir.replace("hrrr", "preload")
            preload_path = os.path.join(preload_dir, f"{time_str}.pkl")
            return os.path.exists(preload_path)
        self.merged_extreme_instances = [inst for inst in self.merged_extreme_instances if check_nwp_path(inst)]
        
        if self.sample_miss:
            self.num_miss = original_num - len(self.merged_extreme_instances)

    def __getitem__(self, index):
        
        if self.use_all:
            raise ValueError
        else:
            instance = self.merged_extreme_instances[index] if self.use_merge else self.extreme_instances[index]
            assert len(instance["times"]) == 2
            surface_t, upper_air_t = _get_preloaded_time_data(instance["times"][0], self.nwp_data_dir, 
                                                                self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                                self.surface_transform, self.upper_air_transform)
            surface_t_1, upper_air_t_1 = _get_preloaded_time_data(instance["times"][1], self.data_dir, 
                                                                self.surface_variables, self.upper_air_variables, self.upper_air_pLevels, 
                                                                self.surface_transform, self.upper_air_transform, 
                                                                use_trans=True)
            target_bbox = torch.zeros((530, 900))
            if self.use_merge:
                for bbox in instance["bboxes"]:
                    y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
                    target_bbox[x_min: x_max, y_min: y_max] = 1.
                types = instance["types_list"]
                stride = instance["strides"]
                bbox = instance["bboxes"]
            else:
                y_min, x_min, y_max, x_max = [eval(e) for e in instance["bbox"]]
                target_bbox[x_min: x_max, y_min: y_max] = 1.
                types = instance["types"]
                stride = instance["stride"]
                bbox = instance["bbox"]
            
            if self.flag == "train":
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
                eval(datetime.strftime(instance["times"][0], '%Y%m%d%H')), eval(datetime.strftime(instance["times"][1], '%Y%m%d%H'))])
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, target_bbox, bbox, types, stride, torch.tensor([
                eval(datetime.strftime(instance["times"][0], '%Y%m%d%H')), eval(datetime.strftime(instance["times"][1], '%Y%m%d%H'))])

    def _get_time_data(self, time, use_trans=True):

        time_str = datetime.strftime(time, '%Y%m%d%H')
        time_dir = join(self.data_dir, time_str[:8])
        var_path_list = [f for f in listdir(time_dir) if f.endswith("grib2") and f"t{time_str[-2:]}z." in f]
        assert len(var_path_list) == 69
        
        surface_data_dict = {}
        upper_air_data_dict = {e: {} for e in self.upper_air_pLevels}
        
        for f in var_path_list:
            d = xr.open_dataset(os.path.join(time_dir, f), engine="cfgrib", indexpath='')
            var_name = list(d.data_vars)[0]
            if 'isobaricInhPa' in d[var_name].coords:
                plevel = int(d[var_name].coords['isobaricInhPa'])
                upper_air_data_dict[plevel][COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            else:
                surface_data_dict[COMPLETE_VARIABLE_MAP[var_name]] = d[var_name].values
            d.close()
        
        surface_data = np.stack([surface_data_dict[v] for v in self.surface_variables], axis=0)
        surface_data = torch.from_numpy(surface_data.astype(np.float32))
        if use_trans:
            surface_data = self.surface_transform(surface_data)
        
        if use_trans:
            upper_air_data = [self.upper_air_transform[pl](
                torch.from_numpy(np.stack(
                    [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))) 
                            for pl in self.upper_air_pLevels]
        else:
            upper_air_data = [torch.from_numpy(np.stack(
                [upper_air_data_dict[pl][v] for v in self.upper_air_variables], axis=0).astype(np.float32))
                            for pl in self.upper_air_pLevels]
        upper_air_data = torch.stack(upper_air_data, dim=1)
        
        return surface_data, upper_air_data

    def __len__(self):
        
        return (len(self.merged_extreme_instances) + len(self.normal_start_time_list) if self.use_all else len(self.merged_extreme_instances)) \
            if self.use_merge else len(self.extreme_instances)

    def get_lat_lon(self):
        
        data = np.load("/home/nihang/ExtremeWeatherForecast/HR-Extreme/index_files/latlon_grid_hrrr.npy") # (530, 900, 2)
        
        return data


def parse_extreme_event_filename(filename):
    
    if filename.endswith(".pth"):
        filename = filename[:-4]
    filename = filename.replace("patch_", "patch-")
    
    last_underscore = filename.rfind("_")
    stride_str = filename[last_underscore + 1:]
    stride = int(stride_str)
    
    remaining_part = filename[:last_underscore]
    second_last_underscore = remaining_part.rfind("_")
    bbox = remaining_part[second_last_underscore + 1:]
    
    remaining_part = filename[:second_last_underscore]
    third_last_underscore = remaining_part.rfind("_")
    time_str = remaining_part[third_last_underscore + 1:]
    time = int(time_str)
    
    first_part = remaining_part[:third_last_underscore]
    type_list = first_part
    type_list = ast.literal_eval(type_list)
    
    # print(type_list, time, bbox, stride)
    # exit(-1)
    
    return type_list, time, bbox, stride

class ExtremeEventTripletData(Dataset):
    
    def __init__(self, data_dir, num_pos=1, num_neg=6, only_begin=False, epoch_seed=0, sample_sel=True):

        self.data_dir = data_dir
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.only_begin = only_begin
        self.instances = []
        self.type2idx = defaultdict(list)
        self.idx2type = defaultdict(list)
        self.class_weights = {}
        self.sample_sel = sample_sel
        self.file_infos = []  # [(idx, file_path, types_list)]
        
        idx = 0
        for file_path in tqdm(os.listdir(data_dir)):
            if not file_path.endswith('.pth'):
                continue
            type_list, time, bbox, stride = parse_extreme_event_filename(file_path)
            if only_begin and stride != 0:
                continue
            self.instances.append(file_path)
            self.idx2type[idx] = type_list
            for t in type_list:
                self.type2idx[t].append(idx)
            self.file_infos.append((idx, file_path, type_list))
            idx += 1

        total = len(self.instances)
        class_counts = {k: len(v) for k, v in self.type2idx.items()}
        max_count = max(class_counts.values())
        self.class_weights = {k: max_count / v for k, v in class_counts.items()}
        print("# num of extreme types: ", len(self.type2idx))

        self.resample_triplets(epoch_seed)
        # exit(-1)

    def resample_triplets(self, seed):

        rng = np.random.default_rng(seed)
        os.makedirs(self.data_dir.replace("pre_embeds", "pooling_embeds"), exist_ok=True)
        sample_path = os.path.join(self.data_dir.replace("pre_embeds", "pooling_embeds"), 
                                   f"contrastive_samples_{self.num_pos}_{self.num_neg}.pth")
        MUST_SAMPLE = False
        if not MUST_SAMPLE and os.path.exists(sample_path):
            triplets = torch.load(sample_path, weights_only=False)
        else:
            triplets = []
            print("Start Sampling...")
            for idx, file_path, types_list in tqdm(self.file_infos):
                
                multi_class_pos_candidates = [[i for i in self.type2idx[anchor_type] if i != idx] for anchor_type in types_list]
                multi_type_all_pos_pool = set(multi_class_pos_candidates[0]).intersection(*multi_class_pos_candidates[1:])
                multi_type_any_pos_pool = set().union(*multi_class_pos_candidates)
                
                for anchor_type in types_list:
                    
                    # positive sampling
                    # pos_pool = [i for i in self.type2idx[anchor_type] if i != idx]
                    pos_pool = list(set([i for i in self.type2idx[anchor_type] if i != idx and len(self.idx2type[i]) == 1]).union(multi_type_all_pos_pool))
                    if len(pos_pool) == 0:
                        if len(multi_type_any_pos_pool) > 0:
                            pos_pool = list(multi_type_any_pos_pool)
                        else:
                            pos_pool = [idx] * self.num_pos
                    pos_indices = rng.choice(
                        pos_pool, 
                        size=self.num_pos, 
                        replace=(len(pos_pool) < self.num_pos)
                    ).tolist()
                        
                    # negative sampling
                    neg_pool = []
                    neg_weights = []
                    for k, v in self.type2idx.items():
                        if anchor_type != "normal" and k == "normal":
                            continue
                        if k not in types_list:
                            # filtered_v = v
                            filtered_v = list(set(v) - multi_type_any_pos_pool)
                            neg_pool.extend(filtered_v)
                            neg_weights.extend([self.class_weights[k]] * len(filtered_v))
                    neg_weights = np.array(neg_weights)
                    neg_weights = neg_weights /neg_weights.sum()
                    assert len(neg_pool) > 0
                    # neg_indices = [idx] * self.num_neg
                    neg_indices = rng.choice(
                        neg_pool, 
                        size=self.num_neg if anchor_type == "normal" else self.num_neg - 1, 
                        replace=(len(neg_pool) < self.num_neg), 
                        p=neg_weights
                    ).tolist()
                    if anchor_type != "normal":
                        neg_indices += rng.choice(self.type2idx["normal"], size=1, replace=False).tolist()

                    triplets.append((idx, pos_indices, neg_indices, anchor_type))
            torch.save(triplets, sample_path)
            # exit(-1)
        self.triplets = triplets
    
    def resample_selected_samples(self, seed, epoch):
        
        if not self.sample_sel:
            return
        rng = np.random.RandomState(seed + epoch)
        self.selected_neg_inds = [rng.choice(self.num_neg, size=2, replace=False) for _ in range(len(self.triplets))]

    def __len__(self):
        
        return len(self.triplets)

    def __getitem__(self, i):
        
        anchor_idx, pos_indices, neg_indices, anchor_type = self.triplets[i]
        if self.sample_sel:
            neg_indices = np.array(neg_indices)[self.selected_neg_inds[i]]

        anchor_file = os.path.join(self.data_dir, self.instances[anchor_idx])
        pos_files = [os.path.join(self.data_dir, self.instances[j]) for j in pos_indices]
        neg_files = [os.path.join(self.data_dir, self.instances[j]) for j in neg_indices]

        anchor_inst = torch.load(anchor_file, weights_only=False)
        pos_insts = [torch.load(f, weights_only=False) for f in pos_files]
        neg_insts = [torch.load(f, weights_only=False) for f in neg_files]

        # return: anchor_emb, [pos_emb], [neg_emb], anchor_type, class_weight
        anchor_embed = anchor_inst["embeds"]
        pos_embeds = [inst["embeds"] for inst in pos_insts]
        neg_embeds = [inst["embeds"] for inst in neg_insts]
        class_weight = self.class_weights[anchor_type]

        return anchor_embed, pos_embeds, neg_embeds, anchor_type, class_weight

def extreme_triplet_collate(batch):
    
    B = len(batch)
    all_hws = [] 
    N, N_pos, N_neg = 0, 0, 0
    for item in batch:
        anchor_embed, pos_embeds, neg_embeds, anchor_type, class_weight = item
        embeds = [anchor_embed] + pos_embeds + neg_embeds
        N_pos, N_neg = len(pos_embeds), len(neg_embeds)
        N = 1 + len(pos_embeds) + len(neg_embeds)
        for e in embeds:
            assert e.shape[-2] * e.shape[-1] > 0
            all_hws.append(e.shape[-2:])

    # 统计最大H, W
    H_max = max([hw[0] for hw in all_hws])
    W_max = max([hw[1] for hw in all_hws])
    D, P = batch[0][0].shape[:2]

    # padding & masking
    batch_embeds = []
    spatial_mask = []
    anchor_types = []
    class_weights = []

    for i, item in enumerate(batch):
        anchor_embed, pos_embeds, neg_embeds, anchor_type, class_weight = item
        embeds = [anchor_embed] + pos_embeds + neg_embeds
        embeds_padded = []
        masks = []

        for j, embed in enumerate(embeds):
            h, w = embed.shape[-2:]
            pad_h = H_max - h
            pad_w = W_max - w

            # (left, right, top, bottom) pad raight & bottom
            pad_dims = (0, pad_w, 0, pad_h)  # (W_left, W_right, H_top, H_bottom)
            embed_padded = pad(embed, pad_dims, value=0)
            embeds_padded.append(embed_padded)

            # mask
            mask = torch.zeros(H_max, W_max, dtype=torch.bool)
            mask[:h, :w] = 1
            masks.append(mask)

        batch_embeds.append(torch.stack(embeds_padded, dim=0))  # (N, D, P, H, W)
        spatial_mask.append(torch.stack(masks, dim=0))  # (N, H, W)
        anchor_types.append(anchor_type)
        class_weights.append(float(class_weight))

    # stack
    batch_embeds = torch.stack(batch_embeds, dim=0)  # (B, N, D, P, H_max, W_max)
    spatial_mask = torch.stack(spatial_mask, dim=0)  # (B, N, H_max, W_max)
    class_weights = torch.tensor(class_weights, dtype=torch.float)  # (B,)
    # assert not torch.isnan(spatial_mask.mean())
   
    return batch_embeds, spatial_mask, class_weights, anchor_types, N_pos, N_neg
 
class ExtremeEventAnchorData(Dataset):
    
    def __init__(self, data_dir, num_pos=1, num_neg=6, only_begin=False, epoch_seed=0):

        self.data_dir = data_dir
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.only_begin = only_begin
        self.file_infos = []  # [(idx, file_path, types_list)]
        
        idx = 0
        for file_path in tqdm(os.listdir(data_dir)):
            if not file_path.endswith('.pth'):
                continue
            type_list, time, bbox, stride = parse_extreme_event_filename(file_path)
            if only_begin and stride != 0:
                continue
            self.file_infos.append((idx, file_path, type_list))
            idx += 1

    def __len__(self):
        
        return len(self.file_infos)

    def __getitem__(self, i):
        
        idx, file_path, type_list = self.file_infos[i]
        anchor_file = os.path.join(self.data_dir, file_path)
        anchor_inst = torch.load(anchor_file, weights_only=False)
        anchor_embed = anchor_inst["embeds"]

        return anchor_embed, type_list
    
def extreme_anchor_collate(batch):
    
    B = len(batch)
    all_hws = [] 
    for item in batch:
        anchor_embed, anchor_type = item
        assert anchor_embed.shape[-2] * anchor_embed.shape[-1] > 0
        all_hws.append(anchor_embed.shape[-2:])

    # 统计最大H, W
    H_max = max([hw[0] for hw in all_hws])
    W_max = max([hw[1] for hw in all_hws])
    D, P = batch[0][0].shape[:2]

    # padding & masking
    batch_embeds = []
    spatial_mask = []
    anchor_types = []

    for i, item in enumerate(batch):
        anchor_embed, anchor_type = item

        h, w = anchor_embed.shape[-2:]
        pad_h = H_max - h
        pad_w = W_max - w

        # (left, right, top, bottom) pad raight & bottom
        pad_dims = (0, pad_w, 0, pad_h)  # (W_left, W_right, H_top, H_bottom)
        embed_padded = pad(anchor_embed, pad_dims, value=0) # (D, P, H, W)
        
        # mask
        mask = torch.zeros(H_max, W_max, dtype=torch.bool)
        mask[:h, :w] = 1 # (H, W)

        batch_embeds.append(embed_padded)
        spatial_mask.append(mask)
        anchor_types.append(anchor_type)

    # stack
    batch_embeds = torch.stack(batch_embeds, dim=0).unsqueeze(1) # (B, 1, D, P, H_max, W_max)
    spatial_mask = torch.stack(spatial_mask, dim=0).unsqueeze(1) # (B, 1, H_max, W_max)
    
    return batch_embeds, spatial_mask, anchor_types

class DataLoaderX(DataLoader):
    
    def __iter__(self):
        
        return BackgroundGenerator(super().__iter__())
    
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, warmup_epochs, cosine_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = cosine_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / (self.warmup_epochs + 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            self.cosine_scheduler.last_epoch = cosine_epoch
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch >= self.warmup_epochs:
            self.cosine_scheduler.step(epoch - self.warmup_epochs)
        else:
            pass

def get_patch_pangu_slice(mask, patch_size):
    
    H, W = mask.shape
    device = mask.device
    
    height, width = H, W
    h_patch_size, w_patch_size = patch_size
    padding_left = padding_right = padding_top = padding_bottom = 0

    h_remainder = height % h_patch_size
    w_remainder = width % w_patch_size

    if h_remainder:
        h_pad = h_patch_size - h_remainder
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

    if w_remainder:
        w_pad = w_patch_size - w_remainder
        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

    H_pad = H + padding_top + padding_bottom
    W_pad = W + padding_left + padding_right
    mask_padded = torch.zeros((H_pad, W_pad), dtype=torch.bool, device=device)
    mask_padded[padding_top:padding_top+H, padding_left:padding_left+W] = mask

    h_coords, w_coords = torch.where(mask_padded)
    if len(h_coords) == 0 or len(w_coords) == 0:
        return (slice(0, 0), slice(0, 0))

    h_min, h_max = h_coords.min().item(), h_coords.max().item()
    w_min, w_max = w_coords.min().item(), w_coords.max().item()

    h_start_idx = h_min // h_patch_size
    h_end_idx = (h_max) // h_patch_size
    w_start_idx = w_min // w_patch_size
    w_end_idx = (w_max) // w_patch_size

    h_slice = slice(h_start_idx, h_end_idx + 1)
    w_slice = slice(w_start_idx, w_end_idx + 1)

    return h_slice, w_slice

def get_patch_slice(bbox, patch_size):

    w_min, h_min, w_max, h_max = [int(e) for e in bbox]
    patch_h, patch_w = patch_size
    
    h_start_idx = h_min // patch_h
    h_end_idx = (h_max - 1) // patch_h
    
    w_start_idx = w_min // patch_w
    w_end_idx = (w_max - 1) // patch_w
    
    h_slice = slice(h_start_idx, h_end_idx + 1)
    w_slice = slice(w_start_idx, w_end_idx + 1)
    
    return h_slice, w_slice

def sample_normal_patch_slices(extreme_slices, H, W):
    
    def get_valid_positions(mask, h, w):

        kernel = torch.ones((1, 1, h, w), dtype=mask.dtype, device=mask.device)
        conv = F.conv2d(mask, kernel, stride=1) # 1, 1, H, W
        valid = (conv[0, 0] == 0).nonzero(as_tuple=False)
        # valid shape: (num_valid, 2), take the last two dims as (top, left)
        if valid.size(0) == 0:
            return None
        positions = valid # (num_valid, 2)
        return positions
    
    mask = torch.zeros((H, W), dtype=torch.float32)
    for h_slice, w_slice in extreme_slices:
        mask[h_slice, w_slice] = 1
    mask = mask.unsqueeze(0).unsqueeze(0)
    normal_slices = []
    for h_slice, w_slice in extreme_slices:
        h_delta = h_slice.stop - h_slice.start
        w_delta = w_slice.stop - w_slice.start
        positions = get_valid_positions(mask, h_delta, w_delta)
        if positions is None or positions.size(0) == 0:
            print(f"No valid position for shape ({h_delta},{w_delta})")
            continue
        idx = random.randint(0, positions.size(0) - 1)
        top, left = positions[idx].tolist()
        normal_slices.append((slice(top, top + h_delta), slice(left, left + w_delta)))
        
    return normal_slices

if __name__ == "__main__":

    # DATA_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/hrrr"
    # MEAN_STD_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/mean_std/2024010100-2024053123"
    # DEBUG_FLAG = False

    # time_span = ["2019010100", "2024123123"]
    # dataset = NOAADataComplete(DATA_DIR, time_span, "val", train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG, preload_flag=False)
    # loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, 
    #                            pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=extreme_collate)
    
    # bar = tqdm(loader)
    # for _ in bar:
    #     continue
    
    DATA_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/NWP/WRF-ARW/hrrr"
    MEAN_STD_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw/mean_std/2024010100-2024053123"
    DEBUG_FLAG = False

    time_span = ["2024010100", "2024123123"]
    dataset = NOAADataComplete(DATA_DIR, time_span, "val", train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG, preload_flag=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    bar = tqdm(loader)
    for _ in bar:
        continue
