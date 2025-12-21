import argparse
import os
import json
from torch.utils.data import DataLoader
from torch import nn
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import pickle
from tdigest import TDigest
from collections import defaultdict

# vscode Relative path
import sys
# sys.path.append("../../")

from Fuxi_freq import FuxiFreq
from Fuxi_tune import FuxiFreqPrompt, FuxiFreqPromptFreq, FuxiFreqRecon
from data_utils import surface_transform, upper_air_transform, surface_inv_transform, upper_air_inv_transform, NOAADataComplete, NOAAExtremeDataComplete, NOAAExtremeDataTestNWP, extreme_collate, NOAACompletePlusExtremeDataComplete
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

parser = argparse.ArgumentParser(description="Fuxi pre-train")
parser.add_argument("--seed", default=42, type=int, help="seed")
parser.add_argument("--data_dir", 
                    default="/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/hrrr", 
                    type=str, help="data dir")
parser.add_argument("--max_threads", default=12, type=int, help="number of threads")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
parser.add_argument("--batch_size", default=12, type=int, help="batch size")
parser.add_argument("--patience", default=5, type=int, help="patience")
parser.add_argument("--parallel", action="store_false", help="use data parallel")
parser.add_argument("--pretrained_model_path", 
                    default='_', 
                    type=str, help="pretrained model path")
parser.add_argument("--mean_std_dir", 
                    default='/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/mean_std/2019010100-2021123123', 
                    type=str, help="mean std dir")
parser.add_argument("--continue_train", action="store_true", help="continue train")
parser.add_argument("--version", default='best', type=str, help="version")
parser.add_argument("--prompt_style", type=str, default="dual", help="prompt style")
parser.add_argument("--pooling_style", type=str, default="mean", help="pooling style")
parser.add_argument("--attn_loss", action="store_true", help="attn loss")
parser.add_argument("--use_part", action="store_true", help="use part")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_pretrained_weights(model, optimizer, scheduler, pretrained_path, ignore_keys=[]):
    print(f"======\n{pretrained_path}\n======")
    checkpoints = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained_dict = checkpoints["model"]
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()

    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and not any([k.startswith(ik) for ik in ignore_keys]):
            filtered_dict[k] = v

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoints["optimizer"])
        scheduler.load_state_dict(checkpoints["scheduler"])
    
    return filtered_dict

class AdvancedStreamingStatistics:
    
    def __init__(self, use_quan=False):
        
        self.use_quan = use_quan
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum = 0
        self.count = 0
        if use_quan:
            self.tdigest = TDigest()
        
    def update(self, new_values):
        
        if len(new_values) == 0:
            return
        
        batch_min = np.min(new_values)
        batch_max = np.max(new_values)
        batch_sum = np.sum(new_values)
        
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)
        self.sum += batch_sum
        self.count += len(new_values)
        
        if self.use_quan:
            self.tdigest.batch_update(new_values)
    
    def get_statistics(self):
        
        if self.count == 0:
            return {}

        mean = self.sum / self.count
        
        return_dict =  {
            'min': self.min_val,
            'max': self.max_val,
            'mean': mean,
            'sum': self.sum, 
            'count': self.count
        }
        if self.use_quan:
            return_dict['quantiles'] = {
                0.0001: self.tdigest.percentile(0.01),
                0.001: self.tdigest.percentile(0.1),
                0.01: self.tdigest.percentile(1),
                0.05: self.tdigest.percentile(5),
                0.1: self.tdigest.percentile(10),
                0.2: self.tdigest.percentile(20),
                0.3: self.tdigest.percentile(30),
                0.7: self.tdigest.percentile(70),
                0.8: self.tdigest.percentile(80),
                0.9: self.tdigest.percentile(90),
                0.95: self.tdigest.percentile(95),
                0.99: self.tdigest.percentile(99),
                0.999: self.tdigest.percentile(99.9),
                0.9999: self.tdigest.percentile(99.99),
            }
        
        return return_dict

def calculate_detect_metrics(pre, gt):
    """
    计算二分类检测指标
    pre: 预测mask [B, H, W] bool tensor
    gt: 真实mask [B, H, W] bool tensor
    返回: (hits, misses, false_alarms, correct_negatives)
    """
    pre_flat = pre.flatten()
    gt_flat = gt.flatten()
    
    if (gt_flat == 1).sum().item() == 0:
        pre_flat = gt_flat
    
    hits = ((pre_flat == 1) & (gt_flat == 1)).sum().item()
    misses = ((pre_flat == 0) & (gt_flat == 1)).sum().item()
    false_alarms = ((pre_flat == 1) & (gt_flat == 0)).sum().item()
    correct_negatives = ((pre_flat == 0) & (gt_flat == 0)).sum().item()
    
    return hits, misses, false_alarms, correct_negatives

def compute_final_metrics(accum):
    results = {}
    for var, var_res in accum.items():
        results[var] = {}
        for thres, thres_res in var_res.items():
            hits = thres_res["hit"]
            misses = thres_res["miss"] 
            false_alarms = thres_res["false_alarm"]
            # others实际上是correct_negatives，但计算CSI不需要
            
            # 防止除零
            pod = hits / (hits + misses) if (hits + misses) > 0 else 0
            far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
            csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
            
            results[var][thres] = {
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "others": thres_res["others"],
                "pod": pod,
                "far": far,
                "csi": csi
            }
    return results

if __name__ == "__main__":
    opt = parser.parse_args()
    SEED = opt.seed
    DATA_DIR = opt.data_dir
    MAX_THREADS = opt.max_threads
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    PATIENCE = opt.patience
    USE_PARALLEL = opt.parallel
    PRETRAINED_MODEL_PATH = opt.pretrained_model_path
    MEAN_STD_DIR = opt.mean_std_dir
    CONTINUE_TRAIN = opt.continue_train
    VERSION = opt.version
    PROMPT_STYLE = opt.prompt_style
    POOLING_STYLE = opt.pooling_style
    ATTN_LOSS = opt.attn_loss
    USE_PART = opt.use_part
    
    set_seed(SEED)
    DEBUG_FLAG = False

    # Climatology directory
    CLIMATOLOGY_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/climatology/2019010100-2022123123_0_0"

    test_time_span = ["2024010100", "2024123123"]
    # test_set = NOAAExtremeDataComplete(DATA_DIR, test_time_span, "test", 
    #                                  train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=MAX_THREADS, 
    #                         pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    test_set = NOAACompletePlusExtremeDataComplete(DATA_DIR, test_time_span, "test", 
                                     train_mean_std_dir=MEAN_STD_DIR, CLIMATOLOGY_DIR=CLIMATOLOGY_DIR, debug_flag=DEBUG_FLAG)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=MAX_THREADS, 
                            pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    print("# test samples: ", len(test_set))
    
    
    CAL_TYPE = False
    DETECT = True
    CAL_THRES = False
    PostFix = f"NEW_{VERSION}"
    if CAL_TYPE:
        PostFix += "_has_type"
    if DETECT:
        PostFix += "_detect"
    if USE_PART:
        PostFix += "_part"
    if "-2021" in MEAN_STD_DIR:
        PostFix += "_19-21"
    fuxi = FuxiFreqPromptFreq(
            weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/space_tokens/2022010100_2022123123", 
            use_space=True)
    if VERSION == "best":
        PRETRAINED_MODEL_PATH = "./checkpoints_tune_pretrain_prompt/2019010100_2022123123_freq_prompt_space_p+x_t5_bs12_steplr/fuxi_best.pth"
    elif VERSION.startswith("epoch_"):
        GET_EPOCH = int(VERSION.lstrip("epoch_"))
        PRETRAINED_MODEL_PATH = f"./checkpoints_tune_pretrain_prompt/2019010100_2022123123_freq_prompt_space_p+x_t5_bs12_steplr/fuxi_epoch_{GET_EPOCH}.pth"
    else:
        raise ValueError
    load_pretrained_weights(fuxi, None, None, PRETRAINED_MODEL_PATH) 
    log_path = f"logs_pretrain_prompt_{PostFix}.json"
    print("########################")
    print("Version: ", VERSION, PostFix)
    print("########################")
    
    # surface_func, surface_variables = surface_transform(os.path.join(MEAN_STD_DIR, "surface_mean.pkl"), 
    #                                                                 os.path.join(MEAN_STD_DIR, "surface_std.pkl"))
    # upper_air_func, upper_air_variables, upper_air_pLevels = upper_air_transform(os.path.join(MEAN_STD_DIR, "upper_air_mean.pkl"), 
    #                                                                                         os.path.join(MEAN_STD_DIR, "upper_air_std.pkl"))
        
    surface_inv_func, surface_variables = surface_inv_transform(os.path.join(MEAN_STD_DIR, "surface_mean.pkl"), 
                                                               os.path.join(MEAN_STD_DIR, "surface_std.pkl"))
    upper_air_inv_func, upper_air_variables, upper_air_pLevels = upper_air_inv_transform(os.path.join(MEAN_STD_DIR, "upper_air_mean.pkl"), 
                                                                                        os.path.join(MEAN_STD_DIR, "upper_air_std.pkl"))
    # print(surface_variables, upper_air_variables, upper_air_pLevels)
    # ['msl', 't2m', 'u10', 'v10'] ['q', 't', 'u', 'v', 'z'] [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    def inv_trans(tensor):
        B, _, H, W = tensor.shape
        tensor_surface, tensor_upper_air = tensor[:, :4], tensor[:, 4:].reshape(B, 5, 13, H, W)
        tensor_surface = surface_inv_func(tensor_surface) # B, C1=4, H, W
        tensor_upper_air = torch.stack([upper_air_inv_func[pl](tensor_upper_air[:, :, i, :, :]) 
                                      for i, pl in enumerate(upper_air_pLevels)], dim=2) # B, C2=5, Pl=13, H, W
        return torch.concat([tensor_surface, tensor_upper_air.flatten(1, 2)], dim=1)

    device = "cuda:0"
    if torch.cuda.is_available():
        fuxi.to(device)
        if USE_PARALLEL:
            fuxi = torch.nn.DataParallel(fuxi, device_ids=[0, 1, 2, 3])
    
    # Initialize accumulators for pixel-level metrics
    fuxi.eval()
    
    detect_dict = {
        # "uv": (["Thunderstorm_Wind", "Marine_Thunderstorm_Wind", "Wind", "Marine_High_Wind", "Marine_Strong_Wind"], [2,3], True), 
        # "uv925": (["Thunderstorm_Wind", "Marine_Thunderstorm_Wind", "Wind", "Marine_High_Wind", "Marine_Strong_Wind"], 
        #           [2,3] + [4+2*13+i for i in range(13)] + [4+3*13+i for i in range(13)], True), 
        # "q925": (["Flood", "Flash_Flood", "Heavy_Rain"], 4+11, True), 
        "t-1": (["Heat"], 1, True),
        "t-2": (["Cold"], 1, False)
    }
    thres_dict = {
        # "uv": [(0, True)], 
        # "uv925": [(0, True)], 
        # "q925": [(0, True)], 
        "t-1": [(311.15, True)], 
        "t-2": [(244.15, False)]
    }
    statistics = {}
    accumulators = {}
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_idx, (test_input_surface, test_input_upper_air, test_target_surface, test_target_upper_air, target_bbox, bboxes, types, stride, times, climatologys) in enumerate(test_bar):
            batch_size = test_input_surface.size(0)
            test_inputs = torch.concat([test_input_surface, test_input_upper_air.flatten(1, 2)], dim=1)
            test_targets = torch.concat([test_target_surface, test_target_upper_air.flatten(1, 2)], dim=1)
            
            if torch.cuda.is_available():
                test_inputs = test_inputs.to(device)
                times = times.to(device)
                test_targets = test_targets.to(device)
                ext_mask = target_bbox.to(device)
            
            if VERSION == "freq":
                test_outputs, _ = fuxi(test_inputs, times)
            else:
                test_outputs, _, _ = fuxi(test_inputs, times)
            
            # Create masks
            ext_mask = ext_mask.unsqueeze(1).expand(-1, 69, -1, -1).bool()
            gen_mask = None
            
            # Calculate metrics for raw data
            test_outputs_raw = inv_trans(test_outputs)
            test_targets_raw = inv_trans(test_targets)
            
            for var, (type_list, var_idx, type_larger) in detect_dict.items():
                B, _, H, W = test_outputs.shape
                type_ext_mask = torch.zeros((B, H, W), device=test_outputs.device) # B, H, W
                for batch_idx, batch_bboxes in enumerate(bboxes):
                    if batch_bboxes is None:
                        assert ext_mask[batch_idx].float().mean() == 0.
                        continue
                    for event_idx, event_bbox in enumerate(batch_bboxes):
                        event_types = types[batch_idx][event_idx]
                        for event_type in event_types:
                            if event_type in type_list:
                                y_min, x_min, y_max, x_max = [eval(e) for e in event_bbox]
                                type_ext_mask[batch_idx, x_min:x_max, y_min:y_max] = 1.
                                break
                if isinstance(var_idx, list):
                    test_outputs_raw[:, var_idx[0]] = (test_outputs_raw[:, var_idx].pow(2).sum(dim=1)).sqrt()
                    test_targets_raw[:, var_idx[0]] = (test_targets_raw[:, var_idx].pow(2).sum(dim=1)).sqrt()
                    var_idx = var_idx[0]
                    
                if CAL_THRES:
                    type_ext_mask_values = test_targets_raw[:, var_idx][type_ext_mask.bool()]
                    type_nor_mask_values = test_targets_raw[:, var_idx][~type_ext_mask.bool()]
                    top_values, _ = torch.topk(type_nor_mask_values.flatten(), type_ext_mask_values.shape[0])
                    if var not in statistics:
                        statistics[var] = {
                            "ext": AdvancedStreamingStatistics(use_quan=True),
                            "nor": AdvancedStreamingStatistics(),
                            "top_nor": AdvancedStreamingStatistics(use_quan=True)
                        }
                    statistics[var]["ext"].update(type_ext_mask_values.cpu().numpy())
                    statistics[var]["nor"].update(type_nor_mask_values.cpu().numpy())
                    statistics[var]["top_nor"].update(top_values.cpu().numpy())
                else:
                    if var not in accumulators:
                        accumulators[var] = {}
                    pre_values = test_outputs_raw[:, var_idx] # B, H, W
                    for thres, thres_larger in thres_dict[var]:
                        if thres not in accumulators[var]:
                            accumulators[var][thres] = {"hit": 0, "miss": 0, "false_alarm": 0, "others": 0}
                        if type_larger != thres_larger: 
                            continue
                        detect_mask = pre_values >= thres if thres_larger else pre_values <= thres # B, H, W
                        res = calculate_detect_metrics(pre=detect_mask, gt=type_ext_mask)
                        accumulators[var][thres]["hit"] += res[0]
                        accumulators[var][thres]["miss"] += res[1]
                        accumulators[var][thres]["false_alarm"] += res[2]
                        accumulators[var][thres]["others"] += res[3]
            # if test_idx > 10:
            #     break            
            test_bar.set_description(desc=f"[testing]")
    
    if not CAL_THRES:
        detect_results = compute_final_metrics(accumulators)
        save_root = "test_logs"
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, log_path), 'w', encoding='utf-8') as f:
            json.dump(detect_results, f, indent=4)
    else:
        thres_results = {}
        for var, var_res in statistics.items():
            thres_results[var] = {
                "ext": var_res["ext"].get_statistics(), 
                "nor": var_res["nor"].get_statistics(), 
                "top_nor": var_res["top_nor"].get_statistics(), 
            }
        save_root = "test_logs"
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "detect_thres.json"), 'w', encoding='utf-8') as f:
            json.dump(thres_results, f, indent=4)
