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
from datetime import datetime, timedelta

# vscode Relative path
import sys
# sys.path.append("../../")

from Fuxi_freq import FuxiFreq
from Fuxi_tune import FuxiFreqPrompt, FuxiFreqPromptFreq, FuxiFreqRecon
from data_utils import surface_transform, upper_air_transform, surface_inv_transform, upper_air_inv_transform, NOAADataComplete, \
    NOAAExtremeDataComplete, NOAAExtremeDataTestNWP, extreme_collate, NOAACompletePlusExtremeDataComplete, NOAACompletePlusExtremeDataCompleteAutoReg
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
parser.add_argument("--version", default='', type=str, help="version")
parser.add_argument("--prompt_style", type=str, default="dual", help="prompt style")
parser.add_argument("--pooling_style", type=str, default="mean", help="pooling style")
parser.add_argument("--attn_loss", action="store_true", help="attn loss")
parser.add_argument("--use_part", action="store_true", help="use part")
parser.add_argument("--autoreg_horizon", default=1, type=int, help="")

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

def calculate_pixel_metrics(pred, target, mask=None):
    """Calculate pixel-level metrics (MSE, MAE) and counts"""
    diff = pred - target
    squared_diff = diff.pow(2)
    abs_diff = diff.abs()
    
    if mask is not None:
        mask_float = mask.float()
        mse_sum = (squared_diff * mask_float).sum(dim=(0, 2, 3))  # Sum over batch and spatial dims
        mae_sum = (abs_diff * mask_float).sum(dim=(0, 2, 3))
        count = mask_float.sum(dim=(0, 2, 3))
    else:
        mse_sum = squared_diff.sum(dim=(0, 2, 3))
        mae_sum = abs_diff.sum(dim=(0, 2, 3))
        count = torch.ones_like(mse_sum) * pred.shape[0] * pred.shape[2] * pred.shape[3]
    
    return mse_sum, mae_sum, count

def calculate_temporal_acc(pred, target, climatology, mask=None):
    
    C, H, W = pred.shape
    
    pred_dev = pred - climatology
    target_dev = target - climatology
    if mask is not None:
        mask_float = mask.float()
        pred_dev, target_dev = pred_dev * mask_float, target_dev * mask_float
        if mask_float.mean() > 0:
            count = 1
        else:
            count = 0
    else:
        count = 1
    
    cov = (pred_dev * target_dev).sum(dim=(1, 2))
    pred_var = pred_dev.pow(2).sum(dim=(1, 2))
    target_var = target_dev.pow(2).sum(dim=(1, 2))
    acc = cov / (torch.sqrt(pred_var) * torch.sqrt(target_var) + 1e-10)
    
    return acc, count

def get_climatology(time_str, climatology_dir):
    """Get climatology data for given time string (YYYYMMDDHH)"""
    hour_of_year = time_str[4:]  # Get MMDDHH part
    clim_path = os.path.join(climatology_dir, f"{hour_of_year}.pkl")
    with open(clim_path, 'rb') as f:
        climatology = pickle.load(f)
    return climatology

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
    HORIZON = opt.autoreg_horizon
    
    set_seed(SEED)
    DEBUG_FLAG = False

    # Climatology directory
    CLIMATOLOGY_DIR = "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/climatology/2019010100-2022123123_0_0"

    test_time_span = ["2024010100", "2024123123"]
    # test_set = NOAAExtremeDataComplete(DATA_DIR, test_time_span, "test", 
    #                                  train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=MAX_THREADS, 
    #                         pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    test_set = NOAACompletePlusExtremeDataCompleteAutoReg(DATA_DIR, test_time_span, "test", autoreg_horizon=HORIZON, 
                                     train_mean_std_dir=MEAN_STD_DIR, CLIMATOLOGY_DIR=CLIMATOLOGY_DIR, debug_flag=DEBUG_FLAG)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=MAX_THREADS, 
                            pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    print("# test samples: ", len(test_set))
    
    
    DETECT = False
    PostFix = f"NEW_{VERSION}_autoreg_{HORIZON}"
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
    accumulators = {
        "norm": {
            "gen": {"mse_sum": 0, "mae_sum": 0, "count": 0, "acc_sum": 0, "acc_count": 0},
            "ext": {"mse_sum": 0, "mae_sum": 0, "count": 0, "acc_sum": 0, "acc_count": 0},
        },
        "raw": {
            "gen": {"mse_sum": 0, "mae_sum": 0, "count": 0, "acc_sum": 0, "acc_count": 0},
            "ext": {"mse_sum": 0, "mae_sum": 0, "count": 0, "acc_sum": 0, "acc_count": 0},
        }
    }
    CAL_TYPE = False
    fuxi.eval()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_input_surface, test_input_upper_air, test_target_surface, test_target_upper_air, target_bbox, bboxes, types, stride, times, climatologys in test_bar:
            batch_size = test_input_surface.size(0)
            test_inputs = torch.concat([test_input_surface, test_input_upper_air.flatten(1, 2)], dim=1)
            test_targets = torch.concat([test_target_surface, test_target_upper_air.flatten(1, 2)], dim=1)
            
            if torch.cuda.is_available():
                test_inputs = test_inputs.to(device)
                times = times.to(device)
                test_targets = test_targets.to(device)
                ext_mask = target_bbox.to(device)
                climatologys = climatologys.to(device)
            
            times = torch.tensor([
                [t, eval((datetime.strptime(str(t), '%Y%m%d%H') + timedelta(hours=1)).strftime('%Y%m%d%H'))]
                for t in times[:, 0].cpu().numpy()]).to(device)
            x = test_inputs
            for _ in range(HORIZON):
                x, _, _ = fuxi(x, times)
                times = torch.tensor([
                    [eval((datetime.strptime(str(t1), '%Y%m%d%H') + timedelta(hours=1)).strftime('%Y%m%d%H')), 
                     eval((datetime.strptime(str(t2), '%Y%m%d%H') + timedelta(hours=1)).strftime('%Y%m%d%H'))]
                    for t1, t2 in times.cpu().numpy()]).to(device)
            test_outputs = x
            
            # Create masks
            ext_mask = ext_mask.unsqueeze(1).expand(-1, 69, -1, -1).bool()
            
            # Calculate metrics for normalized data
            mse_sum_gen_norm, mae_sum_gen_norm, count_gen_norm = calculate_pixel_metrics(
                test_outputs, test_targets)
            mse_sum_ext_norm, mae_sum_ext_norm, count_ext_norm = calculate_pixel_metrics(
                test_outputs, test_targets, ext_mask)
            
            # Update accumulators for normalized data
            accumulators["norm"]["gen"]["mse_sum"] += mse_sum_gen_norm.detach().cpu()
            accumulators["norm"]["gen"]["mae_sum"] += mae_sum_gen_norm.detach().cpu()
            accumulators["norm"]["gen"]["count"] += count_gen_norm.detach().cpu()
            accumulators["norm"]["ext"]["mse_sum"] += mse_sum_ext_norm.detach().cpu()
            accumulators["norm"]["ext"]["mae_sum"] += mae_sum_ext_norm.detach().cpu()
            accumulators["norm"]["ext"]["count"] += count_ext_norm.detach().cpu()
            
            for i in range(batch_size):
                
                # Calculate temporal ACC for normalized data
                acc_sum_gen_norm, acc_count_gen_norm = calculate_temporal_acc(test_outputs[i], test_targets[i], climatologys[i])
                acc_sum_ext_norm, acc_count_ext_norm = calculate_temporal_acc(test_outputs[i], test_targets[i], climatologys[i], ext_mask[i])
                accumulators["norm"]["gen"]["acc_sum"] += acc_sum_gen_norm.detach().cpu()
                accumulators["norm"]["gen"]["acc_count"] += acc_count_gen_norm
                accumulators["norm"]["ext"]["acc_sum"] += acc_sum_ext_norm.detach().cpu()
                accumulators["norm"]["ext"]["acc_count"] += acc_count_ext_norm
    
            test_bar.set_description(desc=f"[testing]")
    
    # Calculate final metrics from accumulators
    def compute_final_metrics(accum):
        results = {}
        
        for norm_type in ["norm"]:
            results[norm_type] = {"all": {}}
            
            # General area metrics
            mse_gen = accum[norm_type]["gen"]["mse_sum"] / accum[norm_type]["gen"]["count"]
            mae_gen = accum[norm_type]["gen"]["mae_sum"] / accum[norm_type]["gen"]["count"]
            rmse_gen = torch.sqrt(mse_gen)
            
            # Extreme area metrics
            mse_ext = accum[norm_type]["ext"]["mse_sum"] / accum[norm_type]["ext"]["count"]
            mae_ext = accum[norm_type]["ext"]["mae_sum"] / accum[norm_type]["ext"]["count"]
            rmse_ext = torch.sqrt(mse_ext)
            
            # ACC metrics
            acc_gen = accum[norm_type]["gen"]["acc_sum"] / accum[norm_type]["gen"]["acc_count"]
            acc_ext = accum[norm_type]["ext"]["acc_sum"] / accum[norm_type]["ext"]["acc_count"]
            
            # Gain metrics
            mse_gain = mse_ext - mse_gen
            mae_gain = mae_ext - mae_gen
            rmse_gain = rmse_ext - rmse_gen
            acc_gain = acc_ext - acc_gen
            
            # Gain ratios
            mse_gain_ratio = mse_gain / mse_gen
            mae_gain_ratio = mae_gain / mae_gen
            rmse_gain_ratio = rmse_gain / rmse_gen
            acc_gain_ratio = acc_gain / acc_gen
            
            # Store results
            results[norm_type]["all"] = {
                "mse_gen": mse_gen.mean().item(),
                "mae_gen": mae_gen.mean().item(),
                "rmse_gen": rmse_gen.mean().item(),
                "acc_gen": acc_gen.mean().item(),
                "mse_ext": mse_ext.mean().item(),
                "mae_ext": mae_ext.mean().item(),
                "rmse_ext": rmse_ext.mean().item(),
                "acc_ext": acc_ext.mean().item(),
                "mse_gain": mse_gain.mean().item(),
                "mae_gain": mae_gain.mean().item(),
                "rmse_gain": rmse_gain.mean().item(),
                "acc_gain": acc_gain.mean().item(),
                "mse_gain_ratio": mse_gain_ratio.mean().item(),
                "mae_gain_ratio": mae_gain_ratio.mean().item(),
                "rmse_gain_ratio": rmse_gain_ratio.mean().item(),
                "acc_gain_ratio": acc_gain_ratio.mean().item()
            }
            
            # Per-variable metrics
            for i, var in enumerate(surface_variables):
                results[norm_type][var] = {
                    "mse_gen": mse_gen[i].item(),
                    "mae_gen": mae_gen[i].item(),
                    "rmse_gen": rmse_gen[i].item(),
                    "acc_gen": acc_gen[i].item(),
                    "mse_ext": mse_ext[i].item(),
                    "mae_ext": mae_ext[i].item(),
                    "rmse_ext": rmse_ext[i].item(),
                    "acc_ext": acc_ext[i].item(),
                    "mse_gain": mse_gain[i].item(),
                    "mae_gain": mae_gain[i].item(),
                    "rmse_gain": rmse_gain[i].item(),
                    "acc_gain": acc_gain[i].item(),
                    "mse_gain_ratio": mse_gain_ratio[i].item(),
                    "mae_gain_ratio": mae_gain_ratio[i].item(),
                    "rmse_gain_ratio": rmse_gain_ratio[i].item(),
                    "acc_gain_ratio": acc_gain_ratio[i].item()
                }
            
            for i, var in enumerate(upper_air_variables):
                for j, pl in enumerate(upper_air_pLevels):
                    var_name = f"{var}_{pl}"
                    idx = 4 + 13 * i + j
                    results[norm_type][var_name] = {
                        "mse_gen": mse_gen[idx].item(),
                        "mae_gen": mae_gen[idx].item(),
                        "rmse_gen": rmse_gen[idx].item(),
                        "acc_gen": acc_gen[idx].item(),
                        "mse_ext": mse_ext[idx].item(),
                        "mae_ext": mae_ext[idx].item(),
                        "rmse_ext": rmse_ext[idx].item(),
                        "acc_ext": acc_ext[idx].item(),
                        "mse_gain": mse_gain[idx].item(),
                        "mae_gain": mae_gain[idx].item(),
                        "rmse_gain": rmse_gain[idx].item(),
                        "acc_gain": acc_gain[i].item(),
                        "mse_gain_ratio": mse_gain_ratio[idx].item(),
                        "mae_gain_ratio": mae_gain_ratio[idx].item(),
                        "rmse_gain_ratio": rmse_gain_ratio[idx].item(),
                        "acc_gain_ratio": acc_gain_ratio[idx].item()
                    }
        
        return results
    
    # Compute universal results
    universal_results = compute_final_metrics(accumulators)
    universal_results["batch_sizes"] = len(test_set)
    
    # Save results
    save_root = "test_logs"
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, log_path), 'w', encoding='utf-8') as f:
        json.dump({"universal": universal_results}, f, indent=4)