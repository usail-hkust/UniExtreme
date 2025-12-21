import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time as time_repo
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from transformers import get_cosine_schedule_with_warmup
import seaborn as sns
from pytorch_wavelets import DWTForward
from skimage.measure import label as sk_label

# vscode Relative path
import sys
# sys.path.append("../../")

from Fuxi import Fuxi
from data_utils import surface_inv_transform, upper_air_inv_transform, NOAADataComplete, NOAAExtremeDataComplete, extreme_collate, DataLoaderX, WarmupCosineScheduler
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

parser = argparse.ArgumentParser(description="Fuxi pre-train")
parser.add_argument("--seed", default=42, type=int, help="seed")
parser.add_argument("--data_dir", default="/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/hrrr", 
                    type=str, help="data dir")
parser.add_argument("--max_threads", default=32, type=int, help="number of threads")
parser.add_argument("--num_epochs", default=100, type=int, help="train epoch number")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--parallel", action="store_false", help="use data parallel")
parser.add_argument("--pretrained_model_path", default='', type=str, help="pretrained model path")
parser.add_argument("--mean_std_dir", 
                    default='/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/mean_std/2019010100-2022123123', 
                    type=str, help="mean std dir")

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_pretrained_weights(model, pretrained_path, ignore_keys=[]):

    pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()

    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and not any([k.startswith(ik) for ik in ignore_keys]):
            filtered_dict[k] = v

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    
    return filtered_dict

def freeze_module(model, activated_dict, freeze_module_names=None, unfreeze_module_names=None):
    
    if freeze_module_names is not None:
        assert unfreeze_module_names is None
        for name, param in model.named_parameters():
            if any([name.startswith(mn) for mn in freeze_module_names]):
                param.requires_grad = False
        return
    
    if unfreeze_module_names is not None:
        assert freeze_module_names is None
        for name, param in model.named_parameters():
            if name in activated_dict:
                param.requires_grad = False
        if len(unfreeze_module_names) > 0:
            for name, param in model.named_parameters():
                if any([name.startswith(mn) for mn in unfreeze_module_names]):
                    param.requires_grad = True
        return

def get_band_thresholds(freq_radius, num_bands, use_log=False, eps=1., according_val=True):

    freq_1d = freq_radius.flatten()
    freq_sorted, _ = torch.sort(freq_1d)
    log_ratio = 1.01 if according_val else 1.15
    
    if according_val:
        max_freq, min_freq = freq_1d.max(), freq_sorted[1]
        if not use_log:
            thresholds = [max_freq * i / num_bands for i in range(num_bands + 1)]
        else:
            r = log_ratio
            total = (r ** num_bands - 1) / (r - 1)
            base = max_freq / total
            # print(base, total, freq_sorted[:5])
            assert base > min_freq
            bandwidths = [base * r ** i for i in range(num_bands)]
            thresholds = [0.0]
            cur = 0.0
            for bw in bandwidths:
                cur += bw.item()
                thresholds.append(cur)  
        thresholds[-1] = max_freq + 1
        return thresholds
    else:
        N = freq_sorted.numel()
        if not use_log:
            indices = [round(i * N / num_bands) for i in range(num_bands + 1)]
        else:
            r = log_ratio
            total = (r ** num_bands - 1) / (r - 1)
            base = N / total
            splits = [base * (r ** i) for i in range(num_bands)]
            indices = [0]
            acc = 0
            for s in splits:
                acc += s
                indices.append(round(acc))
            indices[-1] = N
        assert indices[1] > 1

        thresholds = [float(freq_sorted[i].item()) if i < N else float(freq_sorted[N - 1].item()) + eps for i in indices]
        
        return thresholds

def fourierTransform(data, use_log=False):
    
    device = data.device
    C, H, W = data.shape
    
    fft2d = torch.fft.rfft2(data, dim=(-2, -1))
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.rfftfreq(W, device=device)
    freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2) # (H, W)
    amplitude = torch.abs(fft2d)
    phase = torch.angle(fft2d)
    energy = amplitude ** 2 # (C, H, W)
    
    def intra_get_band_splits(freq_radius, num_bands=9):

        freq_sorted, indices_sorted = torch.sort(freq_radius)
        log_ratio = 1.3
        band_idx = []
        
        T = freq_sorted.numel()
        r = log_ratio
        total = (r ** num_bands - 1) / (r - 1)
        base = T / total
        splits = [base * (r ** i) for i in range(num_bands)]
        indices = [0]
        acc = 0
        for s in splits:
            acc += s
            indices.append(round(acc))
        assert indices[1] > 1
        indices[-1] = T
            
        band_idx.append(torch.arange(0, 1).long())
        indices[0] = 1
        for i in range(num_bands):
            band_idx.append(torch.arange(indices[i], indices[i + 1]).long())
        band_real_idx = [indices_sorted[inds] for inds in band_idx]
            
        return band_real_idx

    freq_radius_flat = freq_radius.flatten() # (H*W)
    if use_log:
        freq_sorted = None
        band_real_idx = intra_get_band_splits(freq_radius_flat)
        energy_flat = energy.reshape(C, -1) # (C, H*W)
        energy_sorted = torch.stack([energy_flat[:, inds].sum(dim=-1) for inds in band_real_idx], dim=-1) # (C, N)
        energy_ratio = torch.cumsum(energy_sorted, dim=-1) # (C, N)
        energy_ratio = energy_ratio / energy_ratio[:, -1:] # (C, N)
        high_freq_areas = 1 - torch.trapz(energy_ratio, dx=1/9, dim=-1)  # (C,)
        avg_amplitude = amplitude.mean(dim=(-2, -1)) # C
    else:
        sorted_idx = torch.argsort(freq_radius_flat) # (H*W)
        freq_sorted = freq_radius_flat[sorted_idx] # (H*W)
        energy_flat = energy.reshape(C, -1) # (C, H*W)
        energy_sorted = energy_flat[:, sorted_idx] # (C, H*W)
        energy_ratio = torch.cumsum(energy_sorted, dim=-1) # (C, H*W)
        energy_ratio = energy_ratio / energy_ratio[:, -1:] # (C, H*W)
        X = (freq_sorted - freq_sorted.min()) / (freq_sorted.max() - freq_sorted.min())
        high_freq_areas = 1 - torch.trapz(energy_ratio, X.expand(C, -1), dim=-1)  # (C,)
        avg_amplitude = amplitude.mean(dim=(-2, -1)) # C
    
    return fft2d, freq_sorted, energy_sorted, high_freq_areas, energy_ratio, avg_amplitude
    
def fourierInvTransform(fft2d, H, W, num_bands, use_log=False):
    
    device = fft2d.device
    C = fft2d.shape[0]
    
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.rfftfreq(W, device=device)
    freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2) # (H, W)

    thresholds = get_band_thresholds(freq_radius, num_bands, use_log=use_log)
    
    max_batch = 64
    recon_list = []
    for start_band in range(0, num_bands, max_batch):
        end_band = min(start_band + max_batch, num_bands)
        batch_size = end_band - start_band
        
        batch_masks = []
        for band in range(start_band, end_band):
            mask = (freq_radius >= thresholds[band]) & (freq_radius < thresholds[band + 1])
            batch_masks.append(mask)
        batch_masks = torch.stack(batch_masks, dim=0)  # (batch_size, H, W)
        batch_masks = batch_masks.unsqueeze(0).expand(C, -1, -1, -1)  # (C, batch_size, H, W)
        
        fft2d_expand = fft2d.unsqueeze(1)  # (C, 1, H, W)
        masked_fft = fft2d_expand * batch_masks  # (C, batch_size, H, W)
        masked_fft = masked_fft.reshape(C * batch_size, H, W // 2 + 1)
        
        recon = torch.fft.irfft2(masked_fft, (H, W), dim=(-2, -1)).real  # (C * batch_size, H, W)
        recon = recon.reshape(C, batch_size, H, W)
        
        recon_list.append(recon)

    recon = torch.concat(recon_list, dim=1)  # (C, num_bands, H, W)
    reconstruction_contributions = recon.abs()  # (C, num_bands, H, W)
    
    return reconstruction_contributions

def sliding_windows(data, window_size, stride):
    
    # data: (..., H, W)
    H, W = data.shape[-2:]
    ws_h, ws_w = window_size
    st_h, st_w = stride
    windows = []
    idxs = []
    for y in range(0, H - ws_h + 1, st_h):
        for x in range(0, W - ws_w + 1, st_w):
            win = data[..., y:y+ws_h, x:x+ws_w]
            windows.append(win)
            idxs.append((y, x))
    windows = torch.stack(windows, dim=0)  # (num_windows, ..., ws_h, ws_w)
    return windows, idxs  # 返回窗口和其左上角索引

def get_window_mask(target_bbox, window_idxs, window_size):
    """
    target_bbox: (H, W) bool
    window_idxs: list of (y, x)
    window_size: (ws_h, ws_w)
    return: list of {'type': 'extreme'/'normal'/'edge'}
    """
    H, W = target_bbox.shape
    ws_h, ws_w = window_size
    win_types = []
    for y, x in window_idxs:
        win_mask = target_bbox[y:y+ws_h, x:x+ws_w]
        total = ws_h * ws_w
        extreme_count = win_mask.sum().item()
        if extreme_count == 0:
            win_types.append("normal")
        elif extreme_count > 0:
            win_types.append("extreme")
        else:
            win_types.append("edge")
    
    return win_types

def extract_windows(data, mask, ws_h=10, ws_w=10):
    """
    data: (C, H, W) torch tensor on cuda
    mask: (H, W) torch bool tensor on cuda, True for extreme region
    Returns:
        windows: (num_windows, C, ws_h, ws_w) torch tensor on same device
        win_types: list of str, "normal" or "extreme"
    """
    device = data.device
    C, H, W = data.shape
    mask_np = mask.cpu().numpy()
    normal_mask_np = (~mask).cpu().numpy()

    # ---- 1. 找normal window ----
    normal_windows = []
    normal_types = []
    for i in range(0, H-ws_h+1, ws_h):
        for j in range(0, W-ws_w+1, ws_w):
            submask = mask_np[i:i+ws_h, j:j+ws_w]
            if submask.sum() == 0:
                window = data[:, i:i+ws_h, j:j+ws_w]
                normal_windows.append(window)
                normal_types.append('normal')
    # ---- 2. 找extreme window ----
    extreme_windows = []
    extreme_types = []

    # 连通分量标记
    labeled_mask, num_labels = sk_label(mask_np, connectivity=1, return_num=True)
    for label_id in range(1, num_labels+1):
        region_mask = (labeled_mask == label_id)
        ys, xs = np.where(region_mask)
        if len(ys) == 0:
            continue
        miny, maxy = ys.min(), ys.max()
        minx, maxx = xs.min(), xs.max()
        region_h, region_w = maxy - miny + 1, maxx - minx + 1
        region_submask = region_mask[miny:maxy+1, minx:maxx+1]

        # 对该区域找所有全为True的最大矩形
        # 动态规划法
        # dp[i][j]: 该点为右下角的最大连续1的高度
        h, w = region_submask.shape
        dp = np.zeros_like(region_submask, dtype=int)
        for j in range(w):
            acc = 0
            for i in range(h):
                if region_submask[i,j]:
                    acc += 1
                else:
                    acc = 0
                dp[i,j] = acc

        # 用单调栈法找最大内接矩形
        max_area = 0
        best_rect = None # (top, left, height, width)
        for i in range(h):
            stack = []
            heights = dp[i]
            lefts = [0]*w
            for j in range(w+1):
                cur_h = heights[j] if j < w else 0
                last_j = j
                while stack and cur_h < stack[-1][0]:
                    height, idx = stack.pop()
                    width = j if not stack else j - stack[-1][1] - 1
                    area = height * width
                    if area > max_area and height >= ws_h and width >= ws_w:
                        # 只考虑能放下至少一个window的矩形
                        max_area = area
                        best_rect = (i-height+1, idx, height, width)
                    last_j = idx
                stack.append((cur_h, last_j))
        if best_rect is None:
            continue
        rect_top, rect_left, rect_h, rect_w = best_rect

        # 计算能放多少个window
        n_h = rect_h // ws_h
        n_w = rect_w // ws_w
        if n_h == 0 or n_w == 0:
            continue
        # 如果只能放下一个window，要求居中
        if n_h == 1 and n_w == 1:
            rect_center_y = rect_top + rect_h//2
            rect_center_x = rect_left + rect_w//2
            region_center_y = int(np.round(np.mean(ys) - miny))
            region_center_x = int(np.round(np.mean(xs) - minx))
            # window左上角
            win_topleft_y = region_center_y - ws_h//2
            win_topleft_x = region_center_x - ws_w//2
            # 保证在rect内
            win_topleft_y = max(rect_top, min(win_topleft_y, rect_top+rect_h-ws_h))
            win_topleft_x = max(rect_left, min(win_topleft_x, rect_left+rect_w-ws_w))
            # 转回全局坐标
            win_y = win_topleft_y + miny
            win_x = win_topleft_x + minx
            window_mask = mask_np[win_y:win_y+ws_h, win_x:win_x+ws_w]
            if window_mask.shape == (ws_h, ws_w) and window_mask.all():
                window = data[:, win_y:win_y+ws_h, win_x:win_x+ws_w]
                extreme_windows.append(window)
                extreme_types.append('extreme')
        else:
            # 可以分成多个window，按左上角填充
            for dh in range(n_h):
                for dw in range(n_w):
                    win_y = miny + rect_top + dh*ws_h
                    win_x = minx + rect_left + dw*ws_w
                    window_mask = mask_np[win_y:win_y+ws_h, win_x:win_x+ws_w]
                    if window_mask.shape == (ws_h, ws_w) and window_mask.all():
                        window = data[:, win_y:win_y+ws_h, win_x:win_x+ws_w]
                        extreme_windows.append(window)
                        extreme_types.append('extreme')

    # 拼接
    all_windows = normal_windows + extreme_windows
    all_types = normal_types + extreme_types
    if all_windows:
        windows_tensor = torch.stack([w.to(device) for w in all_windows])
    else:
        windows_tensor = torch.empty((0, C, ws_h, ws_w), device=device)
    return windows_tensor, all_types

def get_extreme_types(bboxes, types, ext_win_idxs, window_size):
    
    ext_types = []
    ext_bboxes = []
    ext_counts = []
    H, W = 530, 900
    ws_h, ws_w = window_size
    MIN_COUNT = 0.1 * (ws_h * ws_w)
    for y, x in ext_win_idxs:
        win_y_end = y + ws_h
        win_x_end = x + ws_w
        type_list = []
        count_list = []
        for bbox, event_type in zip(bboxes, types):
            assert isinstance(event_type, list)
            y_min, x_min, y_max, x_max = [eval(e) for e in bbox]
            
            overlap_x_min = max(x_min, y)
            overlap_y_min = max(y_min, x)
            overlap_x_max = min(x_max, win_y_end)
            overlap_y_max = min(y_max, win_x_end)
            if overlap_y_min < overlap_y_max and overlap_x_min < overlap_x_max:
                extreme_count = (overlap_y_max - overlap_y_min) * (overlap_x_max - overlap_x_min)
            else:
                extreme_count = 0
            count_list.append(extreme_count)
            
            if extreme_count >= MIN_COUNT:
                type_list.append(event_type)
        if len(type_list) == 0:
            type_list.append(types[count_list.index(max(count_list))])
        type_list = list(set([t for ts in type_list for t in ts]))
        ext_types.append(type_list)
        ext_counts.append(count_list)
        
    return ext_types, ext_counts

def explore_frequency_window_fine(surface, upper_air, target_bbox, bboxes, types, strides, times, 
                            save_path=f"./freq_logs/fft2_window/", time_pos=1, 
                            window_size=(10, 10), window_stride=(10, 10), use_log=False):

    data = torch.concat([surface, upper_air.flatten(1, 2)], dim=1) # B, C, H, W
    B, C, H, W = data.shape
    ws_h, ws_w = window_size
    st_h, st_w = window_stride

    for idx in range(B):
        data_img = data[idx]  # (C, H, W)
        bbox_img = target_bbox[idx]  # (H, W)
        win_data, win_idxs = sliding_windows(data_img, window_size, window_stride)  # (num_windows, C, ws_h, ws_w)
        num_windows = win_data.shape[0]
        win_types = get_window_mask(bbox_img, win_idxs, window_size)  # list of str, len=num_windows
        ext_types, ext_counts = get_extreme_types(bboxes[idx], types[idx], [idxs for i, idxs in enumerate(win_idxs) if win_types[i] == "extreme"], window_size)
        nor_win_data = [win_data[i] for i, t in enumerate(win_types) if t == "normal"]
        num_extremes = len(ext_types)
        nor_win_data = random.sample(nor_win_data, num_extremes)
        ext_win_data = [win_data[i] for i, t in enumerate(win_types) if t == "extreme"]
        ext_types = [list(set(ts)) for ts in ext_types]
        
        all_win_data = torch.stack(nor_win_data + ext_win_data, dim=0)
        all_types = [["normal"]] * num_extremes + ext_types
        AN = all_win_data.shape[0]
        fft2d, freq_sorted, energy_sorted, high_freq_areas, energy_ratio, avg_amplitude = fourierTransform(all_win_data.flatten(0, 1), use_log=use_log) 
        fft2d = fft2d.reshape(AN, C, -1).cpu() # complex
        hfas = high_freq_areas.reshape(AN, C).cpu() # real
        avgamps = avg_amplitude.reshape(AN, C).cpu() # real

        freq_results = {}
        for i, ts in enumerate(all_types):
            w_d, hfa, avgamp = fft2d[i], hfas[i], avgamps[i]
            for t in ts:
                if t not in freq_results:
                    freq_results[t] = {"k": [], "v":[], "ID": []}
                freq_results[t]['k'].append((hfa, avgamp))
                freq_results[t]['v'].append(w_d)
                freq_results[t]['ID'].append(i)
        for t, kvs in freq_results.items():
            type_save_path = os.path.join(save_path, f"{t}")
            os.makedirs(type_save_path, exist_ok=True)
            
            k_save_name = os.path.join(type_save_path, f"{times[idx][time_pos].item()}_k.pth")
            v_save_name = os.path.join(type_save_path, f"{times[idx][time_pos].item()}_v.pth")
            torch.save({"k_hfa": torch.stack([k[0] for k in kvs["k"]], dim=0), 
                        "k_avgamp": torch.stack([k[1] for k in kvs["k"]], dim=0), 
                        "ID": kvs["ID"], "v_path": v_save_name}, k_save_name)
            torch.save({"v": torch.stack(kvs["v"], dim=0)}, v_save_name)
        
            # for i in range(len(kvs["k"])):
            #     k = kvs["k"][i]
            #     v = kvs["v"][i]
            #     ID = kvs["ID"][i]
                
            #     k_save_name = os.path.join(type_save_path, f"{times[idx][time_pos].item()}_{ID}_k.pth")
            #     v_save_name = os.path.join(type_save_path, f"{times[idx][time_pos].item()}_{ID}_v.pth")
            #     torch.save({"k": k, "ID": ID, "v_path": v_save_name}, k_save_name)
            #     torch.save({"v": v}, v_save_name)

def worker(val_input_surface, val_input_upper_air, target_bbox, bboxes, types, strides, times, save_path, gpu_id, mode=None, log_flag=False):

    gpu_id = gpu_id % 4
    torch.cuda.set_device(gpu_id)
    val_input_surface = val_input_surface.cuda(gpu_id)
    val_input_upper_air = val_input_upper_air.cuda(gpu_id)
    target_bbox = target_bbox.cuda(gpu_id)
    explore_frequency_window_fine(val_input_surface, val_input_upper_air, target_bbox, bboxes, types, strides, times, save_path, use_log=log_flag)

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn') 
    
    opt = parser.parse_args()
    SEED = opt.seed
    DATA_DIR = opt.data_dir
    MAX_THREADS = opt.max_threads
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    USE_PARALLEL = opt.parallel
    PRETRAINED_MODEL_PATH = opt.pretrained_model_path
    MEAN_STD_DIR = opt.mean_std_dir
    print("pretrained path", PRETRAINED_MODEL_PATH)
    print("mean_std path", MEAN_STD_DIR)
    
    set_seed(SEED)
    DEBUG_FLAG = False
    
    # val_time_span = ["2021010100", "2021123123"]
    val_time_span = ["2022010100", "2022123123"]
    val_set = NOAAExtremeDataComplete(DATA_DIR, val_time_span, "valid", 
                                      train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=min(BATCH_SIZE, MAX_THREADS), 
                             pin_memory=False, persistent_workers=True, prefetch_factor=2, collate_fn=extreme_collate)
    
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, target_bbox, bboxes, types, strides, times in val_bar:
            batch_size = val_input_surface.size(0)
            
            processes = []
            for i in range(batch_size):
                p = multiprocessing.Process(target=worker, args=(
                    val_target_surface[i:i + 1], val_target_upper_air[i:i + 1], target_bbox[i:i + 1],
                    bboxes[i:i + 1], types[i:i + 1], strides[i:i + 1], times[i:i + 1], 
                    f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/freq_tokens/{val_time_span[0]}_{val_time_span[1]}", i))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
            # break    
    
    print("done")
    
    root_dir = f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/freq_tokens/{val_time_span[0]}_{val_time_span[1]}"
    type_path_list = os.listdir(root_dir)
    for type_path in tqdm(type_path_list):
        print(type_path)
        type_path = os.path.join(root_dir, type_path)
        k_hfa_list = []
        k_avgamp_list = []
        k_ID_list = []
        v_list = []
        v_path_list = []
        token_path_list = os.listdir(type_path)
        for token_path in tqdm(token_path_list):
            if '_k.' in token_path:
                k_data = torch.load(os.path.join(type_path, token_path), weights_only=False)
                k_hfa_list.append(k_data['k_hfa'])
                k_avgamp_list.append(k_data['k_avgamp'])
                k_ID_list.append(k_data["ID"])
                v_path_list.append([k_data['v_path']] * len(k_data["ID"]))
                v_data = torch.load(os.path.join(type_path, k_data['v_path']), weights_only=False)
                v_list.append(v_data['v'])
                # print(k_data['k_hfa'].shape, k_data['k_avgamp'].shape, len(k_data["ID"]), k_data['v_path'], v_data['v'].shape)
        k_data_hfa = torch.concat(k_hfa_list, dim=0)
        k_data_avgamp = torch.concat(k_avgamp_list, dim=0)
        k_ID_list = [b for a in k_ID_list for b in a]
        v_path_list = [b for a in v_path_list for b in a]
        v_data = torch.concat(v_list, dim=0)
        torch.save({'hfa': k_data_hfa, 'avgamp': k_data_avgamp, "v_path_list": v_path_list, "ID": k_ID_list}, os.path.join(type_path, "all_ks.pth"))
        torch.save({'v': v_data}, os.path.join(type_path, "all_vs.pth"))
    