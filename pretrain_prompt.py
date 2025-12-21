import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import time as time_repo
from sklearn.metrics import average_precision_score
from transformers import get_cosine_schedule_with_warmup

# vscode Relative path
import sys
# sys.path.append("../../")

from Fuxi_freq import FuxiFreq
from Fuxi_tune import FuxiFreqPrompt, FuxiFreqPromptFreq, FuxiFreqRecon
from data_utils import surface_inv_transform, upper_air_inv_transform, NOAADataComplete, NOAAExtremeDataComplete, NOAACompletePlusExtremeDataComplete, extreme_collate, get_patch_slice
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

parser = argparse.ArgumentParser(description="Fuxi pre-train")
parser.add_argument("--seed", default=42, type=int, help="seed")
parser.add_argument("--data_dir", default="/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/hrrr", 
                    type=str, help="data dir")
parser.add_argument("--max_threads", default=15, type=int, help="number of threads")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--parallel", action="store_false", help="use data parallel")
parser.add_argument("--pretrained_model_path", default='', type=str, help="pretrained model path")
parser.add_argument("--continue_train", action="store_true", help="continue train")
parser.add_argument("--patience", default=5, type=int, help="patience")
parser.add_argument("--prompt_style", type=str, default="dual", help="prompt style")
parser.add_argument("--pooling_style", type=str, default="mean", help="pooling style")
parser.add_argument("--attn_loss", action="store_true", help="attn loss")
parser.add_argument("--no_two_prompts", action="store_true", help="two")
parser.add_argument("--version", type=str, default='', help="version")
parser.add_argument("--steplr", action="store_true", help="step lr")
parser.add_argument("--attn_nonor", action="store_true", help="no normal in attn loss")
parser.add_argument("--attn_weight", action="store_true", help="no normal in attn loss")

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
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
    assert len(filtered_dict) > 0

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoints["optimizer"])
        scheduler.load_state_dict(checkpoints["scheduler"])
    
    return filtered_dict

def sliding_windows(data, window_size=(10, 10), stride=(10, 10)):
    
    # data: (..., H, W)
    H, W = data.shape[-2:]
    ws_h, ws_w = window_size
    st_h, st_w = stride
    # windows = []
    idxs = []
    for y in range(0, H - ws_h + 1, st_h):
        for x in range(0, W - ws_w + 1, st_w):
            # win = data[..., y:y+ws_h, x:x+ws_w]
            # windows.append(win)
            idxs.append((y, x))
    # windows = torch.stack(windows, dim=1)
    windows = None
    return windows, idxs

def get_window_types(target_bbox, bboxes, types, window_idxs, window_size=(10, 10)):
    
    H, W = target_bbox.shape
    ws_h, ws_w = window_size
    win_types = []
    MIN_COUNT = 0.1 * (ws_h * ws_w)
    for y, x in window_idxs:
        win_mask = target_bbox[y:y+ws_h, x:x+ws_w]
        total = ws_h * ws_w
        extreme_count = win_mask.sum().item()
        if extreme_count == 0:
            win_types.append(["normal"])
        elif extreme_count > 0:
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
            win_types.append(type_list)
        else:
            raise ValueError
    
    return win_types

if __name__ == "__main__":
    
    opt = parser.parse_args()
    SEED = opt.seed
    DATA_DIR = opt.data_dir
    MAX_THREADS = opt.max_threads
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    USE_PARALLEL = opt.parallel
    PRETRAINED_MODEL_PATH = opt.pretrained_model_path
    CONTINUE_TRAIN = opt.continue_train
    ATTN_LOSS = opt.attn_loss
    NO_TWO_PROMPTS = opt.no_two_prompts
    VERSION = opt.version
    STEPLR = opt.steplr
    ATTN_NONOR = opt.attn_nonor
    ATTN_WEIGHTED = opt.attn_weight
    print("pretrained path", PRETRAINED_MODEL_PATH, VERSION)
    
    PATIENCE = opt.patience
    PROMPT_STYLE = opt.prompt_style
    POOLING_STYLE = opt.pooling_style
    print(PROMPT_STYLE, POOLING_STYLE, VERSION)
    
    set_seed(SEED)
    DEBUG_FLAG = False

    train_time_span, val_time_span = ["2019010100", "2022123123"], ["2024010100", "2024123123"]
    MEAN_STD_DIR = f'/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/raw/mean_std/2019010100-2022123123'
    if ATTN_LOSS:
        train_set =  NOAACompletePlusExtremeDataComplete(DATA_DIR, train_time_span, "train", 
                                            train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_THREADS, 
                                   pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    else:
        train_set = NOAADataComplete(DATA_DIR, train_time_span, "train", debug_flag=DEBUG_FLAG)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_THREADS, 
                                pin_memory=False, persistent_workers=False, prefetch_factor=2)
    val_set = NOAAExtremeDataComplete(DATA_DIR, val_time_span, "valid", 
                                      train_mean_std_dir=MEAN_STD_DIR, debug_flag=DEBUG_FLAG)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=MAX_THREADS, 
                             pin_memory=False, persistent_workers=False, prefetch_factor=2, collate_fn=extreme_collate)
    print("# train samples: ", len(train_set))
    print("# val samples: ", len(val_set))
    
    if VERSION == "freq_prompt":
        PostFix = f'{VERSION}_{PROMPT_STYLE}_{POOLING_STYLE}'
        if BATCH_SIZE != 16:
            PostFix += f"_bs{BATCH_SIZE}"
        fuxi = FuxiFreqPrompt(prompt_style=PROMPT_STYLE, pooling_style=POOLING_STYLE, two_prompts=not NO_TWO_PROMPTS, 
            weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/pooling_embeds/Fuxi_freq/2022010100_2022123123"
            # weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/pooling_embeds/Fuxi_freq/2022010100_2022123123_no_normal"
            # weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/pooling_embeds/raw/2022010100_2022123123_no_normal/mean.pth"
        )
    elif VERSION == "freq":
        PostFix = f'{VERSION}'
        if BATCH_SIZE != 16:
            PostFix += f"_bs{BATCH_SIZE}"
        fuxi = FuxiFreq()
    elif VERSION == "freq_prompt_freq":
        PostFix = f'{VERSION}_p+x_t5'
        if BATCH_SIZE != 8:
            PostFix += f"_bs{BATCH_SIZE}"
        fuxi = FuxiFreqPromptFreq(
            weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/freq_tokens/2022010100_2022123123")
    elif VERSION == "freq_prompt_space":
        PostFix = f'{VERSION}_p+x_t5'
        if BATCH_SIZE != 8:
            PostFix += f"_bs{BATCH_SIZE}"
        fuxi = FuxiFreqPromptFreq(
            weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/space_tokens/2022010100_2022123123", 
            use_space=True)
    elif VERSION == "freq_recon":
        PostFix = f'{VERSION}'
        if BATCH_SIZE != 16:
            PostFix += f"_bs{BATCH_SIZE}"
        fuxi = FuxiFreqRecon()
    else:
        raise ValueError
    
    if VERSION.startswith("freq_prompt"):
        if ATTN_LOSS:
            PostFix += '_kl'
            if ATTN_NONOR:
                PostFix += '_nonor'
            if ATTN_WEIGHTED:
                PostFix += '_balance'
        if NO_TWO_PROMPTS:
            PostFix += "_one"
    if STEPLR:
        PostFix += "_steplr"
    checkpoint_path = f"checkpoints_tune_pretrain_prompt/{train_time_span[0]}_{train_time_span[1]}_{PostFix}"
    save_root = "pretrain_prompt_logs"
    save_root += f"/{train_time_span[0]}_{train_time_span[1]}"
    log_path = f"logs_{PostFix}.csv"
    print(save_root, log_path, checkpoint_path)
    trainable_params = sum(p.numel() for p in fuxi.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in fuxi.parameters())
    trainable_ratio = trainable_params / total_params
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable ratio: {trainable_ratio:.2%}")
    # exit(-1)
        
    criterion_1 = nn.L1Loss(reduction='none')
    criterion_2 = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, fuxi.parameters()), lr=1e-3, weight_decay=3e-6)
    if STEPLR:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7)
    results = {'loss': [], 'fore_loss': [], 'attn_loss': [], 'val_metric': [], "ext_metric": [], "gen_metric": [], 
            'val_metric_l1': [], "ext_metric_l1": [], "gen_metric_l1": [], 
            "best_flag": []}
    best_val = float('inf')
    best_epoch = -1
    break_flag = False
    patience_count = 0
    START_EPOCH = 1
    
    if ATTN_LOSS:
        if VERSION == "freq_prompt_freq" or VERSION == "freq_prompt_space":
            event_classes = fuxi.freq_filter.prompt_attn.event_classes
        elif VERSION =="freq_prompt":
            event_classes = fuxi.prompt_attn.event_classes
        else:
            raise ValueError
        event2idx = {c: i for i, c in enumerate(event_classes)}
        class_counts, event_counts = train_set.class_counts, train_set.event_counts
        # class_counts, event_counts = val_set.class_counts, val_set.event_counts
        pos_counts = torch.tensor([class_counts[e] if e in class_counts else 0 for e in event_classes]).float()
        if "normal" in event_classes:
            event_counts += pos_counts[event2idx["normal"]]
        assert pos_counts.min() > 0
        valid_mask = pos_counts > 0
        pos_weights = torch.zeros_like(pos_counts)
        pos_weights[valid_mask] = event_counts / pos_counts[valid_mask]
        # pos_weights /= pos_weights.max()

    if CONTINUE_TRAIN:
        file_path = os.path.join(save_root, log_path)
        data_frame = pd.read_csv(file_path, index_col="Epoch")
        last_epoch = data_frame.index[-1]
        results = data_frame.to_dict(orient='list')
        assert last_epoch > 0
        min_val_metric_row = data_frame.loc[data_frame['val_metric'].idxmin()]
        best_val = min_val_metric_row['val_metric']
        best_epoch = min_val_metric_row.name 
        
        last_true_index = data_frame[data_frame['best_flag'] == True].index.max()
        if pd.isna(last_true_index):
            patience_count = 0
            raise ValueError
        else:
            last_true_position = data_frame.index.get_loc(last_true_index)
            patience_count = (data_frame.iloc[last_true_position + 1:]['best_flag'] == False).sum()
        
        assert patience_count < PATIENCE
        model_load_path = f"{checkpoint_path}/fuxi_epoch_{last_epoch}.pth"
        load_pretrained_weights(fuxi, optimizer, scheduler, model_load_path)
        print("# continued trainable parameters: ", sum(param.numel() for param in fuxi.parameters() if param.requires_grad))
        print(f"last epoch: {last_epoch}, best epoch: {best_epoch}, best val: {best_val}, patience count: {patience_count}")
        
        # scheduler._reduce_lr(last_epoch)
        
    if torch.cuda.is_available():
        fuxi.cuda()
        if USE_PARALLEL:
            fuxi = torch.nn.DataParallel(fuxi, device_ids=[0, 1, 2, 3])
        criterion_1.cuda()
        criterion_2.cuda() 
        if CONTINUE_TRAIN:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
         
    for epoch in range(START_EPOCH, NUM_EPOCHS + START_EPOCH):
        
        torch.manual_seed(SEED + epoch)
        if CONTINUE_TRAIN and epoch <= last_epoch:
            continue
        
        if epoch > 0:
            train_bar = tqdm(train_loader)
            running_results = {"batch_sizes": 0, "loss": 0, "fore_loss": 0, "attn_loss": 0}
            fuxi.train()
            for train_elements in train_bar:
                if ATTN_LOSS:
                    input_surface, input_upper_air, target_surface, target_upper_air, target_bbox, bboxes, types, stride, times = train_elements
                else:
                    input_surface, input_upper_air, target_surface, target_upper_air, times = train_elements
                batch_size = input_surface.size(0)
                inputs = torch.concat([input_surface, input_upper_air.flatten(1, 2)], dim=1)
                targets = torch.concat([target_surface, target_upper_air.flatten(1, 2)], dim=1)
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    # target_bbox = target_bbox.cuda()
                    times = times.cuda()
                optimizer.zero_grad()
                if VERSION =="freq_prompt":
                    if ATTN_LOSS:
                        outputs, _, attn_weights = fuxi(inputs, times)
                    else:
                        outputs, _, _ = fuxi(inputs, times)
                elif VERSION == "freq":
                    outputs, _ = fuxi(inputs, times)
                elif VERSION == "freq_prompt_freq" or VERSION == "freq_prompt_space":
                    if ATTN_LOSS:
                        outputs, _, attn_weights = torch.utils.checkpoint.checkpoint(fuxi, inputs, times, use_reentrant=False)
                    else:
                        outputs, _, _ = torch.utils.checkpoint.checkpoint(fuxi, inputs, times, use_reentrant=False)
                elif VERSION == "freq_recon":
                    outputs, _, outputs_recon = fuxi(inputs, times)
                else:
                    raise ValueError
                fore_loss = criterion_1(outputs, targets).mean()
                
                if VERSION == "freq_recon":
                    recon_loss = criterion_2(outputs_recon, inputs).mean()
                if ATTN_LOSS:
                    if VERSION == "freq_prompt_freq" or VERSION == "freq_prompt_space":
                        win_data, win_idxs = sliding_windows(targets)
                        attn_gt = torch.zeros(batch_size, len(win_idxs), len(event_classes))
                        if ATTN_WEIGHTED:
                            attn_pos_weights = torch.zeros(batch_size, len(win_idxs))
                        for i in range(batch_size):
                            win_types = get_window_types(target_bbox[i], bboxes[i], types[i], win_idxs)
                            for j in range(len(win_idxs)):
                                if ATTN_NONOR and "normal" in win_types[j]:
                                    continue
                                fill_event_idxs = [event2idx[e] for e in win_types[j]] 
                                attn_gt[i, j, fill_event_idxs] = 1.
                                if ATTN_WEIGHTED:
                                    attn_pos_weights[i, j] = event_counts / sum([pos_counts[event_i] for event_i in fill_event_idxs])
                        if ATTN_WEIGHTED:
                            attn_gt = attn_gt.reshape(-1, len(event_classes)).cuda()
                            attn_pos_weights = attn_pos_weights.reshape(-1, 1).cuda()
                            attn_gt_smooth = torch.where(attn_gt == 1, 0.9 + 0.1 / len(event_classes), 0.1 / len(event_classes))
                            attn_loss = 0.01 * (F.kl_div(torch.log(attn_weights + 1e-10), attn_gt_smooth, reduction="none") * attn_pos_weights).mean()
                        else:
                            attn_gt = attn_gt.reshape(-1, len(event_classes)).cuda()
                            attn_mask = attn_gt.sum(dim=-1) > 0
                            if ATTN_NONOR:
                                attn_mask.sum(dim=-1).min() == 0
                            else:
                                attn_mask.sum(dim=-1).min() > 0
                            attn_gt, attn_weights = attn_gt[attn_mask], attn_weights[attn_mask]
                            attn_gt_smooth = torch.where(attn_gt == 1, 0.9 + 0.1 / len(event_classes), 0.1 / len(event_classes))
                            attn_loss = 0.01 * F.kl_div(torch.log(attn_weights + 1e-10), attn_gt_smooth, reduction='mean')
                    elif VERSION == "freq_prompt":
                        attn_gt = torch.zeros(batch_size, 66, 112, len(event_classes))
                        nor_tensor = torch.zeros(len(event_classes))
                        nor_tensor[event2idx["normal"]] = 1.
                        for inst_idx, bbox_inst in enumerate(bboxes):
                            for sample_idx, bbox in enumerate(bbox_inst):
                                h_slice, w_slice = get_patch_slice(bbox, (8, 8))
                                event_types = types[inst_idx][sample_idx]
                                for event_type in event_types:
                                    attn_gt[inst_idx, h_slice, w_slice, event2idx[event_type]] = 1.
                            if "normal" in event_classes and not ATTN_NONOR:
                                ext_mask = attn_gt.sum(dim=-1) > 0
                                attn_gt[~ext_mask] = nor_tensor
                        attn_gt = attn_gt.cuda()
                        if not NO_TWO_PROMPTS:
                            attn_gt = attn_gt.reshape(-1, len(event_classes)).unsqueeze(-2).expand(-1, 2, -1).reshape(-1, len(event_classes))
                            attn_weights = attn_weights.reshape(-1, len(event_classes))
                        else:
                            attn_gt, attn_weights = attn_gt.reshape(-1, len(event_classes)), attn_weights.reshape(-1, len(event_classes))
                        attn_mask = attn_gt.sum(dim=-1) > 0
                        attn_gt, attn_weights = attn_gt[attn_mask], attn_weights[attn_mask]
                        attn_gt_smooth = torch.where(attn_gt == 1, attn_gt * 0.9 + 0.1 / len(event_classes), 0.1 / len(event_classes))
                        # attn_loss = 0.001 * (F.binary_cross_entropy(attn_weights, attn_gt_smooth, reduction='none') * (attn_gt * (pos_weights.unsqueeze(0).cuda() - 1) + 1)).mean()
                        attn_loss = 0.01 * F.kl_div(torch.log(attn_weights + 1e-10), attn_gt_smooth, reduction='none').mean()
                    else:
                        raise ValueError
                else:
                    attn_loss = torch.tensor(0.)
                loss = fore_loss + attn_loss
                if torch.isnan(loss):
                    exit(-1)
                loss.backward()
                optimizer.step()

                running_results["loss"] += loss.item() * batch_size
                running_results["fore_loss"] += fore_loss.item() * batch_size
                running_results["attn_loss"] += attn_loss.item() * batch_size
                running_results["batch_sizes"] += batch_size
                train_bar.set_description(desc="[%d/%d] Loss: %.3e(%.3e+%.3e) Lr: %.3e" % 
                                        (epoch, NUM_EPOCHS, 
                                        running_results["loss"] / running_results["batch_sizes"], 
                                        running_results["fore_loss"] / running_results["batch_sizes"], 
                                        running_results["attn_loss"] / running_results["batch_sizes"], 
                                        optimizer.param_groups[0]['lr']))
                # break
            del inputs, targets, outputs
            if ATTN_LOSS and VERSION.startswith("freq_prompt"):
                del attn_weights, attn_gt, attn_gt_smooth
            torch.cuda.empty_cache()

        fuxi.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "val_metric": 0, "ext_metric": 0, "gen_metric": 0, 
                              'val_metric_l1': 0, "ext_metric_l1": 0, "gen_metric_l1": 0} 
            for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, target_bbox, bboxes, types, stride, times in val_bar:
                batch_size = val_input_surface.size(0)
                val_inputs = torch.concat([val_input_surface, val_input_upper_air.flatten(1, 2)], dim=1)
                val_targets = torch.concat([val_target_surface, val_target_upper_air.flatten(1, 2)], dim=1)
                if torch.cuda.is_available():
                    val_inputs = val_inputs.cuda()
                    val_targets = val_targets.cuda()
                    target_bbox = target_bbox.cuda()
                    times = times.cuda()
                if VERSION.startswith("freq_prompt"):
                    val_outputs, _, _ = fuxi(val_inputs, times)
                elif VERSION == "freq":
                    val_outputs, _ = fuxi(val_inputs, times)
                else:
                    raise ValueError
                
                ext_mask = target_bbox.unsqueeze(1).expand(-1, 69, -1, -1)
                all_mse = criterion_2(val_outputs, val_targets)
                val_ext_metric = ((all_mse * ext_mask).sum(dim=(-2, -1)) / (ext_mask.sum(dim=(-2, -1)) + 1e-10)).detach().mean().cpu().item()
                val_gen_metric = all_mse.detach().mean().cpu().item()
                val_metric = val_ext_metric
                all_l1 = criterion_1(val_outputs, val_targets)
                val_ext_metric_l1 = ((all_l1 * ext_mask).sum(dim=(-2, -1)) / (ext_mask.sum(dim=(-2, -1)) + 1e-10)).detach().mean().cpu().item()
                val_gen_metric_l1 = all_l1.detach().mean().cpu().item()
                val_metric_l1 = val_ext_metric_l1
                
                valing_results["val_metric"] += val_metric * batch_size
                valing_results["ext_metric"] += val_ext_metric * batch_size
                valing_results["gen_metric"] += val_gen_metric * batch_size
                valing_results["val_metric_l1"] += val_metric_l1 * batch_size
                valing_results["ext_metric_l1"] += val_ext_metric_l1 * batch_size
                valing_results["gen_metric_l1"] += val_gen_metric_l1 * batch_size
                valing_results["batch_sizes"] += batch_size
                val_bar.set_description(desc="[validating] Val MSE: %.3e Val Ext. MSE: %.3e Val Gen. MSE: %.3e" % 
                                        (valing_results["val_metric"] / valing_results["batch_sizes"], 
                                        valing_results["ext_metric"] / valing_results["batch_sizes"], 
                                        valing_results["gen_metric"] / valing_results["batch_sizes"]))
                # break
        del val_inputs, val_targets, val_outputs, all_mse, all_l1, ext_mask, target_bbox
        torch.cuda.empty_cache()

        if epoch == 0:
            results["loss"].append(0.)
            results["fore_loss"].append(0.)
            results["attn_loss"].append(0.)
        else:
            results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
            results["fore_loss"].append(running_results["fore_loss"] / running_results["batch_sizes"])
            results["attn_loss"].append(running_results["attn_loss"] / running_results["batch_sizes"])
        val_metric = valing_results["val_metric"] / valing_results["batch_sizes"]
        results["val_metric"].append(val_metric)
        ext_metric = valing_results["ext_metric"] / valing_results["batch_sizes"]
        results["ext_metric"].append(ext_metric)
        gen_metric = valing_results["gen_metric"] / valing_results["batch_sizes"]
        results["gen_metric"].append(gen_metric)
        val_metric_l1 = valing_results["val_metric_l1"] / valing_results["batch_sizes"]
        results["val_metric_l1"].append(val_metric_l1)
        ext_metric_l1 = valing_results["ext_metric_l1"] / valing_results["batch_sizes"]
        results["ext_metric_l1"].append(ext_metric_l1)
        gen_metric_l1 = valing_results["gen_metric_l1"] / valing_results["batch_sizes"]
        results["gen_metric_l1"].append(gen_metric_l1)
        if STEPLR:
            scheduler.step()
        else:
            # scheduler.step(val_metric)
            scheduler.step(val_metric_l1)
        
        if epoch > 0 and epoch % 1 == 0:
            os.makedirs(f"{checkpoint_path}", exist_ok=True)
            if USE_PARALLEL:
                torch.save({
                        "model": fuxi.module.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "scheduler": scheduler.state_dict()
                    }, f"{checkpoint_path}/fuxi_epoch_{epoch}.pth")
            else:
                torch.save({
                        "model": fuxi.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "scheduler": scheduler.state_dict()
                    }, f"{checkpoint_path}/fuxi_epoch_{epoch}.pth")
        if epoch == 0 or val_metric_l1 < best_val:
            best_val = val_metric_l1
            best_epoch = epoch
            patience_count = 0
            results["best_flag"].append(True)
            if epoch > 0:
                if USE_PARALLEL:
                    torch.save({
                            "model": fuxi.module.state_dict(), 
                            "optimizer": optimizer.state_dict(), 
                            "scheduler": scheduler.state_dict()
                        }, f"{checkpoint_path}/fuxi_best.pth")
                else:
                    torch.save({
                            "model": fuxi.state_dict(), 
                            "optimizer": optimizer.state_dict(), 
                            "scheduler": scheduler.state_dict()
                        }, f"{checkpoint_path}/fuxi_best.pth")
        else:
            patience_count += 1
            results["best_flag"].append(False)
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}")
                break_flag = True
        
        # print(results)
        data_frame = pd.DataFrame(data=results, index=range(START_EPOCH, START_EPOCH + len(results["loss"])))
        os.makedirs(save_root, exist_ok=True)
        data_frame.to_csv(os.path.join(save_root, log_path), index_label="Epoch")
        torch.cuda.empty_cache()
        if break_flag:
            break
        # break
