import os
import math
import random
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from prompt_attention import PromptDualAttentionFreq, PromptDualAttentionSpace

EPS = 1e-5

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # torch.backends.cudnn.benchmark = False     # 关闭优化算法, 保证可复现
    os.environ['PYTHONHASHSEED'] = str(seed)

class TimeEmbedding(nn.Module):
    
    def __init__(self, emb_dim, year_range=(2016, 2025)):
        super().__init__()
        self.year_min, self.year_max = year_range
        self.year_emb = nn.Embedding(self.year_max - self.year_min + 1, emb_dim)
        self.month_emb = nn.Embedding(12 + 1, emb_dim)
        self.day_emb = nn.Embedding(31 + 1, emb_dim)
        self.hour_emb = nn.Embedding(24, emb_dim)

        self.use_periodic = True
        if self.use_periodic:
            self.periodic_dim = emb_dim // 3
            self.periodic_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        
        # x: (batch_size, 2)
        x = x[:, 0]
        years = x // 1000000
        months = (x // 10000) % 100
        days = (x // 100) % 100
        hours = x % 100
        
        year_emb = self.year_emb(years - self.year_min)
        month_emb = self.month_emb(months)
        day_emb = self.day_emb(days)
        hour_emb = self.hour_emb(hours)
        
        # emb = year_emb + month_emb + day_emb + hour_emb
        emb = month_emb + day_emb + hour_emb
        
        if self.use_periodic:
            
            def periodic(x, period):
                x = x.float() / period
                feat = [torch.sin(2 * torch.pi * x * i).unsqueeze(-1) for i in range(1, self.periodic_dim // 2 + 1)]
                feat += [torch.cos(2 * torch.pi * x * i).unsqueeze(-1) for i in range(1, self.periodic_dim // 2 + 1)]
                return torch.concat(feat, dim=-1)
            
            p_month = periodic(months, 12)
            p_day = periodic(days, 31)
            p_hour = periodic(hours, 24)
            # p_year = periodic(years - self.year_min, self.year_max - self.year_min + 1)
            # periodic_feat = torch.concat([p_year, p_month, p_day, p_hour], dim=-1).to(x.device)
            periodic_feat = torch.concat([p_month, p_day, p_hour], dim=-1).to(x.device)
            emb = emb + self.periodic_proj(periodic_feat)
        
        return emb  # shape: (B, D)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_band_splits(freq_radius, num_bands, log_ratio=None, use_log=False, eps=1., accord_val=False, add_init=True):

    freq_sorted, indices_sorted = torch.sort(freq_radius)
    band_idx = []
    if use_log:
        assert log_ratio is not None
        log_ratio = log_ratio[0] if accord_val else log_ratio[1]
    
    if accord_val:
        max_freq, min_freq = freq_sorted.max(), freq_sorted[1]
        if not use_log:
            thresholds = [max_freq * i / num_bands for i in range(num_bands + 1)]
        else:
            r = log_ratio
            total = (r ** num_bands - 1) / (r - 1)
            base = max_freq / total
            print(base, max_freq, min_freq, freq_sorted)
            assert base > min_freq
            bandwidths = [base * r ** i for i in range(num_bands)]
            thresholds = [0.0]
            cur = 0.0
            for bw in bandwidths:
                cur += bw.item()
                thresholds.append(cur)  
        thresholds[-1] = max_freq + eps
        
        if use_log and add_init:
            band_idx.append(torch.arange(0, 1).long())
            thresholds[0] = min_freq
        for i in range(num_bands):
            mask = (freq_sorted >= thresholds[i]) & (freq_sorted < thresholds[i + 1])
            band_idx.append(torch.nonzero(mask, as_tuple=True)[0])
    else:
        T = freq_sorted.numel()
        if not use_log:
            indices = [round(i * T / num_bands) for i in range(num_bands + 1)]
        else:
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
            
        if use_log and add_init:
            band_idx.append(torch.arange(0, 1).long())
            indices[0] = 1
        for i in range(num_bands):
            band_idx.append(torch.arange(indices[i], indices[i + 1]).long())
    
    band_real_idx = [indices_sorted[inds] for inds in band_idx]
        
    return indices_sorted, band_idx, band_real_idx


class BandPassFilterRaw(nn.Module):
    
    def __init__(self, H, W, C, time_dim=72, num_bands=32, band_scale="log", accord_val=False, band_kernel="beta", process_step="space", batch_size=None):
        
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.N = num_bands
        self.process_step = process_step
        self.current_epoch = -1
        self.batch_size = batch_size
        
        self.T = H * (W // 2 + 1)
        freq_y = torch.fft.fftfreq(H)
        freq_x = torch.fft.rfftfreq(W)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2)
        self.freq_radius_flat = freq_radius.flatten() # T
        # self.sorted_idx = torch.argsort(self.freq_radius_flat) # T: sorted idx -> real idx
        
        self.band_scale = band_scale
        if band_scale == "uniform":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands - 1, log_ratio=None, use_log=False, accord_val=accord_val)
        elif band_scale == "log":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands - 1, log_ratio=(1.025, 1.2), use_log=True, accord_val=accord_val)
        else:
            raise ValueError
        
        self.band_kernel = band_kernel
        if band_kernel == "rec":
            pass
        elif band_kernel == "gauss":
            self.all_fr = self.freq_radius_flat # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.std_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.std_predictor.weight, 0.)
            nn.init.constant_(self.std_predictor.bias, 1.)
        elif band_kernel == "beta":
            self.all_fr = (self.freq_radius_flat - self.freq_radius_flat.min()) / (self.freq_radius_flat.max() - self.freq_radius_flat.min()) # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.albeta_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.albeta_predictor.weight, 0.)
            nn.init.constant_(self.albeta_predictor.bias, -5.)
        elif band_kernel == "learn":
            # self.kernel_params = nn.Parameter(torch.ones((C, self.N, self.T)))
            self.point_predictor = nn.Linear(2, self.N)
        else:
            raise ValueError
        
        self.time_embed_layer = TimeEmbedding(time_dim)
        self.cnn = nn.Sequential(nn.Conv2d(1, self.N, 1), nn.LeakyReLU())
        self.fc_t = nn.Sequential(nn.Linear(time_dim, self.N), nn.LeakyReLU())
        self.fuse = nn.Linear(self.N * 2, self.N)
        self.ffn = Mlp(self.C)
        self.norm_ffn = nn.LayerNorm(C)
        
    def freq_direct_trans(self, x):
        
        return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

    def freq_inverse_trans(self, fft2d, s):

        return torch.fft.irfft2(fft2d, s, dim=(-2, -1), norm="ortho").real
    
    def route(self, x, time_embeds):
        
        B, C, H, W = x.shape
        x = x.flatten(0, 1)[:, None, :, :]
        w = self.cnn(x).reshape(B, C, self.N, H, W) # B, C, N, H, W
        t = self.fc_t(time_embeds) # B, N
        weight = self.fuse(
            torch.concat(
                [w, t[:, None, :, None, None].expand(-1, C, -1, H, W)], 
                dim=2).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3) # B, C, N, H, W

        return weight
    
    def forward(self, x, times):

        B, C, H, W = x.shape
        fft2d = self.freq_direct_trans(x) # B, C, H, ...
        freq_feats = torch.view_as_real(fft2d.reshape(B, C, self.T)) # B, C, T, 2
        
        time_embeds = self.time_embed_layer(times) # B, D
        band_weight = F.softmax(self.route(x, time_embeds), dim=2) * self.N # B, C, N, H, W
        if self.process_step == "freq":
            band_weight = self.freq_direct_trans(band_weight).reshape(B, C, self.N, self.T) # B, C, N, T
        
        if self.batch_size is not None:
            if self.process_step == "space":
                y = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.zeros(B, C, cur_batch_size, self.T, device=band_weight.device)
                        for i, band_idx in enumerate(range(batch_idx * self.batch_size, end)):
                            indices = self.band_real_idx[band_idx]
                            kernel_band[:, :, i, indices.to(band_weight.device)] = 1
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice] * batch_y).sum(dim=2) # B, C, H, W      
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice] * batch_y).sum(dim=2) # B, C, H, W
                else:
                    raise ValueError
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                weighted_kernel_band = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    weighted_kernel_band = torch.zeros(B, C, self.T, device=band_weight.device, dtype=torch.complex64)
                    for i, indices in enumerate(self.band_real_idx):
                        weighted_kernel_band[:, :, indices.to(band_weight.device)] = band_weight[:, :, i, indices.to(band_weight.device)]
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        weighted_kernel_band += (band_weight[:, :, batch_slice] * kernel_band).sum(dim=2) # B, C, T
                else:
                    raise ValueError
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        else:
            if self.band_kernel == "rec":
                kernel_band = torch.zeros(B, C, self.N, self.T, device=band_weight.device)
                for i, indices in enumerate(self.band_real_idx):
                    kernel_band[:, :, i, indices.to(band_weight.device)] = 1
            elif self.band_kernel == "gauss":
                std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                std = std.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                kernel_band = torch.exp(- (all_fr - band_means) ** 2 / (2 * std ** 2)) # (B, C, N, T)
            elif self.band_kernel == "beta":
                # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                alpha = band_means * (albeta - 2) + 1 # (B, C, N, 1)
                beta = (1 - band_means) * (albeta - 2) + 1 # (B, C, N, 1)
                peak = (band_means ** (alpha - 1)) * ((1 - band_means) ** (beta - 1)) # (B, C, N, 1)
                assert peak.min() > 0
                kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, N, T)
            elif self.band_kernel == "learn":
                kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)) # B, C, N, T
                kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, N, T
            else:
                raise ValueError
            
            if self.process_step == "space":
                fft2d = fft2d.unsqueeze(2).expand(-1, -1, self.N, -1, -1)
                y = self.freq_inverse_trans(fft2d * kernel_band.reshape(B, C, self.N, H, -1), (H, W)) # (B, C, N, H, W)
                y = (band_weight * y).sum(dim=2) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                weighted_kernel_band = (band_weight * kernel_band).sum(dim=2) # B, C, T
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        
        return y


class BandPassFilterWin(nn.Module):
    
    def __init__(self, C, win_size=(10, 10), time_dim=72, num_bands=10, band_scale="log", accord_val=False, band_kernel="beta", process_step="freq", batch_size=None):
        
        super().__init__()
        H = win_size[0]
        W = win_size[1]
        self.C = C
        self.N = num_bands
        self.win_size = win_size
        self.process_step = process_step
        self.current_epoch = -1
        self.batch_size = batch_size
        
        self.T = H * (W // 2 + 1)
        freq_y = torch.fft.fftfreq(H)
        freq_x = torch.fft.rfftfreq(W)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2)
        self.freq_radius_flat = freq_radius.flatten() # T
        # self.sorted_idx = torch.argsort(self.freq_radius_flat) # T: sorted idx -> real idx
        
        self.band_scale = band_scale
        if band_scale == "uniform":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands, log_ratio=None, use_log=False, accord_val=accord_val)
        elif band_scale == "log":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands - 1, log_ratio=(None, 1.3), use_log=True, accord_val=accord_val)
        else:
            raise ValueError
        
        self.band_kernel = band_kernel
        if band_kernel == "rec":
            pass
        elif band_kernel == "gauss":
            self.all_fr = self.freq_radius_flat # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.std_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.std_predictor.weight, 0.)
            nn.init.constant_(self.std_predictor.bias, 1.)
        elif band_kernel == "beta":
            self.all_fr = (self.freq_radius_flat - self.freq_radius_flat.min()) / (self.freq_radius_flat.max() - self.freq_radius_flat.min()) # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.albeta_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.albeta_predictor.weight, 0.)
            nn.init.constant_(self.albeta_predictor.bias, -5.)
        elif band_kernel == "learn":
            # self.kernel_params = nn.Parameter(torch.ones((C, self.N, self.T)))
            self.point_predictor = nn.Linear(2, self.N)
        else:
            raise ValueError
        
        self.time_embed_layer = TimeEmbedding(time_dim)
        # self.fc_x = nn.Sequential(nn.Linear(win_size[0] * win_size[1], self.N), nn.LeakyReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.N, 3, stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc_t = nn.Sequential(nn.Linear(time_dim, self.N), nn.LeakyReLU())
        self.fuse = nn.Linear(self.N * 2, self.N)
        self.ffn = Mlp(self.C)
        self.norm_ffn = nn.LayerNorm(C)
    
    def split_windows(self, x):

        B, C, H, W = x.shape
        H_w, W_w = self.win_size
        assert H % H_w == 0 and W % W_w == 0, "H and W must be divisible by H_w and W_w"
        nH = H // H_w
        nW = W // W_w

        # (B, C, H, W) -> (B, C, nH, H_w, nW, W_w)
        x = x.view(B, C, nH, H_w, nW, W_w)
        # (B, C, nH, H_w, nW, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # (B, nH, nW, C, H_w, W_w) -> (B * nH * nW, C, H_w, W_w)
        windows = x.reshape(-1, C, H_w, W_w)
        
        return windows

    def merge_windows(self, x, B, H, W):

        _, C, H_w, W_w = x.shape
        nH = H // H_w
        nW = W // W_w
        # (B * nH * nW, C, H_w, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.view(B, nH, nW, C, H_w, W_w)
        # (B, nH, nW, C, H_w, W_w) -> (B, C, nH, H_w, nW, W_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # (B, C, nH, H_w, nW, W_w) -> (B, C, H, W)
        out = x.reshape(B, C, H, W)
        
        return out
    
    def freq_direct_trans(self, x):
        
        return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

    def freq_inverse_trans(self, fft2d, s):

        return torch.fft.irfft2(fft2d, s, dim=(-2, -1), norm="ortho").real
    
    def route(self, x, time_embeds):
        
        B, C, H, W = x.shape
        # x = x.reshape(B, C, -1)
        # w = self.fc_x(x) # B, C, N
        x = x.reshape(B * C, 1, H, W)
        w = self.cnn(x).reshape(B, C, self.N) # B, C, N
        t = self.fc_t(time_embeds) # B, N
        t = t[:, None, None, :].expand(-1, B // t.shape[0], C, -1).reshape(B, C, self.N)
        weight = self.fuse(torch.concat([w, t], dim=2)) # B, C, N

        return weight
    
    def forward(self, x, times):

        B_ori, C, H_ori, W_ori = x.shape
        x = self.split_windows(x) 
        B, _, H, W = x.shape
        fft2d = self.freq_direct_trans(x) # B, C, H, ...
        freq_feats = torch.view_as_real(fft2d.reshape(B, C, self.T)) # B, C, T, 2
        
        time_embeds = self.time_embed_layer(times) # B, D
        band_weight = F.softmax(self.route(x, time_embeds), dim=2) * self.N # B, C, N
        # band_weight = torch.ones(B, C, self.N).to(time_embeds.device)
        
        if self.batch_size is not None:
            if self.process_step == "space":
                y = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.zeros(B, C, cur_batch_size, self.T, device=band_weight.device)
                        for i, band_idx in enumerate(range(batch_idx * self.batch_size, end)):
                            indices = self.band_real_idx[band_idx]
                            kernel_band[:, :, i, indices.to(band_weight.device)] = 1
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W      
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                weighted_kernel_band = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    weighted_kernel_band = torch.zeros(B, C, self.T, device=band_weight.device, dtype=torch.complex64)
                    for i, indices in enumerate(self.band_real_idx):
                        weighted_kernel_band[:, :, indices.to(band_weight.device)] = band_weight[:, :, i]
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T                
                else:
                    raise ValueError
            
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        else:
            if self.band_kernel == "rec":
                kernel_band = torch.zeros(B, C, self.N, self.T, device=band_weight.device)
                for i, indices in enumerate(self.band_real_idx):
                    kernel_band[:, :, i, indices.to(band_weight.device)] = 1
            elif self.band_kernel == "gauss":
                std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                std = std.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                kernel_band = torch.exp(- (all_fr - band_means) ** 2 / (2 * std ** 2)) # (B, C, N, T)
            elif self.band_kernel == "beta":
                # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                alpha = band_means * (albeta - 2) + 1 # (B, C, N, 1)
                beta = (1 - band_means) * (albeta - 2) + 1 # (B, C, N, 1)
                peak = (band_means ** (alpha - 1)) * ((1 - band_means) ** (beta - 1)) # (B, C, N, 1)
                try:
                    assert peak.min() > 0
                except:
                    print(albeta.min().item(), albeta.max().item(), "###", freq_feats.min().item(), freq_feats.max().item(), "###", peak.min().item())
                    print(self.albeta_predictor.weight.max().item(), self.albeta_predictor.weight.min().item(), "&&", self.albeta_predictor.bias.max().item(), self.albeta_predictor.bias.min().item())
                    raise ValueError
                kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, N, T)
            elif self.band_kernel == "learn":
                kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)) # B, C, N, T
                kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, N, T
                # exit(-1)
            else:
                raise ValueError
            
            if self.process_step == "space":
                band_weight = band_weight[:, :, :, None, None]
                fft2d = fft2d.unsqueeze(2).expand(-1, -1, self.N, -1, -1)
                y = self.freq_inverse_trans(fft2d * kernel_band.reshape(B, C, self.N, H, -1), (H, W)) # (B, C, N, H, W)
                y = (band_weight * y).sum(dim=2) # (B, C, H, W)s
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                band_weight = band_weight[:, :, :, None]
                weighted_kernel_band = (band_weight * kernel_band).sum(dim=2) # B, C, T
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        
        y = self.merge_windows(y, B_ori, H_ori, W_ori)
        band_weight = band_weight.reshape(B_ori, -1, C, self.N)
        
        return y, band_weight


class BandPassFilterWinFreq(nn.Module):
    
    def __init__(self, C, win_size=(10, 10), time_dim=72, num_bands=10, band_scale="log", accord_val=False, band_kernel="beta", process_step="freq", batch_size=None, 
                 prompt_style=None, weather_embed_path=None):
        
        super().__init__()
        H = win_size[0]
        W = win_size[1]
        self.C = C
        self.N = num_bands
        self.win_size = win_size
        self.process_step = process_step
        self.current_epoch = -1
        self.batch_size = batch_size
        
        self.T = H * (W // 2 + 1)
        freq_y = torch.fft.fftfreq(H)
        freq_x = torch.fft.rfftfreq(W)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2)
        self.freq_radius_flat = freq_radius.flatten() # T
        # self.sorted_idx = torch.argsort(self.freq_radius_flat) # T: sorted idx -> real idx
        
        self.band_scale = band_scale
        if band_scale == "uniform":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands, log_ratio=None, use_log=False, accord_val=accord_val)
        elif band_scale == "log":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands - 1, log_ratio=(None, 1.3), use_log=True, accord_val=accord_val)
        else:
            raise ValueError
        
        self.band_kernel = band_kernel
        if band_kernel == "rec":
            pass
        elif band_kernel == "gauss":
            self.all_fr = self.freq_radius_flat # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.std_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.std_predictor.weight, 0.)
            nn.init.constant_(self.std_predictor.bias, 1.)
        elif band_kernel == "beta":
            self.all_fr = (self.freq_radius_flat - self.freq_radius_flat.min()) / (self.freq_radius_flat.max() - self.freq_radius_flat.min()) # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.albeta_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.albeta_predictor.weight, 0.)
            nn.init.constant_(self.albeta_predictor.bias, -5.)
        elif band_kernel == "learn":
            # self.kernel_params = nn.Parameter(torch.ones((C, self.N, self.T)))
            self.point_predictor = nn.Linear(2, self.N)
        else:
            raise ValueError
        
        self.time_embed_layer = TimeEmbedding(time_dim)
        # self.fc_x = nn.Sequential(nn.Linear(win_size[0] * win_size[1], self.N), nn.LeakyReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.N, 3, stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc_t = nn.Sequential(nn.Linear(time_dim, self.N), nn.LeakyReLU())
        self.fuse = nn.Linear(self.N * 2, self.N)
        self.ffn = Mlp(self.C)
        self.norm_ffn = nn.LayerNorm(C)
        
        assert prompt_style == "dual"
        self.prompt_attn = PromptDualAttentionFreq(2 * self.C, self.C, self.T, weather_embed_path)
    
    def prompting(self, x):
        
        B, C, H, W_ = x.shape
        x = x.reshape(B, C, self.T)
        prompts, attn_weights = self.prompt_attn(x)
        x = (x + prompts).reshape(B, C, H, W_)
        
        return x, attn_weights
    
    def split_windows(self, x):

        B, C, H, W = x.shape
        H_w, W_w = self.win_size
        assert H % H_w == 0 and W % W_w == 0, "H and W must be divisible by H_w and W_w"
        nH = H // H_w
        nW = W // W_w

        # (B, C, H, W) -> (B, C, nH, H_w, nW, W_w)
        x = x.view(B, C, nH, H_w, nW, W_w)
        # (B, C, nH, H_w, nW, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # (B, nH, nW, C, H_w, W_w) -> (B * nH * nW, C, H_w, W_w)
        windows = x.reshape(-1, C, H_w, W_w)
        
        return windows

    def merge_windows(self, x, B, H, W):

        _, C, H_w, W_w = x.shape
        nH = H // H_w
        nW = W // W_w
        # (B * nH * nW, C, H_w, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.view(B, nH, nW, C, H_w, W_w)
        # (B, nH, nW, C, H_w, W_w) -> (B, C, nH, H_w, nW, W_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # (B, C, nH, H_w, nW, W_w) -> (B, C, H, W)
        out = x.reshape(B, C, H, W)
        
        return out
    
    def freq_direct_trans(self, x):
        
        return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

    def freq_inverse_trans(self, fft2d, s):

        return torch.fft.irfft2(fft2d, s, dim=(-2, -1), norm="ortho").real
    
    def route(self, x, time_embeds):
        
        B, C, H, W = x.shape
        # x = x.reshape(B, C, -1)
        # w = self.fc_x(x) # B, C, N
        x = x.reshape(B * C, 1, H, W)
        w = self.cnn(x).reshape(B, C, self.N) # B, C, N
        t = self.fc_t(time_embeds) # B, N
        t = t[:, None, None, :].expand(-1, B // t.shape[0], C, -1).reshape(B, C, self.N)
        weight = self.fuse(torch.concat([w, t], dim=2)) # B, C, N

        return weight
    
    def forward(self, x, times):

        B_ori, C, H_ori, W_ori = x.shape
        x = self.split_windows(x) 
        B, _, H, W = x.shape
        fft2d = self.freq_direct_trans(x) # B, C, H, ...
        fft2d, attn_weights = self.prompting(fft2d)
        freq_feats = torch.view_as_real(fft2d.reshape(B, C, self.T)) # B, C, T, 2
        
        time_embeds = self.time_embed_layer(times) # B, D
        band_weight = F.softmax(self.route(x, time_embeds), dim=2) * self.N # B, C, N
        # band_weight = torch.ones(B, C, self.N).to(time_embeds.device)
        
        if self.batch_size is not None:
            if self.process_step == "space":
                y = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.zeros(B, C, cur_batch_size, self.T, device=band_weight.device)
                        for i, band_idx in enumerate(range(batch_idx * self.batch_size, end)):
                            indices = self.band_real_idx[band_idx]
                            kernel_band[:, :, i, indices.to(band_weight.device)] = 1
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W      
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                weighted_kernel_band = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    weighted_kernel_band = torch.zeros(B, C, self.T, device=band_weight.device, dtype=torch.complex64)
                    for i, indices in enumerate(self.band_real_idx):
                        weighted_kernel_band[:, :, indices.to(band_weight.device)] = band_weight[:, :, i]
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T                
                else:
                    raise ValueError
            
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        else:
            if self.band_kernel == "rec":
                kernel_band = torch.zeros(B, C, self.N, self.T, device=band_weight.device)
                for i, indices in enumerate(self.band_real_idx):
                    kernel_band[:, :, i, indices.to(band_weight.device)] = 1
            elif self.band_kernel == "gauss":
                std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                std = std.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                kernel_band = torch.exp(- (all_fr - band_means) ** 2 / (2 * std ** 2)) # (B, C, N, T)
            elif self.band_kernel == "beta":
                # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                alpha = band_means * (albeta - 2) + 1 # (B, C, N, 1)
                beta = (1 - band_means) * (albeta - 2) + 1 # (B, C, N, 1)
                peak = (band_means ** (alpha - 1)) * ((1 - band_means) ** (beta - 1)) # (B, C, N, 1)
                try:
                    assert peak.min() > 0
                except:
                    print(albeta.min().item(), albeta.max().item(), "###", freq_feats.min().item(), freq_feats.max().item(), "###", peak.min().item())
                    print(self.albeta_predictor.weight.max().item(), self.albeta_predictor.weight.min().item(), "&&", self.albeta_predictor.bias.max().item(), self.albeta_predictor.bias.min().item())
                    raise ValueError
                kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, N, T)
            elif self.band_kernel == "learn":
                kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)) # B, C, N, T
                kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, N, T
                # exit(-1)
            else:
                raise ValueError
            
            if self.process_step == "space":
                band_weight = band_weight[:, :, :, None, None]
                fft2d = fft2d.unsqueeze(2).expand(-1, -1, self.N, -1, -1)
                y = self.freq_inverse_trans(fft2d * kernel_band.reshape(B, C, self.N, H, -1), (H, W)) # (B, C, N, H, W)
                y = (band_weight * y).sum(dim=2) # (B, C, H, W)s
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                band_weight = band_weight[:, :, :, None]
                weighted_kernel_band = (band_weight * kernel_band).sum(dim=2) # B, C, T
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        
        y = self.merge_windows(y, B_ori, H_ori, W_ori)
        band_weight = band_weight.reshape(B_ori, -1, C, self.N)
        
        return y, band_weight, attn_weights


class BandPassFilterWinSpace(nn.Module):
    
    def __init__(self, C, win_size=(10, 10), time_dim=72, num_bands=10, band_scale="log", accord_val=False, band_kernel="beta", process_step="freq", batch_size=None, 
                 prompt_style=None, weather_embed_path=None):
        
        super().__init__()
        H = win_size[0]
        W = win_size[1]
        self.C = C
        self.N = num_bands
        self.win_size = win_size
        self.process_step = process_step
        self.current_epoch = -1
        self.batch_size = batch_size
        
        self.T = H * (W // 2 + 1)
        freq_y = torch.fft.fftfreq(H)
        freq_x = torch.fft.rfftfreq(W)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2)
        self.freq_radius_flat = freq_radius.flatten() # T
        # self.sorted_idx = torch.argsort(self.freq_radius_flat) # T: sorted idx -> real idx
        
        self.band_scale = band_scale
        if band_scale == "uniform":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands, log_ratio=None, use_log=False, accord_val=accord_val)
        elif band_scale == "log":
            self.sorted_idx, self.band_idx, self.band_real_idx = get_band_splits(
                self.freq_radius_flat, num_bands - 1, log_ratio=(None, 1.3), use_log=True, accord_val=accord_val)
        else:
            raise ValueError
        
        self.band_kernel = band_kernel
        if band_kernel == "rec":
            pass
        elif band_kernel == "gauss":
            self.all_fr = self.freq_radius_flat # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.std_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.std_predictor.weight, 0.)
            nn.init.constant_(self.std_predictor.bias, 1.)
        elif band_kernel == "beta":
            self.all_fr = (self.freq_radius_flat - self.freq_radius_flat.min()) / (self.freq_radius_flat.max() - self.freq_radius_flat.min()) # (T,)
            self.band_means = torch.stack([self.all_fr[indices[(indices.shape[0] - 1) // 2]] for indices in self.band_real_idx]) # (N,)
            self.all_fr = nn.Parameter(self.all_fr, requires_grad=False)
            self.band_means = nn.Parameter(self.band_means, requires_grad=False)
            self.albeta_predictor = nn.Linear(self.T * 2, self.N)
            nn.init.constant_(self.albeta_predictor.weight, 0.)
            nn.init.constant_(self.albeta_predictor.bias, -5.)
        elif band_kernel == "learn":
            # self.kernel_params = nn.Parameter(torch.ones((C, self.N, self.T)))
            self.point_predictor = nn.Linear(2, self.N)
        else:
            raise ValueError
        
        self.time_embed_layer = TimeEmbedding(time_dim)
        # self.fc_x = nn.Sequential(nn.Linear(win_size[0] * win_size[1], self.N), nn.LeakyReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.N, 3, stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc_t = nn.Sequential(nn.Linear(time_dim, self.N), nn.LeakyReLU())
        self.fuse = nn.Linear(self.N * 2, self.N)
        self.ffn = Mlp(self.C)
        self.norm_ffn = nn.LayerNorm(C)
        
        assert prompt_style == "dual"
        self.prompt_attn = PromptDualAttentionSpace(self.C, self.C, self.win_size, weather_embed_path)
    
    def prompting(self, x):
        
        B, C, H, W = x.shape
        prompts, attn_weights = self.prompt_attn(x)
        x = x + prompts
        
        return x, attn_weights
    
    def split_windows(self, x):

        B, C, H, W = x.shape
        H_w, W_w = self.win_size
        assert H % H_w == 0 and W % W_w == 0, "H and W must be divisible by H_w and W_w"
        nH = H // H_w
        nW = W // W_w

        # (B, C, H, W) -> (B, C, nH, H_w, nW, W_w)
        x = x.view(B, C, nH, H_w, nW, W_w)
        # (B, C, nH, H_w, nW, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # (B, nH, nW, C, H_w, W_w) -> (B * nH * nW, C, H_w, W_w)
        windows = x.reshape(-1, C, H_w, W_w)
        
        return windows

    def merge_windows(self, x, B, H, W):

        _, C, H_w, W_w = x.shape
        nH = H // H_w
        nW = W // W_w
        # (B * nH * nW, C, H_w, W_w) -> (B, nH, nW, C, H_w, W_w)
        x = x.view(B, nH, nW, C, H_w, W_w)
        # (B, nH, nW, C, H_w, W_w) -> (B, C, nH, H_w, nW, W_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # (B, C, nH, H_w, nW, W_w) -> (B, C, H, W)
        out = x.reshape(B, C, H, W)
        
        return out
    
    def freq_direct_trans(self, x):
        
        return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

    def freq_inverse_trans(self, fft2d, s):

        return torch.fft.irfft2(fft2d, s, dim=(-2, -1), norm="ortho").real
    
    def route(self, x, time_embeds):
        
        B, C, H, W = x.shape
        # x = x.reshape(B, C, -1)
        # w = self.fc_x(x) # B, C, N
        x = x.reshape(B * C, 1, H, W)
        w = self.cnn(x).reshape(B, C, self.N) # B, C, N
        t = self.fc_t(time_embeds) # B, N
        t = t[:, None, None, :].expand(-1, B // t.shape[0], C, -1).reshape(B, C, self.N)
        weight = self.fuse(torch.concat([w, t], dim=2)) # B, C, N

        return weight
    
    def forward(self, x, times, get_bfw=False):

        B_ori, C, H_ori, W_ori = x.shape
        x = self.split_windows(x) 
        B, _, H, W = x.shape
        x, attn_weights = self.prompting(x)
        fft2d = self.freq_direct_trans(x) # B, C, H, ...
        freq_feats = torch.view_as_real(fft2d.reshape(B, C, self.T)) # B, C, T, 2
        
        time_embeds = self.time_embed_layer(times) # B, D
        band_weight = F.softmax(self.route(x, time_embeds), dim=2) * self.N # B, C, N
        # band_weight = torch.ones(B, C, self.N).to(time_embeds.device)
        
        if self.batch_size is not None:
            if self.process_step == "space":
                y = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.zeros(B, C, cur_batch_size, self.T, device=band_weight.device)
                        for i, band_idx in enumerate(range(batch_idx * self.batch_size, end)):
                            indices = self.band_real_idx[band_idx]
                            kernel_band[:, :, i, indices.to(band_weight.device)] = 1
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W      
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        cur_batch_size = end - batch_idx * self.batch_size
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        batch_y = self.freq_inverse_trans(
                            fft2d.unsqueeze(2).expand(-1, -1, cur_batch_size, -1, -1) * kernel_band.reshape(B, C, cur_batch_size, H, -1), 
                            (H, W)) # (B, C, NB, H, W)
                        y += (band_weight[:, :, batch_slice, None, None] * batch_y).sum(dim=2) # B, C, H, W
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                weighted_kernel_band = 0.
                num_batches = math.ceil(self.N / self.batch_size)
                if self.band_kernel == "rec":
                    weighted_kernel_band = torch.zeros(B, C, self.T, device=band_weight.device, dtype=torch.complex64)
                    for i, indices in enumerate(self.band_real_idx):
                        weighted_kernel_band[:, :, indices.to(band_weight.device)] = band_weight[:, :, i]
                elif self.band_kernel == "gauss":
                    std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    std = std.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = torch.exp(- (all_fr - band_means[:, :, batch_slice]) ** 2 / (2 * std[:, :, batch_slice] ** 2)) # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "beta":
                    # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                    albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                    all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                    band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        alpha = band_means[:, :, batch_slice] * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        beta = (1 - band_means[:, :, batch_slice]) * (albeta[:, :, batch_slice] - 2) + 1 # (B, C, NB, 1)
                        peak = (band_means[:, :, batch_slice] ** (alpha - 1)) * ((1 - band_means[:, :, batch_slice]) ** (beta - 1)) # (B, C, NB, 1)
                        assert peak.min() > 0
                        kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, NB, T)
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T
                elif self.band_kernel == "learn":
                    for batch_idx in range(num_batches):
                        end = min((batch_idx + 1) * self.batch_size, self.N)
                        batch_slice = slice(batch_idx * self.batch_size, end)
                        kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)[:, :, batch_slice, :]) # B, C, NB, T
                        kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, NB, T
                        weighted_kernel_band += (band_weight[:, :, batch_slice, None] * kernel_band).sum(dim=2) # B, C, T                
                else:
                    raise ValueError
            
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        else:
            if self.band_kernel == "rec":
                kernel_band = torch.zeros(B, C, self.N, self.T, device=band_weight.device)
                for i, indices in enumerate(self.band_real_idx):
                    kernel_band[:, :, i, indices.to(band_weight.device)] = 1
            elif self.band_kernel == "gauss":
                std = EPS + F.softplus(self.std_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                std = std.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                kernel_band = torch.exp(- (all_fr - band_means) ** 2 / (2 * std ** 2)) # (B, C, N, T)
            elif self.band_kernel == "beta":
                # albeta = 2 + F.softplus(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = 2 + 70 * torch.sigmoid(self.albeta_predictor(freq_feats.reshape(B, C, -1))) # (B, C, N)
                albeta = albeta.unsqueeze(-1) # (B, C, N, 1)
                all_fr = self.all_fr[None, None, None, :]  # (1, 1, 1, T)
                band_means = self.band_means[None, None, :, None]  # (1, 1, N, 1)
                alpha = band_means * (albeta - 2) + 1 # (B, C, N, 1)
                beta = (1 - band_means) * (albeta - 2) + 1 # (B, C, N, 1)
                peak = (band_means ** (alpha - 1)) * ((1 - band_means) ** (beta - 1)) # (B, C, N, 1)
                try:
                    assert peak.min() > 0
                except:
                    print(albeta.min().item(), albeta.max().item(), "###", freq_feats.min().item(), freq_feats.max().item(), "###", peak.min().item())
                    print(self.albeta_predictor.weight.max().item(), self.albeta_predictor.weight.min().item(), "&&", self.albeta_predictor.bias.max().item(), self.albeta_predictor.bias.min().item())
                    raise ValueError
                kernel_band = (all_fr ** (alpha - 1)) * ((1 - all_fr) ** (beta - 1)) / peak # (B, C, N, T)
            elif self.band_kernel == "learn":
                kernel_band = EPS + F.softplus(self.point_predictor(freq_feats).permute(0, 1, 3, 2)) # B, C, N, T
                kernel_band = kernel_band / kernel_band.max(dim=-1, keepdim=True).values # B, C, N, T
                # exit(-1)
            else:
                raise ValueError
            
            if self.process_step == "space":
                band_weight = band_weight[:, :, :, None, None]
                fft2d = fft2d.unsqueeze(2).expand(-1, -1, self.N, -1, -1)
                y = self.freq_inverse_trans(fft2d * kernel_band.reshape(B, C, self.N, H, -1), (H, W)) # (B, C, N, H, W)
                y = (band_weight * y).sum(dim=2) # (B, C, H, W)s
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            elif self.process_step == "freq":
                band_weight = band_weight[:, :, :, None]
                weighted_kernel_band = (band_weight * kernel_band).sum(dim=2) # B, C, T
                y = self.freq_inverse_trans(fft2d * weighted_kernel_band.reshape(B, C, H, -1), (H, W)) # (B, C, H, W)
                y = y + self.ffn(self.norm_ffn(y.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) # (B, C, H, W)
            else:
                raise ValueError
        
        y = self.merge_windows(y, B_ori, H_ori, W_ori)
        band_weight = band_weight.reshape(B_ori, -1, C, self.N)
        # attn_weights = attn_weights.reshape(B_ori, -1, attn_weights.shape[-1])
        
        if get_bfw:
            fine_kernel_band = kernel_band.sum(dim=2)[:, :, self.sorted_idx] # B, C, T
            coarse_kernel_band = torch.stack([fine_kernel_band[:, :, inds].mean(dim=-1) for inds in self.band_idx], dim=-1) # B, C, N
            return y, band_weight, attn_weights, fine_kernel_band.reshape(B_ori, -1, C, self.T), coarse_kernel_band.reshape(B_ori, -1, C, self.N)
        else:
            return y, band_weight, attn_weights, None, None


if __name__ == '__main__':
    
    # set_seed(42)
    # torch.cuda.reset_peak_memory_stats()
    # # B, D, H, W = 4, 69, 530, 900
    # B, D, H, W = 4, 512, 67, 113
    # times = torch.tensor([2022010100, 2022010101]).unsqueeze(0).expand(B, -1).to("cuda:1")
    # inputs = torch.randn(B, D, H, W).to("cuda:1")
    # model = BandPassFilterRaw(H, W, D, num_bands=32, accord_val=False, batch_size=16)
    # model.to("cuda:1")
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Model parameters: {total_params:,}")
    # output = model(inputs, times)
    # print(inputs.shape, output.shape, output.mean())
    # print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    B, D, H, W = 4, 69, 530, 900
    times = torch.tensor([2022010100, 2022010101]).unsqueeze(0).expand(B, -1).to("cuda:1")
    inputs = torch.randn(B, D, H, W).to("cuda:1")
    model = BandPassFilterWin(D, win_size=(10, 10), num_bands=10, accord_val=False, batch_size=None)
    model.to("cuda:1")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    output = model(inputs, times)
    print(inputs.shape, output.shape, output.mean())
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")