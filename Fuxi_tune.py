import os
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from torch.utils.checkpoint import checkpoint

from Fuxi_freq import FuxiFreq, UTransformer
from prompt_attention import PromptAttention, PromptDualAttention
from freq_utils import BandPassFilterWinFreq, BandPassFilterWinSpace 

class Mean2DPooling(nn.Module):
    
    def forward(self, batch_embeds, spatial_mask):
        """
        batch_embeds: (B, N, D, H, W)
        spatial_mask: (B, N, H, W), bool or 0/1
        Returns: (B, N, D)
        """
        
        B, N, D, H, W = batch_embeds.shape
        mask = spatial_mask[:, :, None, :, :].float() # (B, N, 1, H, W)
        masked = batch_embeds * mask
        valid_count = mask.sum(dim=(3, 4), keepdim=False)  # (B, N, 1)
        pooled = masked.sum(dim=(3, 4)) / valid_count  # (B, N, D)
        
        return pooled
    
class AttnPooling(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        """
        x: (B, N, D, H, W)
        mask: (B, N, H, W)
        Returns: (B, N, D, 1)
        """
        B, N, D, H, W = x.shape
        mask = mask.float()
        x_reshaped = x.permute(0, 1, 3, 4, 2).reshape(B, N, H*W, D)  # (B, N, H*W, D)
        
        mask_reshaped = mask.reshape(B, N, H*W, 1)  # (B, N, H*W, 1)
        valid_count = mask_reshaped.sum(dim=2, keepdim=True)  # (B, N, 1, 1)
        masked_x = x_reshaped * mask_reshaped
        mean_x = masked_x.sum(dim=2, keepdim=True) / valid_count  # (B, N, 1, D)
        
        query = self.query_proj(mean_x)  # (B, N, 1, D)
        key = self.key_proj(x_reshaped)  # (B, N, H*W, D)
        attn = torch.einsum('bnqd,bnkd->bnqk', query, key) # (B, N, 1, H*W)
        attn = attn.masked_fill(mask_reshaped.transpose(2, 3) == 0, float('-inf'))
        attn = F.softmax(attn / (query.shape[-1] ** 0.5), dim=-1)  # (B, N, 1, H*W)
        pooled = torch.einsum('bnqk,bnkd->bnqd', attn, x_reshaped)  # (B, N, 1, D)
        pooled = pooled.squeeze(2)  # (B, N, D)
        
        return pooled

class FuxiAdapter(FuxiFreq):
    
    def __init__(self):
        
        super().__init__(use_adapter=True)
        
class FuxiPrompt(FuxiFreq):
    
    def __init__(self, prompt_style, pooling_style, weather_embed_path, two_prompts=False):
        
        super().__init__()
        
        self.prompt_style = prompt_style
        if prompt_style == "single":
            self.prompt_attn = PromptAttention(embed_dim=self.embed_dim, 
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        elif prompt_style == "dual":
            self.prompt_attn = PromptDualAttention(embed_dim=self.embed_dim, 
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        else:
            raise ValueError
        
        self.prompt_gate = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.p_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.x_fc = nn.Linear(self.embed_dim, self.embed_dim)
        # nn.init.constant_(self.prompt_gate[0].weight, 0)
        # nn.init.constant_(self.prompt_gate[0].bias, -4)
        
        self.two_prompts = two_prompts
        if two_prompts:
            if prompt_style == "single":
                self.prompt_attn_2 = PromptAttention(embed_dim=self.embed_dim, 
                                                    weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
            elif prompt_style == "dual":
                self.prompt_attn_2 = PromptDualAttention(embed_dim=self.embed_dim, 
                                                    weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
            else:
                raise ValueError
            
            self.prompt_gate_2 = nn.Sequential(
                nn.Conv2d(2 * self.embed_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.p_fc_2 = nn.Linear(self.embed_dim, self.embed_dim)
            self.x_fc_2 = nn.Linear(self.embed_dim, self.embed_dim)
            # nn.init.constant_(self.prompt_gate_2[0].weight, 0)
            # nn.init.constant_(self.prompt_gate_2[0].bias, -4)
        
    def forward(self, x, times):
        
        if self.freq_mode == "raw":
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
            x = self.freq_filter(x, times)
        elif self.freq_mode == "win":
            # x, bws = self.freq_filter(x, times)
            x, bws = checkpoint(self.freq_filter, x, times, use_reentrant=False)
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        else:
            raise ValueError
        
        if self.prompt_style == "single":
            prompt, entropy_loss, attn_weights, sim_loss = self.prompt_attn(x) # [B C Lat Lon]
        elif self.prompt_style == "dual":
            prompt, entropy_loss, attn_weights, sim_loss = self.prompt_attn(x, times) # [B C Lat Lon]
        else:
            raise ValueError
        prompt = self.p_fc(F.leaky_relu(prompt.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        # prompt = self.p_fc(F.leaky_relu(prompt.permute(0, 2, 3, 1)) + F.leaky_relu(self.x_fc(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        gate = self.prompt_gate(torch.cat([x, prompt], dim=1)) # [B, 2*C, Lat, Lon] -> # [B, 1, Lat, Lon]
        x = x + prompt * gate # [B, C, Lat, Lon] 
        
        x = self.u_transformer(x)
        if self.two_prompts:
            if self.prompt_style == "single":
                prompt, entropy_loss_2, attn_weights_2, sim_loss_2 = self.prompt_attn_2(x) # [B C Lat Lon]
            elif self.prompt_style == "dual":
                prompt, entropy_loss_2, attn_weights_2, sim_loss_2 = self.prompt_attn_2(x, times) # [B C Lat Lon]
            else:
                raise ValueError
            prompt = self.p_fc_2(F.leaky_relu(prompt.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            # prompt = self.p_fc_2(F.leaky_relu(prompt.permute(0, 2, 3, 1)) + F.leaky_relu(self.x_fc_2(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
            gate = self.prompt_gate_2(torch.cat([x, prompt], dim=1)) # [B, 2*C, Lat, Lon] -> # [B, 1, Lat, Lon]
            x = x + prompt * gate # [B, C, Lat, Lon]
            if entropy_loss is not None:
                entropy_loss = torch.stack([entropy_loss, entropy_loss_2], dim=0).unsqueeze(0)
            sim_loss = torch.stack([sim_loss, sim_loss_2], dim=0).unsqueeze(0)
            attn_weights = torch.stack([attn_weights, attn_weights_2], dim=-2)
        
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)
    
        return x, bws, entropy_loss, gate, attn_weights, sim_loss

class FuxiFreqPrompt(FuxiFreq):
    
    def __init__(self, prompt_style, pooling_style, weather_embed_path, two_prompts=False):
        
        super().__init__()
        
        self.prompt_style = prompt_style
        if prompt_style == "single":
            self.prompt_attn = PromptAttention(embed_dim=self.embed_dim,
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        elif prompt_style == "dual":
            self.prompt_attn = PromptDualAttention(embed_dim=self.embed_dim, 
                                                # use_random=True, 
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        else:
            raise ValueError
        
        self.p_fc = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.two_prompts = two_prompts
        if two_prompts:
            if prompt_style == "single":
                self.prompt_attn_2 = PromptAttention(embed_dim=self.embed_dim,
                                                    weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
            elif prompt_style == "dual":
                self.prompt_attn_2 = PromptDualAttention(embed_dim=self.embed_dim, 
                                                    # use_random=True, 
                                                    weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
            else:
                raise ValueError
            
            self.p_fc_2 = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, times):
        
        if self.freq_mode == "raw":
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
            x = self.freq_filter(x, times)
        elif self.freq_mode == "win":
            # x, bws = self.freq_filter(x, times)
            x, bws = checkpoint(self.freq_filter, x, times, use_reentrant=False)
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        else:
            raise ValueError
        
        if self.prompt_style == "single":
            prompt, entropy_loss, attn_weights, _ = self.prompt_attn(x) # [B C Lat Lon]
        elif self.prompt_style == "dual":
            prompt, entropy_loss, attn_weights, _ = self.prompt_attn(x, times) # [B C Lat Lon]
        else:
            raise ValueError
        prompt = self.p_fc(F.leaky_relu(prompt.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = x + prompt # [B, C, Lat, Lon] 
        
        x = self.u_transformer(x)
        if self.two_prompts:
            if self.prompt_style == "single":
                prompt, entropy_loss_2, attn_weights_2, _ = self.prompt_attn_2(x) # [B C Lat Lon]
            elif self.prompt_style == "dual":
                prompt, entropy_loss_2, attn_weights_2, _ = self.prompt_attn_2(x, times) # [B C Lat Lon]
            else:
                raise ValueError
            prompt = self.p_fc_2(F.leaky_relu(prompt.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            x = x + prompt # [B, C, Lat, Lon]
            if entropy_loss is not None:
                entropy_loss = torch.stack([entropy_loss, entropy_loss_2], dim=0).unsqueeze(0)
            attn_weights = torch.stack([attn_weights, attn_weights_2], dim=-2)
        
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)
    
        return x, bws, attn_weights

class FuxiFreqPromptFreq(FuxiFreq):
    
    def __init__(self, weather_embed_path, use_space=False, prompt_style="dual", in_shape=(1, 69, 530, 900)):
        
        super().__init__(in_shape=in_shape)
        
        self.prompt_style = prompt_style
        if use_space:
            self.freq_filter = BandPassFilterWinSpace(self.in_shape[1], win_size=(10, 10), prompt_style=prompt_style, weather_embed_path=weather_embed_path)
        else:
            self.freq_filter = BandPassFilterWinFreq(self.in_shape[1], win_size=(10, 10), prompt_style=prompt_style, weather_embed_path=weather_embed_path)
        
    def forward(self, x, times, get_bfw=False):
        
        x, bws, attn_weights, bfws, cbfws = self.freq_filter(x, times, get_bfw)
        # x, bws, attn_weights = checkpoint(self.freq_filter, x, times, use_reentrant=False)
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1, 3, 4)
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        # x = checkpoint(self.cube_embedding, x, use_reentrant=False).squeeze(2)  # B C Lat Lon
        
        x = self.u_transformer(x)
        # x = checkpoint(self.u_transformer, x, use_reentrant=False)
        
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)

        if get_bfw:
            return x, bws, attn_weights, (bfws, cbfws)
        else:
            return x, bws, attn_weights

class FuxiFreqRecon(FuxiFreq):
    
    def __init__(self):
        
        super().__init__()
    
    def recon_branch(self, x):
        
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)
        
        return x
        
    def forward(self, x, times):
        
        x_ori = x
        # x, bws = self.freq_filter(x, times)
        x, bws = checkpoint(self.freq_filter, x, times, use_reentrant=False)
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1, 3, 4)
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        # x = checkpoint(self.cube_embedding, x, use_reentrant=False).squeeze(2)  # B C Lat Lon
        
        x_recon = self.recon_branch(x)
        
        x = self.u_transformer(x)
        # x = checkpoint(self.u_transformer, x, use_reentrant=False)
        
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)
    
        return x, bws, x_recon


class FuxiPromptRaw(FuxiFreq):
    
    def __init__(self, prompt_style, pooling_style, weather_embed_path):
        
        super().__init__()
        
        self.prompt_style = prompt_style
        if prompt_style == "single":
            self.prompt_attn = PromptAttention(embed_dim=69, in_dim=69, 
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        elif prompt_style == "dual":
            self.prompt_attn = PromptDualAttention(embed_dim=69, in_dim=69, 
                                                weather_embed_path=os.path.join(weather_embed_path, f"{pooling_style}.pth"))
        else:
            raise ValueError
        
        self.prompt_gate = nn.Sequential(
            nn.Conv2d(69 * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.prompt_gate[0].weight, 0)
        nn.init.constant_(self.prompt_gate[0].bias, -4)
        
    def forward(self, x, times):
        
        if self.prompt_style == "single":
            prompt, entropy_loss, attn_weights = self.prompt_attn(x) # [B C H W]
        elif self.prompt_style == "dual":
            prompt, entropy_loss, attn_weights = self.prompt_attn(x, times) # [B C H W]
        else:
            raise ValueError
        ratio = torch.norm(x, p=2, dim=1) / torch.norm(prompt, p=2, dim=1)
        print('ratio', ratio.max().item(), ratio.mean().item(), ratio.min().item())
        gate = self.prompt_gate(torch.cat([x, prompt], dim=1)) # [B, 2*C, H, W] -> # [B, 1, H, W]
        x = x + prompt * gate # [B, C, H, W] 
        
        if self.freq_mode == "raw":
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
            x = self.freq_filter(x, times)
        elif self.freq_mode == "win":
            x, bws = self.freq_filter(x, times)
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1, 3, 4)
            B, _, _, _, _ = x.shape
            _, patch_lat, patch_lon = self.patch_size
            Lat, Lon = self.input_resolution
            Lat, Lon = Lat * 2, Lon * 2
            x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        else:
            raise ValueError
        
        x = self.u_transformer(x)
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear", align_corners=True)
    
        return x, bws, entropy_loss, gate, attn_weights

class FuxiPatternInit(FuxiFreq):
    
    def __init__(self, pooling_type="attn"):
        
        super().__init__()
        self.pooling_type = pooling_type
        
        if self.pooling_type == "mean":
            self.pooling_layer = Mean2DPooling()
        elif self.pooling_type == "attn":
            self.pooling_layer = AttnPooling(self.embed_dim)
        else:
            raise ValueError
        
    def forward(self, *args):
        
        assert len(args) == 3
        _, _, pre_embed = args
        
        if pre_embed:
            x, times, _ = args
            if self.freq_mode == "raw":
                x = x.unsqueeze(1)
                x = x.permute(0, 2, 1, 3, 4)
                B, _, _, _, _ = x.shape
                _, patch_lat, patch_lon = self.patch_size
                Lat, Lon = self.input_resolution
                Lat, Lon = Lat * 2, Lon * 2
                x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
                x = self.freq_filter(x, times)
            elif self.freq_mode == "win":
                x, bws = self.freq_filter(x, times)
                x = x.unsqueeze(1)
                x = x.permute(0, 2, 1, 3, 4)
                B, _, _, _, _ = x.shape
                _, patch_lat, patch_lon = self.patch_size
                Lat, Lon = self.input_resolution
                Lat, Lon = Lat * 2, Lon * 2
                x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
            else:
                raise ValueError
            
            return x
        else:
            batch_embeds, spatial_mask, _ = args # (B, N, D, H_max, W_max), (B, N, H_max, W_max)
            pooling_embeds = self.pooling_layer(batch_embeds, spatial_mask) # (B, N, D)
            
            return pooling_embeds
        
if __name__ == '__main__':
    import time
    torch.cuda.reset_peak_memory_stats()
    # inputs = torch.randn(1, 69, 530, 900).to("cuda")
    # targets = torch.randn(1, 69, 530, 900).to("cuda")
    inputs = torch.randn(1, 69, 720, 1440).to("cuda")
    targets = torch.randn(1, 69, 720, 1440).to("cuda")
    input_times = torch.tensor([[2024010100, 2024010101]]).long().to("cuda")
    model = FuxiFreqPromptFreq(
            in_shape=(1, 69, 720, 1440),
            weather_embed_path=f"/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data_down_2/HRRR/extreme/space_tokens/2022010100_2022123123", 
            use_space=True).to("cuda")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    TRAINING = False
    if TRAINING:
        # 训练模式
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        
        outputs, _, _ = model(inputs, input_times)
        loss = F.mse_loss(outputs, targets)  # 修正为mse_loss
        
        # backward和optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Training loss: {loss.item():.6f}")
        
    else:
        # 推理模式
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                outputs, _, _ = model(inputs, input_times)
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = times[1:]
        avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"All inference times: {[f'{t*1000:.2f}ms' for t in times]}")
        
    print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")  # 修正变量名
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")