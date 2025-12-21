import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

def orthogonalize(prompt, x):
    
    proj = (x * prompt).sum(dim=1, keepdim=True) / (x ** 2).sum(dim=1, keepdim=True) # [B, D, Pl'+1, H', W'] 
    orthogonal_prompt = prompt - proj * x # [B, D, Pl'+1, H', W'] 
    
    return orthogonal_prompt

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

class PromptAttention(nn.Module):
    
    def __init__(self, embed_dim, weather_embed_path, num_prompts=18, use_uniform=False, use_prior="weather", in_dim=None):
        
        super().__init__()
        
        if in_dim is None:
            in_dim = embed_dim
        
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.weather_embed_path = weather_embed_path
        self.use_uniform = use_uniform
        self.use_prior = use_prior
        # if use_prior == "weather":
        #     event_dim = embed_dim
        event_dim = embed_dim
        self.event_dim = event_dim

        self._init_prompt_pool()
        self.query_proj = nn.Linear(in_dim, embed_dim)
        self.key_proj = nn.Linear(event_dim, embed_dim)
        self.value_proj = nn.Linear(event_dim, in_dim)
        
        self.norm_x = nn.LayerNorm(in_dim)

    def _init_prompt_pool(self):
        
        if self.use_prior == "weather":
            weather_embeds_dict = torch.load(self.weather_embed_path, weights_only=True)
            weather_embeds_dict = {k: torch.stack(v, dim=0).mean(dim=0) for k, v in weather_embeds_dict.items()}
            weather_embeds = torch.stack(list(weather_embeds_dict.values()), dim=0)
            self.prompt_pool = nn.Parameter(weather_embeds, requires_grad=True)
            print("Extreme Events: ", list(weather_embeds_dict.keys()))
        elif self.use_prior == "weather-rand":
            weather_embeds_dict = torch.load(self.weather_embed_path, weights_only=True)
            weather_embeds_dict = {k: torch.stack(v, dim=0).mean(dim=0) for k, v in weather_embeds_dict.items()}
            weather_embeds = torch.stack(list(weather_embeds_dict.values()), dim=0)
            rand_embeds = torch.randn(weather_embeds.shape[0], weather_embeds.shape[1])
            weather_embeds = torch.concat([weather_embeds, rand_embeds], dim=0)
            self.prompt_pool = nn.Parameter(weather_embeds, requires_grad=True)
            print("Extreme Events: ", list(weather_embeds_dict.keys()), self.prompt_pool.shape)
        elif self.use_prior == "sentence":
            event_embeds_param, event_names = load_event_embeddings_as_parameter()
            self.prompt_pool = nn.Parameter(event_embeds_param, requires_grad=False)
            # self.prompt_pool = nn.Parameter(event_embeds_param[0].unsqueeze(0).expand(event_embeds_param.shape[0], -1), requires_grad=False)
            print("Extreme Events: ", event_names)
        else:
            embeds = torch.randn(self.num_prompts, self.event_dim)
            self.prompt_pool = nn.Parameter(embeds)
            # self.prompt_pool = nn.Parameter(embeds[0].unsqueeze(0).expand(embeds.shape[0], -1))
            print("prompt shape", self.prompt_pool.shape)
            
        with torch.no_grad():
            normalized_prompts = F.normalize(self.prompt_pool, p=2, dim=1)
            sim_matrix = torch.matmul(normalized_prompts, normalized_prompts.T)
            # print("Prompt Pool Cosine Similarity Matrix:\n", sim_matrix)
            print("Prompt Pool Cosine Similarity", sim_matrix.min(), sim_matrix.mean(), len(self.prompt_pool))
            print("Norm", torch.norm(self.prompt_pool, p=2, dim=1))
            print("=============")
        
    def forward(self, x):
    
        # x: [B, D, H, W]
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, D) # [B, N, D]，N=H*W
        x = self.norm_x(x)

        queries = self.query_proj(x) # [B, N, D]
        prompts = self.prompt_pool   # [M, D]
        test_ps = F.normalize(prompts, p=2, dim=-1)
        sim_mat = torch.einsum("md,nd->mn", test_ps, test_ps)
        sim_loss = sim_mat.mean()
        keys = self.key_proj(prompts)    # [M, D]
        values = self.value_proj(prompts) # [M, D]

        # attention: [B, N, D] x [M, D]T -> [B, N, M]
        attn_scores = torch.einsum('bnd,md->bnm', queries, keys) # [B, N, M]
        attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=2) # [B, N, M]
        y = torch.einsum('bnm,md->bnd', attn_weights, values) # [B, N, D]

        y = y.reshape(B, H, W, D).permute(0, 3, 1, 2)

        if self.use_uniform:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-20)).sum(dim=2) / torch.log(torch.tensor(attn_weights.shape[-1])) # [B, N]
            entropy_loss = -entropy.mean()
        else:
            entropy_loss = None

        return y, entropy_loss, attn_weights, sim_loss

class PromptDualAttention(nn.Module):
    
    def __init__(self, embed_dim, weather_embed_path, use_uniform=False, use_random=False, max_prompts_per_class=20, in_dim=None):
        
        super().__init__()
        
        if in_dim is None:
            in_dim = embed_dim
        
        self.embed_dim = embed_dim
        self.weather_embed_path = weather_embed_path
        print("### prompting tokens path: ", weather_embed_path)
        self.use_uniform = use_uniform
        self.use_random = use_random
        self.max_prompts_per_class = max_prompts_per_class

        self._init_prototype_pool()
        # self._init_rand_pool()
        
        # self.query_proj_prot = nn.Linear(in_dim - in_dim % 6 + in_dim, embed_dim)
        self.query_proj_prot = nn.Linear(in_dim, embed_dim)
        self.key_proj_prot = nn.Linear(embed_dim, embed_dim)
        self.value_proj_prot = nn.Linear(embed_dim, embed_dim)  
        
        self.time_embed_layer = TimeEmbedding(in_dim - in_dim % 6)      
        
        self.query_proj = nn.Linear(in_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, in_dim)
        
        self.norm_x = nn.LayerNorm(in_dim)
        self.norm_p_1 = nn.LayerNorm(embed_dim)
        self.norm_p_2 = nn.LayerNorm(embed_dim)

    def _init_prototype_pool(self):
        
        # Load weather embeddings and process per-class prototypes
        loaded_data = torch.load(self.weather_embed_path, weights_only=True)
        weather_embeds = {}
        for k, v in loaded_data.items():
            weather_embeds[k] = torch.stack(v, dim=0)

        prototypes = {}
        self.event_classes = list(weather_embeds.keys())
        print("Dual Extreme Events:", self.event_classes)
        
        # # print debug
        # class_num_embeds = [(k, weather_embeds[k].shape[0]) for k in self.event_classes]
        # class_num_embeds.sort(key=lambda x: x[1])
        # print("Extreme Events and their counts (sorted):")
        # for k, n in class_num_embeds:
        #     print(f"{k}: {n}")
        # exit(-1)

        embed_dir = os.path.dirname(self.weather_embed_path)
        kmeans_save_path = os.path.join(embed_dir, f"prototypes_kmeans_{self.max_prompts_per_class}.pt")
        MUST_CAL = True
        if not MUST_CAL and os.path.exists(kmeans_save_path):
            print(f"Loading precomputed kmeans from {kmeans_save_path}")
            prototypes = torch.load(kmeans_save_path, weights_only=True)
        else:
            for class_name in self.event_classes:
                samples = weather_embeds[class_name] # [N, event_dim]
                num_samples = samples.size(0)
                if num_samples <= self.max_prompts_per_class:
                    prototypes[class_name] = samples
                else:
                    samples_np = samples.numpy()
                    kmeans = KMeans(
                        n_clusters=self.max_prompts_per_class,
                        random_state=0,
                        n_init='auto'
                    ).fit(samples_np)
                    centers = torch.tensor(kmeans.cluster_centers_, dtype=samples.dtype)
                    prototypes[class_name] = centers # [max_prompts_per_class, event_dim]
            torch.save(prototypes, kmeans_save_path)
            print(f"Saved kmeans clustering result to {kmeans_save_path}")
                
        # self.prototypes = {k: nn.Parameter(v, requires_grad=False) for k, v in prototypes} # M -> N, D
        
        # padding & masking
        max_N = max([v.shape[0] for v in prototypes.values()])
        M = len(self.event_classes)
        stacked_prototypes = []
        prototype_masks = []
        for class_name in self.event_classes:
            proto = prototypes[class_name] # [N, D]
            N = proto.shape[0]
            if N < max_N:
                pad = torch.zeros((max_N - N, self.embed_dim), dtype=proto.dtype, device=proto.device)
                proto_padded = torch.concat([proto, pad], dim=0) # [max_N, D]
                mask = torch.cat([torch.ones(N, dtype=torch.bool), torch.zeros(max_N - N, dtype=torch.bool)]) # [max_N]
            else:
                proto_padded = proto
                mask = torch.ones(max_N, dtype=torch.bool)
            stacked_prototypes.append(proto_padded)
            prototype_masks.append(mask)
        
        if self.use_random:
            self.prompt_pool_rand = nn.Parameter(torch.randn(M, self.embed_dim), requires_grad=True) # M, D
            for idx, ps in enumerate(stacked_prototypes):
                num_rands = self.max_prompts_per_class // 4
                rand_prompts = torch.randn(num_rands, self.embed_dim)
                stacked_prototypes[idx] = torch.concat([ps, rand_prompts], dim=0)
                prototype_masks[idx] = torch.concat([prototype_masks[idx], torch.ones(num_rands, dtype=torch.bool)])
            
        self.prototypes = nn.Parameter(torch.stack(stacked_prototypes, dim=0), requires_grad=True) # [M, max_N, D]
        self.prototype_masks = torch.stack(prototype_masks, dim=0) # [M, max_N]
        
        print("Prototypes Shape: ", self.prototypes.shape)
    
    def _init_rand_pool(self):
        
        loaded_data = torch.load(self.weather_embed_path, weights_only=True)
        weather_embeds = {}
        for k, v in loaded_data.items():
            weather_embeds[k] = torch.stack(v, dim=0)

        prototypes = {}
        self.event_classes = list(weather_embeds.keys())
        print("Dual Extreme Events:", self.event_classes)
        M = len(self.event_classes)
            
        self.prototypes = nn.Parameter(
            torch.randn(M, self.max_prompts_per_class, self.embed_dim), requires_grad=True) # [M, max_N, D]
        self.prototype_masks = torch.ones(M, self.max_prompts_per_class) # [M, max_N]
        
        print("Prototypes Shape: ", self.prototypes.shape)

    def prompt_pool_learn(self, x):
        
        queries = self.query_proj_prot(x) # [B, D]
        prompts = self.prototypes # [M, N, D]
        # prompts = F.normalize(prompts, p=2, dim=-1)
        mask = self.prototype_masks.to(prompts.device) # [M, N]
        keys = self.key_proj_prot(prompts) # [M, N, D]
        values = self.value_proj_prot(prompts) # [M, N, D]

        attn_scores = torch.einsum('bd,mnd->bmn', queries, keys)  # [B, M, N]
        attn_scores = attn_scores.masked_fill(mask[None, :, :] == 0, float('-inf'))  # [B, M, N]
        attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1)  # [B, M, N]
        y = torch.einsum('bmn,mnd->bmd', attn_weights, values)  # [B, M, D]
        if self.use_random:
            y = torch.concat([y, self.prompt_pool_rand.unsqueeze(0).expand(queries.shape[0], -1, -1)], dim=1) # [B, 2*M, D]
        # y = F.normalize(y, p=2, dim=-1)
        # y = self.norm_p_1(y)

        return y

    def forward(self, x, times):
        
        # x: [B, D, H, W]
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, D) # [B, N, D]，N=H*W
        x = self.norm_x(x)
        
        # time_embeds = self.time_embed_layer(times) # [B, D]
        # prompts = self.prompt_pool_learn(torch.concat([time_embeds, x.mean(dim=1)], dim=-1)) # [B, M, D]
        prompts = self.prompt_pool_learn(x.mean(dim=1)) # [B, M, D]
        test_ps = F.normalize(prompts, p=2, dim=-1)
        sim_mat = torch.einsum("bmd,bnd->bmn", test_ps, test_ps)
        sim_loss = sim_mat.mean()
        
        queries = self.query_proj(x) # [B, N, D]
        keys = self.key_proj(prompts) # [B, M, D]
        values = self.value_proj(prompts) # [B, M, D]

        # attention: [B, N, D] x [B, M, D] -> [B, N, M]
        attn_scores = torch.einsum('bnd,bmd->bnm', queries, keys) # [B, N, M]
        attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=2) # [B, N, M]
        y = torch.einsum('bnm,bmd->bnd', attn_weights, values) # [B, N, D]
        # y = self.norm_p_2(y)

        y = y.reshape(B, H, W, D).permute(0, 3, 1, 2)

        if self.use_uniform:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-20)).sum(dim=2) # [B, N]
            entropy_loss = -entropy.mean()
        else:
            entropy_loss = None
        
        # entropy = -(attn_weights * torch.log(attn_weights + 1e-20)).sum(dim=2) # [B, N]
        # theory_max = torch.log(torch.tensor(attn_weights.shape[2], dtype=torch.float))
        # theory_min = 0.0
        # actual_max = entropy.max()
        # actual_min = entropy.min()
        # print(f"tmax: {theory_max.item():.3f}, rmax: {actual_max.item():.3f}, rmin: {actual_min.item():.3f} mean: {entropy.mean().item():.3f}")

        return y, entropy_loss, attn_weights, sim_loss


class MultiscalePooling1D(nn.Module):
    
    def __init__(self, C, reduction_ratio=8):
        
        super().__init__()
        
        self.conv = nn.Conv1d(C, C//reduction_ratio, kernel_size=3, padding=1)
        self.out_proj = nn.Linear(C//reduction_ratio, C)

    def forward(self, x):
        
        BNT, T, C = x.shape
        x = x.transpose(1, 2)  # (B*N, C, T)
        x = self.conv(x) # (B*N, C', T)
        x = x.mean(-1)  # (B*N, C')
        return self.out_proj(x)  # (B*N, C)

class PromptDualAttentionFreq(nn.Module):
    
    def __init__(self, in_dim, embed_dim, num_freq, weather_embed_path, max_prompts_per_class=5, use_out_proj='p+x'):
        
        super().__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_freq = num_freq
        self.weather_embed_path = weather_embed_path
        print("### prompting tokens path: ", weather_embed_path)
        self.max_prompts_per_class = max_prompts_per_class
        self.use_out_proj = use_out_proj

        self._init_prototype_pool()
        self.pool_layer = MultiscalePooling1D(2 * embed_dim)
        
        self.query_proj_prot = nn.Linear(in_dim, in_dim)
        self.key_proj_prot = nn.Linear(in_dim, in_dim)
        self.value_proj_prot = nn.Linear(2 * embed_dim, 2 * embed_dim)      
        
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(2 * embed_dim, 2 * embed_dim)
        
        self.norm_x = nn.LayerNorm(in_dim)
        self.x_proj = nn.Linear(2 * embed_dim, 2 * embed_dim)
        self.y_proj = nn.Linear(2 * embed_dim, 2 * embed_dim)
        
    def _init_prototype_pool(self):
        
        C, T = self.embed_dim, self.num_freq
        weather_save_path = os.path.join(self.weather_embed_path, "weather_kvs.pt")
        if os.path.exists(weather_save_path):
            wd = torch.load(weather_save_path, weights_only=True)
            weather_ks, weather_vs = wd["ks"], wd["vs"]
        else:
            type_path_list = os.listdir(self.weather_embed_path)
            weather_ks, weather_vs = {}, {}
            for type_name in tqdm(type_path_list):
                if '.pt' in type_name:
                    continue
                type_path = os.path.join(self.weather_embed_path, type_name)
                k_data = torch.load(os.path.join(type_path, "all_ks.pth"), weights_only=True)
                v_data = torch.load(os.path.join(type_path, "all_vs.pth"), weights_only=True)
                weather_ks[type_name] = torch.stack([k_data['hfa'], k_data['avgamp']], dim=-1).flatten(-2, -1) # N, C*2
                weather_vs[type_name] = torch.view_as_real(v_data["v"]) # N, C, T, 2
            torch.save({"ks": weather_ks, "vs": weather_vs}, weather_save_path)

        self.event_classes = list(weather_ks.keys())
        print("Dual Extreme Events:", self.event_classes, "Max Classes:", self.max_prompts_per_class)
        
        if self.max_prompts_per_class > 1:
            kmeans_save_path = os.path.join(self.weather_embed_path, f"prototypes_kmeans_vs_{self.max_prompts_per_class}.pt")
            MUST_CAL = False
            if not MUST_CAL and os.path.exists(kmeans_save_path):
                print(f"Loading precomputed kmeans from {kmeans_save_path}")
                prototypes = torch.load(kmeans_save_path, weights_only=True)
            else:
                prototypes = {}
                for class_name in tqdm(self.event_classes):
                    samples = weather_vs[class_name].reshape(-1, C * T * 2) # [N, event_dim]
                    num_samples = samples.size(0)
                    if num_samples <= self.max_prompts_per_class:
                        prototypes[class_name] = samples
                    else:
                        samples_np = samples.numpy()
                        kmeans = KMeans(
                            n_clusters=self.max_prompts_per_class,
                            random_state=0,
                            n_init='auto'
                        ).fit(samples_np)
                        centers = torch.tensor(kmeans.cluster_centers_, dtype=samples.dtype)
                        prototypes[class_name] = centers # [max_prompts_per_class, event_dim]
                torch.save(prototypes, kmeans_save_path)
                print(f"Saved kmeans clustering result to {kmeans_save_path}")
                
            # padding & masking
            max_N = max([v.shape[0] for v in prototypes.values()])
            M = len(self.event_classes)
            stacked_prototypes = []
            prototype_masks = []
            for class_name in tqdm(self.event_classes):
                proto = prototypes[class_name] # [N, D]
                N = proto.shape[0]
                if N < max_N:
                    pad = torch.zeros((max_N - N, C * T * 2), dtype=proto.dtype, device=proto.device)
                    proto_padded = torch.concat([proto, pad], dim=0) # [max_N, D]
                    mask = torch.cat([torch.ones(N, dtype=torch.bool), torch.zeros(max_N - N, dtype=torch.bool)]) # [max_N]
                else:
                    proto_padded = proto
                    mask = torch.ones(max_N, dtype=torch.bool)
                stacked_prototypes.append(proto_padded)
                prototype_masks.append(mask)
                
            self.prototypes_vs = nn.Parameter(
                torch.stack(stacked_prototypes, dim=0).reshape(-1, max_N, C, T, 2).permute(0, 1, 3, 2, 4).reshape(-1, max_N, T, C * 2), 
                requires_grad=False)
            self.prototype_masks = torch.stack(prototype_masks, dim=0) # [M, max_N]
            print("Prototypes Shape: ", self.prototypes_vs.shape, type(self.prototypes_vs))
            # exit(-1)
        else:
            stacked_prototypes = []
            for class_name in tqdm(self.event_classes):
                samples = weather_vs[class_name].reshape(-1, C * T * 2) # [N, event_dim]
                stacked_prototypes.append(samples.mean(dim=0))
            self.prototypes_vs = nn.Parameter(
                torch.stack(stacked_prototypes, dim=0).reshape(-1, C, T, 2).permute(0, 2, 1, 3).reshape(-1, T, C * 2), 
                requires_grad=False)
    
    def prompt_pool_learn(self, x):
        
        if self.max_prompts_per_class > 1:
            queries = self.query_proj_prot(x) # [B, C*2]
            prompts_vs = self.prototypes_vs # [M, N, T, C*2]
            M, N, T, C_2 = prompts_vs.shape
            prompts_ks = self.pool_layer(self.prototypes_vs.reshape(-1, T, C_2)).reshape(M, N, C_2) # [M, N, C*2]
            mask = self.prototype_masks.to(prompts_ks.device) # [M, N]
            keys = self.key_proj_prot(prompts_ks) # [M, N, C*2]
            values = self.value_proj_prot(prompts_vs) # [M, N, T, C*2]

            attn_scores = torch.einsum('bd,mnd->bmn', queries, keys)  # [B, M, N]
            attn_scores = attn_scores.masked_fill(mask[None, :, :] == 0, float('-inf')) # [B, M, N]
            attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1) # [B, M, N]
            y_vs = torch.einsum('bmn,mntc->bmtc', attn_weights, values) # [B, M, T, C*2]
            y_ks = self.pool_layer(y_vs.reshape(-1, T, C_2)).reshape(-1, M, C_2) # [B, M, C*2]
        else:
            y_vs = self.prototypes_vs.unsqueeze(0).expand(x.shape[0], -1, -1, -1) # [B, M, T, C*2]
            B, M, T, C_2 = y_vs.shape
            y_ks = self.pool_layer(y_vs.reshape(-1, T, C_2)).reshape(-1, M, C_2) # [B, M, C*2]

        return y_ks, y_vs

    def forward(self, x):
        
        B, C, T = x.shape
        x = torch.view_as_real(x.permute(0, 2, 1)).reshape(B, T, -1)
        x = self.norm_x(x)
        x_q = self.pool_layer(x) # [B, C*2]
    
        prompts_ks, prompts_vs = self.prompt_pool_learn(x_q) # [B, M, C*2], [B, M, T, C*2]
        queries = self.query_proj(x_q) # [B, C*2]
        keys = self.key_proj(prompts_ks) # [B, M, C*2]
        values = self.value_proj(prompts_vs) # [B, M, T, C*2]
        
        attn_scores = torch.einsum('bnd,bmd->bnm', queries.unsqueeze(1), keys) # [B, 1, M]
        attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1) # [B, 1, M]
        y = torch.einsum('bnm,bmtc->bntc', attn_weights, values).squeeze() # [B, T, C*2]
        if self.use_out_proj is not None:
            if self.use_out_proj == "p":
                 y = self.y_proj(y) # [B, T, C*2]
            elif self.use_out_proj == "p+x":
                y = self.y_proj(y + x) # [B, T, C*2]
            elif self.use_out_proj == "p+x+t":
                y = self.y_proj(y + self.x_proj(x)) # [B, T, C*2]
            else:
                raise ValueError
        y = torch.view_as_complex(y.reshape(B, -1, self.embed_dim, 2).permute(0, 2, 1, 3)) # [B, C, T]

        return y, attn_weights.squeeze()


class MultiscalePooling2D(nn.Module):
    
    def __init__(self, C, reduction_ratio=8):
        
        super().__init__()
        
        self.conv = nn.Conv2d(C, C//reduction_ratio, kernel_size=3, padding=1)
        self.out_proj = nn.Linear(C//reduction_ratio, C)

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = self.conv(x) # (B, C', H, W)
        x = x.mean(dim=(-2, -1))  # (B, C')
        return self.out_proj(x)  # (B, C)

class PromptDualAttentionSpace(nn.Module):
    
    def __init__(self, in_dim, embed_dim, in_shape, weather_embed_path, max_prompts_per_class=5, use_out_proj='p+x'):
        
        super().__init__()
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.in_shape = in_shape
        self.weather_embed_path = weather_embed_path
        print("### prompting tokens path: ", weather_embed_path)
        self.max_prompts_per_class = max_prompts_per_class
        self.use_out_proj = use_out_proj

        self._init_prototype_pool()
        self.pool_layer = MultiscalePooling2D(embed_dim)
        
        self.query_proj_prot = nn.Linear(in_dim, in_dim)
        self.key_proj_prot = nn.Linear(in_dim, in_dim)
        self.value_proj_prot = nn.Linear(embed_dim, embed_dim)      
        
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm_x = nn.LayerNorm(in_dim)
        self.x_proj = nn.Linear(embed_dim, embed_dim)
        self.y_proj = nn.Linear(embed_dim, embed_dim)

    def _init_prototype_pool(self):
        
        C, H, W = self.embed_dim, self.in_shape[0], self.in_shape[1]
        weather_save_path = os.path.join(self.weather_embed_path, "weather_kvs.pt")
        if os.path.exists(weather_save_path):
            wd = torch.load(weather_save_path, weights_only=True)
            weather_ks, weather_vs = wd["ks"], wd["vs"]
        else:
            type_path_list = os.listdir(self.weather_embed_path)
            weather_ks, weather_vs = {}, {}
            for type_name in tqdm(type_path_list):
                if '.pt' in type_name:
                    continue
                type_path = os.path.join(self.weather_embed_path, type_name)
                k_data = torch.load(os.path.join(type_path, "all_ks.pth"), weights_only=True)
                v_data = torch.load(os.path.join(type_path, "all_vs.pth"), weights_only=True)
                weather_ks[type_name] = torch.stack([k_data['hfa'], k_data['avgamp']], dim=-1).flatten(-2, -1) # N, C*2
                weather_vs[type_name] = v_data["v"] # N, C, H, W
            torch.save({"ks": weather_ks, "vs": weather_vs}, weather_save_path)

        self.event_classes = list(weather_ks.keys())
        print("Dual Extreme Events:", self.event_classes, "Max Classes:", self.max_prompts_per_class)
        
        if self.max_prompts_per_class > 1:
            kmeans_save_path = os.path.join(self.weather_embed_path, f"prototypes_kmeans_vs_{self.max_prompts_per_class}.pt")
            MUST_CAL = False
            if not MUST_CAL and os.path.exists(kmeans_save_path):
                print(f"Loading precomputed kmeans from {kmeans_save_path}")
                prototypes = torch.load(kmeans_save_path, weights_only=True)
            else:
                prototypes = {}
                for class_name in tqdm(self.event_classes):
                    samples = weather_vs[class_name].reshape(-1, C * H * W) # [N, event_dim]
                    num_samples = samples.size(0)
                    if num_samples <= self.max_prompts_per_class:
                        prototypes[class_name] = samples
                    else:
                        samples_np = samples.numpy()
                        kmeans = KMeans(
                            n_clusters=self.max_prompts_per_class,
                            random_state=0,
                            n_init='auto'
                        ).fit(samples_np)
                        centers = torch.tensor(kmeans.cluster_centers_, dtype=samples.dtype)
                        prototypes[class_name] = centers # [max_prompts_per_class, event_dim]
                torch.save(prototypes, kmeans_save_path)
                print(f"Saved kmeans clustering result to {kmeans_save_path}")
                
            # padding & masking
            max_N = max([v.shape[0] for v in prototypes.values()])
            M = len(self.event_classes)
            stacked_prototypes = []
            prototype_masks = []
            for class_name in tqdm(self.event_classes):
                proto = prototypes[class_name] # [N, D]
                N = proto.shape[0]
                if N < max_N:
                    pad = torch.zeros((max_N - N, C * H * W), dtype=proto.dtype, device=proto.device)
                    proto_padded = torch.concat([proto, pad], dim=0) # [max_N, D]
                    mask = torch.cat([torch.ones(N, dtype=torch.bool), torch.zeros(max_N - N, dtype=torch.bool)]) # [max_N]
                else:
                    proto_padded = proto
                    mask = torch.ones(max_N, dtype=torch.bool)
                stacked_prototypes.append(proto_padded)
                prototype_masks.append(mask)
                
            self.prototypes_vs = nn.Parameter(
                torch.stack(stacked_prototypes, dim=0).reshape(M, max_N, C, H, W), 
                requires_grad=False)
            self.prototype_masks = torch.stack(prototype_masks, dim=0) # [M, max_N]
            print("Prototypes Shape: ", self.prototypes_vs.shape, type(self.prototypes_vs))
        else:
            stacked_prototypes = []
            for class_name in tqdm(self.event_classes):
                samples = weather_vs[class_name].reshape(-1, C * T * 2) # [N, event_dim]
                stacked_prototypes.append(samples.mean(dim=0))
            self.prototypes_vs = nn.Parameter(
                torch.stack(stacked_prototypes, dim=0).reshape(-1, C, T, 2).permute(0, 2, 1, 3).reshape(-1, T, C * 2), 
                requires_grad=False)
        
    def prompt_pool_learn(self, x):
        
        if self.max_prompts_per_class > 1:
            queries = self.query_proj_prot(x) # [B, C]
            prompts_vs = self.prototypes_vs # [M, N, C, H, W]
            M, N, C, H, W = prompts_vs.shape
            prompts_ks = self.pool_layer(self.prototypes_vs.reshape(-1, C, H, W)).reshape(M, N, C) # [M, N, C]
            mask = self.prototype_masks.to(prompts_ks.device) # [M, N]
            keys = self.key_proj_prot(prompts_ks) # [M, N, C]
            values = self.value_proj_prot(prompts_vs.reshape(M, N, C, -1).permute(0, 1, 3, 2)) # [M, N, H*W, C]

            attn_scores = torch.einsum('bd,mnd->bmn', queries, keys)  # [B, M, N]
            attn_scores = attn_scores.masked_fill(mask[None, :, :] == 0, float('-inf')) # [B, M, N]
            attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1) # [B, M, N]
            y_vs = torch.einsum('bmn,mntc->bmtc', attn_weights, values).reshape(-1, M, H, W, C).permute(0, 1, 4, 2, 3) # [B, M, C, H, W]
            y_ks = self.pool_layer(y_vs.reshape(-1, C, H, W)).reshape(-1, M, C) # [B, M, C]
        else:
            y_vs = self.prototypes_vs.unsqueeze(0).expand(x.shape[0], -1, -1, -1. -1) # [B, M, C, H, W]
            B, M, C, H, W = y_vs.shape
            y_ks = self.pool_layer(y_vs.reshape(-1, C, H, W)).reshape(-1, M, C) # [B, M, C]

        return y_ks, y_vs

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.norm_x(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, C, H, W]
        x_q = self.pool_layer(x) # [B, C]

        prompts_ks, prompts_vs = self.prompt_pool_learn(x_q) # [B, M, C], [B, M, C, H, W]
        queries = self.query_proj(x_q) # [B, C]
        keys = self.key_proj(prompts_ks) # [B, M, C]
        values = self.value_proj(prompts_vs.reshape(B, -1, C, H * W).permute(0, 1, 3, 2)) # [B, M, H*W, C]

        attn_scores = torch.einsum('bnd,bmd->bnm', queries.unsqueeze(1), keys) # [B, 1, M]
        attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1) # [B, 1, M]
        y = torch.einsum('bnm,bmtc->bntc', attn_weights, values).squeeze() # [B, H*W, C]

        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        if self.use_out_proj is not None:
            if self.use_out_proj == "p":
                 y = self.y_proj(y) # [B, H*W, C]
            elif self.use_out_proj == "p+x":
                y = self.y_proj(y + x) # [B, H*W, C]
            elif self.use_out_proj == "p+x+t":
                y = self.y_proj(y + self.x_proj(x)) # [B, H*W, C]
            else:
                raise ValueError
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]

        return y, attn_weights.squeeze()


if __name__ == "__main__":
    
    L = TimeEmbedding(66)
    L(torch.tensor([[2024010100, 2024010101]]))
    
    
    
    
# class PromptDualAttentionFreq(nn.Module):
    
#     def __init__(self, in_dim, embed_dim, weather_embed_path, max_prompts_per_class=20):
        
#         super().__init__()
        
#         self.in_dim = in_dim
#         self.embed_dim = embed_dim
#         self.weather_embed_path = weather_embed_path
#         print("### prompting tokens path: ", weather_embed_path)
#         self.max_prompts_per_class = max_prompts_per_class

#         self._init_prototype_pool()
        
#         self.query_proj_prot = nn.Linear(in_dim, in_dim)
#         self.key_proj_prot = nn.Linear(in_dim, in_dim)
#         self.value_proj_prot = nn.Linear(2 * embed_dim, 2 * embed_dim)      
        
#         self.query_proj = nn.Linear(in_dim, in_dim)
#         self.key_proj = nn.Linear(in_dim, in_dim)
#         self.value_proj = nn.Linear(2 * embed_dim, 2 * embed_dim)
        
#         self.norm_x = nn.LayerNorm(in_dim)
        
#     def calculate_keys(self, x, freq_radius_flat):
        
#         B, C, T = x.shape
#         amplitude = torch.abs(x)
#         phase = torch.angle(x)
#         energy = amplitude ** 2
#         sorted_idx = torch.argsort(freq_radius_flat)
#         freq_sorted = freq_radius_flat[sorted_idx]
#         energy_sorted = energy[:, :, sorted_idx]
#         energy_ratio = torch.cumsum(energy_sorted, dim=-1)
#         energy_ratio = energy_ratio / energy_ratio[:, :, -1:]
#         X = (freq_sorted - freq_sorted.min()) / (freq_sorted.max() - freq_sorted.min())
#         high_freq_areas = 1 - torch.trapz(energy_ratio, X.expand(B, C, -1).to(x.device), dim=-1)
#         avg_amplitude = amplitude.mean(dim=-1)
#         queries = torch.stack([high_freq_areas, avg_amplitude], dim=-1).reshape(B, C * 2)
        
#         return queries

#     def _init_prototype_pool(self):
        
#         # Load weather embeddings and process per-class prototypes
#         type_path_list = os.listdir(self.weather_embed_path)
#         weather_ks, weather_vs = {}, {}
#         C, T = None, None
#         for type_name in tqdm(type_path_list):
#             if '.pt' in type_name:
#                 continue
#             type_path = os.path.join(self.weather_embed_path, type_name)
#             k_data = torch.load(os.path.join(type_path, "all_ks.pth"), weights_only=True)
#             v_data = torch.load(os.path.join(type_path, "all_vs.pth"), weights_only=True)
#             weather_ks[type_name] = torch.stack([k_data['hfa'], k_data['avgamp']], dim=-1).flatten(-2, -1) # N, C*2
#             _, C, T = v_data["v"].shape
#             weather_vs[type_name] = torch.view_as_real(v_data["v"]) # N, C, T, 2

#         self.event_classes = list(weather_ks.keys())
#         print("Dual Extreme Events:", self.event_classes)

#         kmeans_save_path = os.path.join(self.weather_embed_path, "prototypes_kmeans_vs.pt")
#         MUST_CAL = False
#         if not MUST_CAL and os.path.exists(kmeans_save_path):
#             print(f"Loading precomputed kmeans from {kmeans_save_path}")
#             prototypes = torch.load(kmeans_save_path, weights_only=True)
#         else:
#             prototypes = {}
#             for class_name in tqdm(self.event_classes):
#                 samples = weather_vs[class_name].reshape(-1, C * T * 2) # [N, event_dim]
#                 num_samples = samples.size(0)
#                 if num_samples <= self.max_prompts_per_class:
#                     prototypes[class_name] = samples
#                 else:
#                     samples_np = samples.numpy()
#                     kmeans = KMeans(
#                         n_clusters=self.max_prompts_per_class,
#                         random_state=0,
#                         n_init='auto'
#                     ).fit(samples_np)
#                     centers = torch.tensor(kmeans.cluster_centers_, dtype=samples.dtype)
#                     prototypes[class_name] = centers # [max_prompts_per_class, event_dim]
#             torch.save(prototypes, kmeans_save_path)
#             print(f"Saved kmeans clustering result to {kmeans_save_path}")
            
#         kmeans_save_path = os.path.join(self.weather_embed_path, "prototypes_kmeans_ks.pt")
#         MUST_CAL = True
#         if not MUST_CAL and os.path.exists(kmeans_save_path):
#             print(f"Loading precomputed kmeans from {kmeans_save_path}")
#             prototypes_ks = torch.load(kmeans_save_path, weights_only=True)
#         else:            
#             prototypes_ks = {}
#             freq_y = torch.fft.fftfreq(10)
#             freq_x = torch.fft.rfftfreq(10)
#             freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
#             freq_radius = torch.sqrt(freq_grid_y ** 2 + freq_grid_x ** 2)
#             self.freq_radius_flat = freq_radius.flatten()
#             for class_name, x in tqdm(prototypes.items()):
#                 x = torch.view_as_complex(x.reshape(-1, C, T, 2))
#                 queries = self.calculate_keys(x, self.freq_radius_flat)
#                 prototypes_ks[class_name] = queries
#             torch.save(prototypes_ks, kmeans_save_path)
#             print(f"Saved kmeans clustering result to {kmeans_save_path}")
        
#         # padding & masking
#         max_N = max([v.shape[0] for v in prototypes.values()])
#         M = len(self.event_classes)
#         stacked_prototypes = []
#         stacked_prototypes_ks = []
#         prototype_masks = []
#         for class_name in tqdm(self.event_classes):
#             proto = prototypes[class_name] # [N, D]
#             proto_k = prototypes_ks[class_name] # [N, D]
#             N = proto.shape[0]
#             if N < max_N:
#                 pad = torch.zeros((max_N - N, C * T * 2), dtype=proto.dtype, device=proto.device)
#                 proto_padded = torch.concat([proto, pad], dim=0) # [max_N, D]
#                 pad_k = torch.zeros((max_N - N, C * 2), dtype=proto.dtype, device=proto.device)
#                 proto_padded_k = torch.concat([proto_k, pad_k], dim=0) # [max_N, D]
#                 mask = torch.cat([torch.ones(N, dtype=torch.bool), torch.zeros(max_N - N, dtype=torch.bool)]) # [max_N]
#             else:
#                 proto_padded = proto
#                 proto_padded_k = proto_k
#                 mask = torch.ones(max_N, dtype=torch.bool)
#             stacked_prototypes.append(proto_padded)
#             stacked_prototypes_ks.append(proto_padded_k)
#             prototype_masks.append(mask)
            
#         self.prototypes_vs = nn.Parameter(
#             torch.stack(stacked_prototypes, dim=0).reshape(-1, max_N, C, T, 2).permute(0, 1, 3, 2, 4).reshape(-1, max_N, T, C * 2), 
#             requires_grad=False)
#         # self.prototypes_ks = nn.Parameter(
#         #     torch.stack(stacked_prototypes_ks, dim=0), 
#         #     requires_grad=False)
#         self.prototype_masks = torch.stack(prototype_masks, dim=0) # [M, max_N]
#         print("Prototypes Shape: ", self.prototypes_vs.shape, type(self.prototypes_vs), 
#               self.prototypes_ks.shape, type(self.prototypes_ks))
    
#     def prompt_pool_learn(self, x):
        
#         queries = self.query_proj_prot(x) # [B, D]
#         prompts_vs = self.prototypes_vs # [M, N, D], [M, N, T, C*2]
#         mask = self.prototype_masks.to(prompts_ks.device) # [M, N]
#         keys = self.key_proj_prot(prompts_ks) # [M, N, D]
#         values = self.value_proj_prot(prompts_vs) # [M, N, T, C*2]

#         attn_scores = torch.einsum('bd,mnd->bmn', queries, keys)  # [B, M, N]
#         attn_scores = attn_scores.masked_fill(mask[None, :, :] == 0, float('-inf')) # [B, M, N]
#         attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=-1) # [B, M, N]
#         y = torch.einsum('bmn,mntc->bmtc', attn_weights, values) # [B, M, T, C*2]
        
#         B, M, T, C_2 = y.shape
#         qs = torch.view_as_complex(y.reshape(-1, T, C_2 // 2, 2)).permute(0, 2, 1)
#         qs = self.calculate_keys(qs.cpu(), self.freq_radius_flat).reshape(B, M, -1).to(y.device)

#         return qs, y

#     def forward(self, x):
        
#         # x: [B, D, H, W]
#         B, D = x.shape
#         x = self.norm_x(x)
    
#         prompts_ks, prompts_vs = self.prompt_pool_learn(x) # [B, M, D], [B, M, T, C*2]
#         queries = self.query_proj(x) # [B, D]
#         keys = self.key_proj(prompts_ks) # [B, M, D]
#         values = self.value_proj(prompts_vs) # [B, M, T, C*2]

#         # attention: [B, N, D] x [B, M, D] -> [B, N, M]
#         attn_scores = torch.einsum('bnd,bmd->bnm', queries.unsqueeze(1), keys) # [B, 1, M]
#         attn_weights = F.softmax(attn_scores / (queries.shape[-1] ** 0.5), dim=2) # [B, 1, M]
#         y = torch.einsum('bnm,bmtc->bntc', attn_weights, values) # [B, 1, T, C*2]
#         y = torch.view_as_complex(y.reshape(B, -1, self.embed_dim, 2).permute(0, 2, 1, 3)) # [B, C, T]

#         return y, attn_weights
