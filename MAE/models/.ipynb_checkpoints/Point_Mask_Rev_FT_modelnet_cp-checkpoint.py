import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from .modules import square_distance, index_points
from torch.nn import Conv2d, Dropout
from .adapter_super import AdapterSuper
import ipdb
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center,center_idx = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        center_idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * num_points
        center_idx = center_idx + center_idx_base
        center_idx = center_idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, idx, center_idx


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #NOTE: prompt
        self.gate = torch.nn.Parameter(torch.zeros(1, 6, 1, 1))

    def forward(self, x, prompt, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        #NOTE: prompt
        # B, N, C = prompt.shape
        kv = self.qkv(prompt).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        prompt_k, prompt_v = kv[1], kv[2]  # make torchscript happy (cannot use tensor as tuple)

        prompt_attn = (q @ prompt_k.transpose(-2, -1)) * self.scale
        prompt_attn = prompt_attn.softmax(dim=-1)
        prompt_attn = self.gate * prompt_attn.softmax(dim=-1)
        prompt_attn = self.attn_drop(prompt_attn)

        prompt_x = (prompt_attn @ prompt_v).transpose(1, 2).reshape(B, -1, C)
        # x = torch.cat((prompt_x, x), dim=1)
        x = x + prompt_x
        ######################
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Attention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or 18 ** -0.5
        self.qkv = nn.Linear(dim, 18*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(18, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, 18 // (self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 18)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_dim=None, drop_rate_adapter=None, num_tokens=None, if_third=False, if_half=False, if_two=False, if_one=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if if_third:
            self.cp_adapter = AdapterSuper(
                embed_dims=384,
                reduction_dims=16,
                drop_rate_adapter=drop_rate_adapter,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
                    )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adapter = AdapterSuper(
                embed_dims=dim,
                reduction_dims=adapter_dim,
                drop_rate_adapter=drop_rate_adapter,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
                    )
        if if_half:
            self.adapter1 = AdapterSuper(
                    embed_dims=dim,
                    reduction_dims=8,
                    drop_rate_adapter=drop_rate_adapter,
                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
                        )
        else:
            self.adapter1 = AdapterSuper(
                    embed_dims=dim,
                    reduction_dims=adapter_dim,#NOTE: re_dims=8
                    drop_rate_adapter=drop_rate_adapter,
                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
                        )
            
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.ad_gate = torch.nn.Parameter(torch.zeros(1))
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.GELU())

        self.prompt_dropout = Dropout(0.1)
        self.num_tokens = num_tokens
        self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens, dim))
        #cp
        if if_third:
            self.cp_act = act_layer()
            self.train_images_features_agg = torch.load("./ckpts/train_f_pos_shape_modelnet.pt")
            self.cp_gate = 0.5
            #self.cp_gate = torch.nn.Parameter(torch.zeros(1))
        ###
        trunc_normal_(self.prompt_embeddings, std=.02)
    
    def pooling(self, knn_x_w, if_maxmean):
        # Feature Aggregation (Pooling)
        ##maxmean
        #lc_x = knn_x_w.max(dim=2)[0] + knn_x_w.mean(dim=2) #NOTE:maxeman/max
        if if_maxmean:
            lc_x = knn_x_w.max(dim=2)[0] + knn_x_w.mean(dim=2) #NOTE:pooling=maxmean
        else:
            lc_x = knn_x_w.max(dim=2)[0]
        ##max
        #lc_x = knn_x_w.max(dim=2)[0]
        #mean
        # lc_x = knn_x_w.mean(dim=2) 
        ##sum
        # lc_x = knn_x_w.sum(dim=2)
        lc_x = self.out_transform(lc_x.permute(0, 2, 1)).permute(0,2,1)
        return lc_x
    
    def propagate(self, xyz1, xyz2, points1, points2, de_neighbors, pro_cof):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, N, D']
            points2: input points data, [B, S, D'']
        Return:
            new_points: upsampled points data, [B, N, D''']
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :de_neighbors], idx[:, :, :de_neighbors]  # [B, N, S]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        weight = weight.view(B, N, de_neighbors, 1)
        interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)#B, N, 6, C->B,N,C
        #new_points = self.norm_linear(torch.cat([points1, interpolated_points], dim=-1)) # B,N,C
        new_points = points1+pro_cof*interpolated_points # B,N,C #NOTE:0.1/0.2/0.3

        return new_points


    # def forward(self, x):
    #     x = x + self.drop_path(self.attn(self.norm1(x)))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x
    def forward(self, x, center1=None, center2=None, neighborhood=None, idx=None, center_idx=None, num_group=None, group_size=None, if_prompt=False, cp_feat=None, cp_conv=None, if_maxmean=None, pro_cof=None, center_cof=None, attn1=None, norm3=None):
        # NOTE prompt with zero-inti attn
        prompt = self.prompt_dropout(self.prompt_embeddings.repeat(center2.shape[0], 1, 1))
        # cp
        if cp_feat != None:
            K = prompt.shape[1] - 2 # kre
            cp_feat_norm = cp_feat / cp_feat.norm(dim=-1, keepdim=True) #[B, 384]
            new_knowledge = cp_feat_norm @ self.train_images_features_agg #[B, 11392]
            new_knowledge_k, idx_k = torch.topk(new_knowledge, K) #[B, 2*K]

            new_knowledge_k = F.softmax(new_knowledge_k, dim=1).unsqueeze(1)
            train_features_k = []
            for p in range(idx_k.shape[0]):
                train_features_k.append(self.train_images_features_agg[:, idx_k[p]].tolist())
            train_features_k = torch.tensor(train_features_k).permute(0, 2, 1).cuda() #[B, K, 384]
            feat_f = torch.matmul(new_knowledge_k, train_features_k) #[B, 1, 384]
            cache_prompt = torch.cat((cp_feat.unsqueeze(1), feat_f, train_features_k), 1)
        
            cache_prompt = self.cp_adapter(cache_prompt)
            prompt = prompt + cache_prompt
        ######################
        x = x + self.attn(self.norm1(x), prompt)
        x = self.ad_gate * self.adapter(self.drop_path(self.mlp(self.norm2(x)))) + x
        
        B,G,_ = x.shape
        ##because of class token 
        cls_x = x[:,0]
        x = x[:,1:]
        G = G-1
        ####
        x_neighborhoods = x.reshape(B*G, -1)[idx, :].reshape(B*center2.shape[1], group_size, -1)
        x_centers = x.reshape(B*G, -1)[center_idx, :].reshape(B, center2.shape[1], -1)
        
        std_xyz = torch.std(neighborhood)
        neighborhood = neighborhood / (std_xyz + 1e-5)
        x_neighborhoods = self.drop_path(attn1(norm3(x_neighborhoods.clone())))+x_neighborhoods.clone()

        vis_x = self.pooling(x_neighborhoods.reshape(B, center2.shape[1], group_size, -1), if_maxmean)+center_cof*x_centers#B,G1,C#NOTE: 0/0.3/0.5
        x = self.propagate(xyz1=center1, xyz2=center2, points1=x, points2=vis_x, de_neighbors=center2.shape[1], pro_cof=pro_cof)
        x = torch.cat((cls_x.unsqueeze(1), x), 1)
        x = self.adapter1(x)    
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., adapter_dim=1024., drop_rate_adapter=0, num_tokens=0., if_third=False, if_half=False, if_two=False, if_one=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                adapter_dim=adapter_dim, drop_rate_adapter=drop_rate_adapter, num_tokens=num_tokens, if_third=if_third, if_half=if_half
                ,if_one=if_one, if_two=if_two
                )
            for i in range(depth)])
        self.attn1 = Attention1(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, pos, center=None, center2=None, neighborhood=None, idx=None, center_idx=None, num_group=None, group_size=None, cp_feat=None, if_maxmean=None, pro_cof=None, center_cof=None):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, center,center2, neighborhood, idx, center_idx, num_group, group_size, cp_feat=cp_feat, if_maxmean=if_maxmean, pro_cof=pro_cof, center_cof=center_cof, attn1=self.attn1, norm3=self.norm3)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
       
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

# finetune model
@MODELS.register_module()
class PointTransformer_modelnet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.adapter_dim = config.adapter_config.adapter_dim
        self.drop_rate_adapter = config.adapter_config.adapter_drop_path_rate
        #self.train_images_features_agg = torch.load("./ckpts/train_f_pos_shape_scan.pt")

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            adapter_dim=self.adapter_dim, 
            drop_rate_adapter=self.drop_rate_adapter,
            num_tokens=10,
            #num_tokens=int(self.num_group//8),#NOTE: 100/ //4/ //8
            if_third=True,
            if_one=False,
            if_two=False,
            if_half = config.if_half,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, cache=False, cp_feat=None, args=None):

        neighborhood, center, idx, center_idx = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        self.group = Group(num_group=int(self.num_group/2), group_size=int(self.group_size/2))
        neighborhood, center_new, idx, center_idx = self.group(center)
        x = self.blocks(x, pos, center, center2=center_new ,neighborhood=neighborhood, idx=idx,center_idx=center_idx, group_size=int(self.group_size/2), cp_feat=cp_feat,  if_maxmean=args.if_maxmean, pro_cof=args.propagate_cof, center_cof=args.center_cof)
        # x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        if cache == True:
            for i in range(len(self.cls_head_finetune) - 1):
                concat_f = self.cls_head_finetune[i](concat_f)
            return concat_f
        ret = self.cls_head_finetune(concat_f)
        return ret
