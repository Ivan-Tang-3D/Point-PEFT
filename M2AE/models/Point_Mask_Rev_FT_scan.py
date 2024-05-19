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
import math
from .adapter_super import AdapterSuper, AdapterSuper_fn
import ipdb
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, out_c, 1)
            )

        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1),
                nn.BatchNorm1d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_c, in_c, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(in_c * 2, out_c, 1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_c, out_c, 1)
            )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.out_c)


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
        center,center_idx = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        if type(xyz) is tuple or type(center) is tuple:
            ipdb.set_trace()
        _, idx = self.knn(xyz, center)  # B G M
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., LoRA_dim=4., drop_rate_LoRA=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        

    def forward(self, x):
        #old_x = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        #old_x = x
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PosPool_Layer(nn.Module):
    def __init__(self, out_channels):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosPool_Layer, self).__init__()
        self.out_channels = out_channels
        

    def forward(self, re_xyz, x):
        B, _, npoint, nsample= re_xyz.shape
        feat_dim = self.out_channels // 6
        wave_length = 1000  ##NOTE:500 nothing
        alpha = 100
        feat_range = torch.arange(feat_dim, dtype=torch.float32).to(re_xyz.device)  # (feat_dim, )
        dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
        position_mat = torch.unsqueeze(alpha * re_xyz, -1)  # (B, 3, npoint, nsample, 1)
        div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
        sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
        cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
        position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
        # position_embedding = torch.stack([sin_mat, cos_mat], dim=5).flatten(4)
        position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
        position_embedding = position_embedding.view(B, self.out_channels, npoint, nsample)  # (B, C, npoint, nsample)

        aggregation_features = x + position_embedding  #NOTE:先乘后加 nothing
        aggregation_features *= position_embedding

        return aggregation_features


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., LoRA_dim=16., drop_rate_LoRA=0.1, prefix_dim=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask * - 100000.0
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        #old_x = x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Attention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights nothing
        self.scale = qk_scale or 18 ** -0.5
        self.qkv = nn.Linear(dim, 18*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(18, dim)
        self.proj_drop = nn.Dropout(proj_drop)
       
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, 18 // (self.num_heads)).permute(2, 0, 3, 1, 4)
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // (self.num_heads*8)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask = mask * float('-inf') 
            mask = mask * - 100000.0
            attn = attn + mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 18)
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C//8)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,adapter_dim=None, drop_rate_adapter=None, num_tokens=None, if_third=False, if_half=False, if_two=False, if_one=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here nothing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if if_third:
            #self.norm_cp = norm_layer(384)
            self.cp_adapter = AdapterSuper(
                embed_dims=384,
                reduction_dims=8,
                drop_rate_adapter=drop_rate_adapter,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
                    )
     
        self.norm2 = norm_layer(dim)
        #self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=False, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
      
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
        self.ad_gate = torch.nn.Parameter(torch.zeros(1))
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.GELU())
        self.pos_layer = PosPool_Layer(dim)

        self.prompt_dropout = Dropout(0.1)
        self.num_tokens = num_tokens
        self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens, dim))
       
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

    def forward(self, x, mask=None, center1=None, center2=None, neighborhood=None, idx=None, center_idx=None, num_group=None, group_size=None, if_prompt=False, cache_prompt=None, cp_conv=None, if_maxmean=None, pro_cof=None, center_cof=None, attn1=None, norm3=None,
                if_third = None, layer_id=None):
        B, G1, G2 = mask.shape
        mask_new = torch.zeros([B,G1+self.num_tokens,G2+self.num_tokens]).cuda()
        mask_new[:, :G1, :G2] = mask
        mask = mask_new # true:not contribute
        # NOTE prompt with zero-inti attn
        prompt = self.prompt_dropout(self.prompt_embeddings.repeat(center2.shape[0], 1, 1))
        # cp
        if cache_prompt != None:
            cache_prompt = self.cp_adapter(cache_prompt)
            prompt = prompt + cache_prompt
        ######################
        x = torch.cat((prompt,x), 1)
        x = x + self.attn(self.norm1(x), prompt, mask)
        x_fn = self.drop_path(self.mlp(self.norm2(x)))
        x = self.ad_gate * self.adapter(x_fn) + x

        prompt = x[:,:self.num_tokens]
        x = x[:, self.num_tokens:]
        B,G,_ = x.shape
        ##new_add
        G = G+self.num_tokens
        prompt_x = torch.cat((prompt,x), dim=1)
        ###

        x_neighborhoods = prompt_x.reshape(B*G, -1)[idx, :].reshape(B*center2.shape[1], group_size, -1)
        x_centers = prompt_x.reshape(B*G, -1)[center_idx, :].reshape(B, center2.shape[1], -1)
      
        std_xyz = torch.std(neighborhood)
        neighborhood = neighborhood / (std_xyz + 1e-5)
        x_neighborhoods = self.drop_path(attn1(norm3(x_neighborhoods.clone())))+x_neighborhoods.clone()

        vis_x = self.pooling(x_neighborhoods.reshape(B, center2.shape[1], group_size, -1), if_maxmean)+center_cof*x_centers#B,G1,C#NOTE: 0/0.3/0.5
  
        x = self.propagate(xyz1=center1, xyz2=center2, points1=x, points2=vis_x, de_neighbors=center2.shape[1], pro_cof=pro_cof)
        #x = x + self.adapter(self.drop_path(self.mlp(self.norm2(x))))
        x = torch.cat((prompt,x), 1)
        x = self.adapter1(x)    #
        # if if_third and layer_id==4:
        #     return x
        return x[:, self.num_tokens:]
        

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0,adapter_dim=1024., drop_rate_adapter=0, num_tokens=0., if_third=False, if_half=False, if_two=False, if_one=False):
        super().__init__()
        
        self.pos_drop = nn.Dropout(p=drop_rate)
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
            embed_dim, num_heads=num_heads, qkv_bias=False, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.if_third = if_third

    def forward(self, x, pos, vis_mask=None, center=None, center2=None, neighborhood=None, idx=None, center_idx=None, num_group=None, group_size=None, cache_prompt=None, if_maxmean=None, pro_cof=None, center_cof=None,layer_num=None):
        for layer_id, block in enumerate(self.blocks):
            x = block(x + pos, vis_mask, center,center2, neighborhood, idx, center_idx, num_group, group_size, cache_prompt=cache_prompt, if_maxmean=if_maxmean, pro_cof=pro_cof, center_cof=center_cof, if_third = self.if_third, attn1=self.attn1, norm3=self.norm3, layer_id=layer_num)
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

# finetune model
@MODELS.register_module()
class PointTransformer_best(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = config
        self.smooth = config.smooth
        self.trans_dim = 384 #config.transformer_config.trans_dim
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.depths = config.transformer_config.depths
        self.num_heads = config.transformer_config.num_heads 
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.cls_dim = config.cls_dim
        self.hd_s1 = config.hd_s1
        self.adapter_dim = config.adapter_config.adapter_dim
        self.drop_rate_adapter = config.adapter_config.adapter_drop_path_rate


        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        self.encoders = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()

        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.encoders.append(Encoder(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.encoders.append(Encoder(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
                
            self.pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.encoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
            ))
        self.masking_radius = [0.32, 0.64, 1.28]
        #self.masking_radius = [0.24, 0.48, 0.96]

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.prompt_cor = nn.Parameter(torch.zeros(16, 3))
        trunc_normal_(self.prompt_cor, std=.02) 
        self.train_images_features_agg = torch.load("./ckpts/train_f_pos_shape.pt")
        self.blocks = nn.ModuleList()
        depth = 0
        for i in range(len(self.depths)):
            if i==0:
                self.blocks.append(TransformerEncoder(
                    embed_dim = self.encoder_dims[i],
                    depth = self.depths[i],
                    drop_path_rate = dpr[depth: depth + self.depths[i]],
                    num_heads = self.num_heads,
                    adapter_dim=self.adapter_dim[i], 
                    drop_rate_adapter=self.drop_rate_adapter,
                    num_tokens=16,
                    #num_tokens=int(self.num_groups[i]//8),#NOTE: 100/ //4/ //8
                    if_third=False,
                    if_one=False,
                    if_two=False,
                    if_half = config.if_half
                ))
            elif i==1:
                self.blocks.append(TransformerEncoder(
                    embed_dim = self.encoder_dims[i],
                    depth = self.depths[i],
                    drop_path_rate = dpr[depth: depth + self.depths[i]],
                    num_heads = self.num_heads,
                    adapter_dim=self.adapter_dim[i], 
                    drop_rate_adapter=self.drop_rate_adapter,
                    num_tokens=16,
                    #num_tokens=int(self.num_groups[i]//8),#NOTE: 100/ //4/ //8
                    if_third=False,
                    if_one=False,
                    if_two=False,
                    if_half = config.if_half
                ))
            else:
                self.blocks.append(TransformerEncoder(
                    embed_dim = self.encoder_dims[i],
                    depth = self.depths[i],
                    drop_path_rate = dpr[depth: depth + self.depths[i]],
                    num_heads = self.num_heads,
                    adapter_dim=self.adapter_dim[i], 
                    drop_rate_adapter=self.drop_rate_adapter,
                    num_tokens=16,
                    #num_tokens=int(self.num_groups[i]//8),#NOTE: 100/ //4/ //8
                    if_third=True,
                    if_one=False,
                    if_two=False,
                    if_half = config.if_half,
                ))
            depth += self.depths[i]
        
        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),#NOTE:get this
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

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

            incompatible = self.load_state_dict(base_ckpt , strict=False)
            if incompatible.missing_keys:
                print_log('missing_keys', logger='Point_Mask_Rev')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Point_Mask_Rev'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Point_Mask_Rev')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Point_Mask_Rev'
                )

            print_log(f'[Point_Mask_Rev] Successful Loading the ckpt from {bert_ckpt_path}', logger='Point_Mask_Rev')
        else:
            print_log('Training from scratch!!!', logger='Point_Mask_Rev')
            self.apply(self._init_weights)

    
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        if self.smooth == 0.:
            loss = self.loss_ce(ret, gt.long())
        else:
            loss = self.smooth_loss(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def smooth_loss(self, pred, gt):
        eps = self.smooth
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

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

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, pts, cache=False, cp_feat=None, args=None):

        neighborhoods, centers, idxs, center_idxs = [], [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx, center_idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx, center_idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # b*g*k
            center_idxs.append(center_idx)
        #new_add
        prompt_cor = self.prompt_cor.repeat(pts.shape[0], 1, 1)
        prompt_pts = torch.cat((prompt_cor, pts), dim=1)
        prompt_neighborhoods, prompt_centers, prompt_idxs, prompt_center_idxs = [], [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx, center_idx = self.group_dividers[i](prompt_pts)
            else:
                neighborhood, center, idx, center_idx = self.group_dividers[i](center)
            prompt_neighborhoods.append(neighborhood)
            prompt_centers.append(center)
            prompt_idxs.append(idx)  # b*g*k
            prompt_center_idxs.append(center_idx)
        ####
        xyz_dist = None
        for i in range(len(centers)):
            if i == 0:
                x_vis = self.encoders[i](neighborhoods[0])  # B G C
            else:
                # print(x_vis.shape)
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                # print(neighborhoods[i].shape)
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                # print(x_vis_neighborhoods.shape)
                x_vis = self.encoders[i](x_vis_neighborhoods)  # B G C

            if self.masking_radius[i] > 0:
                mask_radius, xyz_dist = self.compute_mask(centers[i], self.masking_radius[i], xyz_dist)
                mask_vis_att = mask_radius

            else:
                mask_vis_att = None

            pos = self.pos_embeds[i](centers[i])

         
            if i+1<=len(prompt_centers)-1:
                x_vis = self.blocks[i](x_vis, pos, mask_vis_att, prompt_centers[i],center2=prompt_centers[i+1] ,neighborhood=prompt_neighborhoods[i+1], idx=prompt_idxs[i+1],center_idx=prompt_center_idxs[i+1], group_size=self.group_sizes[i+1], cache_prompt=None, if_maxmean=args.if_maxmean, pro_cof=args.propagate_cof, center_cof=args.center_cof, layer_num=i)
            elif i==len(prompt_centers)-1:
            # else:
                self.group = Group(num_group=int(self.num_groups[i]/2), group_size=self.group_sizes[i])
                neighborhood_prompt, center_new_prompt, idx_prompt, center_idx_prompt = self.group(prompt_centers[i])
                ####cp_feat
                K = prompt_cor.shape[1] - 2 # kre
                cp_feat_norm = cp_feat[i] / cp_feat[i].norm(dim=-1, keepdim=True) #[B, 384]
                new_knowledge = cp_feat_norm @ self.train_images_features_agg #[B, 11392]
                new_knowledge_k, idx_k = torch.topk(new_knowledge, K) #[B, 2*K]
                new_knowledge_k = F.softmax(new_knowledge_k, dim=1).unsqueeze(1)
                train_features_k = []
                for p in range(idx_k.shape[0]):
                    train_features_k.append(self.train_images_features_agg[:, idx_k[p]].tolist())
                train_features_k = torch.tensor(train_features_k).permute(0, 2, 1).cuda() #[B, K, 384]
                feat_f = torch.matmul(new_knowledge_k, train_features_k) #[B, 1, 384]
                cache_prompt = torch.cat((cp_feat[i].unsqueeze(1), feat_f, train_features_k), 1)
                ######
                x_vis = self.blocks[i](x_vis, pos, mask_vis_att, prompt_centers[i],center2=center_new_prompt ,neighborhood=neighborhood_prompt, idx=idx_prompt,center_idx=center_idx_prompt, group_size=self.group_sizes[i],cache_prompt=cache_prompt,  if_maxmean=args.if_maxmean, pro_cof=args.propagate_cof, center_cof=args.center_cof) #cp_feat=cp_feat[i],  if_maxmean=args.if_maxmean, pro_cof=args.propagate_cof, center_cof=args.center_cof)

            ####
            
           # x_vis = self.blocks[i](x_vis, pos, mask_vis_att)
            
        x_vis = self.norm(x_vis)

        # concat_f = x_vis.mean(1) + x_vis.max(1)[0]
        concat_f = torch.cat((x_vis.mean(1), x_vis.max(1)[0]), dim=1)
        if cache == True:
            for i in range(len(self.cls_head_finetune) - 1):
                concat_f = self.cls_head_finetune[i](concat_f)
            return concat_f
        ret = self.cls_head_finetune(concat_f)

        return ret
