import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 num_heads, qkv_bias, 
                 qk_scale, attn_drop, 
                 proj_drop,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper, self).__init__()
    
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims

        self.dropout = nn.Dropout(p=drop_rate_adapter)
        self.identity = False

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)

            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)


    def set_sample_config(self, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
            self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]

            self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
            self.sampled_bias_1 =  self.ln2.bias


    def forward(self, x, identity=None):
        if self.identity:
            return x

        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)

        if identity is None:
            identity = x
        return identity + out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops

class AdapterSuper_f(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 num_heads, qkv_bias, 
                 qk_scale, attn_drop, 
                 proj_drop,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper_f, self).__init__()
    
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims

        self.dropout = nn.Dropout(p=drop_rate_adapter)
        self.identity = False

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)

            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)


    def set_sample_config(self, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            # import pdb;pdb.set_trace()
            self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
            self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]

            self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
            self.sampled_bias_1 =  self.ln2.bias


    def forward(self, x, identity=None):
        if self.identity:
            return x
        
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)

        if identity is None:
            identity = x
        return out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops