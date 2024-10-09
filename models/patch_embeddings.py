import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

class StandardPatchEmbedding(nn.Module):
    def __init__(self, inner_dim, num_channels: list, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
        patch_dim = sum(num_channels) * self.patch_size ** 2  
        self.linear_layer = nn.Linear(patch_dim, inner_dim)
    
    def forward(self, x):
        x = rearrange(x, 'b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.linear_layer(x)
        # B*N_H*N_W x T x d
        # print("Standard patch embedding", x.shape)
        return x

class ModalityConcatenation(nn.Module):
    def __init__(self, inner_dim, num_channels: list, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_modalities = len(num_channels)
        
        self.patch_embed = nn.ModuleList(
            StandardPatchEmbedding(inner_dim, [i], patch_size)
            for i in self.num_channels
        )
        
        #self.norm_layer = nn.LayerNorm(inner_dim)
        
    def forward(self, x):
        # split by modality
        modality_input = []

        channel_idx = self.num_channels # list of number of channels with 0 prepended
        for i in range(self.num_modalities):
            x1 = x[:,:, sum(channel_idx[:i]) : sum(channel_idx[:i+1])]
            modality_input += [self.patch_embed[i](x1)]
        
        # concatenate modality tokens in temporal dimension
        x = torch.cat(modality_input, dim=1)
        #x = self.norm_layer(x)
        return x
    


class MultiModalityEmbedding(nn.Module):
    def __init__(self, inner_dim, num_channels: list, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_modalities = len(num_channels)
        
        # one separate patch embedding for each modality
        self.patch_embed = nn.ModuleList(
            StandardPatchEmbedding(inner_dim, [i], patch_size)
            for i in self.num_channels
        )
        
    def forward(self, x):
        # split by modality
        modality_input = []

        channel_idx = self.num_channels # list of number of channels 
        for i in range(self.num_modalities):
            x1 = x[:,:, sum(channel_idx[:i]) : sum(channel_idx[:i+1])]
            modality_input += [self.patch_embed[i](x1)]
        
        # concatenate modality tokens in time dimension: one token time series for each modality
        x = modality_input # 3 x B*H*W x T x d
        #x = torch.stack(modality_input, axis=1)
        
        return x
