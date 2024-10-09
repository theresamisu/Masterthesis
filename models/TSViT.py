"""
TSViT Implementation
Adapted from: https://github.com/michaeltrs/DeepSatModels
Authors: Michail Tarasiou and Erik Chavez and Stefanos Zafeiriou
License: Apache License 2.0
"""
#TODO license
import sys
sys.path.append("..")
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pytorch_lightning as pl
from patch_embeddings import StandardPatchEmbedding, ModalityConcatenation

class ChannelEncoder(nn.Module):
    def __init__(self, inner_dim, num_channels: int, channel_depth, heads, dim_head, scale_dim, dropout):
        super().__init__()
        # enumerate channels
        self.pos_encoding = nn.Parameter(torch.randn(1, num_channels, inner_dim))
        # channel encoder
        self.channel_transformer = Transformer(inner_dim, channel_depth, heads, dim_head,
                                                inner_dim * scale_dim, dropout)
        
    def forward(self, x):
        # input: B x C x d
        x += self.pos_encoding
        x = self.channel_transformer(x)
        x = torch.mean(x, axis=1)
        return x
        
class ChannelEncoding(nn.Module):
    def __init__(self, inner_dim, num_channels: list, patch_size, channel_depth, heads, dim_head, scale_dim, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels

        # one linear transformation per channel
        patch_dim = self.patch_size ** 2  
        self.linear_layer = nn.Linear(patch_dim, inner_dim)

        self.channel_encoder = ChannelEncoder(inner_dim, sum(num_channels), channel_depth, heads, dim_head, scale_dim, dropout)

    def forward(self, x):
        B, T, C, H, W = x.shape # batch size, number of time steps, (total) number of channels, total height and width of image (e.g. 80x80)
        # group time points together -> work on all time points simultaneously
        # tokenize (cut into patches and project to d) each channel separately
        x = rearrange(x, 'b t c (h p1) (w p2) -> (b h w t) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.linear_layer(x)
        
        h, w = x.shape[-2:] 
        # process each token series 
        x = self.channel_encoder(x) # returns only class token: H*W*T x 1 x d
        # turn back into time series
        x = rearrange(x, ' (b h w t) d -> (b h w) t d', b=B, t=T, h=h)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x = b x n x d (n=(T+K) temporal or n=(h*w) (spatial) token sequence length)
        b, n, _, h = *x.shape, self.heads
        # tuple of 3 tensors with each bxnx(heads*dim_head) dimension
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split by heads -> b x head x n x dim_head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # attention between each of the n tokens: nxn. per head: hxnxn
        attn = dots.softmax(dim=-1)
        
        # weightening the token values V (hxnxd) with a (hxnxn) -> out = hxnxd 
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # reshape back to nx head*dim_head
        out = rearrange(out, 'b h n d -> b n (h d)')
        # project from inner dimension (head*dim_head) back to d
        out = self.to_out(out)

        return out

class TSViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config, img_res=80,  num_channels=[4], num_classes=16, max_seq_len=37, patch_embedding="Standard"): # model config must contain: patch_size=2, d=128, temporal_depth=6, spatial_depth=2, n_heads=4, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4
        """
        num_channels list
        """
        super().__init__()
        self.name="TSViT_"+patch_embedding
        
        self.patch_size = model_config['patch_size'] # p1=p2
        self.num_patches_1d = img_res // (self.patch_size) # h=w
        self.num_classes = num_classes # K
        self.dim = model_config['dim'] # d

        # layer number of temporal encoder
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth'] 
        else:
            self.temporal_depth = model_config['depth'] 
        # layer number of spatial encoder
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.channel_depth = model_config['channel_depth']

        # transformer encoder parameters
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.scale_dim = model_config['scale_dim']
        
        
        self.patch_embedding = patch_embedding
        self.num_modalities = len(num_channels)
        print(patch_embedding)
        if patch_embedding == "Standard":
            self.to_patch_embedding = StandardPatchEmbedding(self.dim, num_channels, self.patch_size)
        elif patch_embedding == "Modality Concatenation":
            self.to_patch_embedding = ModalityConcatenation(self.dim, num_channels, self.patch_size)
        elif patch_embedding == "Channel Encoding":
            self.to_patch_embedding = ChannelEncoding(self.dim, num_channels, self.patch_size, self.channel_depth, self.heads, self.dim_head, self.scale_dim, self.dropout)
        
        # temporal position encoding: project temp position one hot encoding [0,365] to d, this is then added to the tokens
        # learn d-dim vector for each time point
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        # temporal class token 1xKxd
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        
        # spatial position embedding: 1x(h*w)xd
        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches_1d ** 2, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)

        self.dropout = nn.Dropout(self.emb_dropout)
        # project back from d to p1xp2 2 dimensional patch 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )


    def forward(self, x):
        B, T, _, H, W = x.shape # B=batch size, T=temporal dimension, C=channel dimension, H,W=spatial dimensions
        
        xt = x[:, :, -1, 0, 0] # BxTx1 = time step: last band in channel dimension encodes the timestep as value in [0,365], e.g. if image has 10 spectral bands, there is an 11th band with only the timestep value everywhere
        x = x[:, :, :-1] # BxTxCxHxW C=only bands
        
        # 1. PATCH EMBEDDING
        # reshape from BxTxCxHxW -> 3 dimensional (B*h*w)xTx(p1*p2*C)
        # h,w is the number of e.g 3x3 patches in vertical and horizontal dimension. 
        # B*h*w means that we are operating over all patches with size Txp1xp2xC simultaneously
        # p1xp2 is the size of the patch (3x3) with C channels, these dimensions are combined bc the patch is then embedded to embedding dimension d
        # by linear projection (p1xp2xC patch) -> d
        # in total BxTxCxHxW -> (B*h*w)xTx(p1*p2*C) -> (B*h*w)xTxd
        x = self.to_patch_embedding(x)
            
        # 2. TEMPORAL ENCODING 
        xt = xt.to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32) # make into one hot encoding BxTx366
        xt = xt.reshape(-1, 366) # (B*T)x365
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim) 
        if self.patch_embedding == "Modality Concatenation":
            temporal_pos_embedding = temporal_pos_embedding.repeat(1,self.num_modalities,1) # repeat for each modality
        
        # add temporal position encoding
        # to Bx(h*w)xTxd = split again batch and spatial dimension (because images within one batch might have different time points)
        num_temporal_tokens = temporal_pos_embedding.shape[1]
        x = x.reshape(B, -1, num_temporal_tokens, self.dim)

        # add temporal position enc to each patch by broadcasting along patch-dimension -> Bx(h*w)xTxd + Bx1xTxd = Bx(h*w)xTxd
        x += temporal_pos_embedding.unsqueeze(1)
        # back to (B*h*w)xTxd token sequence
        x = x.reshape(-1, num_temporal_tokens, self.dim)        
        
        cls_temporal_tokens = repeat(self.temporal_token, '() K d -> b K d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        
        # 3. SPATIAL ENCODER
        # to prepare for spatial encoder reshape to (B*K)x(w*h)xd
        # in detail: (B*h*w)xKxd -> Bx(h*w)xKxd -> BxKx(h*w)xd -> (B*K)x(h*w)xd
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        # add spatial position encoding 1x(h*w)xd (broadcast along class dimension): (B*K)x(h*w)xd + (1x(h*w)xd)
        x += self.space_pos_embedding

        # apply spatial encoder (do not add image class token because we want to do only semantic segmentation)
        x = self.space_transformer(self.dropout(x))
        
        # 4. TO PIXEL PROBABILITIES
        # project back to spatial dimensions (d -> p1xp2)
        # reshape to (B*K*h*w)xd and project to (B*K*h*w)xp1xp2. This produces the class probabilities (K) for each pixel (p1xp2) in each of the (h*w) patches for each image in batch B
        x = rearrange(x, '(b k) (h w) d-> (b k h w) d', h=self.num_patches_1d, k=self.num_classes, w=self.num_patches_1d)
        x = self.mlp_head(x)
        
        # assemble original image extent HxW from amount of patches (hxw) and patch size (p1xp2)
        x = rearrange(x, '(b k h w) (p1 p2) -> b k (h p1) (w p2)', h=self.num_patches_1d, k=self.num_classes, w=self.num_patches_1d, p1=self.patch_size)
        return x
    
import os
from utils.gpu_mars_dict import gpu_mapping
if __name__ == "__main__":
    pl.seed_everything(35)
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu_mapping["0"]

    res = 80 # has to be divisible by patch_size
    max_seq_len = 37 # has to be divisible by patch_size_time
    channels = [2,10,4]
    num_classes = 16
    batch_size = 1
    patch_size = 2
    print(f"creating {(res/patch_size)} * {(res/patch_size)} = {(res/patch_size)**2} patches of size {patch_size} x {patch_size}")

    x = torch.rand((batch_size, max_seq_len, res, res, sum(channels))) # BxTxHxWxC
    B,T,H,W,C = x.shape

    # add channel that contains time steps
    time_points = torch.randint(low=0,high=365,size=(max_seq_len,))
    time_channel = time_points.repeat(B,H,W,1).permute(0,3,1,2) # BxTxHxW
    x = torch.cat((x, time_channel[:,:,:,:,None]), dim=4) # BxTxHxWxC + BxTxHxWx1 = BxTx(C+1)xHxW
    # last layer should contain only the value of the timestep for fixed T
    for t in range(T):
        assert int(np.unique(x[:,t,:,:,-1].numpy(), return_counts=True)[0][0]) == time_points.numpy()[t]
    
    model_config = {'patch_size': patch_size, 'patch_size_time': 1, 'patch_time': 4,
                    'dim': 128, 'temporal_depth': 6, 'spatial_depth': 2, 'channel_depth': 4,
                    'heads': 4, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}
    print(f"model configuration: {channels} input dim, {max_seq_len} sequencelength")
    model = TSViT(img_res=res, num_channels=channels, model_config=model_config, num_classes=num_classes, patch_embedding="Channel Encoding").to("cuda")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    x = x.permute(0,1,4,2,3).to("cuda")
    print("\ninput", x.shape, "\n")
    out = model(x)
    
    print("total memory usage", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
    
    print("Shape of out :", out.shape)  # [B, num_classes, H, W]
    print('Trainable Parameters: %.3fM' % parameters)
    
    