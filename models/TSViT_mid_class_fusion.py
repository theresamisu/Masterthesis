"""
TSViT Implementation
Adapted from: https://github.com/michaeltrs/DeepSatModels
Authors: Michail Tarasiou and Erik Chavez and Stefanos Zafeiriou
License: ??? (not specified in github)
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
from patch_embeddings import MultiModalityEmbedding

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
        # print(x.shape)
        for attn, ff in self.layers:
            # print("before attention\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
            x = attn(x) + x
            # print("after attention\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
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
        #print("attn", x.shape)
        b, n, _, h = *x.shape, self.heads
        # tuple of 3 tensors with each bxnx(heads*dim_head) dimension
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split by heads -> b x head x n x dim_head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        #print("Q^T K/ sqrt(d_head)",dots.shape)
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
        self.name="TSViT"
        
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
        self.to_patch_embedding = MultiModalityEmbedding(self.dim, num_channels, self.patch_size)
        
        # temporal position encoding: project temp position one hot encoding [0,365] to d, this is then added to the tokens
        # learn d-dim vector for each time point
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        # temporal class token Mx1xKxd
        self.temporal_token = nn.Parameter(torch.randn(self.num_modalities, 1, self.num_classes, self.dim))
        # temporal encoder
        self.temporal_transformers = nn.ModuleList( 
            Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
            for _ in range(self.num_modalities))
        
        # spatial position embedding: 1x(h*w)xd
        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches_1d ** 2, self.dim))
        # spatial encoder
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)

        self.dropout = nn.Dropout(self.emb_dropout)
        # project back from d to p1xp2 2 dimensional patch 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )


    def forward(self, x):
        # print("input memory usage\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        # print("x", x.shape, x.nelement() * x.element_size()* 1e-06, "Mb", x.dtype,  x.element_size()* x.nelement())
        B, T, _, H, W = x.shape # B=batch size, T=temporal dimension, C=channel dimension, H,W=spatial dimensions
        
        xt = x[:, :, -1, 0, 0] # BxTx1 = time step: last band in channel dimension encodes the timestep as value in [0,365], e.g. if image has 10 spectral bands, there is an 11th band with only the timestep value everywhere
        x = x[:, :, :-1] # BxTxCxHxW C=only spectral bands
        # print("input\t", x.shape,  torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        
        # 1. PATCH EMBEDDING
        # reshape from BxTxCxHxW -> 3 dimensional (B*h*w)xTx(p1*p2*C)
        # h,w is the number of e.g 3x3 patches in vertical and horizontal dimension. 
        # B*h*w means that we are operating over all patches with size Txp1xp2xC simultaneously
        # p1xp2 is the size of the patch (3x3) with C channels, these dimensions are combined bc the patch is then embedded to embedding dimension d
        # by linear projection (p1xp2xC patch) -> d
        # so in total BxTxCxHxW -> (B*h*w)xTx(p1*p2*C) -> (B*h*w)xTxd
        # print(x.shape)
        x = self.to_patch_embedding(x)
        # print("patch embedding\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
            
        # 2. TEMPORAL ENCODING 
        xt = xt.to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32) # make into one hot encoding BxTx366
        
        # print("temporal vector\t", torch.cuda.memory_allocated(xt.device)* 1e-06, "Mb")
        xt = xt.reshape(-1, 366) # (B*T)x366
        # P_T[t,:] = linear projection from 366-dim one-hot-encoding to d (embedding dimension). reshape back to BxTxd
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim) 
        # add temporal position encoding to each modality
        for mod_idx, x_mod in enumerate(x):
            # to Bx(h*w)xTxd = split again batch and list of patches
            x_mod = x_mod.reshape(B, -1, T, self.dim)

            # add temporal position encoding to each patch by broadcasting along patch-dimension -> Bx(h*w)xTxd + Bx1xTxd = Bx(h*w)xTxd
            x_mod += temporal_pos_embedding.unsqueeze(1)
            # back to (B*h*w)xTxd
            x_mod = x_mod.reshape(-1, T, self.dim)        
            
            # create class tokens (h*w)xKxd and broadcast along b=B*h*w
            cls_temporal_tokens = repeat(self.temporal_token[mod_idx], '() K d -> b K d', b=B * self.num_patches_1d ** 2)
            # prepend to feature tokens in Temporal dimension -> (B*h*w)x(K+T)xd
            x_mod = torch.cat((cls_temporal_tokens, x_mod), dim=1)
            # print("class token added\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
            # apply temporal transformer encoder
            x_mod = self.temporal_transformers[mod_idx](x_mod)
            # keep only the class tokens -> (B*h*w)xKxd
            x[mod_idx] = x_mod[:, :self.num_classes]

        # stack class tokens for each modality -> Mx(B*h*w)xKxd
        x = torch.stack(x)
        # take mean over modalities -> (B*h*w)xKxd
        x = torch.mean(x, axis=0)

        # 3. SPATIAL ENCODER
        # to prepare for spatial encoder reshape to (B*K)x(w*h)xd
        # in detail: (B*h*w)xKxd -> Bx(h*w)xKxd -> BxKx(h*w)xd -> (B*K)x(h*w)xd
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        # add spatial position encoding 1x(h*w)xd (broadcast along class dimension): (B*K)x(h*w)xd + (1x(h*w)xd)
        x += self.space_pos_embedding

        # print("pre spatial transformer\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        # print("spatial token series size", x.shape, x.nelement() * x.element_size()* 1e-06, "Mb", x.dtype,  x.element_size(), x.nelement())
        # apply spatial encoder (do not add image class token because we want to do only semantic segmentation)
        x = self.space_transformer(self.dropout(x))
        # print("after spatial transformer\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        
        # 4. TO PIXEL PROBABILITIES
        # project back to spatial dimensions (d -> p1xp2)
        # reshape to (B*K*h*w)xd and project to (B*K*h*w)xp1xp2. This produces the class probabilities (K) for each pixel (p1xp2) in each of the (h*w) patches for each image in batch B
        x = rearrange(x, '(b k) (h w) d-> (b k h w) d', h=self.num_patches_1d, k=self.num_classes, w=self.num_patches_1d)
        #x = x.reshape(-1, self.dim)
        x = self.mlp_head(x)
        #print("classification mlp\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        
        # assemble original image extent HxW from amount of patches (hxw) and patch size (p1xp2)
        x = rearrange(x, '(b k h w) (p1 p2) -> b k (h p1) (w p2)', h=self.num_patches_1d, k=self.num_classes, w=self.num_patches_1d, p1=self.patch_size)
        #print("final\t", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
        return x

    
import os
from utils.gpu_mars_dict import gpu_mapping
if __name__ == "__main__":
    pl.seed_everything(35)
    #os.environ["CUDA_VISIBLE_DEVICES"]= gpu_mapping["1"]

    res = 80 # has to be divisible by patch_size
    max_seq_len = 37 # has to be divisible by patch_size_time
    channels = [2,10,4]
    num_classes = 14
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
                    'dim': 128, 'temporal_depth': 6, 'spatial_depth': 2, 'channel_depth': 2,
                    'heads': 4, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}
    print(f"model configuration: {channels} input dim, {max_seq_len} sequencelength")
    model = TSViT(img_res=res, num_channels=channels, model_config=model_config, num_classes=num_classes, patch_embedding="Modality Concatenation").to("cuda:1")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    x = x.permute(0,1,4,2,3).to("cuda:1")
    print("\ninput", x.shape, "\n")
    out = model(x)
    
    print("total memory usage", torch.cuda.memory_allocated(x.device)* 1e-06, "Mb")
    
    print("Shape of out :", out.shape)  # [B, num_classes, H, W]
    print('Trainable Parameters: %.3fM' % parameters)
    