import sys
sys.path.append("./models")

from simple_3dcnn import C3D
from utae import UTAE
from tempCNN import TempCNN
from TSViT import TSViT
from TSViT_cross_attn import TSViT as TSViT_cross_attn
from TSViT_synchr_class_token import TSViT as TSViT_scf
import torch
import numpy as np

def get_channels(modalities: str):
    """
    Number of channels for each modality 
    options: "S2", "S1", "S"=S1+S2, "P5"=Planet5, "P"=Planet, "ALL"=Planet+S1+S2, "ALL_5"=Planet5+S1+S2, "PS2"=Planet+S2
    """
    S1_channels, S2_channels, P_channels = 0, 0, 0
    if modalities == "S2":
        S2_channels = 10
    elif modalities == "S1":
        S1_channels = 2
    elif modalities == "S":
        S2_channels = 10
        S1_channels = 2
    elif modalities == "PS2":
        S2_channels = 10
        P_channels = 4
    elif modalities == "P":
        P_channels = 4
    elif modalities == "ALL":
        S2_channels = 10
        S1_channels = 2
        P_channels = 4
    return S1_channels, S2_channels, P_channels


def get_bands(batch, modalities):
    """
    returns modalities as one tensor 
    """
    timepoints = batch["timepoints"]
    if modalities == "S2":
        inputs = batch["s2_bands"]
        labels = batch["s_labels"]
    elif modalities == "S1":
        inputs = batch["s1_bands"]
        labels = batch["s_labels"]
    elif modalities == "P":
        inputs = batch["p_bands"]
        labels = batch["p_labels"]
    elif modalities == "S":
        inputs = torch.cat((batch["s1_bands"], batch["s2_bands"]), 2) # concatenate channels -> 3nd dimension 
        labels = batch["s_labels"]
    elif modalities == "PS1":
        inputs = torch.cat((batch["s1_bands"], batch["p_bands"]), 2) # concatenate channels -> 3nd dimension   
        labels = batch["p_labels"]  
    elif modalities == "PS2":
        inputs = torch.cat((batch["s2_bands"], batch["p_bands"]), 2) # concatenate channels -> 3nd dimension   
        labels = batch["p_labels"]  
    elif modalities == "ALL" or modalities == "ALL_5":
        inputs = torch.cat((batch["s1_bands"], batch["s2_bands"], batch["p_bands"]), 2) # concatenate channels -> 3nd dimension 
        labels = batch["p_labels"]
    #print("input shape after fusion:", inputs.shape)
    # add time channel
    B, T, C, H, W = inputs.shape
    time_channel = timepoints.repeat(H,W,1,1).permute(2,3,0,1) # BxTxHxW
    inputs = torch.cat((torch.as_tensor(inputs), time_channel[:,:,None,:,:]), dim=2) # BxTxCxHxW + BxTxCxHxW = BxTx(C+1)xHxW
    # B x T x C x H x W
    return inputs, labels
    
def load_model(config, timesteps=37, img_size=80):
    """
    config: config object with settings
    timesteps: number of timesteps if specific dates have been chosen before. None otherwise, then all available timesteps will be used
    """
    backbone = config.backbone
    S1_channels, S2_channels, P_channels = get_channels(config.modalities)
    
    num_classes = config.num_classes
    model = None

    bands = S1_channels+S2_channels+P_channels
    if backbone=="C3D":
        # basic early fusion model with a 3DCNN backbone (S1_t == S2_t == P_t)
        kernel_size = config.c3d_k_size
        temp_k_size = config.c3d_temp_k_size
        print(f"model configuration: {bands} bands, {timesteps} timesteps, kernel size = ({temp_k_size, kernel_size, kernel_size})")
        model = C3D(bands=bands, labels=num_classes, temp_k_size=temp_k_size, k_size=kernel_size, timepoints=timesteps)
    elif backbone=="U-TAE":
        print(f"model configuration: {bands} bands, {timesteps} timesteps, encoder widths={config.utae_encoder_widths}, decoder widths={config.utae_decoder_widths}, n_heads={config.utae_n_head}")
        model = UTAE(bands, encoder_widths=config.utae_encoder_widths, decoder_widths=config.utae_decoder_widths, timesteps=timesteps, n_head=config.utae_n_head, str_conv_k=config.utae_str_conv_k, str_conv_p=config.utae_str_conv_p, d_model=config.utae_d_model, positional_encoding="sinus", out_conv=config.utae_out_conv)
    elif backbone=="tempcnn":
        print(f"model configuration: {bands} input dim, {timesteps} sequencelength")
        model = TempCNN(input_dim=bands, kernel_size=config.tempcnn_temp_k_size, num_classes=num_classes, sequencelength=timesteps)
    elif backbone=="TSViT":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength")
        print(model_config)
        model = TSViT(model_config=model_config, img_res=img_size, num_channels=[bands], num_classes=num_classes) 
    elif backbone=="TSViT_mc":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength, {S1_channels}, {S2_channels}, {P_channels}")
        print(model_config)
        model = TSViT(model_config=model_config, img_res=img_size, num_channels=[S1_channels, S2_channels, P_channels], num_classes=num_classes, patch_embedding="Modality Concatenation") 
    elif backbone=="TSViT_scf":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength, {S1_channels}, {S2_channels}, {P_channels}")
        print(model_config)
        model = TSViT_scf(model_config=model_config, img_res=img_size, num_channels=[S1_channels, S2_channels, P_channels], num_classes=num_classes) 
    elif backbone=="TSViT_channel_enc":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength, {S1_channels}, {S2_channels}, {P_channels}")
        print(model_config)
        model = TSViT(model_config=model_config, img_res=img_size, num_channels=[S1_channels, S2_channels, P_channels], num_classes=num_classes, patch_embedding="Channel Encoding") 
    elif backbone=="TSViT_cross_attn":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength, {S1_channels}, {S2_channels}, {P_channels}")
        print(model_config)
        model = TSViT_cross_attn(model_config=model_config, img_res=img_size, num_channels=[S1_channels, S2_channels, P_channels], num_classes=num_classes) 
    elif backbone=="TSViT_ensemble":
        model_config = {'patch_size': config.tsvit_patch_size, 'patch_size_time': config.tsvit_patch_size_time, 'patch_time': config.tsvit_patch_time,
                    'dim': config.tsvit_dim, 'temporal_depth': config.tsvit_temporal_depth, 'spatial_depth': config.tsvit_spatial_depth, 'channel_depth': config.tsvit_channel_depth,
                    'heads': config.tsvit_heads, 'pool': config.tsvit_pool, 'dim_head': config.tsvit_dim_head, 'dropout': config.tsvit_dropout, 'emb_dropout': config.tsvit_emb_dropout,
                    'scale_dim': config.tsvit_scale_dim, 'depth': config.tsvit_depth}
        print(f"model configuration: {bands} num_channels, {img_size}x{img_size} image size, {num_classes} classes, {timesteps} sequencelength")
        print(model_config)
        ensemble_size = 3
        model = [TSViT(model_config=model_config, img_res=img_size, num_channels=[bands], num_classes=num_classes) for i in range(ensemble_size)]
    else:
        raise ValueError(f"model type {backbone} is not supported. Use \"tempcnn\", \"C3D\", \"TSViT\", \"TSViT_scf\", \"TSViT_cross_attn\", \"TSViT_channel_enc\", \"TSViT_mc\" or \"U-TAE\" as backbone")
    return model

if __name__ == '__main__':
    x = np.random.rand(52)