import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    project_name: str = "eoekoland"
    run_name: str

    max_epochs: int = 10
    batch_size: int = 8
    num_workers: int = 16
    num_classes: int = 15
    lr: float = 1e-04
    loss: str = "CE" # "CE-W" = Cross entropy with class weights, "CE" = cross entropy without class weights, "F" = Focal loss without class weights
    factor: int = 1

    backbone: str  # "C3D" = three dimensional CNN, U-TAE, tsvit, tempcnn
    modalities: str  # "S2", "S1", "S"=S1+S2, "P"=Planet, "ALL"=Planet+S1+S2, "PS2"=Planet+S2
    
    data_path: str = "/mnt/storagecube/data/projects/eoekoland/data/CT_CODE_condensed_patches"
    checkpoint: str = None
    gradient_accumulation: int = 1

    patchsize: int # rescale to this size

    # C3D
    c3d_k_size: int = 3
    c3d_temp_k_size: int = 3
    
    #tempcnn
    tempcnn_temp_k_size: int = 7
    
    # utae
    utae_encoder_widths: list = [64,64,64,128] # less layers, less filters
    utae_decoder_widths: list = [32,32,64,128]
    utae_n_head: int = 16 
    utae_d_model: int = 256 
    utae_str_conv_k: int = 4
    utae_str_conv_p: int = 1
    utae_str_conv_s: int = 2
    utae_out_conv: list = [32,15] 

    #TSViT
    tsvit_patch_size: int = 2
    tsvit_patch_size_time: int = 1
    tsvit_patch_time: int = 4
    tsvit_dim: int = 128 
    tsvit_temporal_depth: int = 6
    tsvit_spatial_depth: int = 2
    tsvit_channel_depth: int = 4
    tsvit_heads: int = 4
    tsvit_pool: str = 'cls'
    tsvit_dim_head: int = 64
    tsvit_dropout: float = 0.
    tsvit_emb_dropout: float = 0.
    tsvit_scale_dim: int = 4
    tsvit_depth: int = 4
        

    @classmethod
    def parse_yaml_raw(cls, config_path: Path, args) -> "Config":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # backbone can also be specified via terminal input
        if args["backbone"] is not None:
            config["backbone"] = args["backbone"]
        if args["batchsize"] is not None:
            config["batch_size"] = args["batchsize"]
        if args["factor"] is not None:
            config["factor"] = args["factor"]

        config = cls.parse_obj(config)
        return config
    
    @classmethod
    def parse_obj(cls, config: dict) -> "Config":
        if config["modalities"] == "S2" or config["modalities"] == "S1":
            config["patchsize"] = 24
            
        else: # planet or fusion involving planet
            config["patchsize"] = 80
            
        config = super(Config, cls).parse_obj(config)
        return config

    def dict(self, **kwargs) -> dict:
        out_dict = super(Config, self).dict()
        return out_dict

    def save_dict(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.dict(), f)
