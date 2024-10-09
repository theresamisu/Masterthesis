"""
U-TAE Implementation
Adapted from: https://github.com/VSainteuf/lightweight-temporal-attention-pytorch
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn
import numpy as np

from ltae import LTAE2d



class UTAE(nn.Module):
    def __init__(self,
                 input_dim,
                 encoder_widths=[64, 64, 64, 128],
                 decoder_widths=[32, 32, 64, 128],
                 out_conv=[32, 15],
                 str_conv_k=4,
                 str_conv_s=2,
                 str_conv_p=1,
                 agg_mode="att_group",
                 encoder_norm="group",
                 n_head=16,
                 d_model=256,
                 d_k=4,
                 encoder=False,
                 return_maps=False,
                 pad_value=0,
                 padding_mode="reflect",
                 timesteps=37,
                 positional_encoding="sinus"
                 ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UTAE, self).__init__()
        self.name="UTAE"
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (decoder_widths[0]
                        if decoder_widths is not None else encoder_widths[0])
        self.stack_dim = (sum(decoder_widths) if decoder_widths is not None
                          else sum(encoder_widths))
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            ) for i in range(self.n_stages - 1))
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            ) for i in range(self.n_stages - 1, 0, -1))
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
            positional_encoding=positional_encoding,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv,
                                  padding_mode=padding_mode)
        self.timerange = torch.tensor(range(timesteps))

    def forward(self,
                input,
                batch_positions=None,
                return_att=False,
                encoder_only=False):
        # input shape BxTxCxHxW
        # print("input:", torch.cuda.memory_allocated(device=input.device)* 1e-05, "Mb")
        
        # discard last channel that contains timestamp
        input = torch.clone(input[:,:,:-1,:,:])
        # print("input", input.shape)

        # inconv: change channel dimension from bands to input dim of first encoder (usually 64)
        # 2 layers: 1. bands -> enc_dim. 2. enc_dim -> end_dim
        # temporally shared block (in parallel for all time points)
        out = self.in_conv.smart_forward(input)
        # print("in_conf", out.shape)
        
        # remember feature maps for shortcut connections to upsampling later
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            # print("sp_enc", i, out.shape)
            feature_maps.append(out)

        if encoder_only:
            return feature_maps

        # TEMPORAL ENCODER
        batch_positions = self.timerange.unsqueeze(0).repeat(
                input.shape[0], 1).to(input.device) # b x T
        # out = d^L = f^L
        out, att = self.temporal_encoder(feature_maps[-1],
                                         batch_positions=batch_positions)
        
        # print("tmp_enc", out.shape)
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            # skip = f^l
            skip = self.temporal_aggregator(feature_maps[-(i + 2)],
                                            attn_mask=att)
            # out = d^l+1, skip = f^l
            out = self.up_blocks[i](out, skip)
            # print("sp_dec", i, out.shape)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            #print("out_conv", out.shape)
            if return_att:
                return out, att
            if self.return_maps:
                return maps[-1]
            else:
                return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(
                    dim=-1)
                if pad_mask.any():
                    temp = (torch.ones(self.out_shape,
                                       device=input.device,
                                       requires_grad=False) * self.pad_value)
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):

    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):  
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            # print("in: ", nkernels[i], "out:", nkernels[i+1])
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                ))
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):

    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):

    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):

    def __init__(self,
                 d_in,
                 d_out,
                 k,
                 s,
                 p,
                 norm="batch",
                 d_skip=None,
                 padding_mode="reflect"):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip # encoder_l
        # 1x1 convolution applied to skip connection = f^l
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_in,
                               out_channels=d_out,
                               kernel_size=k,
                               stride=s,
                               padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        #print("conv1 up block")
        # concatenation d_out + d (skip connection + up(d^l+1))
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], # 128->64
                               norm=norm,
                               padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], # 64->64
                               norm=norm,
                               padding_mode=padding_mode)

    def forward(self, input, skip):
        # upsample input d^l+1 to spatial dimension of e^l * a^l (skip)
        out = self.up(input)
        # apply 1x1 convolution to get f^l and concatenate with channels of d^l+1  
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        # D_l convolutions to output
        out = self.conv1(out)
        # why adding out? (where is this in the paper?)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    """
    multiplies upsampled attention mask with feature map e^l of spatial encoder at level l
    sums over temporal dimension
    input 
    x = BxTxCxHxW
    att = hxBxTxHxW
    returns f^l with shape C_l x W_l x H_l
    """
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:],
                                       mode="bilinear",
                                       align_corners=False)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(size=x.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False)(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None,
                                                              None]
                return out
        else:
            if self.mode == "att_group":
                # x = BxTxCxHxW
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w) # (h*B)xTxHxW
                # resize attention mask to size of input x
                if x.shape[-2] > w:
                    attn = nn.Upsample(size=x.shape[-2:],
                                       mode="bilinear",
                                       align_corners=False)(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                
                # group channels by heads -> C/h channels per head h 
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTx(C/h)xHxW
                
                # print("memory:", torch.cuda.memory_allocated(out.device)* 1e-05, "Mb", out.shape, attn.shape)
                out = attn[:, :, :, None, :, :] * out # broadcast over channels hxBxTx1xHxW
                
                out = out.sum(dim=2)  # sum on temporal dim -> hxBx(C/h)xHxW
                out = torch.cat([group for group in out], dim=1)  # concatenate channel groups again -> BxCxHxW
                # where is 1x1 convolution???
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW (no channel grouping anymore)
                # upsample attention to spatial resolution of input x (=feature map e^l)
                attn = nn.Upsample(size=x.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False)(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1) # broadcast attn over channel dimension, sum over temporal dimension
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


if __name__ == "__main__":
    x = torch.ones((1, 37, 11, 80, 80), dtype=torch.float32).to("cuda")
    model = UTAE(10, out_conv=[32, 10], d_model=128, timesteps=37).to("cuda")
    # summary(model, (37, 11, 80, 80), batch_size=4, device="cuda") # bcthw
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    #for name, p in model.named_parameters():
    #    print(name, "\t", p.size(), np.prod(p.size()))
    print('Trainable Parameters: %.3fM' % parameters)
    print("\ninput", x.shape, "\n")

    model(x)