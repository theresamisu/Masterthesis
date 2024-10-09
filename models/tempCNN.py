import os

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from einops import rearrange

"""
Pytorch re-implementation of Pelletier et al. 2019
https://github.com/charlotte-pel/temporalCNN

https://www.mdpi.com/2072-4292/11/5/523
"""

__all__ = ['TempCNN']

class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, sequencelength=45, kernel_size=7, hidden_dims=128, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelength={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"
        self.name = "TempCNN"
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        # require BxCxT: from BxCxTxHxW to B*H*WxCxT (put spatial dimensions to batch dimension -> apply to each pixel. each pixel now has a temporal and a spectral dimension)
        # "spatially shared block" (apply same convolution to all pixel (share weights))
        x = x[:,:,:-1,:,:] # discard last channel that contains timestamp

        b,t,c,h,w = x.shape
        # pixel level CxT
        x = rearrange(x, 'b t c h w -> (b h w) c t')
        #x = x.view(b*h*w, c, t)
        
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.logsoftmax(x) # class probabilities
        # B x classes x H x W
        x = rearrange(x, '(b h w) c -> b c h w', h=h, w=w)
        #x = x.view(b, self.num_classes, h, w)
        return x

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == "__main__":
    x = torch.ones((2, 52, 5, 103, 101), dtype=torch.float32).to("cuda")
    model = TempCNN(4, 10, 52, 7, 128).to("cuda")
    # summary(model, (7, 52, 200, 200), batch_size=4, device="cuda") # bcthw
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    for name, p in model.named_parameters():
        print(name, "\t", p.size(), np.prod(p.size()))
    print('Trainable Parameters: %.3fM' % parameters)
    print("\ninput", x.shape, "\n")

    print("output:", model(x).shape)