import torch.nn as nn
import torch as t
import numpy as np
import torch

class C3D(nn.Module):
    """3D Convolutional Neural Network for classification of time series data. """
    def __init__(self, bands=None, labels=None, timepoints=None, temp_k_size=3, k_size=3, return_maps = False) -> None:
        super().__init__()
        
        self.name="C3D"
        print(f"C3D: basic C3D with kernel ({temp_k_size, k_size, k_size})")
        pad = int((k_size-1)/2)
        temp_pad = int((temp_k_size-1)/2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.c1 = nn.Conv3d(bands, 64, kernel_size=(temp_k_size, k_size, k_size), padding=(temp_pad, pad, pad))
        self.p1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.c2 = nn.Conv3d(64, 128, kernel_size=(temp_k_size, k_size, k_size), padding=(temp_pad, pad, pad))
        self.p2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dc1 = nn.ConvTranspose3d(128,
                                      128,
                                      kernel_size=(1, k_size, k_size),
                                      stride=(1, 2, 2),
                                      padding=pad,
                                      dilation=1,
                                      output_padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(128)
        self.bn1c = nn.Conv3d(128, 128, kernel_size=(1, k_size, k_size), padding=pad)
        
        self.dc2 = nn.ConvTranspose3d(128, #
                                      64, #128
                                      kernel_size=(1, k_size, k_size),
                                      stride=(1, 2, 2),
                                      padding=pad,
                                      dilation=1,
                                      output_padding=(0, 1, 1))
        self.bn2c = nn.Conv3d(64, 64, kernel_size=(1, k_size, k_size), padding=pad)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.classifier = nn.Conv3d(64,
                                    labels,
                                    kernel_size=(int(timepoints / 4), 1, 1))
        self.return_maps = return_maps

    def forward(self, x, **kwargs):
        """ forward pass through the network

        Args:
            x (Tensor): input tensor B x T x C x W x H

        Returns:
            Tensor: Tensor with class predictions
        """
        x = x.permute(0,2,1,3,4) # BxTxCxHxW -> BxCxTxHxW

        x = x[:,:-1,:,:,:] # discard last channel that contains timestamp   

        x = self.p1(self.relu(self.c1(x)))
        x = self.p2(self.relu(self.c2(x)))
        
        x = self.bn1(self.relu(self.dc1(x)))
        x = self.relu(self.bn1c(x))
        
        x = self.bn2(self.relu(self.dc2(x)))
        x = self.relu(self.bn2c(x))
        
        if self.return_maps:
            return x
        
        x = self.classifier(x)
        x = t.squeeze(x, dim=2)
        
        assert not t.isnan(x).any() and not t.isinf(x).any()
        # B x C x W x H
        
        return x

if __name__ == "__main__":
    x = torch.ones((2, 52, 5, 103, 101), dtype=torch.float32).to("cuda")
    model = C3D(4, 10, 52, 3, 3, False).to("cuda")
    # summary(model, (7, 52, 200, 200), batch_size=4, device="cuda") # bcthw
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    for name, p in model.named_parameters():
        print(name, "\t", p.size(), np.prod(p.size()))
    print('Trainable Parameters: %.3fM' % parameters)
    print("\ninput", x.shape, "\n")

    print("output:", model(x).shape)