import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    """Positional encoding for timeseries sequences."""
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

class LearnableFourierPositionalEncoding(nn.Module):
    """Source https://github.com/jnyborg/tpe/blob/main/models/ltae.py"""
    def __init__(self, m=1, f=384, h=32, d=768, max_pos=10000, n_head=16):
        """
        Re-implementation of Learnable Fourier Features from https://arxiv.org/abs/2106.02795
        """

        super().__init__()

        assert f % 2 == 0

        self.wr = nn.Linear(m, f//2, bias=False)
        self.max_pos = max_pos

        self.mlp = nn.Sequential(
                nn.Linear(f, h),
                nn.GELU(),
                nn.Linear(h, d)
        )
        self.scale = f**-0.5
        self.n_head = n_head

    def forward(self, x):
        x = x.unsqueeze(2).float() / self.max_pos  # normalize to [0, 1]

        x = self.wr(x)
        x = self.scale * torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        x = self.mlp(x)
        x = torch.cat([x for _ in range(self.n_head)], dim=2)

        return x


class RNNPositionalEncoding(nn.Module):
    """Source https://github.com/jnyborg/tpe/blob/main/models/ltae.py"""
    def __init__(self, d_model, n_head, sinusoid=True, max_pos=10000):
        super().__init__()
        dim = d_model // n_head
        self.sinusoid = sinusoid
        
        if self.sinusoid:
            self.position_enc = PositionalEncoder(dim, T=max_pos)
            input_dim = dim
        else:
            input_dim = 1

        self.rnn = nn.GRU(input_dim, dim, batch_first=True)
        self.mlp = nn.Linear(dim, dim)
        self.n_head = n_head
        self.max_pos = max_pos
        
    def forward(self, x):
        x = self.position_enc(x) if self.sinusoid else x.unsqueeze(2) / self.max_pos
        x, _ = self.rnn(x)
        x = self.mlp(x)
        x = torch.cat([x for _ in range(self.n_head)], dim=2)
        return x
    
def get_positional_encoding(max_len, d_model, T=1000.0):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
