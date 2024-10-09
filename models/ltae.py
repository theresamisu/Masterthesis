import copy

import numpy as np
import torch
import torch.nn as nn

from pe import LearnableFourierPositionalEncoding, PositionalEncoder, RNNPositionalEncoding


class LTAE2d(nn.Module):

    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=50,
        return_att=False,
        positional_encoding="sinus",
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.mlp[0] = self.d_model #!!!!
            self.inconv = None
        assert self.mlp[0] == self.d_model      


        if positional_encoding == "fourier":
            print("fourier positional encoding")
            dim = self.d_model // n_head
            self.positional_encoder = LearnableFourierPositionalEncoding(m=1, f=dim, h=32, d=dim, max_pos=T, n_head=n_head)
        elif positional_encoding == "rnn":
            print("rnn positional encoding")
            self.positional_encoder = RNNPositionalEncoding(self.d_model, n_head, sinusoid=True, max_pos=T)
        elif positional_encoding == "sinus":
            print("sinusoid positional encoding")
            self.positional_encoder = PositionalEncoder(self.d_model // n_head,
                                                        T=T,
                                                        repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head,
                                                  d_k=d_k,
                                                  d_in=self.d_model)
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend([
                nn.Linear(self.mlp[i], self.mlp[i + 1]),
                nn.BatchNorm1d(self.mlp[i + 1]),
                nn.ReLU(),
            ])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        # bhw X t X d

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            # print("bp", bp.shape, "input", x.shape)
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head # number of heads
        self.d_k = d_k # key dimension
        self.d_in = d_in # number of channels E. channels per head = d_in/n_head = E/H

        # master query: K x 1, h times -> H x K
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k) # map input channels to keys for each head
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))

    def forward(self, v, return_comp=False):
        # v = value = B x T x E, E=Channels ?
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size() 

        # make master query for each element in batch
        # -> (H*B) x K
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        # compute keys from values -> B x T x (H*K) and view as B x T x H x K
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        # permute to H x B x T x K and view as H*B x T x K (combine batch and head dimension)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # split channels of values into H groups, one for each head (channel grouping) (dim=-1 -> last dimension is E=channel dimension)
        # combine again head and batch dimension -> (H*B) x T x E' =E/H
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, return_comp=return_comp
            )
        else:
            output, attn = self.attention(q, k, v, return_comp=return_comp)
        # shape of attention: from (H*B) x 1 x T -> H x B x (1) x T
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        # from (H*B) x 1 x E' -> H x B x (1) x E' (=E/H)
        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, return_comp=False):
        """
        q = (H*B) x K
        k = (H*B) x T x K
        v = (H*B) x T x E'

        return output = (H*B) x 1 x E'
        attn = (H*B) x 1 x T
        """
        # (H*B) x 1 x K \times (H*B) x K x T -> attn = (H*B) x 1 x T
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # output = (H*B) x 1 x E'
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn