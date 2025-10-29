# implementation of PINNsformer
# paper: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2307.11833

import torch
import torch.nn as nn

from setpinn.util import get_clones


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            *[
                nn.Linear(d_model, d_ff),
                WaveAct(),
                nn.Linear(d_ff, d_ff),
                WaveAct(),
                nn.Linear(d_ff, d_model),
            ]
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True
        )
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.enc_attn = None

    def forward(self, x):
        x2 = self.act1(x)
        attn = self.attn(x2, x2, x2)
        x = x + attn[0]
        self.enc_attn = attn[1]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class PINNSFormer_Enc(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, in_dim=2):
        super(PINNSFormer_Enc, self).__init__()

        self.linear_emb = nn.Linear(in_dim, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            *[
                nn.Linear(d_model, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, d_out),
            ]
        )

    def forward(self, *v):
        if len(v) > 1:
            src = torch.cat(v, dim=-1)
        else:
            src = v[0]
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        output = self.linear_out(e_outputs)
        return output
