import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class PINNGPT(nn.Module):
    """
    A neural network model for solving the Burgers' equation.
    This model includes an embedding layer, attention mechanism, and a feed-forward neural network.
    
    Note that the attention mechanism is causal, meaning that the model can only see the past data. This is the most important characteristic of a time series model.
    
    The model can in theory be made more complex like the GPT model, but this is a simple version as a proof of concept. Also, a complex model would require more data.
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.emb = nn.Linear(2, 128, bias=False)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(128, 3 * 128, bias=False)
        # output projection
        self.c_proj = nn.Linear(128, 128, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.n_head = 1
        self.n_embd = 128
        self.dropout = 0.0
        
        self.nn = nn.Sequential(
            nn.Linear(128, 64),
            WaveAct(),
            nn.Linear(64, 32),
            WaveAct(),
            nn.Linear(32, 16),
            WaveAct(),
            nn.Linear(16, 1),
        )
        
        self.register_buffer("bias", torch.tril(torch.ones(100, 100)).view(1, 1, 100, 100))

    def forward(self, *v):
        """
        Forward pass of the model.
        Args:
            *v: Variable length input list, expected to be tensors.
        Returns:
            Tensor: Output of the neural network.
            
        Important: For now, I have given position (x) and time (t) as input. When there is some actual data, a past token needs to be given as input along with time.
        
        Note that the positional embeddings are not used here as the variable "t" takes care of the time (position) information.
        """
        input = torch.concat(v, dim=-1)
        y = self.emb(input)
        B, T, C = (
            y.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(y).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return self.nn(y)