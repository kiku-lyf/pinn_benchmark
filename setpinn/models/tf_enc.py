import torch
import torch.nn as nn
import copy
def get_clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.Tanh()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.mhs = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        out = self.ln_1(x)
        x = x + self.mhs(out, out, out)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, in_dim=2):
        super(Transformer, self).__init__()
        
        self.N = N

        self.linear_emb = nn.Linear(in_dim, d_model)

        self.layers = get_clones(Block(d_model, heads=heads), N)

        self.linear_out = nn.Sequential(
            *[
                nn.Linear(d_model, d_hidden),
                nn.Tanh(),
                nn.Linear(d_hidden, d_model),
                nn.Tanh(),
                nn.Linear(d_model, d_out),
            ]
        )

    def forward(self, *v):
        if len(v) > 1:
            src = torch.cat(v, dim=-1)
        else:
            src = v[0]
        src = self.linear_emb(src)
        for i in range(self.N):
            src = self.layers[i](src)
        output = self.linear_out(src)
        return output
