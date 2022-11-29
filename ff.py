import torch
from torch import nn
import torch.nn.functional as F


class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.00, activation="gelu"):
        assert activation in ["gelu", "relu"]
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x1 = self.dropout(self.activation(self.ff1(x)))
        x1 = self.dropout(self.activation(self.ff2(x1)))
        return x + x1


class MLP(nn.Module):
    def __init__(self, d_inp: int, d_hidden: int, n_layers: int, d_output: int):
        super().__init__()
        assert n_layers >= 1
        self.in_layer = nn.Linear(d_inp, d_hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 1)]
        )
        self.out_layer = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out_layer(x)
        return x


class FeedForwardEncoder(nn.Module):
    def __init__(
        self, input_dim, d_model=128, d_ff=512, dropout=0.00, activation="gelu"
    ):
        super().__init__()
        self.traj_emb = nn.Linear(input_dim, d_model)
        self.traj_block = FFBlock(d_model, d_ff, dropout=dropout, activation=activation)
        self.traj_last = nn.Linear(d_model, d_model)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = d_model

    def forward(self, states, pad_mask=None):
        traj_emb = self.dropout(self.activation(self.traj_emb(states)))
        traj_emb = self.traj_block(traj_emb)
        traj_emb = self.dropout(self.activation(self.traj_last(traj_emb)))
        return traj_emb
