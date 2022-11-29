from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


class FullAttention(nn.Module):
    def __init__(
        self,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # expand heads dimension
            scores.masked_fill_(attn_mask, -torch.inf)

        # nan_to_num because some timesteps are fully padded
        A = torch.nan_to_num(self.dropout(torch.softmax(scale * scores, dim=-1)))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V, A


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_queries_keys,
        d_values,
        n_heads,
        dropout_qkv=0.0,
    ):
        super().__init__()
        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)
        out, attn = self.attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out, attn


class TransformerLayer(nn.Module):
    """
    Pre-Norm Self/Cross Attention
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff,
        dropout_ff=0.1,
        activation="gelu",
        norm="layer",
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, self_seq, self_mask=None, cross_seq=None, cross_mask=None):
        q1 = self.norm1(self_seq)
        q1, self_attn = self.self_attention(
            queries=q1,
            keys=q1,
            values=q1,
            attn_mask=self_mask,
        )
        self_seq = self_seq + q1

        if self.cross_attention is not None:
            q1 = self.norm2(self_seq)
            q1, cross_attn = self.cross_attention(
                queries=q1,
                keys=cross_seq,
                values=cross_seq,
                attn_mask=cross_mask,
            )
            self_seq = self_seq + q1
        else:
            cross_attn = None

        q1 = self.norm3(self_seq)
        q1 = self.dropout_ff(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq, {"self_attn": self_attn, "cross_attn": cross_attn}


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        d_model: int = 200,
        d_ff: int = 600,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.0,
        dropout_qkv: float = 0.05,
        activation: str = "gelu",
        norm: str = "layer",
    ):
        super().__init__()
        assert activation in ["gelu", "relu"]

        # embedding
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim=d_model)
        self.ff1_state = nn.Linear(input_dim, d_model)
        self.ff2_state = nn.Linear(d_model, d_model)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout_emb)

        def make_attn():
            return AttentionLayer(
                attention=FullAttention(attention_dropout=dropout_attn),
                d_model=d_model,
                d_queries_keys=d_model // n_heads,
                d_values=d_model // n_heads,
                n_heads=n_heads,
                dropout_qkv=dropout_qkv,
            )

        def make_layer():
            return TransformerLayer(
                self_attention=make_attn(),
                cross_attention=None,
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model

    def make_attn_mask(self, x, pad_mask):
        batch, length, dim = x.shape
        # mask future tokens
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool), diagonal=1
        ).to(x.device)
        causal_mask = causal_mask.repeat(batch, 1, 1)
        full_mask = torch.max(causal_mask, pad_mask)
        return full_mask

    @property
    def emb_dim(self):
        return self.d_model

    def forward(self, states, pad_mask):
        batch, length, dim = states.shape
        mask = self.make_attn_mask(x=states, pad_mask=pad_mask)
        pos_idxs = torch.arange(length).to(states.device).long()
        pos_emb = self.position_embedding(pos_idxs)
        pos_emb = repeat(pos_emb, f"length d_model -> {batch} length d_model")
        traj_emb = self.dropout(self.activation(self.ff1_state(states)))
        traj_emb = self.dropout(self.ff2_state(traj_emb))
        traj_emb = traj_emb + pos_emb

        # self-attention
        for layer in self.layers:
            traj_emb, attn = layer(self_seq=traj_emb, self_mask=mask)
        traj_emb = self.norm(traj_emb)
        return traj_emb
