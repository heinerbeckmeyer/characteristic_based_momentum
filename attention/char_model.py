# %%
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
from entmax import Sparsemax, Entmax15
import math


# %% Auxiliary functions
class TimeDistributed(nn.Module):
    """Takes any module and stacks the time dimension (dim=1) with the batch dimension (=0) before applying processing.
    From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    """

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *tuple(x.shape[2:]))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape y
        if len(x.size()) == 4:
            y = y.contiguous().view(x.size(0), -1, *tuple(x.shape[-2:]))  # (samples, timesteps, output_size)
        else:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)

        return y


# %% Attention
class PositionalEmbedding(nn.Module):
    """Uses sin/cos positional encoding following the paper by Vaswani et al. (2017), Attention is all you need
    See http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding."""

    def __init__(
        self,
        d_model,
        dropout: float,
        max_len=1000,
        freq=10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(freq) / d_model))
        pe[:, 0::2] = torch.sin(position * div)  # Even terms
        pe[:, 1::2] = torch.cos(position * div)  # Odd terms
        pe = pe.unsqueeze(0)  # (1 x max_len x embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe.requires_grad_(False)  # in 4D!
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        d_model: int,
        dropout_p: float,
        alpha: float = 1.0,
        retrieve_attention_mask: bool = False,
    ):
        super().__init__()

        self.retrieve_attention_mask = retrieve_attention_mask
        self.heads = heads
        self.d_k = d_model // heads

        self.query = self.PrepareMHA(heads, d_model, self.d_k)
        self.key = self.PrepareMHA(heads, d_model, self.d_k)
        self.value = self.PrepareMHA(heads, d_model, self.d_k)

        if alpha == 1.5:
            self.normalization = Entmax15(dim=-2)
        elif alpha == 2.0:
            self.normalization = Sparsemax(dim=-2)
        else:
            self.normalization = nn.Softmax(dim=-2)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_p)
        self.scale = 1 / torch.sqrt(torch.tensor(self.d_k).to(torch.float32))  # Eq (10)

    class PrepareMHA(nn.Module):
        def __init__(self, heads: int, d_model: int, d_k: int):
            super().__init__()
            self.linear = nn.Linear(d_model, d_k * heads)
            self.heads = heads
            self.d_k = d_k

        def forward(self, x: torch.Tensor):
            """x: B x seq x (...) x (d_k * heads)"""
            shape = x.shape
            x = self.linear(x)  # B x seq x (...) x (d_k * heads)
            x = x.view(*shape[:-1], self.heads, self.d_k)  # B x seq x (...) x heads x d_k
            return x

    def forward(
        self, query, key, value, attn_einsum: str, value_einsum: str, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # run input through linear layers for comparison
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # shape = query.shape

        # obtain attention
        # NOTE: F.softmax(torch.einsum("TN,tN->NT", q, k), dim=-1) is the same as
        # F.softmax(torch.einsum("TN,tN->NTt", q, k).sum(dim=-1), dim=-1)
        attn = torch.einsum(attn_einsum, query, key)
        attn = attn * self.scale

        # mask inputs
        if mask is not None:
            attn = attn.masked_fill(~mask.bool().to(query.device), float("-inf"))

        # normalize attention matrix
        attn = self.normalization(attn)
        # attn = self.dropout(attn)

        # multiply attention with values
        value = torch.einsum(value_einsum, attn, value)

        # concat attention heads
        value = value.reshape(*value.shape[:-2], -1)

        if self.retrieve_attention_mask:
            attn = attn.detach()
        else:
            attn = None

        return value, attn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        d_model: int,
        d_ffn: int,
        dropout_p: float,
        d_rotate: int = 0,
        alpha: float = 1.0,
        retrieve_attention_mask: bool = False,
    ):
        super().__init__()

        self.d_rotate = d_rotate
        if d_rotate > 0:
            self.rotate = nn.LazyLinear(d_rotate)
        self.mha = MultiHeadAttention(
            heads=heads,
            d_model=d_model,
            dropout_p=dropout_p,
            alpha=alpha,
            retrieve_attention_mask=retrieve_attention_mask,
        )
        self.mha_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, attn_einsum: str, value_einsum: str, mask: torch.Tensor = None):
        # attention
        residual = x.clone()
        x, attention_mask = self.mha.forward(
            query=x,
            key=x,
            value=x,
            mask=mask,
            attn_einsum=attn_einsum,
            value_einsum=value_einsum,
        )
        x = self.mha_norm(self.dropout(x) + residual)  # ResNet

        # ffn:
        residual = x.clone()
        x = self.ffn.forward(x)
        x = self.ffn_norm(self.dropout(x) + residual)

        return x, attention_mask


# %%
# Other building blocks
class CategoricalEmbedding(nn.Module):
    def __init__(self, d_model: int, n_groups: int, time_distributed: bool):
        super().__init__()

        if time_distributed:
            self.emb = TimeDistributed(nn.Embedding(num_embeddings=n_groups, embedding_dim=d_model))
        else:
            self.emb = nn.Embedding(num_embeddings=n_groups, embedding_dim=d_model)

    def forward(self, x):
        return self.emb(x)


class LinearEmbedding(nn.Module):
    def __init__(self, d_model: int, time_distributed: bool, relu: bool = False, pos_only: bool = False):
        super().__init__()

        self.relu = relu
        if pos_only:
            if time_distributed:
                self.emb = TimeDistributed(PosLinear(1, d_model, bias=False))
            else:
                self.emb = PosLinear(1, d_model, bias=False)

        else:
            if time_distributed:
                self.emb = TimeDistributed(nn.Linear(1, d_model, bias=False))
            else:
                self.emb = nn.Linear(1, d_model, bias=False)

    def forward(self, x):
        if self.relu:
            return F.relu(self.emb(x))
        else:
            return self.emb(x)


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias: bool = True):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        self.use_bias = bias

    def forward(self, x):
        if self.use_bias:
            return torch.matmul(x, torch.clamp(self.weight, 0, 10)) + self.bias
        else:
            return torch.matmul(x, torch.clamp(self.weight, 0, 10))


# %%
# Model
class CharacteristicWeightedMomentum_ClassModel(nn.Module):
    def __init__(
        self,
        N_characteristics: int,
        N_returns: int,
        n_groups: int,
        n_attention_blocks: int,
        heads: int,
        d_model: int,
        d_ffn: int,
        dropout_p: float,
        alpha: float = 1.0,
        retrieve_attention_mask: bool = False,
    ):
        super().__init__()
        self.N_characteristics = N_characteristics
        self.N_returns = N_returns

        # ---- characteristics
        self.char_embedding = nn.ModuleList(
            [
                CategoricalEmbedding(d_model=d_model, n_groups=n_groups + 1, time_distributed=False)
                for _ in range(N_characteristics)
            ]
        )
        self.char_attention_block = nn.ModuleList(
            [
                AttentionBlock(
                    heads=heads,
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout_p=dropout_p,
                    alpha=alpha,
                    retrieve_attention_mask=retrieve_attention_mask,
                )
                for _ in range(n_attention_blocks)
            ]
        )

        # ---- returns
        self.r_emb = LinearEmbedding(d_model=d_model, time_distributed=False, relu=False, pos_only=False)
        self.return_positional_emb = PositionalEmbedding(d_model=d_model, dropout=dropout_p, max_len=N_returns)

        # ---- interaction between experience and demographics
        self.extract_weights = nn.Linear(N_characteristics, N_returns)
        self.norm_weights = nn.Softmax(dim=1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, characteristics, returns):
        attn_dem_to_experience = None

        # ---- characteristics
        # first, embed characteristics
        embedded_characteristics = []
        for i in range(self.N_characteristics):
            embedded_characteristics.append(self.char_embedding[i](characteristics[:, i]))
        embedded_characteristics = torch.stack(embedded_characteristics, dim=1)

        # then, extract interaction effects between demographics
        attn_dem_to_experience = {}
        for i, m in enumerate(self.char_attention_block):
            embedded_characteristics, attn = m.forward(
                x=embedded_characteristics,
                mask=None,
                attn_einsum="BNHD,BMHD->BNMH",
                value_einsum="BNMH,BMHD->BNHD",
            )
            attn_dem_to_experience[i] = attn

        # ---- returns
        # add positional encoding to input returns
        embedded_returns = self.r_emb(returns[..., None])
        embedded_return = self.return_positional_emb(embedded_returns)

        # ---- interaction between experience and demographics
        # in a first step, create attention matrix of dem conditional on exp
        # then renormalize that (as it is multiplied by values from dem)
        # and multiply with values from exp to get exp conditional on dem

        # --- comparative weighting
        queries = self.query.forward(embedded_characteristics)
        keys = self.key.forward(embedded_return)
        values = returns.clone()

        weights = torch.einsum("BCD,BRD->BR", queries, keys)
        weights = weights / math.sqrt(keys.shape[-1])
        weights = self.norm_weights.forward(weights)
        out = torch.einsum("BR,BR->B", weights, values)
        #out = out[:, None]  # expand to Bx1

        # ---- outputs
        return out, weights

# %%
# Model
class CharacteristicWeightedMomentum(nn.Module):
    def __init__(
        self,
        N_characteristics: int,
        N_returns: int,
        d_model: int,
        dropout_p: float,
        inverse: bool = False,
    ):
        super().__init__()
        self.N_characteristics = N_characteristics
        self.N_returns = N_returns
        self.inverse = inverse

        # ---- characteristics: BxF --> Bxemb
        self.char_mlp = nn.Sequential(
            nn.Linear(N_characteristics, N_characteristics),
            nn.GELU(),
            nn.Linear(N_characteristics, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, d_model),
        )

        # ---- returns: BxT --> BxTxemb
        self.r_emb = LinearEmbedding(d_model=d_model, time_distributed=False, relu=False, pos_only=False)
        self.return_positional_emb = PositionalEmbedding(d_model=d_model, dropout=dropout_p, max_len=N_returns)

        # ---- interaction between experience and demographics
        #self.extract_weights = nn.Linear(N_characteristics, N_returns)
        self.norm_weights = nn.Softmax(dim=1)
        #self.query = nn.Linear(d_model, d_model)
        #self.key = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, characteristics, returns):
        # Values - Plain returns
        values = returns.clone()
        if self.inverse:
            values = -1 * values

        # Queries - Characteristics
        chars = self.char_mlp(characteristics)

        # Keys - Embedded Returns
        embedded_returns = self.r_emb(returns[..., None])
        keys = self.return_positional_emb(embedded_returns)


        weights = torch.einsum("BD,BRD->BR", chars, keys)
        weights = weights / math.sqrt(keys.shape[-1])
        weights = self.norm_weights.forward(weights)
        out = torch.einsum("BR,BR->B", weights, values)
        #out = out[:, None]  # expand to Bx1

        # ---- outputs
        return out, weights


# %%
# params = dict(
#     N_characteristics=10,
#     N_returns=230,
#     #n_groups=100,
#     #n_attention_blocks=1,
#     #heads=1,
#     d_model=16,
#     #d_ffn=16,
#     dropout_p=0,
#     #alpha=1.0,
# )
# model = CharacteristicWeightedMomentum(**params)

# B = 25
# # characteristics = torch.randint(low=0, high=params["n_groups"], size=(B, params["N_characteristics"]))
# characteristics = torch.rand(size=(B, params["N_characteristics"])) - 0.5
# returns = torch.randn(size=(B, params["N_returns"]))

# model.forward(characteristics=characteristics, returns=returns)


# %%
