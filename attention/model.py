# %%
# TORCH
import torch
import torch.nn as nn

import math

from entmax import Sparsemax, Entmax15


# %%
class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        dropout_p: float,
        d_embedding: int,
        n_lags: int,
        freq: float = 10000.0,
    ):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos = torch.arange(0, n_lags, dtype=torch.float).unsqueeze(1)

        two_i = torch.arange(0, d_embedding, step=2).float()
        divisor = torch.exp(two_i * (-math.log(freq)) / d_embedding)

        pe = torch.zeros(n_lags, d_embedding)
        pe[:, 0::2] = torch.sin(pos * divisor)
        pe[:, 1::2] = torch.cos(pos * divisor)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor, shape [B, F, d_embedding]
        """
        self.pe = self.pe.to(x.device)
        x = x + self.pe.requires_grad_(False)

        return self.dropout(x)


# %%
class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_embedding: int,
        do_pos_encoding: bool,
        n_lags: int,
        retrieve_attention_mask: bool = False,
        alpha: float = 1.0,
        dropout_p: float = 0.1,
    ):

        super().__init__()
        self.retrieve_attention_mask = retrieve_attention_mask
        self.d_k = d_embedding  # NOTE heads = 1 assumed
        self.do_pos_encoding = do_pos_encoding

        # "Feature" embedding, i.e. one embedding per feature
        # self.lin_embed = nn.ModuleList(
        #     [nn.Linear(1, d_embedding, bias=False) for _ in range(n_lags)]
        # )  # output shape: (B,F, d_embedding)
        self.lin_embed = nn.Linear(1, d_embedding, bias=False)

        # Positional encoding
        if self.do_pos_encoding:
            self.positionEncoding = PositionalEmbedding(
                dropout_p=dropout_p, d_embedding=d_embedding, n_lags=n_lags, freq=10000
            )

        # --- Prepare Attention
        # via LL
        # self.keys = nn.Linear(d_embedding, d_embedding, bias=False)
        # self.queries = nn.Linear(d_embedding, d_embedding, bias=False)

        # or MLP
        self.keys = nn.Sequential(
            nn.Linear(d_embedding, 2 * d_embedding),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2 * d_embedding, d_embedding),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_embedding, d_embedding),
        )
        self.queries = nn.Sequential(
            nn.Linear(d_embedding, 2 * d_embedding),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2 * d_embedding, d_embedding),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_embedding, d_embedding),
        )

        # ---- Scaling
        self.scale = 1 / torch.sqrt(torch.tensor(self.d_k).to(torch.float32))

        # --- Normalization
        if alpha == 1.5:
            self.normalization = Entmax15(dim=-1)
        elif alpha == 2.0:
            self.normalization = Sparsemax(dim=-1)
        else:
            self.normalization = nn.Softmax(dim=-1)

        # --- Output
        self.output = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        values = x.clone()
        # ---- Embedding
        # output shape: (B,F,d_embedding)
        # embedding = []
        # for i in range(len(self.lin_embed)):
        #     embedding.append(self.lin_embed[i](x[:, i : i + 1]))
        # x = torch.stack(embedding, dim=1)
        x = self.lin_embed(x[..., None])

        # ---- Positional Embedding
        if self.do_pos_encoding:
            x = self.positionEncoding(x)

        # ---- Run through linear layers/mlp
        queries = self.queries(x)
        keys = self.keys(x)

        # ---- Obtain attention matrix
        att = torch.einsum("bqe,bke->bqk", queries, keys)
        # queries shape: (B, query_len, embed) here: query_len = F
        # keys shape: (B, key_len, embed) here: key_len = F
        # # att shape: (B, query_len, key_len)
        # att shape: (B, key_len)

        # ---- Scale
        att = att * self.scale

        # Dimension Reduction
        # From (BxFxF) to Bx1xF
        # NOTE: Use sum instead of mean as it is computational less expensive
        att = torch.mean(att, dim=1, keepdim=False)
        # att shape: (B, 1, key_len)
        # att shape: (B, key_len)

        # Normalize
        # NOTE Do we need dropout?
        att = self.normalization(att)

        # Multiply with values
        out = torch.einsum("bk,bk->b", att, values)
        # NOTE: value_len = key_len which is important here
        # values shape: (B, value_len) here: value_len = F
        # output shape: (B, 1)

        if self.retrieve_attention_mask:
            att = att.squeeze(dim=1)
            att = att.detach()
        else:
            att = None

        # Remove last dimension
        # out = out.squeeze(dim=-1)
        out = self.output(out[:, None]).squeeze(dim=-1)
        return out, att


# x = 2*torch.rand(size=(100, 10)) -1
# self = SingleHeadAttention(d_embedding=8, pos_encoding = False, n_lags=x.shape[1])
# self.forward(x)

# %%
