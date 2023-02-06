# %%
import pandas as pd
import numpy as np

# TORCH
import torch
import torch.nn as nn

from entmax import Sparsemax, Entmax15

# %%
class CateroricalEmbedding(nn.Module):
    """
    Embedding for categorical features:
    Input:
        - d_embedding: Dimension of each embedding vector
        - n_groups: Number of unique values the feature can take
    """

    def __init__(self, d_embedding: int, n_groups: int):
        super().__init__()
        # --- Use Pytorch Embedding Module
        self.embed = nn.Embedding(num_embeddings=n_groups, embedding_dim=d_embedding)

    def forward(self, x):
        return self.embed(x)



# %%
class LinearEmbedding(nn.Module):

    def __init__(self,d_embedding):
        super().__init__()

        self.embedding = nn.Linear(1, d_embedding)

    def forward(self, x):
        x = self.embedding(x)

        return x


# %%
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_embedding: int,
        heads: int,
        retrieve_attention_mask: bool = False,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.d_embedding = d_embedding
        self.heads = heads
        self.retrieve_attention_mask = retrieve_attention_mask
        self.d_k = d_embedding // heads

        assert (self.d_k * heads == d_embedding), "d_embedding must be divisible by n_heads"
        
        # ---- Input Preparation for MHA
        self.values = self.PrepareMHA(heads, d_embedding, self.d_k)
        self.keys = self.PrepareMHA(heads, d_embedding, self.d_k)
        self.queries = self.PrepareMHA(heads, d_embedding, self.d_k)
        
        # Get Normalization 
        if alpha == 1.5:
            self.normalization = Entmax15(dim=3)
        elif alpha == 2.0:
            self.normalization = Sparsemax(dim=3)
        else:
            self.normalization = nn.Softmax(dim=3)

        self.fc_out = nn.Linear(d_embedding, d_embedding)
        self.scale = 1 / torch.sqrt(torch.tensor(self.d_k).to(torch.float32))

    class PrepareMHA(nn.Module):
        def __init__(self, heads: int, d_embedding: int, d_k: int):
            super().__init__()
            self.linear = nn.Linear(d_embedding, d_k * heads, bias = False) #NOTE Do we need a BIAS here?
            self.heads = heads
            self.d_k = d_k
        
        def forward(self, x: torch.Tensor):
            shape = x.shape
            x = self.linear(x) 
            # Split embedding dimension into heads x d_k
            x = x.reshape(*shape[:-1], self.heads, self.d_k)

            return x

    def forward(self, queries, keys, values):
        #----  Run through linear layers
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        # ---- Obtain attention matrix
        attn = torch.einsum("bqhd,bkhd->bhqk", queries, keys)
        # queries shape: (B, query_len, heads, d_k), here: query_len = n_factors
        # keys shape: (B, key_len, heads, d_k), here: key_len = n_features
        # attn shape: (B, heads, query_len, key_len)
        
        # Scale
        attn = attn * self.scale
        # Normalize (last dimension (3))
        attn = self.normalization(attn)

        # ---- Multiply with values
        out = torch.einsum("bhqk,bkhd->bqhd",attn,values)
        # NOTE: key_len = value_len which is important here
        # values shape: (B, value_len, heads, d_k), here: value_len = n_feat
        # output shape: (B, query_len, heads, d_k), then concat to (B, query_len, heads*d_k)
        out = out.reshape(*out.shape[:-2],-1)

        if self.retrieve_attention_mask:
            attn = attn.detach()
        else:
            attn = None
        
        return out, attn


# %%
class AttentionBlock_withReduction(nn.Module):
    """
        Q: BxFqxEmb
        K: BxFkxEmb
        V: BxFvxEmb

        Output of Attention Layer is BXFqxEmb. So we have three options:
            1. Set d_embedding=1 and reduce query size via a LL. 
             --> We thus could only use one head.
            2. Keep embedding and concatenate along the feature dimension. 
               We would thus affectively increase the feature dimension first to then 
               decrease the query size via a LL . 
             --> We thus could only use one head.
            3. Keep embedding and reduce query size via a LL and obtain ouput shape of
               Bxn_FactorsxEmb. Then flatten embbeding dimension via LL.
              --> Interpretability issues?
    """
    def __init__(
        self,
        heads: int,
        d_embedding: int,
        dropout_p: float,
        n_factors: int = 0,
        forward_expansion: int = 1,
        retrieve_attention_mask: bool = False,
        alpha: float = 1.0
    ):

        super().__init__()
        self.n_factors = n_factors
        
        # Reduce Query Size
        if self.n_factors > 0:
            self.linear_reduction = nn.LazyLinear(n_factors)

        self.attention = MultiHeadAttention(
            heads=heads,
            d_embedding=d_embedding,
            alpha=alpha,
            retrieve_attention_mask=retrieve_attention_mask
        )
        self.mha_norm = nn.LayerNorm(d_embedding)
        self.ffn = nn.Sequential(
            nn.Linear(d_embedding, d_embedding*forward_expansion),
            nn.GELU(),
            nn.Linear(d_embedding*forward_expansion, d_embedding)
        )
        self.ffn_norm = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self,x):
        
        if self.n_factors>0:
            queries = x.clone()
            # NOTE: Reduction of Query size is in Feature dimension. 
            # We need to permute the queries first and the reverse the permutation
            # x shape: (B,F,d_embedding)
            queries = self.linear_reduction(queries.permute(0,2,1))
            queries = queries.permute(0,2,1)

            x, attention_mask = self.attention(
                queries=queries, keys=x, values=x,
            )
            x = self.mha_norm(self.dropout(x)) # NOTE There is no residual connection here?
        else:
            residual = x.clone()
            x, attention_mask = self.attention(
                queries=x, keys=x, values=x,
            )
            x = self.mha_norm(self.dropout(x) + residual)

        # --- FFN
        residual = x.clone()
        x = self.ffn(x)
        x = self.ffn_norm(self.dropout(x) + residual) # NOTE Should we switch Droput/Norm here?

        return x, attention_mask



# %%
class AttentionEncoder(nn.Module):
    def __init__(
        self,
        d_embedding: int,
        n_factors: int,
        n_features: int,
        heads: int,
        dropout_p: float,
        n_attention_blocks: int = 1,
        forward_expansion: int = 1,
        alpha: float = 1.0,
        retrieve_attention_mask: bool = False,
    ):
        super().__init__()
        
        # ---- Feat embeddding, i.e. one embedding per feature
        self.lin_embed = nn.ModuleList(
            [LinearEmbedding(d_embedding=d_embedding) for _ in range(n_features)]
        )   # output shape: (B,F,d_embedding)

        # ---- Attention Block
        self.attention_block = nn.ModuleList(
            [
                AttentionBlock_withReduction(
                    heads=heads,
                    d_embedding=d_embedding,
                    n_factors=0,
                    dropout_p=dropout_p,
                    alpha=alpha,
                    forward_expansion=forward_expansion,
                    retrieve_attention_mask=retrieve_attention_mask
                )
                for _ in range(n_attention_blocks-1)
            ]
        )
        # Add Last Block with Reduction
        self.attention_block.append(
            AttentionBlock_withReduction(
                heads=heads,
                d_embedding=d_embedding,
                n_factors=n_factors,
                dropout_p=dropout_p,
                alpha=alpha,
                forward_expansion=forward_expansion,
                retrieve_attention_mask=retrieve_attention_mask
            )   # output shape: (B, n_factors, d_embedding)
        )
        


        # ---- Output LL to reduce embedding dimension
        # NOTE: Should we instead reshape to (B, n_factors*d_embedding) first? 
        self.output = nn.Linear(d_embedding,1)

    def forward(self, x):
        embedding = []
        for i in range(len(self.lin_embed)):
            embedding.append(self.lin_embed[i](x[:,i:i+1]))
        x = torch.stack(embedding, dim=1)

        # --- Transformer
        attn_mask = {}
        for i in range(len(self.attention_block)):
            x, attn= self.attention_block[i](x)
            attn_mask[i] = attn

        # --- Output
        x = self.output(x)
        x = torch.squeeze(x)

        return x, attn_mask

# x = 2*torch.rand(size=(100, 10)) -1
# self = AttentionEncoder(n_features=10,dropout_p=0.1,d_embedding=6,n_factors=3,heads=2, n_attention_blocks=1)
# self.forward(x)


# %%
class AttentionDecoder(nn.Module):
    def __init__(
        self,
        d_embedding: int,
        n_features: int,
        n_factors: int,
        heads: int,
        dropout_p: float,
        n_attention_blocks: int = 1,
        forward_expansion: int = 1,
        alpha: float = 1.0,
        retrieve_attention_mask: bool = False,
    ):
        super().__init__()
        
        # ---- Linear Layer to expand Embedding dimension
        #self.emb_exp = nn.Linear(1,d_embedding)

        # ---- Embedding, i.e. one embedding per factor
        self.lin_embed = nn.ModuleList(
            [LinearEmbedding(d_embedding=d_embedding) for _ in range(n_factors)]
        )   # output shape: (B,F,d_embedding)

        # ---- Attention Block
        self.attention_block = nn.ModuleList()
        # Add Block with expansion first
        self.attention_block.append(
            AttentionBlock_withReduction(
                heads=heads,
                d_embedding=d_embedding,
                n_factors=n_features,
                dropout_p=dropout_p,
                alpha=alpha,
                forward_expansion=forward_expansion,
                retrieve_attention_mask=retrieve_attention_mask
            )   # output shape: (B, n_factors, d_embedding)
        )
        # Add rest
        for _ in range(n_attention_blocks-1):
            self.attention_block.append(
                AttentionBlock_withReduction(
                    heads=heads,
                    d_embedding=d_embedding,
                    n_factors=0,
                    dropout_p=dropout_p,
                    alpha=alpha,
                    forward_expansion=forward_expansion,
                    retrieve_attention_mask=retrieve_attention_mask
                )
            )

        
        # ---- Output LL to reduce embedding dimension
        # NOTE: Should we instead reshape to (B, n_factors*d_embedding) first? 
        self.output = nn.Linear(d_embedding,1)

    def forward(self, x):
        # x = self.emb_exp(x.unsqueeze(dim=-1))

        embedding = []
        for i in range(len(self.lin_embed)):
            embedding.append(self.lin_embed[i](x[:,i:i+1]))
        x = torch.stack(embedding, dim=1)

        # --- Transformer
        attn_mask = {}
        for i in range(len(self.attention_block)):
            x, attn= self.attention_block[i](x)
            attn_mask[i] = attn

        # --- Output
        x = self.output(x)
        x = torch.squeeze(x)

        return x, attn_mask


# x = 2*torch.rand(size=(100, 3)) -1
# self = AttentionDecoder(n_features=10,dropout_p=0.1,d_embedding=6,heads=2, n_attention_blocks=1)
# self.forward(x)
# %%
class SimpleEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        internal_sizes: list,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        # --- Check input
        if not all([i >= j for i, j in zip(internal_sizes[:-1], internal_sizes[1:])]):
            raise ValueError("Encoder: Internal sizes should be strictly decreasing.")

        # --- MLP
        self.fcs = nn.ModuleList()

        if internal_sizes:
            self.fcs.append(nn.Linear(n_features, internal_sizes[0]))
            for i in range(len(internal_sizes) - 1):
                self.fcs.append(nn.Linear(internal_sizes[i], internal_sizes[i + 1]))

            # --- Output Layer
            self.fcs.append(nn.Linear(internal_sizes[-1], n_factors))
        else:
            self.fcs.append(nn.Linear(n_features,n_factors))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # --- Apply MLP
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            if i < (len(self.fcs) - 1):
                x = self.dropout(self.relu(x))  # Don't apply to last layer.

        return x


# x = torch.randint(1, 9, size=(100, 10))
# self = SimpleEncoder(n_features=10,n_groups=10,dropout_p=0.1,d_embedding=3,n_factors=3,internal_sizes=[30,20,10])
# self.forward(x)

# %%
class SVDEncoder(nn.Module):
    def __init__(
        self,
        n_factors: int,
        seed: int = 123,
    ):
        torch.manual_seed(seed)

        super().__init__()
        self.n_factors = n_factors


    def forward(self, x):
        # ---- Get Principal Components
        # U, S, _ = torch.linalg.svd(x)
        # # NOTE: This some how raises a RuntimeError after some epochs as
        # the algorithm fails to converged due to too many repeated singular values
        # --> matrix ill-conditioned?
        U, S, _ = torch.svd_lowrank(x, q=self.n_factors)
        x = torch.mm(U[:, : self.n_factors], torch.diag(S[: self.n_factors]))

        return x


# x = torch.randint(1, 9, size=(100, 10))
# self = SVDEncoder(n_features=10,n_groups=10,d_embedding=3,n_factors=3)
# self.forward(x)

# %%
class DegenerateDecoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        # dropout_p: float = 0.0,
    ):
        super().__init__()

        # self.dropout = nn.Dropout(p=dropout_p)
        # --- Output Layer
        self.lin_out = nn.Linear(n_factors, n_features)

    def forward(self, x):
        # ---- Apply Output Layer
        x = self.lin_out(x)

        return x


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        internal_sizes: list,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.internal_sizes = internal_sizes
        
        # --- Check input
        if not all([i <= j for i, j in zip(internal_sizes[:-1], internal_sizes[1:])]):
            raise ValueError("Decoder: Internal sizes should be strictly increasing.")

        # --- MLP
        if self.internal_sizes:
            self.fcs = nn.ModuleList()
            self.fcs.append(nn.Linear(n_factors, internal_sizes[0]))
            for i in range(len(internal_sizes) - 1):
                self.fcs.append(nn.Linear(internal_sizes[i], internal_sizes[i + 1]))

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_p)

            # --- Output Layer
            self.lin_out = nn.Linear(internal_sizes[-1], n_features)
        else:
            self.lin_out = nn.Linear(n_factors, n_features)


    def forward(self, x):

        if self.internal_sizes:
            # --- Apply MLP
            for i in range(len(self.fcs)):
                x = self.fcs[i](x)
                x = self.dropout(self.relu(x))

        # ---- Apply Output Layer
        x = self.lin_out(x)

        return x


# x = torch.randint(1, 9, size=(100, 3)).type(torch.FloatTensor)
# self = SimpleDecoder(n_features=10,n_groups=10,dropout_p=0.1,internal_sizes=[5,8,10],n_factors=3)
# self.forward(x)


class SimpleModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        encoder_sizes: list,
        decoder_sizes: list,
        dropout_p: float = 0.0,
        retrieve_factors: bool = False,
    ):

        super().__init__()
        self.retrieve_factors = retrieve_factors
        self.encoder = SimpleEncoder(
            n_features=n_features,
            n_factors=n_factors,
            internal_sizes=encoder_sizes,
            dropout_p=dropout_p,
        )
        self.decoder = SimpleDecoder(
            n_factors=n_factors,
            n_features=n_features,
            internal_sizes=decoder_sizes,
        )


    def forward(self, x):
        factors = None
       
        # Apply model
        x = self.encoder(x)
        if self.retrieve_factors:
            factors = x.clone()
        x = self.decoder(x)

        return x, factors, None, None

class AttentionModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        d_embedding: int,
        heads: int,
        n_attention_blocks: int = 1,
        n_attention_blocks_decoder: int = 1, 
        dropout_p: float = 0.1,
        alpha: float = 1.0,
        forward_expansion: int = 1,
        retrieve_attention_mask: bool = False,
        retrieve_factors: bool = False,
    ):
        super().__init__()
        self.retrieve_factors = retrieve_factors

        self.encoder = AttentionEncoder(
            d_embedding=d_embedding,
            n_factors=n_factors,
            n_features=n_features,
            dropout_p=dropout_p,
            n_attention_blocks=n_attention_blocks,
            alpha=alpha,
            retrieve_attention_mask=retrieve_attention_mask,
            heads=heads,
            forward_expansion=forward_expansion,
        )
        self.decoder = AttentionDecoder(
            d_embedding=d_embedding,
            n_features=n_features,
            heads=heads,
            n_factors=n_factors,
            n_attention_blocks=n_attention_blocks_decoder,
            forward_expansion=forward_expansion,
            alpha=alpha,
            dropout_p=dropout_p,
            retrieve_attention_mask=retrieve_attention_mask,
        )

    def forward(self, x):
        factors = None

        x, attn_mask_encoder = self.encoder(x)
        if self.retrieve_factors:
            factors = x.clone()

        x, attn_mask_decoder= self.decoder(x)

        return x, factors, attn_mask_encoder, attn_mask_decoder

# x = 2*torch.rand(size=(100, 10)) -1
# self = AttentionModel(n_features=10,n_factors=3,dropout_p=0.1,d_embedding=6,heads=2, n_attention_blocks=1)
# self.forward(x)

# %%
class SVDModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        decoder_sizes: list,
        retrieve_factors: bool = False,
    ):

        super().__init__()
        self.retrieve_factors = retrieve_factors

        self.encoder = SVDEncoder(n_factors=n_factors)
        self.decoder = SimpleDecoder(
            n_factors=n_factors,
            n_features=n_features,
            internal_sizes=decoder_sizes,
        )

    def forward(self, x):
        factors = None

        # Apply model
        x = self.encoder(x)
        if self.retrieve_factors:
            factors = x.clone()
        x = self.decoder(x)

        return x, factors, None, None


# -----------------------------------------------------------------------------------------------
# %%
# train_dataset = ModelDataset(
#     filename="../04_results/kelly_chars_std=hist_n_groups=100.pq",
#     name="train",
#     val_pct=0.3
# )
# train_data_loader = data.DataLoader(train_dataset,batch_size=8,drop_last=True)
# %%
