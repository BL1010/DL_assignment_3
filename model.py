"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import gdown
import spacy


# ══════════════════════════════════════════════════════════════════════
#  SCALING DOT PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d_k = K.size(-1)

    scores = torch.matmul(
        Q,
        K.transpose(-2, -1),
    )  / math.sqrt(d_k)

    if mask is not None:

        mask = mask.to(torch.bool)

        scores = scores.masked_fill(
            mask,
            torch.finfo(scores.dtype).min,
        )

    attn_w = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASKS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    return (
        (src == pad_idx)
        .unsqueeze(1)
        .unsqueeze(2)
        .bool()
    )


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    B, seq_len = tgt.shape

    pad_mask = (
        (tgt == pad_idx)
        .unsqueeze(1)
        .unsqueeze(2)
    )

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=tgt.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    causal_mask = (
        causal_mask
        .unsqueeze(0)
        .unsqueeze(1)
    )

    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):

        B = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        seq_q = Q.size(1)
        seq_k = K.size(1)

        Q = Q.view(
            B,
            seq_q,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)

        K = K.view(
            B,
            seq_k,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)

        V = V.view(
            B,
            seq_k,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)

        out, attn_w = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
        )

        out = self.dropout(out)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, seq_q, self.d_model)
        )

        return self.Wo(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000,
        learned = False, 
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.learned = learned 
        
        

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
                0,
                max_len,
                dtype=torch.float,
            ).unsqueeze(1)

        div_term = torch.exp(
                torch.arange(
                    0,
                    d_model,
                    2,
                    dtype=torch.float,
                )
                * (-math.log(10000.0) / d_model)
            )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        
        seq_len = x.size(1) 
        
        
         
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn = self.mha(x, x, x, mask)

        x = self.norm1(
            x + self.dropout(attn)
        )

        ffn = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ffn)
        )

        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.ffn = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask,
    ):

        attn1 = self.self_attn(
            x,
            x,
            x,
            tgt_mask,
        )

        x = self.norm1(
            x + self.dropout(attn1)
        )

        attn2 = self.cross_attn(
            x,
            memory,
            memory,
            src_mask,
        )

        x = self.norm2(
            x + self.dropout(attn2)
        )

        ffn = self.ffn(x)

        x = self.norm3(
            x + self.dropout(ffn)
        )

        return x


# ══════════════════════════════════════════════════════════════════════
#  STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(layer)
            for _ in range(N)
        ])

        self.norm = nn.LayerNorm(
            layer.mha.d_model
        )

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(layer)
            for _ in range(N)
        ])

        self.norm = nn.LayerNorm(
            layer.self_attn.d_model
        )

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask,
    ):

        for layer in self.layers:
            x = layer(
                x,
                memory,
                src_mask,
                tgt_mask,
            )

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=7853,
        tgt_vocab_size=5893,
        d_model=128,
        N=2,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        load_weights=True,
        learned_positional_encoding=False,
        checkpoint_path="checkpoints/best_model.pt",
    ):
        super().__init__()

        self.d_model = d_model
        self.learned_positional_encoding = learned_positional_encoding

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            dropout,
            learned=learned_positional_encoding,
        )

        self.output_proj = nn.Linear(
            d_model,
            tgt_vocab_size,
        )

        enc_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
        )

        dec_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
        )

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        try:
            self.spacy_de = spacy.load(
                "de_core_news_sm"
            )
        except:
            self.spacy_de = spacy.blank("de")

        src_vocab_path = "vocab/src_vocab.pkl"
        tgt_vocab_path = "vocab/tgt_vocab.pkl"

        if (
            os.path.exists(src_vocab_path)
            and os.path.exists(tgt_vocab_path)
        ):

            with open(src_vocab_path, "rb") as f:
                self.src_vocab = pickle.load(f)

            with open(tgt_vocab_path, "rb") as f:
                self.tgt_vocab = pickle.load(f)

        else:

            self.src_vocab = None
            self.tgt_vocab = None

        self._reset_parameters()

        if (
            load_weights
            and os.path.exists(checkpoint_path)
        ):

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device_name,
            )

            if "model_state_dict" in checkpoint:

                self.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            else:

                self.load_state_dict(checkpoint)

            print(
                f"Loaded weights from {checkpoint_path}"
            )

    # ───────────────────────────────────────────────

    def _reset_parameters(self):

        for p in self.parameters():

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ───────────────────────────────────────────────

    def encode(self, src, src_mask):

        x = (
            self.src_embedding(src)
            * math.sqrt(self.d_model)
        )

        x = self.pos_enc(x)

        return self.encoder(x, src_mask)

    # ───────────────────────────────────────────────

    def decode(
        self,
        memory,
        src_mask,
        tgt,
        tgt_mask,
    ):

        x = (
            self.tgt_embedding(tgt)
            * math.sqrt(self.d_model)
        )

        x = self.pos_enc(x)

        x = self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask,
        )

        return self.output_proj(x)

    # ───────────────────────────────────────────────

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
    ):

        memory = self.encode(
            src,
            src_mask,
        )

        return self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask,
        )

    # ───────────────────────────────────────────────
    # INFERENCE
    # ───────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        german_sentence,
        max_len=100,
    ):

        self.eval()

        tokens = [
            tok.text.lower()
            for tok in self.spacy_de.tokenizer(
                german_sentence
            )
        ]

        src_indices = [self.sos_idx]

        for token in tokens:

            src_indices.append(
                self.src_vocab.stoi.get(
                    token,
                    self.src_vocab.stoi["<unk>"],
                )
            )

        src_indices.append(self.eos_idx)

        src_tensor = torch.tensor(
            [src_indices],
            dtype=torch.long,
            device=self.device_name,
        )

        src_mask = make_src_mask(
            src_tensor,
            self.pad_idx,
        ).to(self.device_name)

        memory = self.encode(
            src_tensor,
            src_mask,
        )

        ys = torch.tensor(
            [[self.sos_idx]],
            dtype=torch.long,
            device=self.device_name,
        )

        for _ in range(max_len):

            tgt_mask = make_tgt_mask(
                ys,
                self.pad_idx,
            ).to(self.device_name)

            out = self.decode(
                memory,
                src_mask,
                ys,
                tgt_mask,
            )

            next_token = torch.argmax(
                out[:, -1],
                dim=-1,
            ).item()

            ys = torch.cat(
                [
                    ys,
                    torch.tensor(
                        [[next_token]],
                        dtype=torch.long,
                        device=self.device_name,
                    ),
                ],
                dim=1,
            )

            if next_token == self.eos_idx:
                break

        output_tokens = ys.squeeze(0).tolist()

        translated_tokens = []

        for idx in output_tokens:

            if idx in [
                self.pad_idx,
                self.sos_idx,
                self.eos_idx,
            ]:
                continue

            translated_tokens.append(
                self.tgt_vocab.itos[idx]
            )

        sentence = " ".join(translated_tokens)

        sentence = sentence.replace(" .", ".")
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" !", "!")
        sentence = sentence.replace(" ?", "?")

        return sentence