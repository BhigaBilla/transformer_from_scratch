"""A small GPT-style (decoder-only) transformer written from scratch.

Only basic PyTorch building blocks are used (nn.Linear, nn.Embedding, nn.LayerNorm,
nn.Dropout). Attention, positional encodings and the block structure are written here.

PyTorch notes for Keras users:
- nn.Module is like a Keras Layer/Model: layers are created in __init__, and
  forward() is like call().
- Tensors are (batch, time, channels) here; .view/.transpose reshape them like tf.reshape/tf.transpose.
- register_buffer stores a tensor that moves with the model to the GPU but is not trained.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def causal_attention(q, k, v, dropout_p=0.0, training=False):
    """Scaled dot-product attention with a causal mask.

    q, k, v: (batch, heads, time, head_dim). Position t may only attend to positions <= t.
    """
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))      # (B, H, T, T)
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    # softmax in float32 for stability under fp16 mixed precision
    weights = scores.float().softmax(dim=-1).to(q.dtype)
    if dropout_p > 0 and training:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v                                           # (B, H, T, head_dim)


def rope_tables(head_dim, context, base=10000.0):
    """cos/sin tables for rotary position embeddings (Su et al. 2021), shape (context, head_dim/2)."""
    inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.arange(context).float()[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs of channels (i, i + head_dim/2) by a position-dependent angle.

    x: (B, H, T, head_dim). Because both q and k are rotated, q.k depends only on
    the distance between the two positions.
    """
    T = x.size(-2)
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def sinusoidal_table(context, d_model):
    """Fixed sin/cos absolute position encodings from 'Attention Is All You Need'."""
    pos = torch.arange(context).float()[:, None]
    i = torch.arange(0, d_model, 2).float()
    angles = pos / (10000 ** (i / d_model))
    table = torch.zeros(context, d_model)
    table[:, 0::2] = angles.sin()
    table[:, 1::2] = angles.cos()
    return table


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.use_rope = cfg.pos == "rope"
        if self.use_rope:
            assert self.head_dim % 2 == 0
            cos, sin = rope_tables(self.head_dim, cfg.context)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = causal_attention(q, k, v, self.attn_dropout, self.training)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # concatenate heads
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = cfg.norm
        self.use_ffn = cfg.use_ffn
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        if cfg.use_ffn:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.ffn = FeedForward(cfg)

    def forward(self, x):
        if self.norm == "pre":
            # normalise the input of each sub-layer; the residual path stays un-normalised
            x = x + self.attn(self.ln1(x))
            if self.use_ffn:
                x = x + self.ffn(self.ln2(x))
        else:
            # original transformer: normalise after adding the residual
            x = self.ln1(x + self.attn(x))
            if self.use_ffn:
                x = self.ln2(x + self.ffn(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.pos in ("rope", "learned", "sinusoidal", "none")
        assert cfg.norm in ("pre", "post")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        elif cfg.pos == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_table(cfg.context, cfg.d_model), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # pre-LN needs a final norm because the residual stream is never normalised;
        # post-LN blocks already end with a LayerNorm
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.norm == "pre" else nn.Identity()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # GPT-2 init: shrink the projections that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # tied weights are counted once by .parameters()
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.context
        x = self.tok_emb(idx)
        if self.cfg.pos == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos == "sinusoidal":
            # as in the 2017 paper, scale embeddings by sqrt(d) so they are not swamped by the +-1 table
            x = x * math.sqrt(self.cfg.d_model) + self.pos_table[:T]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
