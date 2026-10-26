import itertools

import pytest
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import GPT, apply_rope, causal_attention, rope_tables

SMALL = dict(vocab_size=64, context=32, n_layer=2, d_model=32, n_head=4, d_ff=64, dropout=0.0)
VARIANTS = list(itertools.product(["rope", "learned", "sinusoidal", "none"], ["pre", "post"]))


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    mine = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


@pytest.mark.parametrize("pos,norm", VARIANTS)
def test_no_future_leak(pos, norm):
    """Changing token t must not change the outputs at positions < t."""
    torch.manual_seed(0)
    model = GPT(ModelConfig(**SMALL, pos=pos, norm=norm)).eval()
    x = torch.randint(0, 64, (1, 20))
    x2 = x.clone()
    x2[0, 12] = (x[0, 12] + 1) % 64
    out1, _ = model(x)
    out2, _ = model(x2)
    assert torch.allclose(out1[0, :12], out2[0, :12], atol=1e-6)
    assert not torch.allclose(out1[0, 12:], out2[0, 12:])


def test_rope_is_relative():
    """After RoPE, q.k depends only on the distance between positions."""
    torch.manual_seed(0)
    cos, sin = rope_tables(16, 64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        seq_q = torch.zeros(1, 1, 64, 16)
        seq_k = torch.zeros(1, 1, 64, 16)
        seq_q[0, 0, m], seq_k[0, 0, n] = q, k
        rq, rk = apply_rope(seq_q, cos, sin), apply_rope(seq_k, cos, sin)
        return (rq[0, 0, m] * rk[0, 0, n]).sum()

    assert torch.allclose(score(10, 3), score(40, 33), atol=1e-4)
    assert not torch.allclose(score(10, 3), score(10, 8), atol=1e-4)


def test_no_position_info_without_pos_is_still_causal_only():
    """With pos='none' the first output depends only on the first token."""
    torch.manual_seed(0)
    model = GPT(ModelConfig(**SMALL, pos="none")).eval()
    a, _ = model(torch.tensor([[5, 1, 2]]))
    b, _ = model(torch.tensor([[5, 9, 9]]))
    assert torch.allclose(a[0, 0], b[0, 0], atol=1e-6)


def test_tied_weights():
    model = GPT(ModelConfig(**SMALL))
    assert model.head.weight is model.tok_emb.weight
    untied = GPT(ModelConfig(**SMALL, tie_weights=False))
    assert untied.num_params() == model.num_params() + 64 * 32


@pytest.mark.parametrize("pos,norm", [("rope", "pre"), ("sinusoidal", "post")])
def test_overfits_one_batch(pos, norm):
    torch.manual_seed(0)
    model = GPT(ModelConfig(**SMALL, pos=pos, norm=norm))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 64, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_shape():
    torch.manual_seed(0)
    model = GPT(ModelConfig(**SMALL)).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)    # longer than context: generation crops the window
