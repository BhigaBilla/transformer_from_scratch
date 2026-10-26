"""Generate stories from a trained run.

  python sample.py --run runs/baseline_s0 --data data --prompt "Once upon a time" -n 3
"""
import argparse
import os

import torch
from tokenizers import Tokenizer

from src.config import ModelConfig
from src.data import EOS
from src.model import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=250)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(a.run, "model.pt"), map_location=device)
    model = GPT(ModelConfig(**ckpt["config"]["model"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)

    torch.manual_seed(a.seed)
    for i in range(a.n):
        idx = torch.tensor([tok.encode(a.prompt).ids], device=device)
        out = model.generate(idx, a.max_tokens, a.temperature, top_k=50, eos_id=eos_id)[0].tolist()
        if eos_id in out:
            out = out[:out.index(eos_id)]
        print(f"--- sample {i + 1} ---\n{tok.decode(out)}\n")


if __name__ == "__main__":
    main()
