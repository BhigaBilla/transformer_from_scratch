"""Training loop: AdamW, linear warmup + cosine decay, gradient clipping, fp16 mixed precision on GPU.

Writes <out_dir>/<run_name>/metrics.json (history + final validation loss) and model.pt.
A run whose metrics.json has "final" set is skipped, so a disconnected Colab can resume
a list of runs without redoing finished ones.
"""
import json
import math
import os
import time

import torch

from .config import ModelConfig, TrainConfig, to_dict
from .data import load_tokens, random_batch, sequential_batches
from .model import GPT


def lr_at(step, cfg: TrainConfig):
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, cfg: TrainConfig):
    # weight decay on matrices only, not on biases / LayerNorm gains
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, tokens, batch_size, context, device, max_batches=None, amp=False):
    model.eval()       # turns dropout off (like training=False in Keras)
    total, n = 0.0, 0
    for x, y in sequential_batches(tokens, batch_size, context, device, max_batches):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        total += loss.item() * x.numel()
        n += x.numel()
    model.train()
    return total / n


def train(run_name, model_cfg: ModelConfig, train_cfg: TrainConfig, data_dir, out_dir, device=None, log=print):
    run_dir = os.path.join(out_dir, run_name)
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            if json.load(f).get("final") is not None:
                log(f"{run_name}: already done, skipping")
                return None
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    torch.manual_seed(train_cfg.seed)
    data_gen = torch.Generator().manual_seed(train_cfg.seed)
    train_tokens = load_tokens(os.path.join(data_dir, "train.bin"))
    val_tokens = load_tokens(os.path.join(data_dir, "val.bin"))

    model = GPT(model_cfg).to(device)
    opt = make_optimizer(model, train_cfg)
    # GradScaler multiplies the loss so fp16 gradients do not underflow to zero
    scaler = torch.amp.GradScaler(enabled=amp)
    metrics = {"run": run_name, **to_dict(model_cfg, train_cfg), "params": model.num_params(),
               "history": [], "diverged": False, "final": None}
    log(f"{run_name}: {model.num_params():,} params on {device}")

    t0 = time.time()
    running = None
    for step in range(train_cfg.steps):
        lr = lr_at(step, train_cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = random_batch(train_tokens, train_cfg.batch_size, model_cfg.context, data_gen, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            metrics["diverged"] = True
            log(f"{run_name}: loss is {loss.item()} at step {step}, stopping")
            break
        # PyTorch training step by hand (what model.fit does inside Keras):
        opt.zero_grad(set_to_none=True)          # clear old gradients
        scaler.scale(loss).backward()            # backprop
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(opt)                         # update weights
        scaler.update()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()

        if (step + 1) % train_cfg.eval_every == 0 or step + 1 == train_cfg.steps:
            val = evaluate(model, val_tokens, train_cfg.batch_size, model_cfg.context, device,
                           train_cfg.eval_batches, amp)
            elapsed = time.time() - t0
            metrics["history"].append({"step": step + 1, "train_loss": running, "val_loss": val,
                                       "lr": lr, "time_s": elapsed})
            log(f"{run_name} step {step + 1}: train {running:.4f} val {val:.4f} ({elapsed:.0f}s)")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=1)

    elapsed = time.time() - t0
    if not metrics["diverged"]:
        full_val = evaluate(model, val_tokens, train_cfg.batch_size, model_cfg.context, device, None, amp)
        metrics["final"] = {"val_loss": full_val, "val_ppl": math.exp(full_val), "train_time_s": elapsed,
                            "tokens_per_s": train_cfg.steps * train_cfg.batch_size * model_cfg.context / elapsed}
        log(f"{run_name}: full validation loss {full_val:.4f} (ppl {math.exp(full_val):.2f})")
        torch.save({"model": model.state_dict(), "config": to_dict(model_cfg, train_cfg)},
                   os.path.join(run_dir, "model.pt"))
    else:
        metrics["final"] = {"val_loss": None, "diverged_at_step": step}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics
