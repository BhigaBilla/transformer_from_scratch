from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context: int = 256
    n_layer: int = 6
    d_model: int = 256
    n_head: int = 8
    d_ff: int = 1024           # hidden size of the feed-forward block (4 x d_model)
    dropout: float = 0.1
    pos: str = "rope"          # "rope" | "learned" | "sinusoidal" | "none"
    norm: str = "pre"          # "pre" (GPT-2 style) | "post" (original 2017 paper)
    tie_weights: bool = True   # share token embedding and output projection
    use_ffn: bool = True       # False = attention-only blocks


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 2500          # 2500 x 64 x 256 = ~41M tokens per run
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200    # 0 = no warmup
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50     # quick eval during training; full val set at the end
    seed: int = 0


def to_dict(model_cfg, train_cfg):
    return {"model": asdict(model_cfg), "train": asdict(train_cfg)}
