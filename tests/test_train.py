import json

import numpy as np

from src.config import ModelConfig, TrainConfig
from src.train import lr_at, train
from tests.test_model import SMALL


def test_lr_schedule():
    cfg = TrainConfig(steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4)
    assert lr_at(0, cfg) == 1e-3 / 100
    assert abs(lr_at(99, cfg) - 1e-3) < 1e-12
    assert abs(lr_at(1000, cfg) - 1e-4) < 1e-12
    assert lr_at(0, TrainConfig(warmup_steps=0)) == TrainConfig().lr


def test_train_writes_metrics_and_skips_when_done(tmp_path):
    rng = np.random.default_rng(0)
    # repeating pattern so the loss can actually go down
    pattern = rng.integers(0, 64, 50).astype(np.uint16)
    np.tile(pattern, 200).tofile(tmp_path / "train.bin")
    np.tile(pattern, 20).tofile(tmp_path / "val.bin")
    mc = ModelConfig(**SMALL)
    tc = TrainConfig(batch_size=8, steps=60, eval_every=30, eval_batches=2, warmup_steps=5, lr=3e-3)

    m = train("smoke", mc, tc, str(tmp_path), str(tmp_path / "runs"), device="cpu", log=lambda s: None)
    assert m["final"]["val_loss"] < m["history"][0]["val_loss"]
    saved = json.loads((tmp_path / "runs" / "smoke" / "metrics.json").read_text())
    assert saved["final"]["val_loss"] == m["final"]["val_loss"]
    assert (tmp_path / "runs" / "smoke" / "model.pt").exists()
    assert train("smoke", mc, tc, str(tmp_path), str(tmp_path / "runs"), device="cpu", log=lambda s: None) is None
