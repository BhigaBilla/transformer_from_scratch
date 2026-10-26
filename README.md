# Transformer from scratch: which design choices matter?

A small GPT-style (decoder-only) transformer written from scratch in PyTorch and trained
on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) using free
Colab GPUs. Controlled ablations then test which parts of the architecture actually matter
at this scale: positional encoding, where LayerNorm goes, LR warmup, number of heads,
depth vs width, weight tying and the feed-forward block.

> Status: code and tests done; training runs in progress. Results below will be filled
> in with measured numbers only.

## What is written by hand

`src/model.py` uses only basic PyTorch layers (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`).
Written here:

- multi-head causal self-attention (scores, mask, softmax, head split/merge)
- positional encodings: rotary (RoPE), learned absolute, fixed sinusoidal, or none
- pre-LN (GPT-2) and post-LN (original 2017 paper) blocks
- GPT-2 style init, weight tying, sampling with temperature and top-k
- training loop: AdamW, warmup + cosine LR, gradient clipping, fp16 mixed precision

The tokenizer is a 4,096-token byte-level BPE trained with the HuggingFace `tokenizers`
library on the training stories only (not the point of the project, so not hand-written).

## Correctness checks (`pytest`)

- hand-written attention equals `F.scaled_dot_product_attention` (to 1e-5)
- no future leakage: changing token t leaves outputs before t unchanged, for all 8
  position x norm variants
- RoPE: the q·k score depends only on relative distance
- the model overfits a single batch; training loop writes metrics and resumes correctly

## Setup

| | |
|---|---|
| Baseline | 6 layers, d_model 256, 8 heads, FFN 1024, context 256, pre-LN, RoPE, tied embeddings, dropout 0.1 |
| Parameters | 5.79M (4.74M excluding embeddings) |
| Training | 2,500 steps x 64 x 256 = ~41M tokens per run; AdamW (0.9, 0.95), lr 1e-3 -> 1e-4 cosine, 200 warmup steps, wd 0.1, clip 1.0 |
| Data | first 600k TinyStories training stories; metric = loss on the full official validation split |
| Hardware | Colab free T4 |
| Seeds | 2 per configuration; differences smaller than the seed spread are treated as noise |

## Ablations

| group | configurations |
|---|---|
| positional encoding | none, learned, sinusoidal, RoPE (baseline) |
| normalisation | pre-LN vs post-LN, each with and without LR warmup |
| heads (d_model fixed at 256) | 1, 4, 8 (baseline), 16 |
| depth vs width (~equal non-embedding params) | 2 x 448, 6 x 256 (baseline), 12 x 176 |
| extra | untied embeddings, no feed-forward block |

## Results

To be filled in from `results.md` after the runs.

## Run it

```bash
pip install -r requirements.txt
pytest
python run.py list                                    # configs and parameter counts
python run.py prepare --data data                     # download + tokenize (~10-15 min)
python run.py train --data data --out runs --group baseline pos --seeds 0 1
python analyze.py --out runs                          # results.md + plots/
python sample.py --run runs/baseline_s0 --data data
```

Or open `colab.ipynb` in Colab with a T4 runtime; it keeps data and results on Google Drive.

## Limitations

- One small dataset and one model scale (~6M params, ~41M tokens). Findings may not
  transfer to large models.
- Two seeds per configuration, so only fairly large differences can be called real.
- Same hyperparameters (learning rate etc.) for every ablation; some variants might do
  better with their own tuning.
