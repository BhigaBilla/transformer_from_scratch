"""TinyStories download, BPE tokenizer training and token files.

prepare() writes to data_dir:
  tokenizer.json   byte-level BPE, 4,096 tokens, trained on train stories only
  train.bin        uint16 token ids of the first N training stories
  val.bin          uint16 token ids of the full official validation split
Stories are separated by the <|endoftext|> token.
"""
import os

import numpy as np
import torch

EOS = "<|endoftext|>"


def train_tokenizer(texts, vocab_size=4096):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_to_bin(tok, texts, path, batch=10_000):
    eos_id = tok.token_to_id(EOS)
    chunks = []
    for i in range(0, len(texts), batch):
        for enc in tok.encode_batch(texts[i:i + batch]):
            chunks.append(np.array(enc.ids + [eos_id], dtype=np.uint16))
    arr = np.concatenate(chunks)
    arr.tofile(path)
    return len(arr)


def prepare(data_dir, n_train_stories=600_000, n_tokenizer_stories=200_000, vocab_size=4096):
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    train = [t.strip() for t in ds["train"].select(range(n_train_stories))["text"]]
    val = [t.strip() for t in ds["validation"]["text"]]

    tok = train_tokenizer(train[:n_tokenizer_stories], vocab_size)
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    n_tr = encode_to_bin(tok, train, os.path.join(data_dir, "train.bin"))
    n_va = encode_to_bin(tok, val, os.path.join(data_dir, "val.bin"))
    print(f"train: {len(train):,} stories, {n_tr:,} tokens; val: {len(val):,} stories, {n_va:,} tokens")


def load_tokens(path):
    # memmap reads from disk on demand instead of loading the whole file into RAM
    return np.memmap(path, dtype=np.uint16, mode="r")


def random_batch(tokens, batch_size, context, gen, device):
    """Random windows; y is x shifted by one token (next-token targets)."""
    ix = torch.randint(len(tokens) - context - 1, (batch_size,), generator=gen).tolist()
    x = torch.stack([torch.from_numpy(tokens[i:i + context].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + context].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def sequential_batches(tokens, batch_size, context, device, max_batches=None):
    """Non-overlapping windows from the start of the file: the same tokens for every run."""
    n_windows = (len(tokens) - 1) // context
    starts = list(range(0, n_windows * context, context))
    if max_batches is not None:
        starts = starts[:max_batches * batch_size]
    for b in range(0, len(starts), batch_size):
        ix = starts[b:b + batch_size]
        x = torch.stack([torch.from_numpy(tokens[i:i + context].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + context].astype(np.int64)) for i in ix])
        yield x.to(device), y.to(device)
