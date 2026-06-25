"""Memmap uint16 shard dataset, nanoGPT-style.

Data layout (produced upstream by a tokenizer pre-pass):
  <data_dir>/train/*.bin   concatenated token-id streams (np.uint16), EOS(=2)
  <data_dir>/val/*.bin     between documents.

Each .bin is `np.array(token_ids, dtype=np.uint16).tofile(path)`. We memmap them
(no full load into RAM) and sample random contiguous blocks of length block_size.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

UINT16_BYTES = 2
EOS_ID = 2  # document separator, per the tokenizer spec (unk=0 bos=1 eos=2 pad=3)


class ShardedBinDataset:
    """Random-block sampler over a directory of uint16 .bin shards for one split.

    Treats all shards in <data_dir>/<split>/ as a pool. Each draw picks a shard
    (weighted by length so sampling is uniform over tokens) and a random start
    offset, returning a (block_size + 1)-token window for next-token prediction.
    """

    def __init__(self, data_dir: str, split: str, block_size: int):
        split_dir = os.path.join(data_dir, split)
        paths = sorted(glob.glob(os.path.join(split_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(
                f"No .bin shards found in {split_dir!r}. Expected uint16 token files."
            )
        self.block_size = block_size
        self.paths = paths

        # Lazily-opened memmaps; lengths (in tokens) computed from file size.
        self._memmaps: list[np.memmap | None] = [None] * len(paths)
        self.lengths = np.array(
            [os.path.getsize(p) // UINT16_BYTES for p in paths], dtype=np.int64
        )
        # Each shard must hold at least one full window.
        usable = self.lengths - (block_size + 1)
        if not (usable >= 0).any():
            raise ValueError(
                f"Every shard in {split_dir!r} is shorter than block_size+1 "
                f"({block_size + 1} tokens)."
            )
        self._usable = np.clip(usable, 0, None)
        total = self._usable.sum()
        self._weights = self._usable / total if total > 0 else None
        self.total_tokens = int(self.lengths.sum())

    def _get_memmap(self, i: int) -> np.memmap:
        # Reopen per access pattern is wasteful; cache one handle per shard.
        if self._memmaps[i] is None:
            self._memmaps[i] = np.memmap(self.paths[i], dtype=np.uint16, mode="r")
        return self._memmaps[i]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Return one (block_size + 1,) int64 window."""
        shard_idx = int(rng.choice(len(self.paths), p=self._weights))
        mm = self._get_memmap(shard_idx)
        max_start = int(self._usable[shard_idx])
        start = int(rng.integers(0, max_start + 1))
        chunk = mm[start : start + self.block_size + 1]
        return chunk.astype(np.int64)


class BatchLoader:
    """Builds (x, y) batches on the target device for train/val splits."""

    def __init__(self, data_dir: str, block_size: int, batch_size: int,
                 device: str = "cuda", seed: int = 1337):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        # Independent RNGs so eval draws never perturb the training data sequence
        # (keeps the train trajectory reproducible regardless of eval cadence).
        self._rngs = {
            "train": np.random.default_rng(seed),
            "val": np.random.default_rng(seed + 1),
        }

        self.datasets: dict[str, ShardedBinDataset] = {}
        for split in ("train", "val"):
            split_dir = os.path.join(data_dir, split)
            if os.path.isdir(split_dir):
                self.datasets[split] = ShardedBinDataset(data_dir, split, block_size)
        if "train" not in self.datasets:
            raise FileNotFoundError(f"No train/ split under {data_dir!r}")

    def has_split(self, split: str) -> bool:
        return split in self.datasets

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) of shape (batch_size, block_size) on the device."""
        ds = self.datasets[split]
        rng = self._rngs[split]
        windows = np.stack([ds.sample(rng) for _ in range(self.batch_size)])
        x = torch.from_numpy(windows[:, :-1])
        y = torch.from_numpy(windows[:, 1:])

        if self.device_type == "cuda":
            # Pinned + non_blocking async copy to overlap H2D with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
