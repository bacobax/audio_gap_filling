"""Dataset producing fixed-length waveform windows for VAE training.

This dataset is similar to ``GapWaveformDataset`` but it returns *all*
possible windows including those that cover the real gap.  It also
performs a random 80/20 split into train and validation subsets.
"""
import random
from typing import Dict, Any, List

import numpy as np
import torch

from .mel_spectrogram_dataset import MelSpectrogramDataset


class VAEWaveformDataset(MelSpectrogramDataset):
    """Return waveform crops for training a VAE."""

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        self.split = split
        super().__init__(config)

        # Number of waveform samples corresponding to one crop
        self.crop_samples = self.crop_frames * self.hop_length
        if len(self.wave) < self.crop_samples:
            pad = self.crop_samples - len(self.wave)
            self.wave = np.pad(self.wave, (0, pad), mode="constant")

        # Build complete list of possible start positions (allowing gap windows)
        all_starts: List[int] = list(range(self.num_frames - self.crop_frames + 1))
        random.shuffle(all_starts)
        split_idx = int(0.8 * len(all_starts))
        self.train_starts = all_starts[:split_idx]
        self.val_starts = all_starts[split_idx:]

    def __len__(self) -> int:  # type: ignore[override]
        starts = self.train_starts if self.split == "train" else self.val_starts
        max_samples = 1500 * self.config["batch_size"]
        return min(len(starts), max_samples)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        starts = self.train_starts if self.split == "train" else self.val_starts
        # Choose a random start each time to increase variety
        start = random.choice(starts)
        start_sample = start * self.hop_length
        end_sample = start_sample + self.crop_samples
        wave = self.wave
        if end_sample > len(wave):
            pad = end_sample - len(wave)
            wave = np.pad(wave, (0, pad), mode="constant")
        crop = wave[start_sample:end_sample]
        return torch.from_numpy(crop).float()
