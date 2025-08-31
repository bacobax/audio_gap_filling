"""Waveform dataset for diffusion training.

This dataset shares the same gap detection and cropping logic as
``MelSpectrogramDataset`` but returns raw waveform segments for
training diffusion models.
"""

import random
from typing import Dict, Any
import numpy as np
import torch

from .mel_spectrogram_dataset import MelSpectrogramDataset


class GapWaveformDataset(MelSpectrogramDataset):
    """Dataset returning waveform slices instead of spectrograms.

    Performs a random 80/20 train/val split over valid_starts
    (windows that do NOT overlap the real gap), similar to VAEWaveformDataset
    but restricted to valid windows only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.split = config.get('split', 'train')
        super().__init__(config)
        # Number of samples corresponding to a crop of ``crop_frames``
        self.crop_samples = self.crop_frames * self.hop_length
        if self.wave.shape[0] < self.crop_samples:
            pad = self.crop_samples - self.wave.shape[0]
            self.wave = np.pad(self.wave, (0, pad), mode="constant")

        # Use only starts that do NOT overlap the detected gap
        valid = list(self.valid_starts)
        random.shuffle(valid)
        split_idx = int(0.8 * len(valid))
        self.train_starts = valid[:split_idx]
        self.val_starts = valid[split_idx:]

    def __len__(self) -> int:  # type: ignore[override]
        starts = self.train_starts if self.split == 'train' else self.val_starts
        step_interval = int(self.config.get("step_interval", 1))
        max_samples = 1500 * self.config["batch_size"] * max(1, step_interval)
        return min(len(starts), max_samples)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        starts = self.train_starts if self.split == 'train' else self.val_starts
        start = random.choice(starts)
        start_sample = start * self.hop_length
        end_sample = start_sample + self.crop_samples
        if end_sample > len(self.wave):
            pad = end_sample - len(self.wave)
            wave = np.pad(self.wave, (0, pad), mode="constant")
        else:
            wave = self.wave
        crop = wave[start_sample:end_sample]
        return torch.from_numpy(crop).float()
