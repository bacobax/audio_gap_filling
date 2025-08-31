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
    """Dataset returning waveform slices instead of spectrograms."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Number of samples corresponding to a crop of ``crop_frames``
        self.crop_samples = self.crop_frames * self.hop_length
        if self.wave.shape[0] < self.crop_samples:
            pad = self.crop_samples - self.wave.shape[0]
            self.wave = np.pad(self.wave, (0, pad), mode="constant")

    def __len__(self) -> int:  # type: ignore[override]
        starts = len(self.valid_starts)
        max_samples = 1500 * self.config["batch_size"]
        return min(starts, max_samples)


    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        start = random.choice(self.valid_starts)
        start_sample = start * self.hop_length
        end_sample = start_sample + self.crop_samples
        if end_sample > len(self.wave):
            pad = end_sample - len(self.wave)
            wave = np.pad(self.wave, (0, pad), mode="constant")
        else:
            wave = self.wave
        crop = wave[start_sample:end_sample]
        return torch.from_numpy(crop).float()
