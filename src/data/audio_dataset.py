import os
import random
from typing import Any, Dict

import torch
import torchaudio

from ..core.base_dataset import BaseDataset


class AudioFolderDataset(BaseDataset):
    """Simple dataset loading audio files from a folder."""

    def _validate_config(self) -> None:
        required = ["folder", "sample_rate", "segment_length"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        if not os.path.isdir(self.config["folder"]):
            raise FileNotFoundError(self.config["folder"])

    def _setup_dataset(self) -> None:
        self.folder = self.config["folder"]
        self.sample_rate = self.config["sample_rate"]
        self.segment_length = self.config["segment_length"]
        exts = self.config.get("extensions", [".wav", ".flac"])
        self.files = [
            os.path.join(self.folder, f)
            for f in sorted(os.listdir(self.folder))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if len(self.files) == 0:
            raise ValueError(f"No audio files found in {self.folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx % len(self.files)]
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(dim=0)  # mono
        if audio.shape[0] >= self.segment_length:
            start = random.randint(0, audio.shape[0] - self.segment_length)
            audio = audio[start : start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.shape[0]))
        return audio
