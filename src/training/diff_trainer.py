import random
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.base_trainer import BaseTrainer
from .edm import EDM


class DiffusionTrainer(BaseTrainer):
    """Minimal diffusion trainer using EDM loss."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or {}

        diff_params_cfg = self.config.get("diff_params", {})
        exp_cfg = {"sample_rate": self.config.get("sample_rate", 16000)}
        args = SimpleNamespace(diff_params=SimpleNamespace(**diff_params_cfg), exp=SimpleNamespace(**exp_cfg))
        self.edm = EDM(args)

        self.gap_min = self.config.get("gap_min_size", 2048)
        self.gap_max = self.config.get("gap_max_size", 8192)
        self.gap_num = self.config.get("gap_num", 1)
        super().__init__(model, train_loader, val_loader, self.config, device, self.config.get("log_dir"))

    def _setup_training_components(self) -> None:
        lr = self.config.get("lr", 2e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = None
        self.criterion = nn.MSELoss()

    def _create_noisy_input(self, audio: torch.Tensor):
        B, T = audio.shape
        mask = torch.ones_like(audio)
        for b in range(B):
            for _ in range(self.gap_num):
                gap_size = random.randint(self.gap_min, self.gap_max)
                start = random.randint(0, T - gap_size)
                end = start + gap_size
                mask[b, start:end] = 0
        context = audio * mask
        noise = torch.randn_like(audio)
        audio_noisy = context + noise * (1 - mask)
        return audio_noisy, context, mask

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        for audio in self.train_loader:
            audio = audio.to(self.device)
            audio_noisy, context, mask = self._create_noisy_input(audio)
            self.optimizer.zero_grad()
            error, _ = self.edm.loss_fn(self.model, audio_noisy, mask=mask, context=context)
            loss = error.mean()
            loss.backward()
            if self.config.get("use_grad_clip", False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
        avg = total_loss / max(1, len(self.train_loader))
        return {"loss": avg}

    def _validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"loss": 0.0}
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for audio in self.val_loader:
                audio = audio.to(self.device)
                audio_noisy, context, mask = self._create_noisy_input(audio)
                error, _ = self.edm.loss_fn(self.model, audio_noisy, mask=mask, context=context)
                total_loss += error.mean().item()
        avg = total_loss / max(1, len(self.val_loader))
        return {"loss": avg}
