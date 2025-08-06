from pprint import pprint
import random
from types import SimpleNamespace
from typing import Any, Dict, Optional

from easydict import EasyDict
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
        args = EasyDict(diff_params=EasyDict(**diff_params_cfg), exp=EasyDict(**exp_cfg))
        print("DIFF TRAINER ARGS")
        pprint(args)
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

    def _edm_inpaint(self, context: torch.Tensor, mask: torch.Tensor, steps: int = 30) -> torch.Tensor:
        """Sample the gap region using the EDM sampler."""
        device = context.device
        t = self.edm.create_schedule(steps).to(device)
        x = context + self.edm.sample_prior(context.shape, t[0])
        for i in range(steps - 1):
            t_i = t[i]
            t_next = t[i + 1]
            gamma = self.edm.get_gamma(t_i.unsqueeze(0)).to(device)
            if gamma.item() > 0:
                x = x + self.edm.sample_prior(x.shape, gamma * t_i)
            den = self.edm.denoiser(x, self.model, t_i, context=context, mask=mask)
            d = (x - den) / t_i
            x = x + (t_next - t_i) * d
        x = self.edm.denoiser(x, self.model, t[-1], context=context, mask=mask)
        return context * mask + x * (1 - mask)

    def _waveform_to_image(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a raw waveform to an RGB image tensor by plotting it with matplotlib.

        The returned tensor has shape (3, H, W) in the [0, 1] range, so it can be
        logged directly via ``self.writer.add_image``.
        """
        import io
        import matplotlib.pyplot as plt
        from PIL import Image
        import torchvision.transforms as transforms

        # Ensure NumPy array on CPU
        wav_np = waveform.detach().cpu().numpy()

        # Plot the waveform
        fig = plt.figure(figsize=(10, 2))
        plt.plot(wav_np)
        plt.axis("off")

        # Save the figure to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Convert the PNG buffer to a tensor
        img = Image.open(buf).convert("RGB")
        img_tensor = transforms.ToTensor()(img)  # (3, H, W), float32 in [0, 1]
        return img_tensor

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
            error, _ = self.edm.loss_fn(self.model, audio_noisy)
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
            for i, audio in enumerate(self.val_loader):
                audio = audio.to(self.device)
                audio_noisy, context, mask = self._create_noisy_input(audio)
                error, _ = self.edm.loss_fn(self.model, audio_noisy, mask=mask, context=context)
                total_loss += error.mean().item()

                if i == 0:
                    sampled = self._edm_inpaint(context, mask)
                    img = self._waveform_to_image(sampled[0].cpu())
                    self.writer.add_image("val/sample", img, self.current_epoch)
        avg = total_loss / max(1, len(self.val_loader))
        return {"loss": avg}
