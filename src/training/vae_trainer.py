"""Trainer for Variational Autoencoder (VAE) models."""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING
import torchaudio
from tqdm import tqdm

if TYPE_CHECKING:
    from lpips import LPIPS  # type: ignore

from ..core.base_trainer import BaseTrainer
from ..utils.metrics import AverageMeter
from ..models.VAE import VAE


class SimpleDiscriminator(nn.Module):
    """Light‑weight 1D discriminator used for adversarial loss."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), -1)


class VAETrainer(BaseTrainer):
    """Training loop for VAE models with perceptual and adversarial losses."""

    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.beta_kl = config.get("beta_kl", 1.0) if config else 1.0
        self.perceptual_loss = config.get("perceptual_loss", False) if config else False
        self.lambda_p = config.get("lambda_p", 0.0) if config else 0.0
        self.lambda_adv = config.get("lambda_adv", 0.0) if config else 0.0
        self.sample_rate = config.get("sample_rate", 16000) if config else 16000

        super().__init__(model, train_loader, val_loader, config, device, config.get("log_dir") if config else None)

        # Set up discriminator and perceptual loss
        self.discriminator = SimpleDiscriminator().to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.get("disc_learning_rate", 1e-4)) if config else optim.Adam(self.discriminator.parameters(), lr=1e-4)

        if self.perceptual_loss:
            from lpips import LPIPS  # type: ignore
            self.lpips_fn = LPIPS(net='vgg').to(self.device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=config.get("n_fft", 1024),
                hop_length=config.get("hop_length", 256),
                n_mels=config.get("n_mels", 80),
            ).to(self.device)
        else:
            self.lpips_fn = None
            self.mel_transform = None

    def _setup_training_components(self) -> None:
        lr = self.config.get("base_learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.bce = nn.BCEWithLogitsLoss()

    # --------------------------- training helpers ---------------------------
    def _compute_perceptual(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.perceptual_loss or self.lpips_fn is None or self.mel_transform is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            spec_recon = self.mel_transform(recon).unsqueeze(1)
            spec_target = self.mel_transform(target).unsqueeze(1)
        spec_recon = spec_recon.repeat(1, 3, 1, 1)
        spec_target = spec_target.repeat(1, 3, 1, 1)
        return self.lpips_fn(spec_recon, spec_target, normalize=True).mean()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = AverageMeter()
        recon_meter = AverageMeter()
        kl_meter = AverageMeter()
        p_meter = AverageMeter()
        adv_meter = AverageMeter()

        pbar = tqdm(
            total=len(self.train_loader),
            desc='Training VAE',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )

        for batch_idx, batch in enumerate(self.train_loader):
            wave = batch
            if isinstance(batch, (list, tuple)):
                wave = batch[0]
            wave = wave.to(self.device).unsqueeze(1)  # [B,1,T]

            recon, mu, logvar, _ = self.model(wave)

            recon_loss = F.mse_loss(recon, wave)
            kl_loss = self.model.kl_loss(mu, logvar, reduction="mean")
            p_loss = self._compute_perceptual(recon, wave)
            pred_fake = self.discriminator(recon)
            adv_loss = self.bce(pred_fake, torch.ones_like(pred_fake))

            loss = recon_loss + self.beta_kl * kl_loss + self.lambda_adv * adv_loss
            if self.perceptual_loss:
                loss = loss + self.lambda_p * p_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update discriminator
            with torch.no_grad():
                recon_detached = recon.detach()
            pred_real = self.discriminator(wave)
            pred_fake_det = self.discriminator(recon_detached)
            real_loss = self.bce(pred_real, torch.ones_like(pred_real))
            fake_loss = self.bce(pred_fake_det, torch.zeros_like(pred_fake_det))
            d_loss = 0.5 * (real_loss + fake_loss)
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

            total_loss.update(loss.item(), wave.size(0))
            recon_meter.update(recon_loss.item(), wave.size(0))
            kl_meter.update(kl_loss.item(), wave.size(0))
            adv_meter.update(adv_loss.item(), wave.size(0))
            if self.perceptual_loss:
                p_meter.update(p_loss.item(), wave.size(0))

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

            if batch_idx == 0:
                self.writer.add_audio('train/original', wave[0].squeeze(0), self.current_epoch, sample_rate=self.sample_rate)
                self.writer.add_audio('train/reconstruction', recon[0].squeeze(0), self.current_epoch, sample_rate=self.sample_rate)
                if self.mel_transform is not None:
                    spec = self.mel_transform(recon[0]).log2()[None]
                    self.writer.add_image('train/recon_spectrogram', spec, self.current_epoch, dataformats='CHW')

        pbar.close()

        metrics = {
            'loss': total_loss.avg,
            'recon_loss': recon_meter.avg,
            'kl_loss': kl_meter.avg,
            'adv_loss': adv_meter.avg,
        }
        if self.perceptual_loss:
            metrics['p_loss'] = p_meter.avg
        return metrics

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = AverageMeter()
        recon_meter = AverageMeter()
        kl_meter = AverageMeter()
        p_meter = AverageMeter()
        adv_meter = AverageMeter()

        pbar = tqdm(
            total=len(self.val_loader),
            desc='Validating VAE',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )

        for batch_idx, batch in enumerate(self.val_loader):
            wave = batch
            if isinstance(batch, (list, tuple)):
                wave = batch[0]
            wave = wave.to(self.device).unsqueeze(1)

            recon, mu, logvar, _ = self.model(wave)
            recon_loss = F.mse_loss(recon, wave)
            kl_loss = self.model.kl_loss(mu, logvar, reduction="mean")
            p_loss = self._compute_perceptual(recon, wave)
            pred_fake = self.discriminator(recon)
            adv_loss = self.bce(pred_fake, torch.ones_like(pred_fake))

            loss = recon_loss + self.beta_kl * kl_loss + self.lambda_adv * adv_loss
            if self.perceptual_loss:
                loss = loss + self.lambda_p * p_loss

            total_loss.update(loss.item(), wave.size(0))
            recon_meter.update(recon_loss.item(), wave.size(0))
            kl_meter.update(kl_loss.item(), wave.size(0))
            adv_meter.update(adv_loss.item(), wave.size(0))
            if self.perceptual_loss:
                p_meter.update(p_loss.item(), wave.size(0))

            if batch_idx == 0:
                self.writer.add_audio('val/original', wave[0].squeeze(0), self.current_epoch, sample_rate=self.sample_rate)
                self.writer.add_audio('val/reconstruction', recon[0].squeeze(0), self.current_epoch, sample_rate=self.sample_rate)
                if self.mel_transform is not None:
                    spec = self.mel_transform(recon[0]).log2()[None]
                    self.writer.add_image('val/recon_spectrogram', spec, self.current_epoch, dataformats='CHW')

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

        pbar.close()

        metrics = {
            'loss': total_loss.avg,
            'recon_loss': recon_meter.avg,
            'kl_loss': kl_meter.avg,
            'adv_loss': adv_meter.avg,
        }
        if self.perceptual_loss:
            metrics['p_loss'] = p_meter.avg
        return metrics
