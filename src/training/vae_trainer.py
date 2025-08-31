"""Trainer for Variational Autoencoder (VAE) models."""
import os
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING
import torchaudio
from tqdm import tqdm

from lpips import LPIPS  # type: ignore

from ..core.base_trainer import BaseTrainer
from ..utils.metrics import AverageMeter
from ..models.VAE import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DiscriminatorWithFeatures(nn.Module):
    """Convolutional discriminator that returns intermediate features."""

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channel_sizes = [32, 64, 128, 256]
        last_channels = in_channels
        for ch in channel_sizes:
            layers.append(nn.Conv1d(last_channels, ch, 15, stride=2, padding=7))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            last_channels = ch
        layers.append(nn.Conv1d(last_channels, 1, 3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        out = self.layers[-1](x)
        out = out.mean(dim=-1)
        return out, feats


class MultiScaleDiscriminator(nn.Module):
    """Stack of discriminators operating at multiple time scales."""

    def __init__(self, n_discriminators: int = 5, in_channels: int = 2) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorWithFeatures(in_channels) for _ in range(n_discriminators)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores: List[torch.Tensor] = []
        features: List[List[torch.Tensor]] = []
        for disc in self.discriminators:
            s, f = disc(x)
            scores.append(s)
            features.append(f)
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return scores, features


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=None, hop_sizes=None, win_lengths=None):
        super().__init__()
        self.fft_sizes = fft_sizes or [512, 1024, 2048]
        self.hop_sizes = hop_sizes or [50, 120, 240]
        self.win_lengths = win_lengths or [240, 600, 1200]
        # Anchor buffer to track current device and a cache for Hann windows
        self.register_buffer('_dummy', torch.tensor(0.), persistent=False)
        self._windows: Dict[int, torch.Tensor] = {}

    def _get_window(self, win: int) -> torch.Tensor:
        """Return a cached Hann window of length `win` on the module's current device."""
        dev = DEVICE
        w = self._windows.get(win)
        if w is None or w.device != dev:
            w = torch.hann_window(win, device=dev)
            self._windows[win] = w
        return w

    def _stft(self, x, fft, hop, win):
        window = self._get_window(win)
        return torch.stft(
            x, n_fft=fft, hop_length=hop, win_length=win,
            window=window, return_complex=True
        ).abs()

    def forward(self, x, y):
        # x,y: [B, 1, T]
        loss = 0.0
        x_ = x.squeeze(1)
        y_ = y.squeeze(1)
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            X = self._stft(x_, fft, hop, win)
            Y = self._stft(y_, fft, hop, win)
            loss += F.l1_loss(X, Y)
        return loss / len(self.fft_sizes)

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
        self.beta_kl = config.get("beta_kl", 1e-4) if config else 1e-4
        self.lambda_adv = config.get("lambda_adv", 0.0) if config else 0.0
        self.lambda_fm = config.get("lambda_fm", 0.0) if config else 0.0
        self.sample_rate = config.get("sample_rate", 44100) if config else 44100
        self.freeze_encoder_epoch = config.get("freeze_encoder_epoch") if config else None
        self.decoder_lr = config.get("decoder_learning_rate", 1.5e-4) if config else 1.5e-4
        self.step_interval = config.get("step_interval", 1) if config else 1
        self.grad_clip = (config.get("grad_clip", 1.0) if config else 1.0)

        super().__init__(model, train_loader, val_loader, config, device, config.get("log_dir") if config else None)

        # Set up discriminator and reconstruction loss
        in_channels = getattr(model.encoder.initial_conv, 'in_channels', 2)
        self.discriminator = MultiScaleDiscriminator(in_channels=in_channels).to(self.device)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get("disc_learning_rate", 3e-4) if config else 3e-4,
        )
        self.scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        mrstft_cfg = config.get("mrstft", {}) if config else {}
        self.recon_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=mrstft_cfg.get("fft_sizes"),
            hop_sizes=mrstft_cfg.get("hop_sizes"),
            win_lengths=mrstft_cfg.get("win_lengths"),
        )

        self.mel_transform_cpu = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.config.get("n_fft", 1024),
            hop_length=self.config.get("hop_length", 256),
            n_mels=self.config.get("n_mels", 80),
        )  # stays on CPU

        # Optional perceptual loss (disabled by default)
        self.perceptual_loss = config.get("perceptual_loss", False) if config else False
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
            self.lambda_p = config.get("lambda_p", 0.0)
            self.lambda_p = config.get("lambda_p", 0.0)
        else:
            self.lpips_fn = None
            self.mel_transform = None
            self.lambda_p = 0.0
        join_path = os.path.join(self.log_dir, "best_model.pt")
        print(f"join path: {join_path}")
        self.checkpoint_path = self.config.get("checkpoint_path", join_path)
        if self.config.get("resume", False):
            print(f"CHECKPOINT path: {self.checkpoint_path}")
            self.current_epoch = self.load_checkpoint(self.checkpoint_path)

    def _setup_training_components(self) -> None:
        lr = self.config.get("base_learning_rate", 1.5e-4)
        weight_decay = self.config.get("weight_decay", 0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.bce = nn.BCEWithLogitsLoss()

    def _extra_state_to_save(self) -> Dict[str, Any]:
        """Extend BaseTrainer checkpoint with discriminator and its optimizer/scheduler."""
        return {
            'discriminator_state_dict': self.discriminator.state_dict() if hasattr(self, 'discriminator') else None,
            'optimizer_d_state_dict': self.optimizer_d.state_dict() if hasattr(self, 'optimizer_d') and self.optimizer_d is not None else None,
            'scheduler_d_state_dict': self.scheduler_d.state_dict() if hasattr(self, 'scheduler_d') and self.scheduler_d is not None else None,
        }

    def _load_extra_state(self, checkpoint: Dict[str, Any]) -> None:
        """Restore discriminator and its optimizer/scheduler from checkpoint if present."""
        disc_sd = checkpoint.get('discriminator_state_dict')
        if disc_sd is not None and hasattr(self, 'discriminator'):
            try:
                self.discriminator.load_state_dict(disc_sd)
            except Exception:
                pass
        optd_sd = checkpoint.get('optimizer_d_state_dict')
        if optd_sd is not None and hasattr(self, 'optimizer_d') and self.optimizer_d is not None:
            try:
                self.optimizer_d.load_state_dict(optd_sd)
            except Exception:
                pass
        sched_d_sd = checkpoint.get('scheduler_d_state_dict')
        if sched_d_sd is not None and hasattr(self, 'scheduler_d') and self.scheduler_d is not None:
            try:
                self.scheduler_d.load_state_dict(sched_d_sd)
            except Exception:
                pass

    # --------------------------- training helpers ---------------------------
    def _compute_perceptual(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.perceptual_loss or self.lpips_fn is None or self.mel_transform is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            spec_recon = self.mel_transform(recon).unsqueeze(1)
            spec_target = self.mel_transform(target).unsqueeze(1)

        spec_recon = spec_recon.squeeze(1).repeat(1, 3, 1, 1)
        spec_target = spec_target.squeeze(1).repeat(1, 3, 1, 1)
        return self.lpips_fn(spec_recon, spec_target, normalize=True).mean()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()

        # Freeze encoder if scheduled
        if self.freeze_encoder_epoch is not None and self.current_epoch == self.freeze_encoder_epoch:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.decoder_lr,
                weight_decay=self.config.get("weight_decay", 0.0),
            )
            if self.config.get('decoder_batch_size') is not None:
                self.train_loader = DataLoader(
                    self.train_loader.dataset,
                    batch_size=self.config['decoder_batch_size'],
                    shuffle=True,
                    num_workers=self.train_loader.num_workers,
                )

        total_loss = AverageMeter()
        recon_meter = AverageMeter()
        kl_meter = AverageMeter()
        p_meter = AverageMeter()
        adv_meter = AverageMeter()
        fm_meter = AverageMeter()

        pbar = tqdm(
            total=len(self.train_loader),
            desc='Training VAE',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_d.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            wave = batch
            if isinstance(batch, (list, tuple)):
                wave = batch[0]
            wave = wave.to(self.device)
            if wave.dim() == 2:
                wave = wave.unsqueeze(1)

            recon, mu, logvar, _ = self.model(wave)

            recon_loss = self.recon_loss_fn(recon, wave)
            kl_loss = self.model.kl_loss(mu, logvar, reduction="mean")
            p_loss = self._compute_perceptual(recon, wave)

            # Freeze discriminator parameters for generator update
            for p in self.discriminator.parameters():
                p.requires_grad_(False)
            pred_fake, feats_fake = self.discriminator(recon)
            for p in self.discriminator.parameters():
                p.requires_grad_(True)

            with torch.no_grad():
                _, feats_real = self.discriminator(wave)

            adv_loss = sum(self.bce(p, torch.ones_like(p)) for p in pred_fake) / len(pred_fake)
            fm_loss = 0.0
            for fr, ff in zip(feats_real, feats_fake):
                for r, f in zip(fr, ff):
                    fm_loss += F.l1_loss(f, r)
            fm_loss = fm_loss / len(feats_real)

            loss = recon_loss + self.beta_kl * kl_loss + self.lambda_adv * adv_loss + self.lambda_fm * fm_loss
            if self.perceptual_loss:
                loss = loss + self.lambda_p * p_loss

            (loss / self.step_interval).backward()

            # Update discriminator
            with torch.no_grad():
                recon_detached = recon.detach()
            pred_real_det, _ = self.discriminator(wave)
            pred_fake_det, _ = self.discriminator(recon_detached)
            real_loss = sum(self.bce(pr, torch.ones_like(pr)) for pr in pred_real_det) / len(pred_real_det)
            fake_loss = sum(self.bce(pf, torch.zeros_like(pf)) for pf in pred_fake_det) / len(pred_fake_det)
            d_loss = 0.5 * (real_loss + fake_loss)
            (d_loss / self.step_interval).backward()

            if (batch_idx + 1) % self.step_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                # Optional gradient clipping to stabilize/limit grad memory
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_d.step()
                self.optimizer_d.zero_grad(set_to_none=True)

            total_loss.update(loss.item(), wave.size(0))
            recon_meter.update(recon_loss.item(), wave.size(0))
            kl_meter.update(kl_loss.item(), wave.size(0))
            adv_meter.update(adv_loss.item(), wave.size(0))
            fm_meter.update(fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss, wave.size(0))
            if self.perceptual_loss:
                p_meter.update(p_loss.item(), wave.size(0))

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1

            if batch_idx == 0:
                with torch.no_grad():
                    print(f"wave shape: {wave.shape}, recon shape: {recon.shape}")
                    self.writer.add_audio('train/original', wave[0, 0].detach().float().cpu(), self.current_epoch, sample_rate=self.sample_rate)
                    self.writer.add_audio('train/reconstruction', recon[0, 0].detach().float().cpu(),self.current_epoch, sample_rate=self.sample_rate)
                    if self.mel_transform is not None:
                        # Use a CPU transform for logging (see #2)
                        spec = self.mel_transform_cpu(recon[0, 0].detach().cpu()).clamp_min(1e-9).log2()
                        # [n_mels, time] -> [C,H,W] expected by add_image (use 1 channel)
                        self.writer.add_image('train/recon_spectrogram', spec.unsqueeze(0), self.current_epoch, dataformats='CHW')  # 1 x H x W
        pbar.close()

        metrics = {
            'loss': total_loss.avg,
            'recon_loss': recon_meter.avg,
            'kl_loss': kl_meter.avg,
            'adv_loss': adv_meter.avg,
            'fm_loss': fm_meter.avg,
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
        fm_meter = AverageMeter()

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
            wave = wave.to(self.device)
            if wave.dim() == 2:
                wave = wave.unsqueeze(1)

            recon, mu, logvar, _ = self.model(wave)
            recon_loss = self.recon_loss_fn(recon, wave)
            kl_loss = self.model.kl_loss(mu, logvar, reduction="mean")
            p_loss = self._compute_perceptual(recon, wave)
            pred_fake, feats_fake = self.discriminator(recon)
            pred_real, feats_real = self.discriminator(wave)
            adv_loss = sum(self.bce(p, torch.ones_like(p)) for p in pred_fake) / len(pred_fake)
            fm_loss = 0.0
            for fr, ff in zip(feats_real, feats_fake):
                for r, f in zip(fr, ff):
                    fm_loss += F.l1_loss(f, r)
            fm_loss = fm_loss / len(feats_real)

            loss = recon_loss + self.beta_kl * kl_loss + self.lambda_adv * adv_loss + self.lambda_fm * fm_loss
            if self.perceptual_loss:
                loss = loss + self.lambda_p * p_loss

            total_loss.update(loss.item(), wave.size(0))
            recon_meter.update(recon_loss.item(), wave.size(0))
            kl_meter.update(kl_loss.item(), wave.size(0))
            adv_meter.update(adv_loss.item(), wave.size(0))
            fm_meter.update(fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss, wave.size(0))
            if self.perceptual_loss:
                p_meter.update(p_loss.item(), wave.size(0))

            if batch_idx == 0:
                self.writer.add_audio('val/original', wave[0][0], self.current_epoch, sample_rate=self.sample_rate)
                self.writer.add_audio('val/reconstruction', recon[0][0], self.current_epoch, sample_rate=self.sample_rate)
                if self.mel_transform is not None:
                    spec = self.mel_transform(recon[0]).log2()[None]
                    self.writer.add_image('val/recon_spectrogram', spec.squeeze(0), self.current_epoch, dataformats='CHW')

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

        pbar.close()

        metrics = {
            'loss': total_loss.avg,
            'recon_loss': recon_meter.avg,
            'kl_loss': kl_meter.avg,
            'adv_loss': adv_meter.avg,
            'fm_loss': fm_meter.avg,
        }
        if self.perceptual_loss:
            metrics['p_loss'] = p_meter.avg
        return metrics
