"""
MAE (Masked Autoencoder) trainer implementation.
"""
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import os
import lpips
from lpips import LPIPS
from ..utils.metrics import AverageMeter

from ..core.base_trainer import BaseTrainer
from ..utils.math_utils import MCD


class MAETrainer(BaseTrainer):
    """
    Trainer for MAE (Masked Autoencoder) models.
    
    This trainer implements the specific training logic for MAE models,
    including the custom loss function and evaluation procedures.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the MAE trainer.
        
        Args:
            model: MAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
            log_dir: Directory for logging
        """
        if config is None:
            print(f"⚠️ No configuration provided, using default values.")
        # Extract MAE-specific configuration
        self.mask_ratio = config.get('mask_ratio', 0.75) if config else 0.75
        self.patch_size = config.get('patch_size', 4) if config else 4
        self.n_mels = config.get('n_mels', 80) if config else 80
        self.l1_weight = config.get('l1_weight', 0.0) if config else 0.0
        self.perceptual_loss = config.get('perceptual_loss', False) if config else False
        self.lambda_p = config.get('lambda_p', 0.0) if config else 0.0
        self.lambda_p_warmup = config.get('lambda_p_warmup', 0)

        # Gradient accumulation: prefer explicit step_interval; fallback to ratio of global:device batch
        self.batch_size = config.get('batch_size', 4) if config else 4
        self.max_device_batch_size = config.get('max_device_batch_size', 512) if config else 512
        self.load_batch_size = min(self.max_device_batch_size, self.batch_size)
        explicit_si = int(config.get('step_interval', 1)) if config else 1
        if explicit_si > 1:
            self.steps_per_update = explicit_si
        else:
            # Fallback to whole-batch equivalence if user didn't set step_interval
            self.steps_per_update = max(1, self.batch_size // max(1, self.load_batch_size))

        super().__init__(model, train_loader, val_loader, config, device, config["log_dir"])

        if self.perceptual_loss:
            print(f"LPIPS, warmup= {self.lambda_p_warmup}, lambda_p= {self.lambda_p}")
            self.lpips_fn = LPIPS(net='vgg').to(self.device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

        self.check_patch_compatibility(train_loader.dataset)

        self.checkpoint_path = self.config.get("checkpoint_path", os.path.join(self.log_dir, "mae_latest.pt"))
        if self.config.get("resume", False):
            self.current_epoch = self.load_checkpoint(self.checkpoint_path)


    def _setup_training_components(self) -> None:
        """Setup optimizer, scheduler, and loss function for MAE training."""
        # Setup optimizer
        base_lr = self.config.get('base_learning_rate', 1.5e-4)
        weight_decay = self.config.get('weight_decay', 0.05)

        # Scale learning rate by batch size
        scaled_lr = base_lr * self.batch_size / 256

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )

        # Setup learning rate scheduler
        total_epoch = self.config.get('total_epoch', 2000)
        warmup_epoch = self.config.get('warmup_epoch', 200)

        def lr_lambda(epoch):
            return min(
                (epoch + 1) / (warmup_epoch + 1e-8),
                0.5 * (math.cos(epoch / total_epoch * math.pi) + 1)
            )

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)  # type: ignore

        # Setup base loss functions
        self.criterion = nn.MSELoss(reduction='none')
        self.l1_criterion = nn.L1Loss(reduction='none')

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = AverageMeter()
        mse_loss = AverageMeter()
        p_loss_meter = None
        if self.perceptual_loss:
            p_loss_meter = AverageMeter()
        step_count = 0
        # zero grads once per accumulation window
        self.optimizer.zero_grad()  # type: ignore

        pbar = tqdm(
            total=len(self.train_loader),
            desc='Training MAE',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )

        for batch in self.train_loader:
            step_count += 1

            # Extract data from batch
            if len(batch) == 9:  # Training mode
                spectrogram_slice, start_idx, end_idx, start_sec, end_sec, _, _, start_gap_sec, end_gap_sec = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            # Move to device
            spectrogram_slice = spectrogram_slice.to(self.device)

            # Forward pass
            predicted_spectrogram, mask = self.model(spectrogram_slice)

            # Log first batch images
            if step_count == 1:
                self.writer.add_image('mae_target_train', spectrogram_slice.squeeze(0)[0], global_step=self.current_epoch)
                self.writer.add_image('mae_predicted_train', predicted_spectrogram.squeeze(0)[0], global_step=self.current_epoch)

            # Compute loss (only on masked regions)
            mse = torch.mean((predicted_spectrogram - spectrogram_slice) ** 2 * mask) / self.mask_ratio
            if self.l1_weight > 0:
                l1 = torch.mean(torch.abs(predicted_spectrogram - spectrogram_slice) * mask) / self.mask_ratio
                loss = (1 - self.l1_weight) * mse + self.l1_weight * l1
            else:
                loss = mse

            mse_loss.update(mse.item(), n=1)

            if self.perceptual_loss and p_loss_meter is not None:
                pred_lpips = (predicted_spectrogram * mask).repeat(1, 3, 1, 1)
                target_lpips = (spectrogram_slice * mask).repeat(1, 3, 1, 1)
                p_loss = self.lpips_fn(pred_lpips, target_lpips, normalize=True).mean() / self.mask_ratio
                self.lambda_p_effective = (
                    self.lambda_p
                    if self.current_epoch >= self.lambda_p_warmup
                    else self.lambda_p * self.current_epoch / max(1, self.lambda_p_warmup)
                )
                logged_total_loss = loss + p_loss
                loss = loss + self.lambda_p_effective * p_loss
                p_loss_meter.update(p_loss.item(), n=1)

            total_loss.update(logged_total_loss.item(), n=1)

            # Backward pass with gradient accumulation
            (loss / self.steps_per_update).backward()

            # Optimizer step when reaching virtual batch
            if step_count % self.steps_per_update == 0 or step_count == len(self.train_loader):
                self.optimizer.step()  # type: ignore
                self.optimizer.zero_grad()  # type: ignore

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

        pbar.close()

        # Step scheduler
        self.scheduler.step()  # type: ignore

        # Save latest checkpoint after every epoch
        self._save_checkpoint("mae_latest.pt", self.current_epoch, {'loss': total_loss.avg})
        if p_loss_meter:
            return {'loss': total_loss.avg, 'mse': mse_loss.avg, 'p_loss': p_loss_meter.avg, 'lambda_p': self.lambda_p_effective}
        else:
            return {'loss': total_loss.avg, 'mse': mse_loss.avg}


    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {'loss': 0.0}

        self.model.eval()
        total_loss = AverageMeter()
        p_loss_meter = AverageMeter() if self.perceptual_loss else None
        mse_loss = AverageMeter()

        pbar = tqdm(
            total=len(self.val_loader),
            desc='Validating MAE',
            unit='batch',
            leave=False,
            dynamic_ncols=True,
            position=1
        )

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Extract data from batch
                if len(batch) == 8:  # Test mode
                    gap_slice, target, start_sec, end_sec, _, _, gap_start, gap_end = batch
                else:
                    raise ValueError(f"Unexpected validation batch size: {len(batch)}")

                # Move to device
                target = target.to(self.device)
                gap_slice = gap_slice.to(self.device)

                # Handle gap indices
                if isinstance(gap_start, torch.Tensor):
                    gap_start = gap_start.squeeze().item()
                if isinstance(gap_end, torch.Tensor):
                    gap_end = gap_end.squeeze().item()

                if gap_end == gap_start:
                    print(f"⚠️ Skipping validation sample {i}: empty gap range")
                    continue

                # Forward pass with masking on the real gap
                predicted, mask = self.model(gap_slice, mask_bounds=(gap_start, gap_end))

                # Compute loss only on gap region
                gap_region_target = target[:, :, :, gap_start:gap_end]
                gap_region_pred = predicted[:, :, :, gap_start:gap_end]

                mse = torch.mean((gap_region_pred - gap_region_target) ** 2)
                if self.l1_weight > 0:
                    l1 = torch.mean(torch.abs(gap_region_pred - gap_region_target))
                    loss = (1 - self.l1_weight) * mse + self.l1_weight * l1
                else:
                    loss = mse
                mse_loss.update(loss.item(), n=1)

                if self.perceptual_loss and p_loss_meter is not None:
                    pred_lpips = gap_region_pred.repeat(1, 3, 1, 1)
                    target_lpips = gap_region_target.repeat(1, 3, 1, 1)
                    p_loss = self.lpips_fn(pred_lpips, target_lpips, normalize=True).mean()
                    lambda_p_effective = (
                        self.lambda_p
                        if self.current_epoch >= self.lambda_p_warmup
                        else self.lambda_p * self.current_epoch / max(1, self.lambda_p_warmup)
                    )
                    loss = loss + lambda_p_effective * p_loss
                    p_loss_meter.update(p_loss.item(), n=1)
                total_loss.update(loss.item(), n=1)

                # Log first batch images
                if i == 0:
                    self.writer.add_image('mae_original_gap', gap_slice.squeeze(0), global_step=self.current_epoch)
                    self.writer.add_image('mae_target', target.squeeze(0), global_step=self.current_epoch)
                    self.writer.add_image('mae_predicted', predicted.squeeze(0), global_step=self.current_epoch)

                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

        pbar.close()

        if p_loss_meter:
            return {'mse': mse_loss.avg, 'p_loss': p_loss_meter.avg, 'loss': total_loss.avg}
        else:

            return {'mse': mse_loss.avg, 'loss': total_loss.avg}
    
    def check_patch_compatibility(self, dataset) -> None:
        """
        Check if patch size is compatible with dataset dimensions.
        
        Args:
            dataset: Dataset to check compatibility with
        """
        crop_frames = dataset.crop_frames
        self.writer.add_scalar('debug/crop_frames', crop_frames, self.current_epoch)
        self.writer.add_scalar('debug/n_mels', self.n_mels, self.current_epoch)
        self.writer.add_scalar('debug/patch_size', self.patch_size, self.current_epoch)
        
        min_patch_size = 1
        max_patch_size = 80
        
        common_divisors = MCD(crop_frames, self.n_mels, min_patch_size, max_patch_size)
        print("COMPATIBLE PATCH SIZES")
        pprint(common_divisors)
        if not (crop_frames % self.patch_size == 0 and self.n_mels % self.patch_size == 0):
            if len(common_divisors) == 0:
                raise Exception(
                    f"No common divisors between {crop_frames} and {self.n_mels} "
                    f"in the range ({min_patch_size}, {max_patch_size})"
                )
            else:
                self.writer.add_text(
                    'error/patch_size_incompatibility',
                    f"Crop frames ({crop_frames}) and n_mels ({self.n_mels}) not divisible by patch size ({self.patch_size}). Try one of these: {common_divisors}",
                    global_step=self.current_epoch
                )
                raise Exception(
                    f"Crop frames {crop_frames} and n_mels {self.n_mels} not divisible "
                    f"by the patch size {self.patch_size}, try these other patch sizes: {common_divisors}"
                )
