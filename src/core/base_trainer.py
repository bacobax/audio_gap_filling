"""
Base trainer class that defines the interface for all training loops in the framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter # type: ignore
import os
from datetime import datetime

import random
import numpy as np  # type: ignore


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the framework.

    This class defines the common interface that all trainers must implement,
    ensuring consistency across different training strategies and enabling easy
    swapping of training implementations.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
            log_dir: Directory for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device or self._get_default_device()
        if self.config.get("resume", False):
            if log_dir is None:
                raise Exception("To resume training provide the log_dir")
        self.log_dir = log_dir or self._get_default_log_dir()

        # Move model to device
        self.model.to(self.device)

        # Gradient accumulation (virtual batch size)
        self.step_interval = int(self.config.get('step_interval', 1))
        if self.step_interval < 1:
            self.step_interval = 1

        # Setup logging
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Declare attributes that will be set by subclasses
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # Setup optimizer, scheduler, and loss function
        self._setup_training_components()

    def _get_default_device(self) -> torch.device:
        """Get the default device for training."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _get_default_log_dir(self) -> str:
        """Get the default logging directory."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        return os.path.join('runs', 'train', f'{self.config["config_filename"].split(".")[0]}-{timestamp}')

    @abstractmethod
    def _setup_training_components(self) -> None:
        """
        Setup optimizer, scheduler, loss function, and other training components.
        """
        pass

    @abstractmethod
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary containing training metrics
        """
        pass

    @abstractmethod
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary containing validation metrics
        """
        pass

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train for

        Returns:
            Dictionary containing training history
        """
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }

        start_epoch = self.current_epoch + 1 if self.config.get('resume', False) else 0
        pbar = tqdm(range(start_epoch, num_epochs), desc="Training Progress", unit="epoch", leave=True, position=0)

        # Early stopping parameters
        patience = self.config.get('early_stop_patience')
        epochs_no_improve = 0

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch()
            training_history['train_losses'].append(train_metrics.get('loss', 0.0))

            postfix = train_metrics

            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()

                postfix_val = {f"val_{k}": v for k, v in val_metrics.items()}
                postfix.update(postfix_val)

                training_history['val_losses'].append(val_metrics.get('loss', 0.0))

                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)

                # Save best model and track improvement
                current_val_loss = val_metrics.get('loss', float('inf'))
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self._save_checkpoint('best_model.pt', epoch, val_metrics)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                self._log_metrics(train_metrics, {}, epoch)

            # Save checkpoint periodically
            if epoch % self.config.get('save_every', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, train_metrics)
            pbar.update(1)
            pbar.set_postfix(postfix)

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                pbar.close()
                break

        # Log hyperparameters and final metrics
        try:
            hparams = self._prepare_hparams()
            metrics = {'best_val_loss': self.best_val_loss}
            self.writer.add_hparams(hparams, metrics)
        except Exception:
            pass

        return training_history

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to tensorboard."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

    def _extra_state_to_save(self) -> Dict[str, Any]:
        """Hook for subclasses to extend checkpoint contents."""
        return {}

    def _load_extra_state(self, checkpoint: Dict[str, Any]) -> None:
        """Hook for subclasses to restore extra checkpoint contents."""
        return None

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a checkpoint with model/optimizers/schedulers/RNG and subclass extras."""
        checkpoint_path = os.path.join(self.log_dir, filename)

        # Core training state
        payload: Dict[str, Any] = {
            'epoch': epoch,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config,
        }

        # RNG states for full reproducibility (best-effort)
        try:
            payload['torch_rng_state'] = torch.get_rng_state()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
        try:
            payload['py_random_state'] = random.getstate()
        except Exception:
            pass
        try:
            if np is not None:
                payload['numpy_random_state'] = np.random.get_state()  # type: ignore
        except Exception:
            pass

        # Allow subclasses (e.g., GANs/VAEs) to append extra state
        try:
            extras = self._extra_state_to_save()
            if isinstance(extras, dict):
                payload.update(extras)
        except Exception:
            pass

        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint and restore model/optimizers/schedulers/RNG and subclass extras.
        Returns the epoch number stored in the checkpoint.
        """
        # AFTER (works on 2.6+, with backward-compat fallback)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # for older torch versions that don't have the weights_only kwarg
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Model & optimizers
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass

        # Bookkeeping
        self.current_epoch = checkpoint.get('current_epoch', checkpoint.get('epoch', 0))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', self.global_step)

        # RNG states (best-effort)
        try:
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
        except Exception:
            pass
        try:
            if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        except Exception:
            pass
        try:
            if 'py_random_state' in checkpoint:
                random.setstate(checkpoint['py_random_state'])
        except Exception:
            pass
        try:
            if np is not None and 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])  # type: ignore
        except Exception:
            pass

        # Let subclasses restore their extra state
        try:
            self._load_extra_state(checkpoint)
        except Exception:
            pass

        return checkpoint.get('epoch', self.current_epoch)
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
