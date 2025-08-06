"""
Base trainer class that defines the interface for all training loops in the framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Handle tensorboard import
try:
    from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    # Create a dummy SummaryWriter class
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def add_hparams(self, *args, **kwargs):
            pass

        def close(self):
            pass

import os
from datetime import datetime


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
                
                # Save best model
                if val_metrics.get('loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                self._log_metrics(train_metrics, {}, epoch)
            
            # Save checkpoint periodically
            if epoch % self.config.get('save_every', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, train_metrics)
            pbar.update(1)
            pbar.set_postfix(postfix)

        
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
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)

    def _prepare_hparams(self) -> Dict[str, Any]:
        """Filter configuration values for TensorBoard hparam logging."""
        hparams: Dict[str, Any] = {}
        for key, value in self.config.items():
            if isinstance(value, (int, float, bool)):
                hparams[key] = value
            else:
                hparams[key] = str(value)
        return hparams
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close() 