"""Minimal package init to avoid heavy imports during testing."""
from .core.base_model import BaseModel
from .core.base_dataset import BaseDataset
from .core.base_trainer import BaseTrainer

__all__ = [
    'BaseModel',
    'BaseDataset',
    'BaseTrainer',
]
