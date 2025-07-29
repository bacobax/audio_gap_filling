"""
Base model class that defines the interface for all models in the framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the framework.
    
    This class defines the common interface that all models must implement,
    ensuring consistency across different architectures and enabling easy
    swapping of model implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Model output(s)
        """
        pass
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get information about trainable parameters.
        
        Returns:
            Dictionary with parameter counts and other info
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            **kwargs: Additional data to save with the model
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            Dictionary containing additional checkpoint data
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        
        # Return additional checkpoint data (excluding model state and config)
        additional_data = {k: v for k, v in checkpoint.items() 
                          if k not in ["model_state_dict", "config"]}
        return additional_data 