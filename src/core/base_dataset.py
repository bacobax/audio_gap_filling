"""
Base dataset class that defines the interface for all datasets in the framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets in the framework.
    
    This class defines the common interface that all datasets must implement,
    ensuring consistency across different data sources and enabling easy
    swapping of dataset implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset with configuration.
        
        Args:
            config: Dictionary containing dataset configuration parameters
        """
        super().__init__()
        self.config = config
        self._validate_config()
        self._setup_dataset()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def _setup_dataset(self) -> None:
        """
        Setup the dataset (load data, create indices, etc.).
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Sample data (tensor or tuple of tensors)
        """
        pass
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a single sample.
        
        Returns:
            Shape of a single sample
        """
        sample = self[0]
        if isinstance(sample, (list, tuple)):
            # Only return shapes for tensor elements
            shapes = []
            for item in sample:
                if hasattr(item, 'shape'):
                    shapes.append(item.shape)
                else:
                    shapes.append(type(item).__name__)
            return tuple(shapes)
        return sample.shape
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            "length": len(self),
            "sample_shape": self.get_sample_shape(),
            "config": self.config
        } 