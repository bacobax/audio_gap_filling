#!/usr/bin/env python3
"""
Example: Adding a Custom Model to the Framework

This example demonstrates how to extend the framework with a custom model
while maintaining compatibility with the existing training pipeline.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.base_model import BaseModel
from src.factory import ModelFactory, TrainingPipeline


class SimpleAutoencoder(BaseModel):
    """
    A simple autoencoder model as an example of extending the framework.
    
    This model demonstrates how to implement the BaseModel interface
    and integrate with the existing training pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simple autoencoder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Extract configuration
        input_size = config.get('input_size', 80 * 380)  # Flattened spectrogram
        hidden_size = config.get('hidden_size', 512)
        latent_size = config.get('latent_size', 128)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = ['input_size', 'hidden_size', 'latent_size']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            **kwargs: Additional arguments (ignored for this simple model)
            
        Returns:
            Tuple containing:
                - Reconstructed tensor
                - Dummy mask (for compatibility with MAE interface)
        """
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Encode
        latent = self.encoder(x_flat)
        
        # Decode
        reconstructed_flat = self.decoder(latent)
        
        # Reshape back to original dimensions
        reconstructed = reconstructed_flat.view_as(x)
        
        # Create dummy mask (all ones) for compatibility
        mask = torch.ones_like(x)
        
        return reconstructed, mask


# Extend the ModelFactory to support our custom model
class ExtendedModelFactory(ModelFactory):
    """Extended model factory that includes our custom model."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'simple_autoencoder':
            return SimpleAutoencoder(config)
        else:
            # Fall back to parent implementation
            return super().create_model(model_type, config)


def create_custom_config() -> Dict[str, Any]:
    """Create a configuration for the custom model."""
    return {
        # Model configuration
        'input_size': 80 * 380,  # Flattened spectrogram size
        'hidden_size': 512,
        'latent_size': 128,
        
        # Training configuration
        'seed': 42,
        'batch_size': 4,
        'max_device_batch_size': 512,
        'base_learning_rate': 0.001,  # Higher learning rate for simpler model
        'weight_decay': 0.01,
        'total_epoch': 5,  # Fewer epochs for demonstration
        'warmup_epoch': 1,
        'save_every': 2,
        
        # Data configuration
        'audio_filename': 'gapped_audio.wav',
        'test_audio_filename': 'wav_test.wav',
        'n_mels': 80,
        'n_fft': 1024,
        'hop_length': 256,
        'gap_percentage': 0.75
    }


def main():
    """Demonstrate the custom model integration."""
    print("🚀 Custom Model Integration Example")
    print("=" * 50)
    
    # Create custom configuration
    config = create_custom_config()
    
    # Test the custom model
    print("\n1. Testing custom model creation...")
    model = ExtendedModelFactory.create_model('simple_autoencoder', config)
    print(f"✅ Custom model created successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(2, 1, 80, 380)  # Batch of 2 spectrograms
    with torch.no_grad():
        output, mask = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Test configuration validation
    print("\n3. Testing configuration validation...")
    try:
        invalid_config = config.copy()
        del invalid_config['input_size']
        model = ExtendedModelFactory.create_model('simple_autoencoder', invalid_config)
    except ValueError as e:
        print(f"✅ Configuration validation working: {e}")
    
    print("\n🎉 Custom model integration successful!")
    print("\nTo use this custom model in training:")
    print("1. Replace ModelFactory with ExtendedModelFactory in factory.py")
    print("2. Update your configuration to use 'simple_autoencoder' as model type")
    print("3. Run training as usual")


if __name__ == "__main__":
    main() 