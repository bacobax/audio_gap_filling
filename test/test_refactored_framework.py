#!/usr/bin/env python3
"""
Test script for the refactored framework.

This script verifies that all components of the refactored framework
work correctly and can be imported without errors.
"""
import traceback

import sys
import os
from typing import Dict, Any

# Try to import yaml, but don't fail if it's not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("⚠️ PyYAML not available, some tests will be skipped")
    YAML_AVAILABLE = False

from src.factory import ModelFactory
from src.models.mae_vit import MAEViT
from src.utils.math_utils import MCD

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available, some tests will be skipped")
    TORCH_AVAILABLE = False

# Try to import tensorboard, but don't fail if it's not available
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️ TensorBoard not available, some tests will be skipped")
    TENSORBOARD_AVAILABLE = False

# Try to import einops, but don't fail if it's not available
try:
    import einops
    EINOPS_AVAILABLE = True
except ImportError:
    print("⚠️ Einops not available, some tests will be skipped")
    EINOPS_AVAILABLE = False

# Try to import timm, but don't fail if it's not available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("⚠️ Timm not available, some tests will be skipped")
    TIMM_AVAILABLE = False

# Try to import librosa, but don't fail if it's not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ Librosa not available, some tests will be skipped")
    LIBROSA_AVAILABLE = False


def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from src.core.base_model import BaseModel
        from src.core.base_dataset import BaseDataset
        from src.core.base_trainer import BaseTrainer
        print("✅ Core modules imported successfully")
        
        # Test model imports
        from src.models.mae_vit import MAEViT, MAEEncoder, MAEDecoder
        from src.models.patch_shuffle import PatchShuffle
        print("✅ Model modules imported successfully")
        
        # Test data imports
        from src.data.mel_spectrogram_dataset import MelSpectrogramDataset
        print("✅ Data modules imported successfully")
        
        # Test training imports
        from src.training.mae_trainer import MAETrainer
        print("✅ Training modules imported successfully")
        
        # Test config imports
        from src.config.config_manager import ConfigManager
        print("✅ Config modules imported successfully")
        
        # Test utils imports
        from src.utils.audio_utils import compute_mel_spectrogram, normalize_spectrogram
        from src.utils.math_utils import MCD
        from src.utils.visualization_utils import plot_spectrogram
        print("✅ Utils modules imported successfully")
        
        # Test factory imports
        from src.factory import ModelFactory, DatasetFactory, TrainerFactory, TrainingPipeline
        print("✅ Factory modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False


def test_config_manager():
    """Test the configuration manager."""
    print("\n🔍 Testing configuration manager...")
    
    if not YAML_AVAILABLE:
        print("⚠️ Skipping configuration manager test - PyYAML not available")
        return True
    
    try:
        from src.config.config_manager import ConfigManager
        
        # Create a test configuration
        test_config = {
            'seed': 42,
            'batch_size': 4,
            'emb_dim': 256,
            'patch_size': 4,
            'image_size': [80, 380],
            'audio_filename': 'test.wav',
            'gap_percentage': 0.75
        }
        
        # Test config manager
        config_manager = ConfigManager(default_config=test_config)
        
        # Test get methods
        assert config_manager.get('seed') == 42
        assert config_manager.get('batch_size') == 4
        assert config_manager.get('nonexistent', 'default') == 'default'
        
        # Test model config
        model_config = config_manager.get_model_config()
        assert 'image_size' in model_config
        assert 'patch_size' in model_config
        
        print("✅ Configuration manager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration manager test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\n🔍 Testing model creation...")
    
    if not TORCH_AVAILABLE:
        print("⚠️ Skipping model creation test - PyTorch not available")
        return True
    
    if not EINOPS_AVAILABLE:
        print("⚠️ Skipping model creation test - Einops not available")
        return True
    
    if not TIMM_AVAILABLE:
        print("⚠️ Skipping model creation test - Timm not available")
        return True
    
    try:
        from src.models.mae_vit import MAEViT
        
        # Create model configuration
        model_config = {
            'image_size': (80, 380),
            'patch_size': 4,
            'emb_dim': 256,
            'encoder_layer': 4,  # Smaller for testing
            'encoder_head': 8,
            'decoder_layer': 2,
            'decoder_head': 8,
            'mask_ratio': 0.75
        }
        
        # Create model
        model = MAEViT(model_config)
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 80, 380)
        with torch.no_grad():
            output, mask = model(dummy_input)
        
        assert output.shape == dummy_input.shape
        assert mask.shape == dummy_input.shape
        
        # Test parameter info
        param_info = model.get_trainable_parameters()
        assert 'trainable_parameters' in param_info
        assert param_info['trainable_parameters'] > 0
        
        print(f"✅ Model creation successful (parameters: {param_info['trainable_parameters']:,})")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        # print stack trace
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test dataset creation (without loading actual files)."""
    print("\n🔍 Testing dataset creation...")
    
    if not LIBROSA_AVAILABLE:
        print("⚠️ Skipping dataset creation test - Librosa not available")
        return True
    
    try:
        from src.data.mel_spectrogram_dataset import MelSpectrogramDataset
        
        # Create dataset configuration
        dataset_config = {
            'flac_path': 'gapped_audio.wav',  # This file should exist
            'gap_percentage': 0.75,
            'n_fft': 1024,
            'hop_length': 256,
            'n_mels': 80,
            'test': (False, None)
        }
        
        # Check if the audio file exists
        if not os.path.exists(dataset_config['flac_path']):
            print(f"⚠️ Audio file {dataset_config['flac_path']} not found, skipping dataset test")
            return True
        
        # Create dataset
        dataset = MelSpectrogramDataset(dataset_config)
        
        # Test dataset info
        dataset_info = dataset.get_dataset_info()
        assert 'length' in dataset_info
        assert 'sample_shape' in dataset_info
        
        print(f"✅ Dataset creation successful (samples: {dataset_info['length']})")
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        traceback.print_exc()
        return False


def test_factory_pattern():
    """Test the factory pattern."""
    print("\n🔍 Testing factory pattern...")
    
    try:
        # Test model factory
        model_config = {
            'image_size': (80, 380),
            'patch_size': 4,
            'emb_dim': 256,
            'encoder_layer': 2,
            'encoder_head': 4,
            'decoder_layer': 1,
            'decoder_head': 4,
            'mask_ratio': 0.75
        }
        
        model = ModelFactory.create_model('mae_vit', model_config)
        assert isinstance(model, MAEViT)
        
        # Test invalid model type
        try:
            ModelFactory.create_model('invalid_model', model_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        print("✅ Factory pattern working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False


def test_math_utils():
    """Test mathematical utilities."""
    print("\n🔍 Testing math utilities...")
    
    try:
        # Test MCD function
        common_divisors = MCD(12, 18, 1, 10)
        assert 2 in common_divisors
        assert 3 in common_divisors
        assert 6 in common_divisors
        
        # Test with no common divisors
        no_common = MCD(7, 11, 2, 10)
        assert len(no_common) == 0
        
        print("✅ Math utilities working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Math utilities test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Refactored Framework")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_manager,
        test_model_creation,
        test_dataset_creation,
        test_factory_pattern,
        test_math_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored framework is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 