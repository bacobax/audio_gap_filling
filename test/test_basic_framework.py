#!/usr/bin/env python3
"""
Basic test script for the refactored framework.

This script tests the core functionality without requiring external dependencies.
"""

import sys
import os
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_file_structure():
    """Test that all required files exist."""
    print("🔍 Testing file structure...")
    
    required_files = [
        'src/__init__.py',
        'src/core/__init__.py',
        'src/core/base_model.py',
        'src/core/base_dataset.py',
        'src/core/base_trainer.py',
        'src/models/__init__.py',
        'src/models/mae_vit.py',
        'src/models/patch_shuffle.py',
        'src/data/__init__.py',
        'src/data/mel_spectrogram_dataset.py',
        'src/training/__init__.py',
        'src/training/mae_trainer.py',
        'src/config/__init__.py',
        'src/config/config_manager.py',
        'src/utils/__init__.py',
        'src/utils/audio_utils.py',
        'src/utils/math_utils.py',
        'src/utils/visualization_utils.py',
        'src/factory.py',
        'main_refactored.py',
        'README_REFACTORED.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True


def test_config_manager():
    """Test the configuration manager without external dependencies."""
    print("\n🔍 Testing configuration manager...")
    
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
        return False


def test_math_utils():
    """Test mathematical utilities."""
    print("\n🔍 Testing math utilities...")
    
    try:
        from src.utils.math_utils import MCD
        
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


def test_audio_utils():
    """Test audio utilities."""
    print("\n🔍 Testing audio utilities...")
    
    try:
        from src.utils.audio_utils import normalize_spectrogram, inverse_normalize_spectrogram
        import numpy as np
        
        # Test normalization functions
        test_spectrogram = np.array([[1.0, 2.0], [3.0, 4.0]])
        min_val = 1.0
        denom = 3.0
        
        normalized = normalize_spectrogram(test_spectrogram, min_val, denom)
        denormalized = inverse_normalize_spectrogram(normalized, min_val, denom)
        
        # Check that denormalization recovers original
        np.testing.assert_array_almost_equal(test_spectrogram, denormalized)
        
        print("✅ Audio utilities working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Audio utilities test failed: {e}")
        return False


def test_factory_pattern():
    """Test the factory pattern without model creation."""
    print("\n🔍 Testing factory pattern...")
    
    try:
        from src.factory import ModelFactory, DatasetFactory, TrainerFactory
        
        # Test invalid model type
        try:
            ModelFactory.create_model('invalid_model', {})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Test invalid dataset type
        try:
            DatasetFactory.create_dataset('invalid_dataset', {})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Test invalid trainer type
        try:
            TrainerFactory.create_trainer('invalid_trainer', None, None)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        print("✅ Factory pattern working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False


def test_base_classes():
    """Test that base classes can be imported and have required methods."""
    print("\n🔍 Testing base classes...")
    
    try:
        from src.core.base_model import BaseModel
        from src.core.base_dataset import BaseDataset
        from src.core.base_trainer import BaseTrainer
        
        # Check that base classes exist and have required methods
        assert hasattr(BaseModel, '_validate_config')
        assert hasattr(BaseModel, 'forward')
        assert hasattr(BaseModel, 'get_trainable_parameters')
        
        assert hasattr(BaseDataset, '_validate_config')
        assert hasattr(BaseDataset, '_setup_dataset')
        assert hasattr(BaseDataset, '__len__')
        assert hasattr(BaseDataset, '__getitem__')
        
        assert hasattr(BaseTrainer, '_setup_training_components')
        assert hasattr(BaseTrainer, '_train_epoch')
        assert hasattr(BaseTrainer, '_validate_epoch')
        
        print("✅ Base classes have required methods")
        return True
        
    except Exception as e:
        print(f"❌ Base classes test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🧪 Testing Refactored Framework (Basic Tests)")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_config_manager,
        test_math_utils,
        test_audio_utils,
        test_factory_pattern,
        test_base_classes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The framework structure is correct.")
        print("\nTo run full tests with PyTorch and TensorBoard:")
        print("1. Install dependencies: pip install torch tensorboard")
        print("2. Run: python test_refactored_framework.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 