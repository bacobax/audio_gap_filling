# Refactored AI Training Framework

This is a professionally refactored version of the original MAE (Masked Autoencoder) training codebase, designed with modern software engineering best practices for AI research and development.

## 🏗️ Architecture Overview

The framework follows a modular, extensible architecture with clear separation of concerns:

```
src/
├── core/                    # Base classes and interfaces
│   ├── base_model.py       # Abstract base model class
│   ├── base_dataset.py     # Abstract base dataset class
│   └── base_trainer.py     # Abstract base trainer class
├── models/                  # Model implementations
│   ├── mae_vit.py         # MAE Vision Transformer
│   └── patch_shuffle.py   # Patch shuffling component
├── data/                    # Dataset implementations
│   └── mel_spectrogram_dataset.py
├── training/                # Training implementations
│   └── mae_trainer.py     # MAE-specific trainer
├── config/                  # Configuration management
│   └── config_manager.py   # YAML config handling
├── utils/                   # Utility functions
│   ├── audio_utils.py     # Audio processing utilities
│   ├── math_utils.py      # Mathematical utilities
│   └── visualization_utils.py
└── factory.py              # Factory classes for easy instantiation
```

## 🚀 Key Features

### 1. **Modular Design**
- **Base Classes**: Abstract interfaces ensure consistency across implementations
- **Component Isolation**: Models, datasets, and trainers are completely independent
- **Easy Extension**: Add new architectures by implementing base interfaces

### 2. **Configuration Management**
- **YAML-based**: Clean, readable configuration files
- **Validation**: Automatic configuration validation with helpful error messages
- **Defaults**: Sensible defaults with easy override capability
- **Type Safety**: Strong typing throughout the codebase

### 3. **Professional Training Pipeline**
- **Checkpointing**: Automatic model saving and loading
- **Logging**: TensorBoard integration for experiment tracking
- **Error Handling**: Robust error handling with informative messages
- **Device Management**: Automatic GPU/MPS/CPU detection and usage

### 4. **Extensibility**
- **Factory Pattern**: Easy creation of new models, datasets, and trainers
- **Plugin Architecture**: Add new components without modifying existing code
- **Interface Contracts**: Clear contracts for all components

## 📦 Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 🎯 Quick Start

### Basic Usage

The simplest way to use the framework is through the high-level training pipeline:

```python
from src.factory import TrainingPipeline

# Create and run training pipeline
pipeline = TrainingPipeline("hparams.yaml")
history = pipeline.run()
```

### Advanced Usage

For more control, you can use the individual components:

```python
from src.config import ConfigManager
from src.models import MAEViT
from src.data import MelSpectrogramDataset
from src.training import MAETrainer
from torch.utils.data import DataLoader

# Load configuration
config_manager = ConfigManager("hparams.yaml")
model_config = config_manager.get_model_config()
data_config = config_manager.get_data_config()

# Create model
model = MAEViT(model_config)

# Create datasets
train_dataset = MelSpectrogramDataset(data_config)
val_config = data_config.copy()
val_config['test'] = (True, 'test_audio.wav')
val_dataset = MelSpectrogramDataset(val_config)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Create trainer
trainer = MAETrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config_manager.config
)

# Train
history = trainer.train(num_epochs=10)
```

## ⚙️ Configuration

The framework uses YAML configuration files. Here's an example `hparams.yaml`:

```yaml
# Training configuration
seed: 42
batch_size: 4
max_device_batch_size: 512
base_learning_rate: 0.00015
weight_decay: 0.05
total_epoch: 8
warmup_epoch: 1
save_every: 10

# Model configuration
image_size: [80, 380]
patch_size: 4
emb_dim: 256
encoder_layer: 32
encoder_head: 16
decoder_layer: 10
decoder_head: 16
mask_ratio: 0.75

# Data configuration
audio_filename: gapped_audio.wav
test_audio_filename: wav_test.wav
n_mels: 80
n_fft: 1024
hop_length: 256
```

## 🔧 Adding New Components

### Adding a New Model

1. **Create the model class**:
```python
from src.core.base_model import BaseModel

class MyNewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your model implementation
    
    def _validate_config(self):
        # Validate configuration
        pass
    
    def forward(self, x, **kwargs):
        # Forward pass implementation
        pass
```

2. **Add to factory**:
```python
# In src/factory.py
@staticmethod
def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
    if model_type.lower() == 'my_new_model':
        return MyNewModel(config)
    # ... existing code
```

### Adding a New Dataset

1. **Create the dataset class**:
```python
from src.core.base_dataset import BaseDataset

class MyNewDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def _validate_config(self):
        # Validate configuration
        pass
    
    def _setup_dataset(self):
        # Setup dataset
        pass
    
    def __len__(self):
        # Return dataset length
        pass
    
    def __getitem__(self, idx):
        # Return sample
        pass
```

2. **Add to factory**:
```python
# In src/factory.py
@staticmethod
def create_dataset(dataset_type: str, config: Dict[str, Any]):
    if dataset_type.lower() == 'my_new_dataset':
        return MyNewDataset(config)
    # ... existing code
```

### Adding a New Trainer

1. **Create the trainer class**:
```python
from src.core.base_trainer import BaseTrainer

class MyNewTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader=None, config=None, device=None, log_dir=None):
        super().__init__(model, train_loader, val_loader, config, device, log_dir)
    
    def _setup_training_components(self):
        # Setup optimizer, scheduler, loss function
        pass
    
    def _train_epoch(self):
        # Training logic
        pass
    
    def _validate_epoch(self):
        # Validation logic
        pass
```

2. **Add to factory**:
```python
# In src/factory.py
@staticmethod
def create_trainer(trainer_type: str, model, train_loader, val_loader=None, config=None, device=None, log_dir=None):
    if trainer_type.lower() == 'my_new_trainer':
        return MyNewTrainer(model, train_loader, val_loader, config, device, log_dir)
    # ... existing code
```

## 🧪 Running Experiments

### Using the Refactored Framework

```bash
# Run with the new framework
python main_refactored.py
```

### Comparing with Original

```bash
# Run with the original framework (for comparison)
python main.py
```

## 📊 Monitoring Training

The framework automatically logs to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open in browser
# http://localhost:6006
```

## 🔍 Key Improvements

### 1. **Code Organization**
- **Before**: Monolithic files with mixed responsibilities
- **After**: Clear separation of concerns with dedicated modules

### 2. **Extensibility**
- **Before**: Hard to add new models or training strategies
- **After**: Easy to extend with new components using base classes

### 3. **Configuration Management**
- **Before**: Hardcoded parameters scattered throughout code
- **After**: Centralized YAML configuration with validation

### 4. **Error Handling**
- **Before**: Basic error handling with unclear messages
- **After**: Comprehensive error handling with helpful diagnostics

### 5. **Testing and Debugging**
- **Before**: Difficult to test individual components
- **After**: Each component can be tested independently

### 6. **Documentation**
- **Before**: Minimal documentation
- **After**: Comprehensive docstrings and type hints

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public methods
- Keep functions small and focused on a single responsibility

### Testing
- Write unit tests for individual components
- Use pytest for testing framework
- Aim for high test coverage

### Documentation
- Update README when adding new features
- Document all configuration options
- Provide examples for common use cases

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** following the development guidelines
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This refactored framework maintains all the original functionality while providing a much more maintainable and extensible codebase. The original work on MAE for audio processing provided the foundation for this professional implementation. 