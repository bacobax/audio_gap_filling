"""
Configuration manager for handling YAML configuration files.
"""
import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration loading, validation, and access.
    
    This class provides a centralized way to handle configuration files,
    with support for validation, defaults, and environment-specific configs.
    """
    
    def __init__(self, config_path: Optional[str] = None, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
            default_config: Default configuration dictionary
        """
        self.config_path = config_path
        self.default_config = default_config or {}
        self.config = {}
        
        if config_path:
            self.load_config(config_path)

        self.config_filename = config_path.split("/")[-1]
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file) or {}

        # Merge with defaults
        self.config = self._merge_configs(self.default_config, loaded_config)
        # Adjust patch size if using pretrained ViT
        if self.get("model.pretrained_ViT", False):
            self.set("model.patch_size", 16)
        return self.config
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            default: Default configuration
            override: Configuration to override defaults
            
        Returns:
            Merged configuration
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration (uses original path if None)
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def validate_config(self, required_keys: Optional[list] = None) -> bool:
        """
        Validate the configuration.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If required keys are missing
        """
        if required_keys is None:
            return True
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""

        model_type = self.get('model.type', 'mae_vit')
        if model_type == 'diffusion_unet':
            cfg = {
                'type': 'diffusion_unet',
                'network': self.get('model.network', {}),
                'exp': {
                    'sample_rate': self.get('data.sample_rate', 16000),
                    'audio_len': self.get('data.segment_length', 65536),
                },
            }
        elif model_type == 'vae':
            cfg = {
                'type': 'vae',
                'input_channels': self.get('model.input_channels', 1),
                'hidden_channels': self.get('model.hidden_channels', 64),
                'latent_dim': self.get('model.latent_dim', 64),
                'kernel_size': self.get('model.kernel_size', 3),
                'num_blocks': self.get('model.num_blocks', 3),
                'downsample_stride': self.get('model.downsample_stride', 2),
                'decoder': {
                    'output_channels': self.get('model.output_channels', 1),
                    'hidden_channels': self.get('model.hidden_channels', 64),
                    'latent_dim': self.get('model.latent_dim', 64),
                    'kernel_size': self.get('model.kernel_size', 3),
                    'num_blocks': self.get('model.num_blocks', 3),
                    'upsample_stride': self.get('model.downsample_stride', 2),
                },
            }
        else:
            cfg = {
                'type': model_type,
                'image_size': self.get('model.image_size', (80, 380)),
                'patch_size': self.get('model.patch_size', 4),
                'emb_dim': self.get('model.emb_dim', 256),
                'encoder_layer': self.get('model.encoder_layer', 32),
                'encoder_head': self.get('model.encoder_head', 16),
                'decoder_layer': self.get('model.decoder_layer', 10),
                'decoder_head': self.get('model.decoder_head', 16),
                'mask_ratio': self.get('training.mask_ratio', 0.75),
                'pretrained_ViT': self.get('model.pretrained_ViT', False),
            }

        return cfg
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        cfg = {
            'seed': self.get('training.seed', 42),
            'batch_size': self.get('training.batch_size', 4),
            'max_device_batch_size': self.get('training.max_device_batch_size', 512),
            'step_interval': self.get('training.step_interval', 1),
            'base_learning_rate': self.get('training.base_learning_rate', 0.00015),
            'weight_decay': self.get('training.weight_decay', 0.05),
            'total_epoch': self.get('training.total_epoch', 8),
            'warmup_epoch': self.get('training.warmup_epoch', 1),
            'save_every': self.get('training.save_every', 10),
            'resume': self.get('training.resume', False),
            'use_muon': self.get('training.use_muon', False),
            'log_dir': self.get('paths.log_dir', None),
            'l1_weight': self.get('training.l1_weight', 0.0),
            'mask_ratio': self.get('training.mask_ratio', 0.75),
            'patch_size': self.get('model.patch_size', 4),
            'n_mels': self.get('data.n_mels', 80),
            'trainer_type': self.get('training.trainer_type', 'mae'),
            'perceptual_loss': self.get('training.perceptual_loss', False),
            'lambda_p' : self.get('training.lambda_p', 0.0),
            'lambda_p_warmup': self.get('training.lambda_p_warmup', 0.0),
            'beta_kl': self.get('training.beta_kl', 1.0),
            'lambda_adv': self.get('training.lambda_adv', 0.0),
            'lambda_fm': self.get('training.lambda_fm', 0.0),
            'disc_learning_rate': self.get('training.disc_learning_rate', 0.0003),
            'freeze_encoder_epoch': self.get('training.freeze_encoder_epoch', None),
            'decoder_batch_size': self.get('training.decoder_batch_size', None),
            'decoder_learning_rate': self.get('training.decoder_learning_rate', self.get('training.base_learning_rate', 0.00015)),
            'mrstft': self.get('training.mrstft', {}),
            'sample_rate': self.get('data.sample_rate', 16000),
            'mixed_precision': self.get('training.mixed_precision', None),
            'config_filename': self.config_filename,
        }
        model_type = self.get('model.type', 'mae_vit')
        if model_type == 'diffusion_unet':
            # Diffusion trainer specific keys
            cfg.update({
                'lr': self.get('training.lr', 5e-5),
                'weight_decay': self.get('training.weight_decay', 1e-3),
                'warmup_steps': self.get('training.warmup_steps', 2000),
                'total_steps': self.get('training.total_steps', 100_000),
                'lambda_ctx': self.get('training.lambda_ctx', 0.1),
                'cfg_dropout_p': self.get('training.cfg_dropout_p', 0.15),
                'gap_percentage': self.get('data.gap_percentage', 0.5),
                'vae': self.get('training.vae', {
                    'input_channels': 1,
                    'hidden_channels': 64,
                    'latent_dim': 64,
                    'kernel_size': 3,
                    'num_blocks': 3,
                    'downsample_stride': 2,
                }),
                'clap': None if self.get('training.clap', None) is None else {
                    'checkpoint': self.get('training.clap.checkpoint'),
                    'audio_model_type': self.get('training.clap.audio_model_type', 'HTSAT-base'),
                    'enable_fusion': self.get('training.clap.enable_fusion', True),
                },
                'cond_dim': self.get('model.network.cond_dim', 256),
            })
            # Keep legacy key for backward compatibility
            cfg["diff_params"] = self.get("diff_params")
        return cfg
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration."""
        cfg = {
            'dataset_type': self.get('data.dataset_type', 'mel_spectrogram'),
            'flac_path': self.get('paths.audio_filename', 'gapped_audio.wav'),
            'test_audio_filename': self.get('paths.test_audio_filename', 'wav_test.wav'),
            'n_mels': self.get('data.n_mels', 80),
            'gap_percentage': self.get('data.gap_percentage', self.get('training.mask_ratio', 0.75)),
            'n_fft': self.get('data.n_fft', 1024),
            'hop_length': self.get('data.hop_length', 256),
            'patch_size': self.get('model.patch_size', self.get('data.patch_size', 16)),
            'sample_rate': self.get('data.sample_rate', 16000),
            'segment_length': self.get('data.segment_length', 65536),
            'folder': self.get('data.folder', 'assets'),
            'batch_size': self.get('training.batch_size', 4),
        }
        return cfg
