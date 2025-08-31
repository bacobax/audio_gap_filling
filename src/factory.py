"""
Factory classes for creating models, datasets, and trainers.
"""
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from pprint import pprint
from .config.config_manager import ConfigManager
from .models.mae_vit import MAEViT
from .models.VAE import VAE, Decoder
from .data.mel_spectrogram_dataset import MelSpectrogramDataset
from .data.audio_dataset import AudioFolderDataset
from .data.vae_waveform_dataset import VAEWaveformDataset
import torch.distributed as dist
import os
from .training.diff_trainer import DiffusionTrainer
from .data.gap_waveform_dataset import GapWaveformDataset
from .training.mae_trainer import MAETrainer
from .training.vae_trainer import VAETrainer

class ModelFactory:
    """Factory for creating model instances."""
    
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
        model_type = model_type.lower()
        if model_type == 'mae_vit':
            return MAEViT(config)
        elif model_type == 'diffusion_unet':
            # Temporal 1-D masked inpainting U-Net operating in latent space
            from .models.inpaint_unet_1d import InpaintUNet1D
            # Configure model according to project defaults
            net_cfg = config.get('network', {})
            return InpaintUNet1D(
                in_channels=net_cfg.get('in_channels', 64 + 64 + 1),
                latents_channels=net_cfg.get('latents_channels', 64),
                channels_per_scale=tuple(net_cfg.get('channels_per_scale', [128, 256, 384, 512, 768])),
                self_attn_scales=tuple(net_cfg.get('self_attn_scales', [3, 4])),
                num_res_blocks=net_cfg.get('num_res_blocks', 2),
                cond_dim=net_cfg.get('cond_dim', 256),
                n_heads=net_cfg.get('n_heads', 8),
                use_gradient_checkpointing=net_cfg.get('gradient_checkpointing', False),
            )
        elif model_type == 'vae':
            decoder_cfg = config.get('decoder', {})
            decoder = Decoder(**decoder_cfg)
            return VAE(
                input_channels=config.get('input_channels', 1),
                hidden_channels=config.get('hidden_channels', 64),
                latent_dim=config.get('latent_dim', 64),
                kernel_size=config.get('kernel_size', 3),
                num_blocks=config.get('num_blocks', 3),
                downsample_stride=config.get('downsample_stride', 2),
                decoder=decoder,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class DatasetFactory:
    """Factory for creating dataset instances."""
    
    @staticmethod
    def create_dataset(dataset_type: str, config: Dict[str, Any]) -> Any:
        """
        Create a dataset instance.
        
        Args:
            dataset_type: Type of dataset to create
            config: Dataset configuration
            
        Returns:
            Dataset instance
        """
        dataset_type = dataset_type.lower()
        if dataset_type == 'mel_spectrogram':
            return MelSpectrogramDataset(config)
        elif dataset_type == 'audio_folder':
            return AudioFolderDataset(config)
        elif dataset_type == 'gap_waveform':
            return GapWaveformDataset(config)
        elif dataset_type == 'vae_waveform':
            return VAEWaveformDataset(config, split=config.get('split', 'train'))
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def create_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """
        Create a data loader.
        
        Args:
            dataset: Dataset to wrap
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


class TrainerFactory:
    """Factory for creating trainer instances."""
    
    @staticmethod
    def create_trainer(
        trainer_type: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> Any:
        """
        Create a trainer instance.
        
        Args:
            trainer_type: Type of trainer to create
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
            log_dir: Directory for logging
            
        Returns:
            Trainer instance
        """
        trainer_type = trainer_type.lower()
        if trainer_type == 'mae':
            return MAETrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
            )
        elif trainer_type == 'diffusion':
            return DiffusionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
            )
        elif trainer_type == 'vae':
            
            return VAETrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
            )
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")


class TrainingPipeline:
    """
    High-level training pipeline that orchestrates the entire training process.
    
    This class provides a simple interface for setting up and running training
    with minimal configuration.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        pprint(self.config)
        
        # Setup device
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        # Create components
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.trainer = None

    def _get_device(self):

        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def setup_model(self) -> None:
        """Setup the model."""
        model_config = self.config_manager.get_model_config()
        model_type = model_config.get('type', 'mae_vit')
        model_config['device'] = str(self.device)
        first_sample = next(iter(self.train_dataset))
        print(f"sample shape: {first_sample.shape}")
        # take the first sample in the train dataset
        model_config['audio_len'] = first_sample.shape[-1]

        print("MODEL CONFIG")
        pprint(model_config)
        self.model = ModelFactory.create_model(model_type, model_config)
        self.model.to(self.device)
        
        # Print model info
        print("Model created successfully")
    
    def setup_data(self) -> None:
        """Setup datasets and data loaders."""
        # Training dataset
        train_config = self.config_manager.get_data_config()
        dataset_type = train_config.get('dataset_type', 'mel_spectrogram')
        train_config['split'] = 'train'
        self.train_dataset = DatasetFactory.create_dataset(dataset_type, train_config)

        # Update image size if dataset provides it
        if hasattr(self.train_dataset, 'get_crop_frames'):
            image_size = (
                self.config_manager.get('data.n_mels', 80),
                self.train_dataset.get_crop_frames(),
            )
            self.config["image_size"] = image_size
            self.config_manager.set('model.image_size', image_size)

        # Validation dataset
        val_config = train_config.copy()
        val_config['split'] = 'val'
        if dataset_type != 'vae_waveform':
            val_config['test'] = (
                True,
                self.config_manager.get('paths.test_audio_filename', 'wav_test.wav')
            )
        self.val_dataset = DatasetFactory.create_dataset(dataset_type, val_config)

        # Create data loaders
        training_config = self.config_manager.get_training_config()
        load_batch_size = min(
            training_config.get('max_device_batch_size', training_config['batch_size']),
            training_config['batch_size'],
        )

        self.train_loader = DatasetFactory.create_dataloader(
            self.train_dataset,
            batch_size=load_batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DatasetFactory.create_dataloader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        print(f"Data setup complete: training and validation datasets created")
    
    def setup_trainer(self) -> None:
        """Setup the trainer."""
        training_config = self.config_manager.get_training_config()

        assert self.model is not None, "Model not setup. Call setup_model() first."
        assert self.train_loader is not None, "Training loader not setup. Call setup_data() first."

        trainer_type = training_config.get('trainer_type', 'mae')
        self.trainer = TrainerFactory.create_trainer(
            trainer_type=trainer_type,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=training_config,
            device=self.device
        )

        # Log chosen configuration file
        try:
            with open(self.config_path, 'r') as f:
                config_text = f.read()
            self.trainer.writer.add_text('config/path', self.config_path, 0)
            self.trainer.writer.add_text('config/file', f"```yaml\n{config_text}\n```", 0)
        except Exception as e:
            print(f"Warning: failed to log configuration file: {e}")
        
        # Check patch compatibility if method is available
        if hasattr(self.trainer, 'check_patch_compatibility'):
            self.trainer.check_patch_compatibility(self.train_dataset)
        
        print("Trainer setup complete")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training pipeline.
        
        Returns:
            Training history
        
        """



        # Only if dist is not initialized
        # if not dist.is_initialized():
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] = '12355'
        #     os.environ['WORLD_SIZE'] = '1'
        #     os.environ['RANK'] = '0'
        #     dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")
        
        num_epochs = self.config_manager.get('training.total_epoch', 8)
        print(f"Starting training for {num_epochs} epochs...")
        
        try:
            history = self.trainer.train(num_epochs)
            print("Training completed successfully!")
            return history
        except KeyboardInterrupt:
            print("Training interrupted by user")
            return {}
        finally:
            self.trainer.close()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Training history
        """
        print("Setting up training pipeline...")
        self.setup_model()
        self.setup_data()
        self.setup_trainer()
        
        return self.train() 
