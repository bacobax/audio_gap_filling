"""
Mel spectrogram dataset for audio processing.
"""
import os
import numpy as np
import torch
import librosa
import random
from typing import Tuple, Optional, Union, Dict, Any
from torch.utils.data import Dataset
from pprint import pprint
from ..core.base_dataset import BaseDataset
from ..utils.audio_utils import compute_mel_spectrogram, normalize_spectrogram


class MelSpectrogramDataset(BaseDataset):
    """
    Dataset for loading and processing mel spectrograms from audio files.
    
    This dataset loads audio files, computes mel spectrograms, and provides
    random crops for training. It supports gap detection and test mode for
    evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mel spectrogram dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        print("DATASET CONFIG")
        pprint(self.config)

        required_keys = ['flac_path', 'gap_percentage']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate file existence
        flac_path = self.config['flac_path']
        if not os.path.isfile(flac_path):
            raise FileNotFoundError(f"Audio file not found: {flac_path}")
        
        # Validate test configuration
        test_config = self.config.get('test', (False, None))
        if test_config[0] and test_config[1] is None:
            raise ValueError("Test mode requires a test filename to be specified.")
    
    def _setup_dataset(self) -> None:
        """Setup the dataset by loading audio and computing spectrograms."""
        # Extract configuration
        self.flac_path = self.config['flac_path']
        self.gap_percentage = self.config['gap_percentage']
        self.n_fft = self.config.get('n_fft', 1024)
        self.hop_length = self.config.get('hop_length', 256)
        self.n_mels = self.config.get('n_mels', 80)
        self.test = self.config.get('test', (False, None))
        
        # Load audio and compute spectrogram
        self.wave, self.sr = self._load_audio(self.flac_path)
        self.mel_db = self._compute_mel_spectrogram(self.wave)
        
        # Detect gap and setup cropping
        self._detect_gap()
        raw_crop = max(1, int(self.gap_frames / self.gap_percentage))
        patch_size = self.config.get("patch_size", 16)  # Default patch size
        self.crop_frames = ((raw_crop + patch_size - 1) // patch_size) * patch_size
        self.context_frames = (self.crop_frames - self.gap_frames) // 2
        self._compute_valid_starts()
        
        # Setup test data if in test mode
        if self.test[0] and self.test[1] is not None:
            self._setup_test_data()
        
        # Handle padding if necessary
        if self.num_frames < self.crop_frames:
            self._pad_spectrogram()
    
    def get_crop_frames(self) -> int:
        """Get the number of frames to crop."""
        return self.crop_frames
    
    def _load_audio(self, path: str) -> Tuple[np.ndarray, int | float]:
        """
        Load audio file.
        
        Args:
            path: Path to the audio file
            
        Returns:
            Tuple containing audio data and sample rate
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        
        wave, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
        return wave, sr
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized mel spectrogram
        """
        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        
        # Store normalization parameters
        self.min_val = mel_db.min()
        self.max_val = mel_db.max()
        self.denom = self.max_val - self.min_val if self.max_val != self.min_val else 1
        
        return normalize_spectrogram(mel_db, self.min_val, self.denom)
    
    def _detect_gap(self) -> None:
        """Detect the largest gap in the spectrogram."""
        min_db = self.mel_db.min()
        silence_cols = (self.mel_db == min_db).all(axis=0)
        
        max_len, cur_len, best_start = 0, 0, 0
        for idx, val in enumerate(silence_cols):
            if val:
                if cur_len == 0:
                    cur_start = idx
                cur_len += 1
            else:
                if cur_len > max_len:
                    max_len, best_start = cur_len, cur_start
                cur_len = 0
        
        if cur_len > max_len:
            max_len, best_start = cur_len, cur_start
        
        self.gap_frames = max_len
        self.gap_start_col = best_start
        self.num_frames = self.mel_db.shape[1]
        
        if self.gap_frames == 0:
            raise ValueError("No silent gap detected in the spectrogram.")
    
    def _compute_valid_starts(self) -> None:
        """Compute valid starting positions for cropping."""
        if self.test[0]:
            start = self.gap_start_col - self.context_frames
            self.valid_starts = [start]
        else:
            gap_end_col = self.gap_start_col + self.gap_frames
            self.valid_starts = [
                s for s in range(self.num_frames - self.crop_frames + 1)
                if not (
                    s <= self.gap_start_col < s + self.crop_frames
                    or s < gap_end_col <= s + self.crop_frames
                )
            ]
            
            if len(self.valid_starts) == 0:
                raise ValueError("No valid windows outside the real gap.")
    
    def _setup_test_data(self) -> None:
        """Setup test data for evaluation."""
        test_wave, _ = self._load_audio(self.test[1]) # type: ignore
        mel_power = librosa.feature.melspectrogram(
            y=test_wave,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        test_mel_db = librosa.power_to_db(mel_power, ref=np.max)
        self.test_mel_db = normalize_spectrogram(test_mel_db, self.min_val, self.denom)
    
    def _pad_spectrogram(self) -> None:
        """Pad the spectrogram if it's shorter than crop_frames."""
        pad = self.crop_frames - self.num_frames
        self.mel_db = np.pad(self.mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=0)
        self.num_frames = self.crop_frames
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return min(len(self.valid_starts), 400) if not self.test[0] else len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Sample data (tensor or tuple of tensors)
        """
        start = random.choice(self.valid_starts)
        end = start + self.crop_frames
        crop = self.mel_db[:, start:end].copy()  # [80, crop_frames]
        
        crop_tensor = torch.from_numpy(crop).float().unsqueeze(0)  # [1, 80, crop_frames]
        
        # Calculate time information
        start_time_sec = start * self.hop_length / self.sr
        end_time_sec = end * self.hop_length / self.sr
        gap_start_sec = self.gap_start_col * self.hop_length / self.sr
        gap_end_sec = (self.gap_start_col + self.gap_frames) * self.hop_length / self.sr
        
        if self.test[0]:
            # Test mode: return target spectrogram and gap information
            target_spectrogram = self.test_mel_db[:, start:end]
            target_spectrogram_slice = torch.from_numpy(target_spectrogram).float().unsqueeze(0)
            
            # Calculate relative gap position
            relative_gap_start = self.gap_start_col - start
            relative_gap_end = self.gap_start_col + self.gap_frames - start
            
            # Clamp for safety
            relative_gap_start = max(0, relative_gap_start)
            relative_gap_end = min(self.crop_frames, relative_gap_end)
            
            return (
                crop_tensor,
                target_spectrogram_slice,
                start_time_sec,
                end_time_sec,
                gap_start_sec,
                gap_end_sec,
                relative_gap_start,
                relative_gap_end,
            ) # type: ignore
        
        # Training mode: return basic information
        return (
            crop_tensor,
            start,
            end,
            start_time_sec,
            end_time_sec,
            self.gap_start_col,
            self.gap_start_col + self.gap_frames,
            gap_start_sec,
            gap_end_sec,
        ) # type: ignore
    
    def reconstruct_spectrogram(self, spectrogram_slice: torch.Tensor, start_frame_idx: int, end_frame_idx: int) -> np.ndarray:
        """
        Reconstruct the full spectrogram from a slice.
        
        Args:
            spectrogram_slice: Spectrogram slice to insert
            start_frame_idx: Starting frame index
            end_frame_idx: Ending frame index
            
        Returns:
            Reconstructed spectrogram
        """
        if spectrogram_slice.ndim == 3:
            spectrogram_slice = spectrogram_slice.squeeze(0)  # [n_mels, crop_frames]
        
        reconstructed = self.mel_db.copy()
        reconstructed[:, start_frame_idx:end_frame_idx] = spectrogram_slice.detach().cpu().numpy()
        return reconstructed
    
    def inverse_normalize(self, normed: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalize a spectrogram.
        
        Args:
            normed: Normalized spectrogram
            
        Returns:
            Denormalized spectrogram
        """
        return normed * self.denom + self.min_val 