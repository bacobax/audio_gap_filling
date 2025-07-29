"""
Audio processing utility functions.
"""
import numpy as np
import librosa
from typing import Tuple


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.
    
    Args:
        audio: Audio data
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel frequency bins
        
    Returns:
        Mel spectrogram in dB scale
    """
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return librosa.power_to_db(mel_power, ref=np.max)


def normalize_spectrogram(spectrogram: np.ndarray, min_val: float, denom: float) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range.
    
    Args:
        spectrogram: Input spectrogram
        min_val: Minimum value for normalization
        denom: Denominator for normalization
        
    Returns:
        Normalized spectrogram
    """
    return (spectrogram - min_val) / denom


def inverse_normalize_spectrogram(normalized: np.ndarray, min_val: float, denom: float) -> np.ndarray:
    """
    Inverse normalize spectrogram from [0, 1] range.
    
    Args:
        normalized: Normalized spectrogram
        min_val: Minimum value for denormalization
        denom: Denominator for denormalization
        
    Returns:
        Denormalized spectrogram
    """
    return normalized * denom + min_val 