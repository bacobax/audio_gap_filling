"""
Visualization utility functions.
"""
import torch
from matplotlib import pyplot as plt


def plot_spectrogram(spectrogram: torch.Tensor, start_sec: float, end_sec: float) -> None:
    """
    Plot a spectrogram as a heatmap.
    
    Args:
        spectrogram: Spectrogram tensor to plot
        start_sec: Start time in seconds
        end_sec: End time in seconds
    """
    plt.imshow(
        spectrogram.squeeze().detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    plt.colorbar(label="Magnitude (arbitrary units)")
    plt.title(
        f"Masked Spectrogram Heatmap ({start_sec}, {end_sec}) duration: {(end_sec - start_sec)}"
    )
    plt.tight_layout()
    plt.show() 