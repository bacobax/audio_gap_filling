
import torch
from matplotlib import pyplot as plt

def MCD(n1, n2, lower_bound, upper_bound):
    common_divisors = []
    for i in range(lower_bound + 1, upper_bound):
        if n1 % i == 0 and n2 % i == 0:
            common_divisors.append(i)
    return common_divisors


def plot_spectrogram(spectrogram, start_sec, end_sec):
    plt.imshow(spectrogram.squeeze().detach().cpu().numpy(), aspect="auto",
               origin="lower", cmap="viridis")
    plt.colorbar(label="Magnitude (arbitrary units)")
    plt.title(
        f"Masked Spectrogram Heatmap ({start_sec}, {end_sec}) duration: {(end_sec - start_sec)}")
    plt.tight_layout()
    plt.show()
