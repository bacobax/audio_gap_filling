import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import plot_spectrogram


class MelSpecRandomCropDataset(Dataset):
    """
    Dataset that:
      • Loads one .flac file
      • Computes an 80-bin mel-spectrogram (log-power in dB)
      • On each __getitem__, returns a random crop of fixed time-length
        shaped (1, 80, crop_frames)
    """

    def __init__(
        self,
        flac_path: str,
        gap_percentage: float,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        test: tuple[bool, str] = (False, None),
    ):
        self.test = test[0]
        self.test_filename = test[1]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gap_percentage = gap_percentage

        self.wave, self.sr = self.load_audio(flac_path)
        self.mel_db = self.compute_mel_spectrogram(self.wave)

        self.detect_gap()
        self.crop_frames = max(1, int(self.gap_frames / self.gap_percentage))
        self.context_frames = (self.crop_frames - self.gap_frames) // 2
        self.compute_valid_starts()

        if self.test and self.test_filename is not None:
            test_wave, _ = self.load_audio(self.test_filename)
            mel_power = librosa.feature.melspectrogram(
                y=test_wave,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            test_mel_db = librosa.power_to_db(mel_power, ref=np.max)
            self.test_mel_db = self.normalize_spectrogram(test_mel_db, self.min_val, self.denom)
        elif self.test:
            raise ValueError("Test mode requires a filename to be specified.")

        if self.num_frames < self.crop_frames:
            pad = self.crop_frames - self.num_frames
            self.mel_db = np.pad(self.mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=0)
            self.num_frames = self.crop_frames

    def load_audio(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        wave, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
        return wave, sr

    def compute_mel_spectrogram(self, audio):
        mel_power = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        self.min_val = mel_db.min()
        self.max_val = mel_db.max()
        self.denom = self.max_val - self.min_val if self.max_val != self.min_val else 1
        return self.normalize_spectrogram(mel_db, self.min_val, self.denom)

    def detect_gap(self):
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
        if self.gap_frames == 0:
            raise ValueError("No silent gap detected in the spectrogram.")
        self.num_frames = self.mel_db.shape[1]

    def compute_valid_starts(self):
        if self.test:
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

    # ───────────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.valid_starts)

    # ───────────────────────────────────────────────────────────────────────────
    def __getitem__(self, _):
        """Return a random [1, 80, crop_frames] tensor, and start/end times in seconds."""
        start = self.valid_starts[_]  # deterministic
        # or to sample randomly use:
        # start = random.choice(self.valid_starts)
        end = start + self.crop_frames
        crop = self.mel_db[:, start:end].copy()              # [80, crop_frames]

        crop_tensor = torch.from_numpy(crop).float().unsqueeze(0)  # [1, 80, crop_frames]
        hop_length = 256  # default value unless overridden
        if hasattr(self, "sr"):
            sr = self.sr
        else:
            sr = 16000
        # Use the hop_length actually used in __init__
        if hasattr(self, "hop_length"):
            hop_length = self.hop_length
        start_time_sec = start * hop_length / sr
        end_time_sec = end * hop_length / sr
        gap_start_sec = self.gap_start_col * hop_length / sr
        gap_end_sec = (self.gap_start_col + self.gap_frames) * hop_length / sr

        if self.test:
            target_spectrogram = self.test_mel_db[:, start:end]
            target_spectrogram_slice = torch.from_numpy(target_spectrogram).float().unsqueeze(0)  # [1, 80, crop_frames]
            return (
                crop_tensor,
                target_spectrogram_slice,
                start_time_sec,
                end_time_sec,
                gap_start_sec,
                gap_end_sec,
                self.gap_start_col,
                self.gap_start_col + self.gap_frames,
            )

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
        )

    def reconstruct_spectrogram(self, spectrogram_slice: torch.Tensor, start_frame_idx: int, end_frame_idx: int):

        if spectrogram_slice.ndim == 3:
            spectrogram_slice = spectrogram_slice.squeeze(0)  # [n_mels, crop_frames]

        reconstructed = self.mel_db.copy()
        reconstructed[:, start_frame_idx:end_frame_idx] = spectrogram_slice.detach().cpu().numpy()
        return reconstructed

    def normalize_spectrogram(self, normal_spectrogram, min_val, denom):

        return (normal_spectrogram - min_val) / denom
    
    def inverse_fn(self, normed: torch.Tensor):
        return normed * self.denom + self.min_val


if __name__ == "__main__":


    mydataset = MelSpecRandomCropDataset(
        gap_percentage=0.25,
        flac_path="gapped_audio.wav",
        hop_length=256,
        n_fft=1024,
        n_mels=80,
        test=(True, "wav_test.wav")  # Set to True if you want to test with a specific file
    )

    dataloader = DataLoader(mydataset, batch_size=1, shuffle=True)

    for i, (gap_slice, target, start_sec, end_sec, _, _, gap_start_sec, gap_end_sec) in enumerate(dataloader):

        plot_spectrogram(gap_slice, start_sec, end_sec)

        plt.imshow(target.squeeze().detach().cpu().numpy(), aspect="auto",
                   origin="lower", cmap="viridis")
        plt.colorbar(label="Magnitude (arbitrary units)")
        plt.title(
            f"Masked Spectrogram Heatmap ({start_sec.item()}, {end_sec.item()}) duration: {(end_sec - start_sec).item()}")
        plt.tight_layout()
        plt.show()