"""
Spectrogram Generation Module.
Computes 128-bin Log-Mel Spectrograms resized to 128x128 matrices,
matching the exact input format expected by the Vision Transformer (ViT) and CNN architectures.
"""

import numpy as np
import librosa
from typing import Tuple


class SpectrogramExtractor:
    """
    Extracts time-frequency representations (STFT and Log-Mel Spectrograms) from audio.
    """

    def __init__(
        self,
        fs: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400
    ):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def compute_log_mel_spectrogram(
        self,
        audio: np.ndarray,
        target_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Computes 128-bin Log-Mel Spectrogram normalized to [0, 1] with shape (128, 128).

        Args:
            audio: 1D audio waveform at 16 kHz.
            target_size: Desired (n_mels, n_frames) matrix size, default (128, 128).

        Returns:
            2D numpy array of shape (128, 128).
        """
        # Ensure minimum length for STFT
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)), mode="constant")

        # Compute Mel Spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=50.0,
            fmax=self.fs / 2.0,
            power=2.0
        )

        # Convert power spectrogram to decibels (Log-Mel)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Interpolate / resize along time axis to achieve exact target_size (128, 128)
        current_mels, current_frames = log_mel.shape
        target_mels, target_frames = target_size

        if current_frames != target_frames or current_mels != target_mels:
            from scipy.ndimage import zoom
            zoom_factors = (target_mels / current_mels, target_frames / current_frames)
            log_mel_128 = zoom(log_mel, zoom_factors, order=1)
        else:
            log_mel_128 = log_mel

        # Normalize to [0.0, 1.0]
        min_val = np.min(log_mel_128)
        max_val = np.max(log_mel_128)
        if (max_val - min_val) > 1e-6:
            norm_log_mel = (log_mel_128 - min_val) / (max_val - min_val)
        else:
            norm_log_mel = np.zeros_like(log_mel_128)

        return norm_log_mel.astype(np.float32)

    def compute_stft_spectrum(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes standard linear Short-Time Fourier Transform (STFT) for visualization.

        Returns:
            Tuple of (frequencies_hz, time_sec, magnitude_db)
        """
        stft_matrix = librosa.stft(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        mag_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
        freqs = librosa.fft_frequencies(sr=self.fs, n_fft=self.n_fft)
        times = librosa.frames_to_time(np.arange(mag_db.shape[1]), sr=self.fs, hop_length=self.hop_length)

        return freqs, times, mag_db
