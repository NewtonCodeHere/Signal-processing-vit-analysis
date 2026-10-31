"""
Audio DSP Filtering & Pre-Processing Module.
Performs audio conditioning, normalization, resampling to 16 kHz,
and high-frequency pre-emphasis filtering for speech feature extraction.
"""

import numpy as np
from scipy import signal
from scipy.signal import resample_poly
from fractions import Fraction
from typing import Tuple


class AudioPreprocessor:
    """
    Standard speech pre-processing pipeline for aviation voice communication.
    """

    def __init__(self, target_fs: int = 16000):
        self.target_fs = target_fs

    def resample_audio(self, audio: np.ndarray, orig_fs: int) -> np.ndarray:
        """
        Resamples input audio to target sampling rate (16 kHz).
        """
        if orig_fs == self.target_fs:
            return audio.astype(np.float32)

        import librosa
        resampled = librosa.resample(audio.astype(np.float32), orig_sr=orig_fs, target_sr=self.target_fs)
        return resampled.astype(np.float32)

    def apply_pre_emphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Applies pre-emphasis high-pass filter: y[n] = x[n] - alpha * x[n-1].
        Flattens speech spectral tilt and emphasizes higher-frequency formants.
        """
        return np.append(audio[0], audio[1:] - alpha * audio[:-1]).astype(np.float32)

    def bandpass_voice_spectrum(
        self,
        audio: np.ndarray,
        low_hz: float = 300.0,
        high_hz: float = 3400.0
    ) -> np.ndarray:
        """
        Applies standard aviation voice bandpass filter (300 Hz - 3400 Hz).
        """
        nyq = 0.5 * self.target_fs
        low_norm = np.clip(low_hz / nyq, 0.001, 0.99)
        high_norm = np.clip(high_hz / nyq, low_norm + 0.001, 0.999)
        sos = signal.butter(4, [low_norm, high_norm], btype="bandpass", output="sos")
        return signal.sosfiltfilt(sos, audio).astype(np.float32)

    def normalize_peak(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Peak normalizes audio to [-target_peak, +target_peak].
        """
        peak = np.max(np.abs(audio))
        if peak > 1e-9:
            return (audio / peak) * target_peak
        return audio
