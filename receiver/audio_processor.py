"""
Audio Processing Module.
Implements post-demodulation audio conditioning:
1. Voice bandpass filtering (300 Hz - 3400 Hz standard civil aviation voice band).
2. DC component removal.
3. Amplitude normalization.
4. Spectral noise reduction.
Provides independent toggle switches for each stage.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any


class AudioProcessor:
    """
    Conditions and filters demodulated analog audio prior to ADC sampling.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def process(
        self,
        audio_signal: np.ndarray,
        enable_bpf: bool = True,
        low_cutoff_hz: float = 300.0,
        high_cutoff_hz: float = 3400.0,
        enable_dc_blocker: bool = True,
        enable_denoiser: bool = True,
        enable_normalize: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes demodulated audio through configurable conditioning stages.

        Args:
            audio_signal: Demodulated audio signal.
            enable_bpf: Apply 300-3400 Hz voice bandpass filter.
            low_cutoff_hz: Lower voice cutoff.
            high_cutoff_hz: Upper voice cutoff.
            enable_dc_blocker: Remove DC bias and sub-audible hum (< 50 Hz).
            enable_denoiser: Apply spectral subtraction noise reduction.
            enable_normalize: Peak normalize output.

        Returns:
            Tuple of (processed_audio, processing_metrics)
        """
        x = audio_signal.copy()
        nyq = 0.5 * self.fs

        # 1. DC Blocker
        if enable_dc_blocker:
            # High-pass filter at 50 Hz to strip DC and power hum
            sos_dc = signal.butter(2, 50.0 / nyq, btype="highpass", output="sos")
            x = signal.sosfiltfilt(sos_dc, x)

        # 2. Voice Band-Pass Filter (300 Hz - 3400 Hz)
        if enable_bpf:
            low_norm = np.clip(low_cutoff_hz / nyq, 0.0001, 0.99)
            high_norm = np.clip(high_cutoff_hz / nyq, low_norm + 0.001, 0.999)
            sos_voice = signal.butter(4, [low_norm, high_norm], btype="bandpass", output="sos")
            x = signal.sosfiltfilt(sos_voice, x)

        # 3. Spectral Subtraction Denoiser
        if enable_denoiser:
            # Estimate stationary noise floor from the quietest 10% frames
            n_fft = 1024
            hop_length = 512
            # Compute Short-Time Fourier Transform
            f, t_spec, zxx = signal.stft(x, fs=self.fs, nperseg=n_fft, noverlap=hop_length)
            mag = np.abs(zxx)
            phase = np.angle(zxx)

            # Noise floor estimated as minimum across time bins
            noise_est = np.percentile(mag, 10, axis=1, keepdims=True)
            # Spectral subtraction with floor
            subtracted_mag = np.maximum(mag - 1.2 * noise_est, 0.05 * mag)

            # Reconstruct time-domain signal via ISTFT
            zxx_clean = subtracted_mag * np.exp(1j * phase)
            _, x_denoised = signal.istft(zxx_clean, fs=self.fs, nperseg=n_fft, noverlap=hop_length)
            
            # Ensure matched length
            min_l = min(len(x), len(x_denoised))
            x[:min_l] = x_denoised[:min_l]

        # 4. Amplitude Normalization
        if enable_normalize:
            peak = np.max(np.abs(x))
            if peak > 1e-9:
                x = x / peak

        # Compute SNR and harmonic power
        p_raw = np.mean(audio_signal ** 2)
        p_proc = np.mean(x ** 2)

        metrics = {
            "enable_bpf": bool(enable_bpf),
            "low_cutoff_hz": float(low_cutoff_hz),
            "high_cutoff_hz": float(high_cutoff_hz),
            "enable_dc_blocker": bool(enable_dc_blocker),
            "enable_denoiser": bool(enable_denoiser),
            "enable_normalize": bool(enable_normalize),
            "output_rms": float(np.sqrt(p_proc)),
            "peak_to_average_ratio": float(np.max(np.abs(x)) / (np.sqrt(p_proc) + 1e-9))
        }

        return x, metrics
