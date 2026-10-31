"""
Frequency Tuning & Downconversion Module.
Models receiver channel selection in the civil aviation VHF voice band (118.000–136.975 MHz).
Simulates Local Oscillator (LO) mixing, intermediate frequency (IF) translation,
and tuning offset errors.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any


class FrequencyTuner:
    """
    Simulates RF-to-IF downconversion and channel tuning for the VHF receiver.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def tune_and_downconvert(
        self,
        t: np.ndarray,
        rf_signal: np.ndarray,
        selected_freq_mhz: float = 121.500,
        carrier_freq_mhz: float = 121.500,
        if_center_hz: float = 40000.0,
        tuning_error_hz: float = 0.0,
        channel_bandwidth_khz: float = 25.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Downconverts the received RF signal to baseband I/Q components and produces
        a tuned Intermediate Frequency (IF) signal.

        Args:
            t: Time vector in seconds.
            rf_signal: Signal from LNA.
            selected_freq_mhz: User-tuned channel frequency (118.000 - 136.975 MHz).
            carrier_freq_mhz: Incoming aircraft carrier frequency (MHz).
            if_center_hz: Intermediate Frequency center in Hz (default 40 kHz).
            tuning_error_hz: Tuning oscillator offset/drift in Hz.
            channel_bandwidth_khz: Channel selectivity filter bandwidth (kHz).

        Returns:
            Tuple of (i_component, q_component, if_signal, tuning_metrics)
        """
        # Frequency offset between incoming signal and receiver tuning
        freq_offset_mhz = carrier_freq_mhz - selected_freq_mhz
        freq_offset_hz = (freq_offset_mhz * 1e6) + tuning_error_hz

        # Local oscillator reference mixing
        lo_freq = if_center_hz + freq_offset_hz
        lo_i = 2.0 * np.cos(2.0 * np.pi * lo_freq * t)
        lo_q = -2.0 * np.sin(2.0 * np.pi * lo_freq * t)

        raw_i = rf_signal * lo_i
        raw_q = rf_signal * lo_q

        # Low-pass filter for channel selection at baseband
        cutoff_hz = (channel_bandwidth_khz * 1000.0) / 2.0
        nyq = 0.5 * self.fs
        norm_cutoff = np.clip(cutoff_hz / nyq, 0.001, 0.99)
        sos_lpf = signal.butter(4, norm_cutoff, btype="lowpass", output="sos")

        filtered_i = signal.sosfiltfilt(sos_lpf, raw_i)
        filtered_q = signal.sosfiltfilt(sos_lpf, raw_q)

        # Reconstruct centered IF signal for downstream envelope detection
        if_signal = (
            filtered_i * np.cos(2.0 * np.pi * if_center_hz * t) -
            filtered_q * np.sin(2.0 * np.pi * if_center_hz * t)
        )

        metrics = {
            "selected_channel_mhz": float(selected_freq_mhz),
            "incoming_channel_mhz": float(carrier_freq_mhz),
            "tuning_error_hz": float(tuning_error_hz),
            "total_offset_hz": float(freq_offset_hz),
            "is_on_channel": bool(abs(freq_offset_hz) < 500.0),
            "channel_bandwidth_khz": float(channel_bandwidth_khz),
            "if_frequency_hz": float(if_center_hz)
        }

        return filtered_i, filtered_q, if_signal, metrics
