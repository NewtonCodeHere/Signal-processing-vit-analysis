"""
RF Front-End Band-Pass Filter Module.
Models the selectivity of the receiver's RF front-end bandpass filter (LC / SAW / Helical).
Rejects out-of-band noise, adjacent channel emissions, and image frequencies.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any


class RFFilter:
    """
    Digital filter representing the RF Front-End Bandpass Filter stage.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def filter_signal(
        self,
        input_signal: np.ndarray,
        center_freq_hz: float = 40000.0,
        bandwidth_khz: float = 25.0,
        filter_order: int = 4,
        filter_type: str = "butterworth",
        passband_ripple_db: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies bandpass filtering centered around center_freq_hz.

        Args:
            input_signal: Incoming RF/IF voltage waveform.
            center_freq_hz: Carrier/IF center frequency in Hz.
            bandwidth_khz: Bandpass filter bandwidth in kHz.
            filter_order: Filter order.
            filter_type: 'butterworth' or 'chebyshev1'.
            passband_ripple_db: Allowed ripple in passband for Chebyshev.

        Returns:
            Tuple of (filtered_signal, filter_metrics)
        """
        bw_hz = bandwidth_khz * 1000.0
        low_cutoff = max(100.0, center_freq_hz - (bw_hz / 2.0))
        high_cutoff = min((self.fs / 2.0) - 100.0, center_freq_hz + (bw_hz / 2.0))

        # Normalized Nyquist frequencies
        nyq = 0.5 * self.fs
        low_norm = low_cutoff / nyq
        high_norm = high_cutoff / nyq

        # Ensure valid normalized range (0 < low < high < 1)
        low_norm = np.clip(low_norm, 0.001, 0.99)
        high_norm = np.clip(high_norm, low_norm + 0.001, 0.999)

        # Design digital filter
        if filter_type.lower() == "chebyshev1":
            sos = signal.cheby1(
                N=filter_order,
                rp=passband_ripple_db,
                Wn=[low_norm, high_norm],
                btype="bandpass",
                output="sos"
            )
        else:
            sos = signal.butter(
                N=filter_order,
                Wn=[low_norm, high_norm],
                btype="bandpass",
                output="sos"
            )

        # Apply zero-phase filtering
        filtered_signal = signal.sosfiltfilt(sos, input_signal)

        # Compute power before and after
        p_in = np.mean(input_signal ** 2)
        p_out = np.mean(filtered_signal ** 2)
        insertion_loss_db = 10.0 * np.log10((p_in / (p_out + 1e-12)) + 1e-12)

        metrics = {
            "center_freq_hz": float(center_freq_hz),
            "bandwidth_khz": float(bandwidth_khz),
            "low_cutoff_hz": float(low_cutoff),
            "high_cutoff_hz": float(high_cutoff),
            "filter_order": int(filter_order),
            "filter_type": filter_type,
            "insertion_loss_db": float(max(0.0, insertion_loss_db)),
            "sos_coefficients": sos
        }

        return filtered_signal, metrics

    def compute_frequency_response(
        self,
        center_freq_hz: float = 40000.0,
        bandwidth_khz: float = 25.0,
        filter_order: int = 4,
        filter_type: str = "butterworth",
        num_points: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the theoretical magnitude and phase response of the filter.

        Returns:
            Tuple of (freq_hz, magnitude_db, phase_degrees)
        """
        bw_hz = bandwidth_khz * 1000.0
        low_cutoff = max(100.0, center_freq_hz - (bw_hz / 2.0))
        high_cutoff = min((self.fs / 2.0) - 100.0, center_freq_hz + (bw_hz / 2.0))

        nyq = 0.5 * self.fs
        low_norm = np.clip(low_cutoff / nyq, 0.001, 0.99)
        high_norm = np.clip(high_cutoff / nyq, low_norm + 0.001, 0.999)

        if filter_type.lower() == "chebyshev1":
            sos = signal.cheby1(filter_order, 0.5, [low_norm, high_norm], btype="bandpass", output="sos")
        else:
            sos = signal.butter(filter_order, [low_norm, high_norm], btype="bandpass", output="sos")

        w, h = signal.sosfreqz(sos, worN=num_points, fs=self.fs)
        magnitude_db = 20.0 * np.log10(np.abs(h) + 1e-12)
        phase_deg = np.angle(h, deg=True)

        return w, magnitude_db, phase_deg
