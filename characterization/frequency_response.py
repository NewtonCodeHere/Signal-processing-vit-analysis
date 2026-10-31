"""
Receiver Frequency Response Characterization.
Analyzes RF/IF filter selectivity, passband ripple, -3dB bandwidth,
stopband attenuation, and filter shape factor.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple
from receiver.rf_filter import RFFilter


class FrequencyResponseAnalyzer:
    """
    Computes frequency response, 3dB bandwidth, and filter selectivity characteristics.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs
        self.rf_filter = RFFilter(fs=self.fs)

    def analyze_filter(
        self,
        center_freq_hz: float = 40000.0,
        bandwidth_khz: float = 25.0,
        filter_order: int = 4,
        filter_type: str = "butterworth",
        num_points: int = 2048
    ) -> Dict[str, Any]:
        """
        Calculates Bode plot data and critical RF filter characteristics.

        Returns:
            Dictionary containing frequency vector, magnitude (dB), phase (deg),
            measured -3dB cutoffs, passband, and selectivity shape factor.
        """
        freq_hz, mag_db, phase_deg = self.rf_filter.compute_frequency_response(
            center_freq_hz=center_freq_hz,
            bandwidth_khz=bandwidth_khz,
            filter_order=filter_order,
            filter_type=filter_type,
            num_points=num_points
        )

        peak_mag_db = np.max(mag_db)
        threshold_3db = peak_mag_db - 3.0
        threshold_20db = peak_mag_db - 20.0

        # Find -3 dB bandwidth indices
        passband_indices = np.where(mag_db >= threshold_3db)[0]
        if len(passband_indices) > 0:
            f_low_3db = float(freq_hz[passband_indices[0]])
            f_high_3db = float(freq_hz[passband_indices[-1]])
            measured_bw_3db_hz = f_high_3db - f_low_3db
        else:
            f_low_3db = center_freq_hz - (bandwidth_khz * 500.0)
            f_high_3db = center_freq_hz + (bandwidth_khz * 500.0)
            measured_bw_3db_hz = bandwidth_khz * 1000.0

        # Find -20 dB bandwidth indices for selectivity shape factor
        stopband_indices = np.where(mag_db >= threshold_20db)[0]
        if len(stopband_indices) > 0:
            bw_20db_hz = float(freq_hz[stopband_indices[-1]] - freq_hz[stopband_indices[0]])
            shape_factor_20_3 = bw_20db_hz / max(1.0, measured_bw_3db_hz)
        else:
            shape_factor_20_3 = 2.5

        return {
            "frequencies_hz": freq_hz.tolist(),
            "magnitude_db": mag_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "center_freq_hz": float(center_freq_hz),
            "configured_bw_khz": float(bandwidth_khz),
            "measured_f_low_3db_hz": float(f_low_3db),
            "measured_f_high_3db_hz": float(f_high_3db),
            "measured_bw_3db_hz": float(measured_bw_3db_hz),
            "shape_factor_20_3": float(shape_factor_20_3),
            "peak_gain_db": float(peak_mag_db),
            "filter_order": int(filter_order),
            "filter_type": filter_type
        }
