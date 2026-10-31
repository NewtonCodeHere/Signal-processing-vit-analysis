"""
ADC Quantization Characterization Module.
Evaluates quantization step size, error distribution histograms,
and compares empirical SQNR against the theoretical 6.02*N + 1.76 dB bound across bit depths.
"""

import numpy as np
from typing import Dict, Any, List
from receiver.adc_simulator import ADCSimulator


class ADCAnalyzer:
    """
    Analyzes analog-to-digital converter performance across various bit resolutions.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs
        self.adc = ADCSimulator(sim_fs=self.fs)

    def sweep_bit_depths(
        self,
        bit_depths: List[int] = [8, 10, 12, 14, 16, 24],
        v_ref: float = 1.0
    ) -> Dict[str, Any]:
        """
        Sweeps ADC bit depths and measures quantization step and SQNR.

        Returns:
            Dictionary with bit depths, steps in mV, theoretical and measured SQNR curves.
        """
        duration = 0.1
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        test_audio = 0.95 * np.sin(2.0 * np.pi * 1000.0 * t)

        steps_mv = []
        theo_sqnr_vals = []
        meas_sqnr_vals = []
        error_sample_histograms = {}

        for bits in bit_depths:
            _, _, q_error, _, metrics = self.adc.quantize_and_sample(
                analog_signal=test_audio,
                resolution_bits=bits,
                v_ref=v_ref,
                target_fs=16000
            )

            steps_mv.append(metrics["quantization_step_mv"])
            theo_sqnr_vals.append(metrics["theoretical_sqnr_db"])
            meas_sqnr_vals.append(metrics["measured_sqnr_db"])

            # Compute error distribution histogram for 8-bit and 16-bit
            if bits in [8, 16]:
                counts, bin_edges = np.histogram(q_error, bins=30, density=True)
                error_sample_histograms[str(bits)] = {
                    "counts": counts.tolist(),
                    "bin_centers": (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist()
                }

        return {
            "bit_depths": bit_depths,
            "quantization_steps_mv": steps_mv,
            "theoretical_sqnr_db": theo_sqnr_vals,
            "measured_sqnr_db": meas_sqnr_vals,
            "error_histograms": error_sample_histograms
        }
