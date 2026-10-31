"""
Noise & Sensitivity Characterization Module.
Performs an automated input SNR sweep to measure output audio SNR,
reconstruction fidelity, and demonstrates the classic AM envelope detection threshold effect.
"""

import numpy as np
from typing import Dict, Any, Tuple
from config.receiver_config import ReceiverParameters
from receiver.simulation_pipeline import ReceiverSimulationEngine


class NoiseAnalyzer:
    """
    Analyzes noise propagation through the complete receiver receiving chain.
    """

    def __init__(self):
        self.engine = ReceiverSimulationEngine()

    def sweep_input_snr(
        self,
        snr_min_db: float = -10.0,
        snr_max_db: float = 40.0,
        step_db: float = 5.0,
        signal_type: str = "tone"
    ) -> Dict[str, Any]:
        """
        Sweeps channel SNR and measures recovered audio SNR and fidelity.

        Returns:
            Dictionary containing input SNR array, output audio SNR array,
            correlation coefficient array, and detected threshold point.
        """
        snr_in_vals = np.arange(snr_min_db, snr_max_db + step_db / 2.0, step_db)
        snr_out_vals = []
        cross_corr_vals = []
        demod_mse_vals = []

        threshold_snr_db = None

        for snr_in in snr_in_vals:
            params = ReceiverParameters(
                signal_type=signal_type,
                tone_freq_hz=1000.0,
                duration_sec=0.2,
                channel_snr_db=float(snr_in),
                modulation_index=0.8,
                adc_resolution_bits=16
            )

            res = self.engine.run_simulation(params=params)
            demod_m = res["stages"]["stage5_demodulator"]["metrics"]

            snr_out = demod_m.get("fidelity_snr_db", 0.0)
            corr = demod_m.get("cross_correlation", 0.0)
            mse = demod_m.get("mean_squared_error", 1.0)

            snr_out_vals.append(float(snr_out))
            cross_corr_vals.append(float(corr))
            demod_mse_vals.append(float(mse))

            # Detect threshold effect: point where correlation drops below 0.70
            if threshold_snr_db is None and corr >= 0.70:
                threshold_snr_db = float(snr_in)

        return {
            "input_snr_db": snr_in_vals.tolist(),
            "output_snr_db": snr_out_vals,
            "cross_correlation": cross_corr_vals,
            "mean_squared_error": demod_mse_vals,
            "detected_threshold_snr_db": threshold_snr_db,
            "linear_region_min_snr_db": 15.0
        }
