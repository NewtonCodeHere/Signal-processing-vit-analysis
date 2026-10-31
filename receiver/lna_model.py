"""
Low Noise Amplifier (LNA) Module.
Simulates the RF pre-amplifier stage of the aviation receiver.
Models small-signal voltage gain, noise figure (NF) contribution via Friis formula,
and non-linear saturation (1 dB compression point P1dB).
"""

import numpy as np
from typing import Tuple, Dict, Any


class LNAModel:
    """
    Mathematical behavioral model of the receiver Low Noise Amplifier.
    """

    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def amplify(
        self,
        input_signal: np.ndarray,
        gain_db: float = 20.0,
        noise_figure_db: float = 2.5,
        p1db_v: float = 2.5,
        bandwidth_hz: float = 25000.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Amplifies input signal with gain, adds internal thermal noise, and models saturation.

        Args:
            input_signal: Input RF signal from the filter.
            gain_db: LNA gain in decibels.
            noise_figure_db: LNA noise figure in decibels.
            p1db_v: Saturation / 1dB compression voltage ceiling.
            bandwidth_hz: Channel bandwidth for thermal noise computation.

        Returns:
            Tuple of (amplified_signal, lna_metrics)
        """
        # Convert gain from dB to linear voltage factor
        linear_voltage_gain = 10.0 ** (gain_db / 20.0)

        # Ideal linear amplification
        v_amplified_linear = input_signal * linear_voltage_gain

        # Internal thermal noise calculation (IEEE definition: F = 1 + T_e/T_0)
        # Noise factor F = 10^(NF/10)
        f_factor = 10.0 ** (noise_figure_db / 10.0)
        k_boltzmann = 1.380649e-23  # J/K
        t_ref = 290.0  # Kelvin (standard reference temperature)
        # Added noise power referred to output: P_n,out = k * T * B * (F - 1) * G_power
        g_power = 10.0 ** (gain_db / 10.0)
        p_noise_added = k_boltzmann * t_ref * bandwidth_hz * (f_factor - 1.0) * g_power

        # Scale thermal noise to typical normalized simulation impedance (50 Ohm normalized to V)
        # For simulation demonstration, inject equivalent proportional Gaussian noise
        noise_voltage_rms = np.sqrt(max(1e-15, p_noise_added * 50.0))
        # Add a baseline physical demonstration floor proportional to (F-1)
        sim_noise_rms = (f_factor - 1.0) * 0.002 * (linear_voltage_gain / 10.0)
        total_noise = self.rng.normal(0.0, sim_noise_rms + noise_voltage_rms, size=len(input_signal))

        v_with_noise = v_amplified_linear + total_noise

        # Non-linear saturation model (smooth soft-clipping via tanh)
        # v_out = V_sat * tanh(v_in_amplified / V_sat)
        v_out = p1db_v * np.tanh(v_with_noise / (p1db_v + 1e-9))

        # Metrics computation
        p_in = np.mean(input_signal ** 2)
        p_out = np.mean(v_out ** 2)
        actual_gain_db = 10.0 * np.log10(p_out / (p_in + 1e-12)) if p_in > 1e-12 else gain_db

        metrics = {
            "configured_gain_db": float(gain_db),
            "actual_gain_db": float(actual_gain_db),
            "linear_gain": float(linear_voltage_gain),
            "noise_figure_db": float(noise_figure_db),
            "noise_factor": float(f_factor),
            "p1db_limit_v": float(p1db_v),
            "peak_output_v": float(np.max(np.abs(v_out))),
            "noise_added_rms_v": float(sim_noise_rms)
        }

        return v_out, metrics
