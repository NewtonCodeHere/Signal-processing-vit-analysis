"""
Gain & Linearity Characterization Module.
Measures Pin vs Pout transfer characteristics, small-signal voltage/power gain,
and detects the 1 dB compression point (P1dB) and amplifier saturation ceiling.
"""

import numpy as np
from typing import Dict, Any, List
from receiver.lna_model import LNAModel


class GainAnalyzer:
    """
    Performs power sweep to measure receiver front-end gain and compression behavior.
    """

    def __init__(self):
        self.lna = LNAModel(seed=42)

    def sweep_input_power(
        self,
        gain_db: float = 20.0,
        noise_figure_db: float = 2.5,
        p1db_v: float = 2.5,
        pin_dbm_range: Tuple[float, float] = (-70.0, 10.0),
        step_db: float = 2.0
    ) -> Dict[str, Any]:
        """
        Sweeps input RF power across a range and measures output power and gain.

        Returns:
            Dictionary containing Pin, Pout, gain curves, and identified P1dB point.
        """
        pin_dbm_vals = np.arange(pin_dbm_range[0], pin_dbm_range[1] + step_db / 2.0, step_db)
        pout_dbm_vals = []
        actual_gain_vals = []

        fs = 200000
        duration = 0.05
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        test_freq = 40000.0

        r_system = 50.0  # Standard 50-Ohm RF reference impedance

        p1db_input_dbm = None
        p1db_output_dbm = None

        for p_in_dbm in pin_dbm_vals:
            # P(W) = 1e-3 * 10^(P_dBm / 10)
            p_in_w = 1e-3 * (10.0 ** (p_in_dbm / 10.0))
            # V_peak = sqrt(2 * R * P)
            v_peak = np.sqrt(2.0 * r_system * p_in_w)

            test_signal = v_peak * np.sin(2.0 * np.pi * test_freq * t)

            v_out, _ = self.lna.amplify(
                input_signal=test_signal,
                gain_db=gain_db,
                noise_figure_db=noise_figure_db,
                p1db_v=p1db_v
            )

            # Coherent fundamental power extraction (standard spectrum analyzer / lock-in measurement)
            ref_sin = np.sin(2.0 * np.pi * test_freq * t)
            ref_cos = np.cos(2.0 * np.pi * test_freq * t)
            v_fund_i = 2.0 * np.mean(v_out * ref_sin)
            v_fund_q = 2.0 * np.mean(v_out * ref_cos)
            v_fund_peak = np.sqrt(v_fund_i ** 2 + v_fund_q ** 2)

            p_out_w = (v_fund_peak ** 2) / (2.0 * r_system)
            p_out_dbm = 10.0 * np.log10(max(1e-18, p_out_w) / 1e-3)
            current_gain_db = p_out_dbm - p_in_dbm

            pout_dbm_vals.append(float(p_out_dbm))
            actual_gain_vals.append(float(current_gain_db))

            # Detect P1dB: gain drops by 1.0 dB relative to configured small-signal gain
            if p1db_input_dbm is None and (gain_db - current_gain_db) >= 1.0:
                p1db_input_dbm = float(p_in_dbm)
                p1db_output_dbm = float(p_out_dbm)

        return {
            "pin_dbm": pin_dbm_vals.tolist(),
            "pout_dbm": pout_dbm_vals,
            "measured_gain_db": actual_gain_vals,
            "small_signal_gain_db": float(gain_db),
            "p1db_input_dbm": p1db_input_dbm,
            "p1db_output_dbm": p1db_output_dbm,
            "saturation_voltage_v": float(p1db_v)
        }
