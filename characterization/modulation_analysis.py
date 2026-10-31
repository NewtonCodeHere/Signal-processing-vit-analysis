"""
Modulation Characterization Module.
Analyzes AM DSB-FC sideband power distribution, modulation efficiency,
and Total Harmonic Distortion (THD) across under-modulation, 100% modulation, and over-modulation.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, List
from receiver.am_modulator import AMModulator
from receiver.am_demodulator import AMDemodulator


class ModulationAnalyzer:
    """
    Performs modulation index sweeps and computes power distribution and harmonic distortion.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs
        self.modulator = AMModulator(fs=self.fs)
        self.demodulator = AMDemodulator(fs=self.fs)

    def sweep_modulation_index(
        self,
        m_range: Tuple[float, float] = (0.1, 1.5),
        step: float = 0.1,
        fm_hz: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Sweeps modulation index m and computes sideband power, efficiency, and THD.

        Returns:
            Dictionary of modulation indices, sideband powers, efficiencies, and THD percentages.
        """
        m_vals = np.arange(m_range[0], m_range[1] + step / 2.0, step)
        carrier_powers = []
        sideband_powers = []
        efficiencies = []
        thd_percentages = []

        duration = 0.1
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        mod_audio = np.cos(2.0 * np.pi * fm_hz * t)

        for m in m_vals:
            # Modulate
            am_sig, m_metrics = self.modulator.modulate(
                t=t,
                modulating_signal=mod_audio,
                carrier_freq_hz=40000.0,
                carrier_amplitude=1.0,
                modulation_index=float(m)
            )

            carrier_powers.append(m_metrics["carrier_power_w"])
            sideband_powers.append(m_metrics["sideband_power_w"])
            efficiencies.append(m_metrics["efficiency_percent"])

            # Demodulate with envelope detector to inspect harmonic distortion
            rec_audio, _ = self.demodulator.demodulate(
                t=t,
                modulated_signal=am_sig,
                method="envelope",
                carrier_freq_hz=40000.0
            )

            # Calculate THD from FFT spectrum
            fft_vals = np.abs(np.fft.rfft(rec_audio))
            fft_freqs = np.fft.rfftfreq(len(rec_audio), 1.0 / self.fs)

            # Find fundamental bin around fm_hz
            fund_bin = np.argmin(np.abs(fft_freqs - fm_hz))
            fund_amp = fft_vals[fund_bin]

            # Sum power of first 5 harmonics
            harmonic_power = 0.0
            for h in range(2, 6):
                h_bin = np.argmin(np.abs(fft_freqs - (h * fm_hz)))
                harmonic_power += (fft_vals[h_bin] ** 2)

            thd = (np.sqrt(harmonic_power) / (fund_amp + 1e-9)) * 100.0
            thd_percentages.append(float(min(100.0, thd)))

        return {
            "modulation_indices": m_vals.tolist(),
            "carrier_powers_w": carrier_powers,
            "sideband_powers_w": sideband_powers,
            "efficiencies_percent": efficiencies,
            "thd_percent": thd_percentages,
            "optimal_linear_m": 0.80,
            "overmodulation_threshold": 1.00
        }
