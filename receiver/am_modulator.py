"""
AM Modulator Module.
Implements standard Double Sideband Full Carrier (AM DSB-FC) modulation:
s(t) = Ac * [1 + m * x_n(t)] * cos(2*pi*fc*t)

Computes power distribution (carrier power, upper/lower sideband power, efficiency)
and handles under-, 100%-, and over-modulation.
"""

import numpy as np
from typing import Dict, Any, Tuple


class AMModulator:
    """
    Simulates the aircraft VHF AM transmitter stage.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def modulate(
        self,
        t: np.ndarray,
        modulating_signal: np.ndarray,
        carrier_freq_hz: float = 40000.0,
        carrier_amplitude: float = 1.0,
        modulation_index: float = 0.70
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Performs AM DSB-FC modulation.

        Args:
            t: Time array (seconds).
            modulating_signal: Baseband speech/audio signal.
            carrier_freq_hz: RF/IF carrier frequency in Hz.
            carrier_amplitude: Peak carrier voltage Ac (V).
            modulation_index: Modulation index m (e.g. 0.70 = 70%).

        Returns:
            Tuple of (am_signal, metrics_dict)
        """
        # Ensure modulating signal is normalized to [-1.0, 1.0]
        peak = np.max(np.abs(modulating_signal))
        if peak > 1e-9:
            x_norm = modulating_signal / peak
        else:
            x_norm = modulating_signal

        # Instantaneous envelope: A(t) = Ac * [1 + m * x_norm(t)]
        envelope = carrier_amplitude * (1.0 + modulation_index * x_norm)

        # Carrier wave: cos(2*pi*fc*t)
        carrier = np.cos(2.0 * np.pi * carrier_freq_hz * t)

        # AM DSB-FC output signal
        am_signal = envelope * carrier

        # Analytical power calculations
        p_carrier = 0.5 * (carrier_amplitude ** 2)
        modulating_pwr = np.mean(x_norm ** 2)
        p_sidebands = 0.5 * (carrier_amplitude ** 2) * (modulation_index ** 2) * modulating_pwr
        p_total = p_carrier + p_sidebands
        efficiency = (p_sidebands / p_total) * 100.0 if p_total > 1e-9 else 0.0

        is_overmodulated = modulation_index > 1.0
        min_envelope = np.min(envelope)

        metrics = {
            "carrier_power_w": float(p_carrier),
            "sideband_power_w": float(p_sidebands),
            "total_power_w": float(p_total),
            "efficiency_percent": float(efficiency),
            "modulation_index": float(modulation_index),
            "is_overmodulated": bool(is_overmodulated),
            "min_envelope_volts": float(min_envelope),
            "peak_envelope_volts": float(np.max(envelope)),
            "carrier_freq_hz": float(carrier_freq_hz)
        }

        return am_signal, metrics
