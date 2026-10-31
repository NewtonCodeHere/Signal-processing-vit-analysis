"""
Unit tests for AM Modulation Engine.
Verifies carrier power, sideband power, modulation efficiency,
and envelope behavior at various modulation depths.
"""

import numpy as np
import pytest
from receiver.am_modulator import AMModulator


def test_am_modulation_powers():
    fs = 200000
    modulator = AMModulator(fs=fs)
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Single tone modulating signal
    fm = 1000.0
    mod_signal = np.cos(2.0 * np.pi * fm * t)
    
    # 100% modulation (m = 1.0), Ac = 2.0 V
    ac = 2.0
    m = 1.0
    am_sig, metrics = modulator.modulate(
        t=t,
        modulating_signal=mod_signal,
        carrier_freq_hz=40000.0,
        carrier_amplitude=ac,
        modulation_index=m
    )
    
    # Theoretical: Pc = Ac^2 / 2 = 2.0^2 / 2 = 2.0 W
    # Psb = (Ac^2 * m^2 * Px) / 2 = (4 * 1 * 0.5) / 2 = 1.0 W
    # Total = 3.0 W
    # Efficiency = Psb / Ptot = 1 / 3 = 33.33%
    assert np.isclose(metrics["carrier_power_w"], 2.0, rtol=1e-2)
    assert np.isclose(metrics["sideband_power_w"], 1.0, rtol=1e-2)
    assert np.isclose(metrics["total_power_w"], 3.0, rtol=1e-2)
    assert np.isclose(metrics["efficiency_percent"], 33.33, rtol=1e-2)
    assert metrics["is_overmodulated"] is False


def test_overmodulation():
    fs = 200000
    modulator = AMModulator(fs=fs)
    t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
    mod_signal = np.cos(2.0 * np.pi * 500.0 * t)
    
    _, metrics = modulator.modulate(
        t=t,
        modulating_signal=mod_signal,
        modulation_index=1.4
    )
    assert metrics["is_overmodulated"] is True
    assert metrics["min_envelope_volts"] < 0.0
