"""
Unit tests for ADC Quantization Simulator.
Verifies quantization steps, error levels, and SQNR agreement with 6.02*N + 1.76 dB.
"""

import numpy as np
import pytest
from receiver.adc_simulator import ADCSimulator


@pytest.mark.parametrize("bits,expected_sqnr_approx", [
    (8, 49.92),
    (10, 61.96),
    (12, 74.00),
    (16, 98.08),
])
def test_adc_quantization_sqnr(bits, expected_sqnr_approx):
    fs = 200000
    adc = ADCSimulator(sim_fs=fs)
    duration = 0.2
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Full scale sine wave
    sine_wave = 0.95 * np.sin(2.0 * np.pi * 1000.0 * t)
    
    _, _, q_error, digital_out, metrics = adc.quantize_and_sample(
        analog_signal=sine_wave,
        resolution_bits=bits,
        v_ref=1.0,
        target_fs=16000
    )
    
    # Theoretical SQNR
    assert np.isclose(metrics["theoretical_sqnr_db"], expected_sqnr_approx, atol=0.2)
    # Measured SQNR should be close to theoretical for full-scale sine (within 3 dB due to 0.95 peak)
    assert abs(metrics["measured_sqnr_db"] - expected_sqnr_approx) < 4.0
    # Quantization step in mV should match 2000 mV / 2^N
    expected_step_mv = (2000.0) / (2 ** bits)
    assert np.isclose(metrics["quantization_step_mv"], expected_step_mv, rtol=1e-3)
