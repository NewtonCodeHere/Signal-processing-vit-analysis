"""
Unit tests for Receiver Characterization modules.
Tests frequency response, gain compression, modulation THD, and noise sweeps.
"""

import numpy as np
import pytest
from characterization.frequency_response import FrequencyResponseAnalyzer
from characterization.gain_analysis import GainAnalyzer
from characterization.modulation_analysis import ModulationAnalyzer
from characterization.noise_analysis import NoiseAnalyzer
from characterization.adc_analysis import ADCAnalyzer


def test_frequency_response_analysis():
    fra = FrequencyResponseAnalyzer(fs=200000)
    res = fra.analyze_filter(center_freq_hz=40000.0, bandwidth_khz=25.0, filter_order=4)
    
    assert res["center_freq_hz"] == 40000.0
    # Measured 3dB bandwidth should be close to 25 kHz
    assert 20000.0 <= res["measured_bw_3db_hz"] <= 30000.0
    assert res["shape_factor_20_3"] > 1.0


def test_gain_and_compression_analysis():
    ga = GainAnalyzer()
    res = ga.sweep_input_power(gain_db=20.0, p1db_v=2.0, pin_dbm_range=(-60.0, 10.0), step_db=5.0)
    
    assert len(res["pin_dbm"]) == len(res["pout_dbm"])
    # Small signal gain should be approx 20 dB at low input powers
    low_power_gain = res["measured_gain_db"][0]
    assert abs(low_power_gain - 20.0) < 2.0


def test_modulation_index_sweep():
    ma = ModulationAnalyzer(fs=200000)
    res = ma.sweep_modulation_index(m_range=(0.2, 1.4), step=0.3)
    
    # Check that sideband power grows with modulation index
    sb_powers = res["sideband_powers_w"]
    assert sb_powers[-1] > sb_powers[0]
    # Check that overmodulation THD increases
    assert res["thd_percent"][-1] > res["thd_percent"][0]


def test_adc_bit_depth_sweep():
    aa = ADCAnalyzer(fs=200000)
    res = aa.sweep_bit_depths(bit_depths=[8, 12, 16])
    
    # SQNR should monotonically increase with bit depth
    assert res["measured_sqnr_db"][2] > res["measured_sqnr_db"][1] > res["measured_sqnr_db"][0]
    # Quantization step should monotonically decrease
    assert res["quantization_steps_mv"][0] > res["quantization_steps_mv"][1] > res["quantization_steps_mv"][2]
