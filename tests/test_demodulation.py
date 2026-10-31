"""
Unit tests for Receiver Pipeline and AM Demodulation.
Verifies end-to-end signal flow from RF Antenna to 16 kHz WAV Audio.
"""

import numpy as np
import pytest
from config.receiver_config import ReceiverParameters
from receiver.simulation_pipeline import ReceiverSimulationEngine


def test_complete_receiver_pipeline_execution():
    engine = ReceiverSimulationEngine()
    params = ReceiverParameters(
        signal_type="tone",
        tone_freq_hz=1000.0,
        carrier_amplitude=1.0,
        modulation_index=0.8,
        duration_sec=0.2,
        channel_snr_db=35.0,
        adc_resolution_bits=16
    )
    
    results = engine.run_simulation(params=params)
    
    # Check that all 8 stages exist and have output signals
    assert "stage1_rf_input" in results["stages"]
    assert "stage2_rf_filter" in results["stages"]
    assert "stage3_lna" in results["stages"]
    assert "stage4_tuning" in results["stages"]
    assert "stage5_demodulator" in results["stages"]
    assert "stage6_audio_proc" in results["stages"]
    assert "stage7_adc" in results["stages"]
    assert "stage8_digital_output" in results["stages"]
    
    # Check digital output sampling rate is exactly 16000 Hz
    dig_stage = results["stages"]["stage8_digital_output"]
    assert dig_stage["sample_rate"] == 16000
    assert len(dig_stage["audio_samples"]) > 0
    
    # Check demodulation correlation with original tone
    demod_metrics = results["stages"]["stage5_demodulator"]["metrics"]
    assert demod_metrics["cross_correlation"] > 0.90
