"""
Analog-to-Digital Converter (ADC) Simulation Module.
Simulates uniform mid-tread quantization, sampling rate conversion,
quantization error calculation, and Signal-to-Quantization-Noise Ratio (SQNR) evaluation.
Supports 8-bit, 10-bit, 12-bit, 16-bit, and 24-bit resolutions.
"""

import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
from typing import Tuple, Dict, Any


class ADCSimulator:
    """
    Simulates analog-to-digital conversion of receiver audio into PCM digital samples.
    """

    def __init__(self, sim_fs: int = 200000):
        self.sim_fs = sim_fs

    def quantize_and_sample(
        self,
        analog_signal: np.ndarray,
        resolution_bits: int = 16,
        v_ref: float = 1.0,
        target_fs: int = 16000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Simulates analog-to-digital conversion.

        Args:
            analog_signal: Continuous-time analog voltage waveform.
            resolution_bits: ADC bit depth N (e.g. 8, 10, 12, 16, 24).
            v_ref: Full-scale reference voltage (+/- v_ref).
            target_fs: Output digital audio sampling frequency in Hz (default 16 kHz).

        Returns:
            Tuple of (
                analog_time,
                quantized_sim_signal,
                quantization_error,
                digital_output_16k,
                adc_metrics
            )
        """
        # Allowed discrete levels: 2^N
        num_levels = 2 ** resolution_bits
        q_step = (2.0 * v_ref) / num_levels

        # Mid-tread uniform quantizer: round to nearest quantum step
        # Input clamped to [-v_ref, v_ref - q_step]
        clamped = np.clip(analog_signal, -v_ref, v_ref - (q_step / 2.0))
        quantized_int = np.round(clamped / q_step)
        quantized_volts = quantized_int * q_step

        # Quantization noise waveform
        quantization_error = quantized_volts - analog_signal

        # Signal power and noise power
        p_sig = np.mean(analog_signal ** 2)
        p_q_noise = np.mean(quantization_error ** 2)

        # Theoretical SQNR = 6.02 * N + 1.76 dB (for full-scale sinusoidal input)
        theoretical_sqnr_db = 6.02 * resolution_bits + 1.76

        # Measured empirical SQNR
        if p_q_noise > 1e-15:
            measured_sqnr_db = 10.0 * np.log10(p_sig / p_q_noise)
        else:
            measured_sqnr_db = theoretical_sqnr_db

        # Downsample to digital target_fs (16 kHz for Speech Emotion Recognition)
        frac = Fraction(target_fs, self.sim_fs).limit_denominator(100)
        digital_audio_resampled = resample_poly(quantized_volts, frac.numerator, frac.denominator)

        # Normalize final digital samples to [-1.0, 1.0]
        peak_dig = np.max(np.abs(digital_audio_resampled))
        if peak_dig > 1e-9:
            digital_audio_resampled = digital_audio_resampled / peak_dig

        metrics = {
            "resolution_bits": int(resolution_bits),
            "num_quantization_levels": int(num_levels),
            "quantization_step_mv": float(q_step * 1000.0),
            "theoretical_sqnr_db": float(theoretical_sqnr_db),
            "measured_sqnr_db": float(measured_sqnr_db),
            "v_ref_volts": float(v_ref),
            "simulation_fs": int(self.sim_fs),
            "output_fs": int(target_fs),
            "num_digital_samples": len(digital_audio_resampled),
            "duration_sec": float(len(digital_audio_resampled) / target_fs)
        }

        # Return simulation-rate waveforms for UI plotting and downsampled 16kHz audio for SER
        t_sim = np.linspace(0, len(analog_signal) / self.sim_fs, len(analog_signal), endpoint=False)
        return t_sim, quantized_volts, quantization_error, digital_audio_resampled, metrics
