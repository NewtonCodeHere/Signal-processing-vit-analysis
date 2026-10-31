"""
AM Demodulator Module.
Recreates the AM demodulation stage of the aviation receiver.
Supports:
1. Physical diode envelope detector with RC discharge dynamics (exhibiting diagonal clipping).
2. Analytic Hilbert / I-Q envelope detector.
3. Synchronous coherent product detector.

Computes MSE, cross-correlation with original audio, and reconstruction fidelity metrics.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any, Optional


class AMDemodulator:
    """
    Simulates AM signal demodulation to recover the pilot audio waveform.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def demodulate(
        self,
        t: np.ndarray,
        modulated_signal: np.ndarray,
        method: str = "envelope",
        rc_us: float = 45.0,
        carrier_freq_hz: float = 40000.0,
        original_audio: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Demodulates the modulated RF/IF signal.

        Args:
            t: Time vector (seconds).
            modulated_signal: Modulated signal from IF/Tuning stage.
            method: 'envelope' (physical diode RC detector), 'hilbert' (analytic), or 'coherent'.
            rc_us: RC discharge time constant in microseconds.
            carrier_freq_hz: Carrier frequency for coherent demodulation.
            original_audio: Optional original baseband audio for error benchmarking.

        Returns:
            Tuple of (demodulated_audio, demod_metrics)
        """
        dt = 1.0 / self.fs

        if method.lower() == "envelope":
            # Physical diode envelope detector with exponential capacitor discharge
            rc_sec = rc_us * 1e-6
            decay_factor = np.exp(-dt / (rc_sec + 1e-12))
            
            rectified = np.abs(modulated_signal)
            envelope = np.zeros_like(rectified)
            v_cap = 0.0
            
            for i in range(len(rectified)):
                v_in = rectified[i]
                if v_in > v_cap:
                    v_cap = v_in  # Diode conducts and charges capacitor
                else:
                    v_cap = v_cap * decay_factor  # Capacitor discharges through R
                envelope[i] = v_cap

            # Remove residual carrier ripple via lowpass filter
            nyq = 0.5 * self.fs
            lpf_cutoff = min(5000.0, 0.45 * nyq)
            sos_post = signal.butter(4, lpf_cutoff / nyq, btype="lowpass", output="sos")
            audio_raw = signal.sosfiltfilt(sos_post, envelope)

        elif method.lower() == "hilbert":
            # Analytic envelope detector using Hilbert transform
            analytic_signal = signal.hilbert(modulated_signal)
            audio_raw = np.abs(analytic_signal)

        elif method.lower() == "coherent":
            # Synchronous product detector: s(t) * 2*cos(2*pi*fc*t) followed by LPF
            carrier_ref = 2.0 * np.cos(2.0 * np.pi * carrier_freq_hz * t)
            multiplied = modulated_signal * carrier_ref
            nyq = 0.5 * self.fs
            sos_lpf = signal.butter(4, 5000.0 / nyq, btype="lowpass", output="sos")
            audio_raw = signal.sosfiltfilt(sos_lpf, multiplied)

        else:
            raise ValueError(f"Unknown demodulation method: {method}")

        # Remove DC component introduced by the full carrier
        audio_ac = audio_raw - np.mean(audio_raw)

        # Normalize recovered audio
        peak = np.max(np.abs(audio_ac))
        if peak > 1e-9:
            audio_norm = audio_ac / peak
        else:
            audio_norm = audio_ac

        # Compare with original audio if provided
        metrics = {
            "demod_method": method,
            "rc_time_constant_us": float(rc_us),
            "dc_offset_volts": float(np.mean(audio_raw)),
            "peak_recovered_volts": float(peak)
        }

        if original_audio is not None and len(original_audio) == len(audio_norm):
            # Peak normalize original
            orig_peak = np.max(np.abs(original_audio))
            orig_norm = original_audio / (orig_peak + 1e-9)

            # Compute cross-correlation to find optimal delay alignment
            corr = signal.correlate(audio_norm, orig_norm, mode="full")
            lags = signal.correlation_lags(len(audio_norm), len(orig_norm), mode="full")
            best_lag = lags[np.argmax(corr)]

            # Compute normalized correlation coefficient rho
            norm_orig = np.sqrt(np.sum(orig_norm ** 2) + 1e-12)
            norm_rec = np.sqrt(np.sum(audio_norm ** 2) + 1e-12)
            cross_corr = np.max(corr) / (norm_orig * norm_rec)

            # Delay-aligned MSE
            if best_lag > 0:
                aligned_rec = audio_norm[best_lag:]
                aligned_orig = orig_norm[:-best_lag]
            elif best_lag < 0:
                aligned_rec = audio_norm[:best_lag]
                aligned_orig = orig_norm[-best_lag:]
            else:
                aligned_rec = audio_norm
                aligned_orig = orig_norm

            min_len = min(len(aligned_rec), len(aligned_orig))
            mse = float(np.mean((aligned_rec[:min_len] - aligned_orig[:min_len]) ** 2))

            # SNR of recovered signal relative to distortion/error
            err_power = mse
            sig_power = float(np.mean(aligned_orig[:min_len] ** 2))
            fidelity_snr_db = 10.0 * np.log10(sig_power / (err_power + 1e-12))

            metrics.update({
                "cross_correlation": float(cross_corr),
                "mean_squared_error": float(mse),
                "delay_samples": int(best_lag),
                "fidelity_snr_db": float(fidelity_snr_db)
            })

        return audio_norm, metrics
