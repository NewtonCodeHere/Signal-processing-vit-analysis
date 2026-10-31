"""
Demodulation Fidelity Characterization Module.
Benchmarks original versus recovered audio, calculates normalized cross-correlation,
delay-aligned MSE, SINAD, and evaluates envelope distortion.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple


class DemodulationAnalyzer:
    """
    Evaluates fidelity and error metrics of recovered audio signals.
    """

    @staticmethod
    def evaluate_fidelity(
        original_audio: np.ndarray,
        recovered_audio: np.ndarray,
        fs: int = 200000
    ) -> Dict[str, Any]:
        """
        Computes cross-correlation, phase delay alignment, MSE, and SINAD.

        Returns:
            Dictionary with correlation, MSE, delay, and SINAD metrics.
        """
        # Peak normalize both signals
        p_orig = np.max(np.abs(original_audio))
        p_rec = np.max(np.abs(recovered_audio))

        x_orig = original_audio / (p_orig + 1e-9)
        x_rec = recovered_audio / (p_rec + 1e-9)

        # Cross-correlation to estimate group delay
        corr = signal.correlate(x_rec, x_orig, mode="full")
        lags = signal.correlation_lags(len(x_rec), len(x_orig), mode="full")
        best_lag = lags[np.argmax(corr)]

        # Normalized correlation coefficient
        norm_orig = np.sqrt(np.sum(x_orig ** 2) + 1e-12)
        norm_rec = np.sqrt(np.sum(x_rec ** 2) + 1e-12)
        cross_corr = float(np.max(corr) / (norm_orig * norm_rec))

        # Align signals for point-by-point error analysis
        if best_lag > 0:
            aligned_rec = x_rec[best_lag:]
            aligned_orig = x_orig[:-best_lag]
        elif best_lag < 0:
            aligned_rec = x_rec[:best_lag]
            aligned_orig = x_orig[-best_lag:]
        else:
            aligned_rec = x_rec
            aligned_orig = x_orig

        min_len = min(len(aligned_rec), len(aligned_orig))
        aligned_rec = aligned_rec[:min_len]
        aligned_orig = aligned_orig[:min_len]

        # Mean Squared Error (MSE)
        mse = float(np.mean((aligned_rec - aligned_orig) ** 2))

        # Error signal: distortion + noise
        error_signal = aligned_rec - aligned_orig
        sig_pwr = float(np.mean(aligned_orig ** 2))
        err_pwr = float(np.mean(error_signal ** 2))

        sinad_db = 10.0 * np.log10(sig_pwr / (err_pwr + 1e-12))

        return {
            "cross_correlation": cross_corr,
            "mean_squared_error": mse,
            "delay_samples": int(best_lag),
            "delay_time_ms": float((best_lag / fs) * 1000.0),
            "sinad_db": float(sinad_db),
            "aligned_original": aligned_orig.tolist()[:1000],  # sample for UI overlay
            "aligned_recovered": aligned_rec.tolist()[:1000]
        }
