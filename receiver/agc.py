"""
Automatic Gain Control (AGC) Module.
Simulates closed-loop dynamic range compression in the aviation receiver audio chain.
Compensates for path loss variations, different aircraft transmitter powers,
and fluctuating pilot vocal loudness.
"""

import numpy as np
from typing import Tuple, Dict, Any


class AGC:
    """
    Automatic Gain Control with independent attack and release dynamics.
    """

    def __init__(self, fs: int = 200000):
        self.fs = fs

    def process(
        self,
        signal: np.ndarray,
        target_level: float = 0.70,
        attack_time_ms: float = 20.0,
        release_time_ms: float = 250.0,
        max_gain_db: float = 30.0,
        min_gain_db: float = -20.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Applies AGC leveling to the input audio waveform.

        Args:
            signal: Demodulated audio signal.
            target_level: Target peak amplitude (e.g. 0.70 V).
            attack_time_ms: Time constant for decreasing gain on signal peaks (ms).
            release_time_ms: Time constant for increasing gain during quiet passages (ms).
            max_gain_db: Maximum allowable gain in dB.
            min_gain_db: Minimum allowable gain in dB.

        Returns:
            Tuple of (leveled_signal, instantaneous_gain_db, agc_metrics)
        """
        dt = 1.0 / self.fs
        alpha_attack = np.exp(-dt / ((attack_time_ms * 1e-3) + 1e-12))
        alpha_release = np.exp(-dt / ((release_time_ms * 1e-3) + 1e-12))

        max_linear_gain = 10.0 ** (max_gain_db / 20.0)
        min_linear_gain = 10.0 ** (min_gain_db / 20.0)

        n = len(signal)
        output = np.zeros(n, dtype=np.float64)
        gain_history = np.zeros(n, dtype=np.float64)

        current_gain = 1.0
        env_detect = 0.0

        for i in range(n):
            x = abs(signal[i])
            # Envelope tracker
            if x > env_detect:
                env_detect = alpha_attack * env_detect + (1.0 - alpha_attack) * x
            else:
                env_detect = alpha_release * env_detect + (1.0 - alpha_release) * x

            # Compute error relative to target
            desired_gain = target_level / (env_detect + 1e-4)
            desired_gain = np.clip(desired_gain, min_linear_gain, max_linear_gain)

            # Smooth gain transition
            if desired_gain < current_gain:
                current_gain = alpha_attack * current_gain + (1.0 - alpha_attack) * desired_gain
            else:
                current_gain = alpha_release * current_gain + (1.0 - alpha_release) * desired_gain

            output[i] = signal[i] * current_gain
            gain_history[i] = 20.0 * np.log10(max(1e-6, current_gain))

        # Dynamic range calculation
        p_in_crest = np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-9)
        p_out_crest = np.max(np.abs(output)) / (np.sqrt(np.mean(output ** 2)) + 1e-9)

        metrics = {
            "target_level": float(target_level),
            "attack_time_ms": float(attack_time_ms),
            "release_time_ms": float(release_time_ms),
            "mean_gain_db": float(np.mean(gain_history)),
            "max_gain_applied_db": float(np.max(gain_history)),
            "min_gain_applied_db": float(np.min(gain_history)),
            "input_crest_factor": float(p_in_crest),
            "output_crest_factor": float(p_out_crest)
        }

        return output, gain_history, metrics
