"""
VHF Radio Channel Model.
Simulates free-space path loss attenuation and Additive White Gaussian Noise (AWGN)
representing the line-of-sight propagation link between the aircraft and ground station antenna.
"""

import numpy as np
from typing import Tuple, Dict, Any


class VHFChannelModel:
    """
    Models VHF wireless propagation channel effects.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def apply_channel(
        self,
        signal: np.ndarray,
        snr_db: float = 25.0,
        attenuation_db: float = 12.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies path loss attenuation and AWGN noise to the transmitted RF signal.

        Args:
            signal: Transmitted AM signal waveform.
            snr_db: Desired Signal-to-Noise Ratio in dB.
            attenuation_db: Path loss attenuation in dB.

        Returns:
            Tuple of (received_signal, channel_metrics)
        """
        # Apply path loss attenuation
        linear_atten = 10.0 ** (-attenuation_db / 20.0)
        attenuated_sig = signal * linear_atten

        # Measure attenuated signal power
        sig_power = np.mean(attenuated_sig ** 2)

        # Calculate required noise power for target SNR
        # SNR = 10 * log10(P_sig / P_noise) => P_noise = P_sig / (10^(SNR/10))
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear if snr_linear > 1e-9 else 1e-6
        noise_std = np.sqrt(noise_power)

        # Generate Gaussian noise
        noise = self.rng.normal(0.0, noise_std, size=len(signal))

        # Channel output
        received_signal = attenuated_sig + noise

        # Verify actual measured SNR
        measured_snr = 10.0 * np.log10(sig_power / (np.mean(noise ** 2) + 1e-12))

        metrics = {
            "attenuation_db": float(attenuation_db),
            "linear_attenuation": float(linear_atten),
            "target_snr_db": float(snr_db),
            "measured_snr_db": float(measured_snr),
            "signal_power_w": float(sig_power),
            "noise_power_w": float(noise_power),
            "noise_rms_v": float(noise_std)
        }

        return received_signal, metrics
