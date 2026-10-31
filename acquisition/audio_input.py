"""
Live Audio Acquisition Module.
Interfaces with host sound cards, USB audio adapters, and microphone inputs
to capture real demodulated audio from the physical receiver module.
"""

import numpy as np
import sounddevice as sd
from typing import List, Dict, Any, Optional, Tuple


class LiveAudioAcquisition:
    """
    Manages live audio capture from physical sound card or USB receiver dongle.
    """

    def __init__(self, default_fs: int = 16000):
        self.default_fs = default_fs

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """
        Enumerates all available audio input host devices.
        """
        devices = sd.query_devices()
        input_devs = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                input_devs.append({
                    "device_id": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "hostapi": dev["hostapi"]
                })
        return input_devs

    def record_audio_block(
        self,
        duration_sec: float = 4.0,
        device_id: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synchronously records a block of audio from the physical input.

        Args:
            duration_sec: Recording duration in seconds.
            device_id: Optional input device index.

        Returns:
            Tuple of (audio_waveform, capture_metrics)
        """
        num_samples = int(duration_sec * self.default_fs)
        try:
            recording = sd.rec(
                frames=num_samples,
                samplerate=self.default_fs,
                channels=1,
                dtype="float32",
                device=device_id
            )
            sd.wait()
            audio_mono = recording.flatten()
            status = "Success"
        except Exception as e:
            audio_mono = np.zeros(num_samples, dtype=np.float32)
            status = f"Hardware audio input error: {str(e)}"

        peak = np.max(np.abs(audio_mono))
        rms = np.sqrt(np.mean(audio_mono ** 2))

        metrics = {
            "status": status,
            "device_id": device_id,
            "sample_rate": self.default_fs,
            "duration_sec": duration_sec,
            "peak_level": float(peak),
            "rms_level": float(rms)
        }

        return audio_mono, metrics
