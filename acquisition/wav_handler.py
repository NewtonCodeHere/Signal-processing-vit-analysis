"""
WAV Audio File Handling Module.
Converts digitized audio arrays into compliant 16-bit PCM WAV files,
generates in-memory audio buffers for browser playback, and handles metadata.
"""

import io
import soundfile as sf
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


class WAVHandler:
    """
    Handles audio file writing, reading, and in-memory byte buffer generation for UI playback.
    """

    @staticmethod
    def save_wav(
        filepath: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        subtype: str = "PCM_16"
    ) -> Dict[str, Any]:
        """
        Saves audio array to a standard WAV file.
        """
        # Ensure audio is float in [-1.0, 1.0]
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(p), audio, sample_rate, subtype=subtype)

        return {
            "filepath": str(p),
            "sample_rate": sample_rate,
            "duration_sec": len(audio) / sample_rate,
            "subtype": subtype,
            "file_size_bytes": p.stat().st_size
        }

    @staticmethod
    def load_wav(filepath: str) -> Tuple[np.ndarray, int, float]:
        """
        Loads a WAV file and returns mono audio array, sample rate, and duration.
        """
        data, fs = sf.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Stereo to mono
        duration = len(data) / fs
        return data.astype(np.float32), fs, duration

    @staticmethod
    def to_in_memory_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        Encodes audio array to an in-memory WAV byte buffer for st.audio playback and download.
        """
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
