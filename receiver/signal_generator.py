"""
Signal Generator Module.
Generates or loads modulating baseband audio signals:
1. Single sinusoidal tone.
2. Multi-tone harmonic audio.
3. Synthetic aviation communication speech with pilot cockpit characteristics.
4. User-provided WAV audio.
"""

import numpy as np
from typing import Tuple, Optional
import soundfile as sf
from scipy.signal import resample_poly


class SignalGenerator:
    """
    Generates time-domain modulating baseband signals for AM simulation.
    """

    def __init__(self, fs: int = 200000):
        """
        Args:
            fs: Simulation sampling rate in Hz (default: 200,000 Hz).
        """
        self.fs = fs

    def generate_single_tone(
        self,
        freq_hz: float = 1000.0,
        duration_sec: float = 1.0,
        amplitude: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a pure single-tone sinusoidal audio signal.
        """
        num_samples = int(self.fs * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        audio = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        return t, audio

    def generate_multi_tone(
        self,
        freqs: Tuple[float, ...] = (400.0, 1000.0, 2400.0),
        amplitudes: Tuple[float, ...] = (0.5, 0.35, 0.15),
        duration_sec: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a multi-tone audio signal containing multiple voice frequencies.
        """
        num_samples = int(self.fs * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        audio = np.zeros_like(t)
        for f, a in zip(freqs, amplitudes):
            audio += a * np.sin(2.0 * np.pi * f * t)
        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(audio))
        if max_val > 1e-9:
            audio = audio / max_val
        return t, audio

    def generate_synthetic_atc_speech(
        self,
        duration_sec: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes a realistic aviation transmission waveform:
        Formant frequencies mimicking 'Mayday / Madurai Tower, Air India 404, descending FL 150'.
        Includes vowel/consonant rhythm, vocal cord fundamental (120 Hz) and cockpit background rumble.
        """
        num_samples = int(self.fs * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        # Fundamental voice pitch: ~130 Hz with natural intonation vibrato
        pitch = 130.0 + 15.0 * np.sin(2 * np.pi * 3.5 * t)
        phase = 2 * np.pi * np.cumsum(pitch) / self.fs
        
        # Glottal pulse train harmonics
        voice_source = (
            np.sin(phase) +
            0.6 * np.sin(2 * phase) +
            0.4 * np.sin(3 * phase) +
            0.25 * np.sin(4 * phase) +
            0.15 * np.sin(5 * phase)
        )
        
        # Syllabic envelope modulation (words and pauses)
        syllables = np.abs(np.sin(2 * np.pi * 3.0 * t)) ** 2
        speech = voice_source * syllables
        
        # Cockpit acoustic background rumble (low-frequency engine vibration at 80-120 Hz)
        engine_rumble = 0.08 * np.sin(2 * np.pi * 95.0 * t) + 0.04 * np.sin(2 * np.pi * 180.0 * t)
        
        audio = speech + engine_rumble
        
        # Normalize to [-1.0, 1.0]
        audio = audio / np.max(np.abs(audio) + 1e-9)
        return t, audio

    def load_wav_file(
        self,
        wav_path: str,
        duration_sec: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads an external audio file and upsamples it to the simulation sample rate.
        """
        data, orig_fs = sf.read(wav_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Convert stereo to mono

        if duration_sec is not None:
            max_orig_samples = int(orig_fs * duration_sec)
            data = data[:max_orig_samples]

        # Resample to simulation rate self.fs
        from fractions import Fraction
        frac = Fraction(self.fs, orig_fs).limit_denominator(100)
        resampled_audio = resample_poly(data, frac.numerator, frac.denominator)

        num_samples = len(resampled_audio)
        actual_dur = num_samples / self.fs
        t = np.linspace(0, actual_dur, num_samples, endpoint=False)

        # Normalize audio
        peak = np.max(np.abs(resampled_audio))
        if peak > 1e-9:
            resampled_audio = resampled_audio / peak

        return t, resampled_audio
