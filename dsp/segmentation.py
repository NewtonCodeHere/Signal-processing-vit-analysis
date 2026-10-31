"""
Audio Segmentation & Voice Activity Detection (VAD) Module.
Implements:
1. Four-second fixed windowing (64,000 samples at 16 kHz) with symmetric padding/trimming.
2. Voice Activity Detection (VAD) using Short-Time Energy and Spectral Centroid.
3. Audio segmentation into discrete transmission bursts.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class AudioSegmenter:
    """
    Handles speech segmentation, windowing, and Voice Activity Detection (VAD).
    """

    def __init__(self, fs: int = 16000, window_duration_sec: float = 4.0):
        self.fs = fs
        self.window_samples = int(fs * window_duration_sec)  # 64,000 samples

    def enforce_fixed_window(self, audio: np.ndarray) -> np.ndarray:
        """
        Pads or trims the audio to exactly window_samples (4 seconds at 16 kHz),
        matching the Project I & II dataset requirement.
        """
        n = len(audio)
        if n == self.window_samples:
            return audio.astype(np.float32)

        if n > self.window_samples:
            # Center crop
            start = (n - self.window_samples) // 2
            return audio[start:start + self.window_samples].astype(np.float32)

        # Symmetric zero padding
        pad_total = self.window_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant").astype(np.float32)

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        frame_len_ms: float = 25.0,
        hop_len_ms: float = 10.0,
        energy_threshold_percentile: float = 35.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Performs Voice Activity Detection (VAD) using Short-Time Energy.

        Returns:
            Tuple of (frame_times, vad_mask, speech_segments_list)
        """
        frame_len = int((frame_len_ms * 1e-3) * self.fs)
        hop_len = int((hop_len_ms * 1e-3) * self.fs)

        num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
        energies = np.zeros(num_frames)
        frame_times = np.zeros(num_frames)

        for i in range(num_frames):
            idx = i * hop_len
            frame = audio[idx:idx + frame_len]
            energies[i] = np.sum(frame ** 2) / frame_len
            frame_times[i] = (idx + frame_len / 2) / self.fs

        # Adaptive threshold based on percentile
        threshold = np.percentile(energies, energy_threshold_percentile)
        vad_mask = energies > threshold

        # Group contiguous speech frames into segments
        segments = []
        in_speech = False
        start_t = 0.0

        for i, is_speech in enumerate(vad_mask):
            t = frame_times[i]
            if is_speech and not in_speech:
                in_speech = True
                start_t = t
            elif not is_speech and in_speech:
                in_speech = False
                dur = t - start_t
                if dur >= 0.25:  # Minimum valid syllable duration
                    segments.append({
                        "start_sec": float(start_t),
                        "end_sec": float(t),
                        "duration_sec": float(dur)
                    })

        if in_speech:
            segments.append({
                "start_sec": float(start_t),
                "end_sec": float(frame_times[-1]),
                "duration_sec": float(frame_times[-1] - start_t)
            })

        return frame_times, vad_mask, segments
