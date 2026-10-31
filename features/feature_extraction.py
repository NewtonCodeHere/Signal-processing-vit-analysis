"""
Integrated Feature Extraction Pipeline Module.
Combines Log-Mel Spectrograms, MFCCs, Chroma, Spectral Centroid,
Spectral Rolloff, and Zero-Crossing Rate into a comprehensive feature set for SER.
"""

import numpy as np
import librosa
from typing import Dict, Any
from dsp.spectrogram import SpectrogramExtractor
from features.mfcc import MFCCExtractor


class FeatureExtractionPipeline:
    """
    Coordinates extraction of all acoustic and spectral representations.
    """

    def __init__(self, fs: int = 16000):
        self.fs = fs
        self.spec_extractor = SpectrogramExtractor(fs=fs, n_mels=128)
        self.mfcc_extractor = MFCCExtractor(fs=fs, n_mfcc=40)

    def extract_all_features(self, audio_16k: np.ndarray) -> Dict[str, Any]:
        """
        Runs complete feature extraction on 16 kHz audio.

        Returns:
            Dictionary containing Log-Mel Spectrogram (128x128), MFCCs,
            Spectral Centroid, Rolloff, ZCR, and summary statistics.
        """
        # Ensure non-empty
        if len(audio_16k) == 0:
            audio_16k = np.zeros(16000, dtype=np.float32)

        # 1. 128x128 Log-Mel Spectrogram
        log_mel_128 = self.spec_extractor.compute_log_mel_spectrogram(audio_16k, target_size=(128, 128))

        # 2. MFCC Features
        mfccs, delta, delta2, mfcc_summary = self.mfcc_extractor.extract_mfcc_features(audio_16k)

        # 3. Spectral Descriptors
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_16k, sr=self.fs)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_16k, sr=self.fs, roll_percent=0.85)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_16k)[0]
        rms_energy = librosa.feature.rms(y=audio_16k)[0]

        # 4. Summary metrics
        metrics = {
            "log_mel_shape": list(log_mel_128.shape),
            "mfcc_shape": list(mfccs.shape),
            "mean_spectral_centroid_hz": float(np.mean(spectral_centroids)),
            "mean_spectral_rolloff_hz": float(np.mean(spectral_rolloff)),
            "mean_zcr": float(np.mean(zero_crossing_rate)),
            "mean_rms_energy": float(np.mean(rms_energy)),
            "mfcc_summary": mfcc_summary
        }

        return {
            "log_mel_128": log_mel_128,
            "mfccs": mfccs,
            "mfcc_delta": delta,
            "mfcc_delta2": delta2,
            "spectral_centroids": spectral_centroids,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate,
            "metrics": metrics
        }
