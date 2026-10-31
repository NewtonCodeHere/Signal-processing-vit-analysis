"""
Mel-Frequency Cepstral Coefficients (MFCC) Extraction Module.
Computes 13 and 40 MFCC coefficients along with first-order Delta (velocity)
and second-order Delta-Delta (acceleration) dynamics.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple


class MFCCExtractor:
    """
    Extracts static and dynamic MFCC feature vectors from speech.
    """

    def __init__(self, fs: int = 16000, n_mfcc: int = 40):
        self.fs = fs
        self.n_mfcc = n_mfcc

    def extract_mfcc_features(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes MFCC, Delta, and Delta-Delta matrices.

        Returns:
            Tuple of (mfcc, delta, delta2, summary_stats)
        """
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode="constant")

        # Static MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.fs, n_mfcc=self.n_mfcc, n_fft=512, hop_length=160)

        # Dynamic delta (first-order derivative)
        delta = librosa.feature.delta(mfccs, order=1)

        # Acceleration delta-delta (second-order derivative)
        delta2 = librosa.feature.delta(mfccs, order=2)

        # Global feature vector: mean and standard deviation across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)

        feature_vector = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std])

        summary = {
            "n_mfcc": int(self.n_mfcc),
            "num_frames": int(mfccs.shape[1]),
            "feature_vector_dim": int(len(feature_vector)),
            "feature_vector": feature_vector.tolist(),
            "mean_mfcc_1": float(mfcc_mean[0]),
            "mean_mfcc_2": float(mfcc_mean[1])
        }

        return mfccs, delta, delta2, summary
