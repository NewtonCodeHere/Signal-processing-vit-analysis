"""
Unit tests for Speech Processing, Feature Extraction, and SER Models.
Verifies 16 kHz conditioning, 4-second windowing, 128x128 Log-Mel matrices,
and forward pass through both ViT and CNN-BiLSTM architectures.
"""

import numpy as np
import torch
import pytest

from dsp.filtering import AudioPreprocessor
from dsp.segmentation import AudioSegmenter
from dsp.spectrogram import SpectrogramExtractor
from features.mfcc import MFCCExtractor
from models.cnn_vit import VerticallyPatchedViT
from models.cnn_bilstm import CNNBiLSTM
from models.predictor import SERPredictor


def test_audio_preprocessor_and_windowing():
    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)

    # 1.5 second dummy audio at 44.1 kHz
    orig_fs = 44100
    t = np.linspace(0, 1.5, int(orig_fs * 1.5), endpoint=False)
    raw_audio = np.sin(2 * np.pi * 500 * t)

    resampled = prep.resample_audio(raw_audio, orig_fs=orig_fs)
    assert len(resampled) == int(1.5 * 16000)

    pre_emph = prep.apply_pre_emphasis(resampled)
    assert len(pre_emph) == len(resampled)

    windowed = seg.enforce_fixed_window(pre_emph)
    assert len(windowed) == 64000  # Exactly 4 seconds at 16 kHz


def test_log_mel_spectrogram_dimensions():
    extractor = SpectrogramExtractor(fs=16000, n_mels=128)
    dummy_audio = np.random.randn(64000).astype(np.float32)

    log_mel = extractor.compute_log_mel_spectrogram(dummy_audio, target_size=(128, 128))
    assert log_mel.shape == (128, 128)
    assert np.min(log_mel) >= 0.0
    assert np.max(log_mel) <= 1.0


def test_vit_forward_pass():
    vit = VerticallyPatchedViT(num_classes=8, in_mels=128, in_frames=128)
    vit.eval()

    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        logits = vit(dummy_input)

    assert logits.shape == (2, 8)


def test_cnn_bilstm_forward_pass():
    model = CNNBiLSTM(num_classes=8)
    model.eval()

    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 8)


def test_ser_predictor_inference():
    predictor = SERPredictor()
    dummy_spec = np.random.rand(128, 128).astype(np.float32)
    dummy_feats = {"mean_rms_energy": 0.08, "mean_spectral_centroid_hz": 1600.0, "mean_zcr": 0.04}

    res = predictor.predict_emotion(dummy_spec, dummy_feats, model_key="model_6_cnn_bilstm")
    assert "dominant_emotion" in res
    assert "confidence" in res
    assert len(res["probabilities"]) == 8
