"""
Speech Emotion Recognition (SER) Predictor Module.
Executes inference using either Model 6 (CNN-BiLSTM) or Model 4 (ViT),
formats class probability distributions, and provides acoustic fallback scoring.
"""

import numpy as np
import torch
from typing import Dict, Any, List
from config.settings import EMOTION_LABELS
from models.registry import load_model_instance, get_model_status


class SERPredictor:
    """
    Inference orchestrator for Speech Emotion Recognition on aviation audio.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cached_models: Dict[str, torch.nn.Module] = {}

    def predict_emotion(
        self,
        log_mel_128: np.ndarray,
        acoustic_features: Dict[str, Any],
        model_key: str = "model_6_cnn_bilstm"
    ) -> Dict[str, Any]:
        """
        Predicts speech emotion from Log-Mel spectrogram.

        Args:
            log_mel_128: (128, 128) normalized Log-Mel Spectrogram matrix.
            acoustic_features: Dictionary of spectral metrics (pitch, RMS, ZCR).
            model_key: 'model_6_cnn_bilstm' or 'model_4_vit'.

        Returns:
            Dictionary containing dominant emotion, confidence, class probabilities,
            and model provenance status.
        """
        # Load or retrieve cached model
        if model_key not in self.cached_models:
            model, weights_loaded = load_model_instance(model_key=model_key, device=self.device)
            self.cached_models[model_key] = (model, weights_loaded)
        else:
            model, weights_loaded = self.cached_models[model_key]

        status_info = get_model_status(model_key)

        # Prepare PyTorch tensor of shape (1, 1, 128, 128)
        tensor_x = torch.from_numpy(log_mel_128).float().unsqueeze(0).unsqueeze(0).to(self.device)

        if weights_loaded:
            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            inference_type = "PyTorch Deep Neural Network (Trained Checkpoint)"
        else:
            # If checkpoint weights are not present on local machine,
            # calculate calibrated acoustic prosody probabilities based on ECE speech science:
            # High energy + High spectral centroid -> Urgency / Anger / Distress
            # Low energy + Low spectral centroid -> Sadness / Fatigue
            # Balanced energy + Moderate centroid -> Neutral / Calm
            rms = acoustic_features.get("mean_rms_energy", 0.05)
            centroid = acoustic_features.get("mean_spectral_centroid_hz", 1500.0)
            zcr = acoustic_features.get("mean_zcr", 0.05)

            # Heuristic energy-prosody prior
            base_scores = np.array([0.25, 0.20, 0.10, 0.10, 0.10, 0.10, 0.08, 0.07], dtype=np.float32)
            if rms > 0.15 and centroid > 2000.0:
                # High arousal (Anger / Urgency / Fear)
                base_scores[4] += 0.45  # Angry / Distress
                base_scores[5] += 0.35  # Fearful / Panic
                base_scores[0] -= 0.15
            elif rms < 0.03:
                # Low arousal (Sad / Fatigue / Calm)
                base_scores[3] += 0.35  # Sad
                base_scores[1] += 0.30  # Calm
                base_scores[0] -= 0.10
            else:
                # Standard pilot calm transmission
                base_scores[0] += 0.40  # Neutral
                base_scores[1] += 0.25  # Calm

            # Softmax normalize
            base_scores = np.maximum(0.01, base_scores)
            probs = base_scores / np.sum(base_scores)
            inference_type = "Acoustic Prosody Baseline (Weights Pending in data/models/)"

        dominant_idx = int(np.argmax(probs))
        dominant_emotion = EMOTION_LABELS[dominant_idx]
        confidence = float(probs[dominant_idx])

        # Format probability distribution
        prob_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

        return {
            "model_name": status_info["name"],
            "model_key": model_key,
            "weights_loaded": weights_loaded,
            "inference_mode": inference_type,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": prob_dict,
            "labels": EMOTION_LABELS
        }
