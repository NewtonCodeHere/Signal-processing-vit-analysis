"""
Model Registry Module.
Manages available deep learning models, verifies model weights on disk,
and provides architecture instantiation for both Model 4 (ViT) and Model 6 (CNN-BiLSTM).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from config.settings import MODELS_DIR, EMOTION_LABELS
from models.cnn_vit import VerticallyPatchedViT
from models.cnn_bilstm import CNNBiLSTM


AVAILABLE_MODELS = {
    "model_6_cnn_bilstm": {
        "name": "Model 6: CNN-BiLSTM (87% Accuracy)",
        "architecture_class": CNNBiLSTM,
        "weights_file": "cnn_bilstm_model6.pt",
        "description": "Hybrid CNN-BiLSTM trained on RAVDESS + CREMAD + TESS. Best generalized model."
    },
    "model_4_vit": {
        "name": "Model 4: Vertically Patched ViT",
        "architecture_class": VerticallyPatchedViT,
        "weights_file": "vit_model4.pt",
        "description": "Vision Transformer with 128x1 vertical time-slice patching and positional encoding."
    }
}


def get_model_status(model_key: str = "model_6_cnn_bilstm") -> Dict[str, Any]:
    """
    Checks if trained checkpoint weights exist in data/models/.
    """
    info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["model_6_cnn_bilstm"])
    weights_path = Path(MODELS_DIR) / info["weights_file"]
    has_weights = weights_path.exists() and weights_path.stat().st_size > 1024

    return {
        "model_key": model_key,
        "name": info["name"],
        "has_weights": has_weights,
        "weights_path": str(weights_path),
        "file_size_bytes": weights_path.stat().st_size if has_weights else 0,
        "description": info["description"]
    }


def load_model_instance(
    model_key: str = "model_6_cnn_bilstm",
    device: str = "cpu"
) -> Tuple[torch.nn.Module, bool]:
    """
    Instantiates the model architecture and loads weights if present.

    Returns:
        Tuple of (model_instance, weights_loaded_boolean)
    """
    info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["model_6_cnn_bilstm"])
    arch_cls = info["architecture_class"]
    model = arch_cls(num_classes=len(EMOTION_LABELS))
    model.to(device)

    weights_path = Path(MODELS_DIR) / info["weights_file"]
    weights_loaded = False

    if weights_path.exists() and weights_path.stat().st_size > 1024:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            weights_loaded = True
        except Exception as e:
            weights_loaded = False

    model.eval()
    return model, weights_loaded
