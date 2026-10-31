"""
Tab 7: Speech Emotion Recognition (SER) Deep Learning Inference.
Integrates Model 6 (CNN-BiLSTM) and Model 4 (Vertically Patched Vision Transformer)
to predict emotion probabilities from the receiver's demodulated speech.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any

from dsp.filtering import AudioPreprocessor
from dsp.segmentation import AudioSegmenter
from features.feature_extraction import FeatureExtractionPipeline
from models.predictor import SERPredictor
from models.registry import AVAILABLE_MODELS, get_model_status
from utils.plotting import plot_emotion_probabilities


def render_emotion_recognition_tab(simulation_results: Dict[str, Any]):
    """
    Renders Tab 7: Emotion Recognition.
    """
    st.subheader("🧠 Speech Emotion Recognition (SER) Neural Classifier")
    st.caption("Classifies the emotional state of the pilot/controller transmission using deep learning architectures from Projects I & II.")

    raw_audio = simulation_results["stages"]["stage8_digital_output"]["audio_samples"]

    # Preprocess & extract features
    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)
    feat_pipe = FeatureExtractionPipeline(fs=16000)

    audio_clean = prep.normalize_peak(prep.apply_pre_emphasis(raw_audio))
    windowed = seg.enforce_fixed_window(audio_clean)
    features = feat_pipe.extract_all_features(windowed)

    # Model Selection Bar
    col_sel, col_stat = st.columns([1.5, 1.5])
    with col_sel:
        model_choice = st.selectbox(
            "Select Neural Network Architecture:",
            list(AVAILABLE_MODELS.keys()),
            format_func=lambda k: AVAILABLE_MODELS[k]["name"],
            index=0
        )
    with col_stat:
        status_info = get_model_status(model_choice)
        if status_info["has_weights"]:
            st.success(f"🟢 Checkpoint Found: `{status_info['weights_path']}`")
        else:
            st.info(f"🔵 Architecture Active (Weights Pending in `data/models/`) | Acoustic Mode Active")

    # Run Prediction
    predictor = SERPredictor()
    pred_res = predictor.predict_emotion(
        log_mel_128=features["log_mel_128"],
        acoustic_features=features["metrics"],
        model_key=model_choice
    )

    dominant_emotion = pred_res["dominant_emotion"]
    confidence = pred_res["confidence"]
    probs = pred_res["probabilities"]

    st.markdown("---")

    # KPI Banner
    col_em, col_conf, col_mode = st.columns([1, 1, 1])
    with col_em:
        st.markdown(f"""
        <div style="background-color: #161B22; border-left: 5px solid #00E676; padding: 15px; border-radius: 8px;">
            <div style="font-size: 13px; color: #8B949E;">PREDICTED PILOT EMOTION</div>
            <div style="font-size: 24px; font-weight: bold; color: #FFFFFF;">{dominant_emotion.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div style="background-color: #161B22; border-left: 5px solid #00E5FF; padding: 15px; border-radius: 8px;">
            <div style="font-size: 13px; color: #8B949E;">CLASSIFICATION CONFIDENCE</div>
            <div style="font-size: 24px; font-weight: bold; color: #00E5FF;">{confidence * 100.0:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_mode:
        st.markdown(f"""
        <div style="background-color: #161B22; border-left: 5px solid #FF9100; padding: 15px; border-radius: 8px;">
            <div style="font-size: 13px; color: #8B949E;">INFERENCE ENGINE</div>
            <div style="font-size: 15px; font-weight: bold; color: #E6EDF3;">{pred_res['inference_mode'][:26]}...</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Probability Distribution Bar Chart
    col_chart, col_arch = st.columns([1.2, 0.8])

    with col_chart:
        fig_prob = plot_emotion_probabilities(probs, dominant_emotion=dominant_emotion)
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_arch:
        st.markdown("#### 🏗 Model Architecture Breakdown")
        if "cnn_bilstm" in model_choice:
            st.markdown("""
            **Model 6: CNN-BiLSTM (87% Test Accuracy)**
            - **Input:** 128×128 Log-Mel Spectrogram
            - **Conv2D Block 1:** 32 filters (3×3), BatchNorm, MaxPool (2,2)
            - **Conv2D Block 2:** 64 filters (3×3), BatchNorm, MaxPool (2,2)
            - **Conv2D Block 3:** 128 filters (3×3), BatchNorm, MaxPool (2,1)
            - **BiLSTM:** 2 Layers, 128 hidden units (Bidirectional)
            - **Attention / Pooling:** Temporal Mean Pooling
            - **Dense Head:** Linear 256 ➔ 128 ➔ 8 Softmax Classes
            """)
        else:
            st.markdown("""
            **Model 4: Vertically Patched ViT (Project II)**
            - **Input:** 128×128 Log-Mel Spectrogram
            - **Conv Stem:** 16 ➔ 32 ➔ 64 ➔ 32 ➔ 16 filters
            - **Vertical Patchify:** 128×1 time-slice strips (2048-dim vectors)
            - **Token Projection:** Linear projection to 256 dimensions
            - **Positional Encoding:** 1D Sinusoidal Position Embeddings
            - **Transformer Encoder:** 3 Layers, 4 Attention Heads
            - **MLP Head:** Dense 256 ➔ 128 ➔ 8 Softmax Classes
            """)

    return pred_res
