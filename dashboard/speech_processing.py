"""
Tab 6: Speech Processing Pipeline & Acoustic Feature Extraction.
Implements:
1. Resampling to 16 kHz & Pre-emphasis filtering.
2. Four-second windowing (64,000 samples).
3. Voice Activity Detection (VAD) & speech segmentation.
4. 128x128 Log-Mel Spectrogram computation.
5. 40-coefficient MFCC extraction with Delta dynamics.
6. Metadata Tagging.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any

from dsp.filtering import AudioPreprocessor
from dsp.segmentation import AudioSegmenter
from features.feature_extraction import FeatureExtractionPipeline
from utils.plotting import plot_spectrogram_heatmap, plot_time_waveform, apply_dark_theme, PLOT_THEME


def render_speech_processing_tab(simulation_results: Dict[str, Any]):
    """
    Renders Tab 6: Speech Processing Pipeline.
    """
    st.subheader("🎙 Speech Processing Pipeline & Feature Extraction")
    st.caption("Conditions digital audio into standardized 4-second chunks at 16 kHz and extracts Log-Mel and MFCC representations for SER.")

    raw_audio = simulation_results["stages"]["stage8_digital_output"]["audio_samples"]
    p = simulation_results["params"]

    # 1. Conditioning & Windowing
    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)

    pre_emph = prep.apply_pre_emphasis(raw_audio)
    windowed_4s = seg.enforce_fixed_window(pre_emph)

    # 2. Voice Activity Detection (VAD)
    frame_times, vad_mask, segments = seg.detect_voice_activity(windowed_4s)

    # 3. Feature Extraction
    feat_pipe = FeatureExtractionPipeline(fs=16000)
    features = feat_pipe.extract_all_features(windowed_4s)

    st.markdown("#### 1. Audio Windowing & Voice Activity Detection (VAD)")
    t_4s = np.linspace(0, 4.0, len(windowed_4s), endpoint=False)

    fig_vad = go.Figure()
    fig_vad.add_trace(go.Scatter(
        x=t_4s, y=windowed_4s, mode="lines",
        line={"color": PLOT_THEME["primary_cyan"], "width": 1.2}, name="Audio (4s Window)"
    ))
    # Overlay VAD decision
    vad_curve = np.interp(t_4s, frame_times, vad_mask.astype(float)) * 0.8
    fig_vad.add_trace(go.Scatter(
        x=t_4s, y=vad_curve, mode="lines",
        line={"color": PLOT_THEME["accent_orange"], "width": 2.0}, name="VAD Speech Flag"
    ))
    fig_vad.update_xaxes(title_text="Time (seconds)")
    fig_vad.update_yaxes(title_text="Amplitude")
    st.plotly_chart(apply_dark_theme(fig_vad, title="4-Second Window & Voice Activity Detection (VAD)"), use_container_width=True)

    st.markdown(f"**Detected Active Speech Bursts:** `{len(segments)} segment(s)`")
    if segments:
        st.dataframe(segments, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 2. Time-Frequency Feature Representations for SER")
    col_mel, col_mfcc = st.columns(2)

    with col_mel:
        st.markdown("**128x128 Log-Mel Spectrogram Matrix**")
        fig_mel = plot_spectrogram_heatmap(features["log_mel_128"], title="Log-Mel Matrix (128 Freq x 128 Time)")
        st.plotly_chart(fig_mel, use_container_width=True)

    with col_mfcc:
        st.markdown("**40 Mel-Frequency Cepstral Coefficients (MFCCs)**")
        fig_mfcc = go.Figure(data=go.Heatmap(
            z=features["mfccs"],
            colorscale="Plasma",
            colorbar={"title": "Coeff Value"}
        ))
        fig_mfcc.update_xaxes(title_text="Frame Index")
        fig_mfcc.update_yaxes(title_text="MFCC Index")
        st.plotly_chart(apply_dark_theme(fig_mfcc, title="40 MFCC Cepstral Heatmap"), use_container_width=True)

    st.markdown("---")

    st.markdown("#### 3. Aviation Dataset Metadata Tagging Generator")
    metadata_record = {
        "UTCTimestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ChannelFrequencyMHz": f"{p.carrier_freq_mhz:.3f}",
        "ModulationFormat": "AM DSB-FC",
        "SamplingRateHz": 16000,
        "BitDepth": p.adc_resolution_bits,
        "WindowDurationSec": 4.0,
        "ActiveSpeechSegments": len(segments),
        "MeasuredSNRdB": f"{simulation_results['stages']['stage5_demodulator']['metrics'].get('fidelity_snr_db', 0):.1f}",
        "AviationAirbandSector": "Civil Aviation Voice Band (ICAO Annex 10)"
    }
    st.json(metadata_record)
