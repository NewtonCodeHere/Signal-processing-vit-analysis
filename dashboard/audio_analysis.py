"""
Tab 5: Audio DSP Analysis & Oscilloscope.
Allows uploading external WAV files or inspecting demodulated receiver audio,
providing time-domain waveforms, 2D FFT spectra, STFT spectrograms, and acoustic metrics.
"""

import streamlit as st
import numpy as np
import io
import soundfile as sf
from typing import Dict, Any

from utils.plotting import plot_time_waveform, plot_frequency_spectrum, plot_spectrogram_heatmap
from dsp.spectrogram import SpectrogramExtractor
from acquisition.wav_handler import WAVHandler


def render_audio_analysis_tab(simulation_results: Dict[str, Any]):
    """
    Renders Tab 5: Audio Analysis.
    """
    st.subheader("🔊 Audio Signal Analysis & Oscilloscope")
    st.caption("Inspect acoustic waveforms, Fourier spectra, spectrograms, and signal dynamic metrics.")

    audio_source = st.radio(
        "Select Audio Source:",
        ["Current Demodulated Receiver Audio (16 kHz)", "Upload Custom Aviation WAV File"],
        horizontal=True
    )

    if audio_source == "Current Demodulated Receiver Audio (16 kHz)":
        audio = simulation_results["stages"]["stage8_digital_output"]["audio_samples"]
        fs = 16000
    else:
        uploaded_file = st.file_uploader("Upload WAV File (e.g., Pilot-ATC recording):", type=["wav"])
        if uploaded_file is not None:
            data, fs = sf.read(io.BytesIO(uploaded_file.read()))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            audio = data.astype(np.float32)
        else:
            st.info("No file uploaded yet. Falling back to demodulated simulation audio.")
            audio = simulation_results["stages"]["stage8_digital_output"]["audio_samples"]
            fs = 16000

    duration = len(audio) / fs
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    crest_factor = float(peak / (rms + 1e-9))
    dynamic_range_db = float(20.0 * np.log10(peak / (rms + 1e-9)))

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sampling Rate", f"{fs} Hz")
    c2.metric("Duration", f"{duration:.2f} s")
    c3.metric("Peak Amplitude", f"{peak:.3f}")
    c4.metric("RMS Level", f"{rms:.3f}")
    c5.metric("Crest Factor", f"{crest_factor:.2f}")

    # Audio Player
    st.markdown("#### 🎧 Audio Playback")
    wav_bytes = WAVHandler.to_in_memory_wav_bytes(audio, sample_rate=fs)
    st.audio(wav_bytes, format="audio/wav")

    st.markdown("---")

    col_w, col_s = st.columns(2)
    t_arr = np.linspace(0, duration, len(audio), endpoint=False)

    with col_w:
        st.markdown("**Acoustic Time-Domain Waveform**")
        fig_time = plot_time_waveform(t_arr, audio, title="Audio Time-Domain Oscilloscope", y_label="PCM Level")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_s:
        st.markdown("**Power Spectrum (FFT)**")
        fig_fft = plot_frequency_spectrum(audio, fs=fs, title="Audio Frequency Spectrum", max_freq_hz=fs/2.0)
        st.plotly_chart(fig_fft, use_container_width=True)

    # Spectrogram View
    st.markdown("**Time-Frequency Spectrogram**")
    spec_extractor = SpectrogramExtractor(fs=fs, n_mels=128)
    log_mel = spec_extractor.compute_log_mel_spectrogram(audio, target_size=(128, 128))
    fig_spec = plot_spectrogram_heatmap(log_mel, title="128-bin Log-Mel Spectrogram")
    st.plotly_chart(fig_spec, use_container_width=True)
