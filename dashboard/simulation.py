"""
Tab 3: Receiver Simulation Lab.
Interactive laboratory environment where students and review panels can manipulate
RF, IF, demodulation, audio conditioning, and ADC parameters, observing dynamic recalculation.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any
from config.receiver_config import ReceiverParameters, VHF_CHANNEL_PRESETS
from receiver.simulation_pipeline import ReceiverSimulationEngine
from utils.plotting import (
    plot_time_waveform,
    plot_dual_waveform_comparison,
    plot_frequency_spectrum
)
from acquisition.wav_handler import WAVHandler


def render_simulation_tab(engine: ReceiverSimulationEngine) -> Dict[str, Any]:
    """
    Renders Tab 3: Interactive Receiver Simulation Lab.
    Returns:
        Updated simulation results dictionary.
    """
    st.subheader("🧪 Receiver Simulation Lab – Interactive DSP Testbench")
    st.caption("Modify parameters below. The simulation engine dynamically recalculates the exact physics and DSP transformations.")

    # Control Parameters Form
    with st.expander("⚙ Interactive Simulation Parameter Controls", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📡 RF & Modulation")
            preset_choice = st.selectbox("VHF Channel Preset:", list(VHF_CHANNEL_PRESETS.keys()), index=0)
            carrier_freq = VHF_CHANNEL_PRESETS[preset_choice]

            sig_type = st.selectbox("Modulating Signal Source:", ["synthetic_atc", "tone", "multitone"], index=0)
            tone_freq = 1000.0
            if sig_type == "tone":
                tone_freq = st.slider("Tone Frequency (Hz):", 300.0, 3400.0, 1000.0, step=50.0)

            mod_index = st.slider("Modulation Index (m):", 0.0, 1.5, 0.75, step=0.05,
                                  help="m < 1: Under-modulation | m = 1: 100% | m > 1: Over-modulation distortion")
            if mod_index > 1.0:
                st.warning("⚠ Over-modulation (m > 1.0) active: Diode envelope detector will exhibit envelope clipping.")

        with col2:
            st.markdown("##### 🌌 Channel & RF Front-End")
            chan_snr = st.slider("Channel SNR (dB):", -10.0, 45.0, 25.0, step=1.0)
            rf_bw = st.select_slider("RF Front-End BW (kHz):", options=[8.33, 12.5, 25.0, 50.0], value=25.0)
            lna_gain = st.slider("LNA Gain (dB):", 0.0, 35.0, 20.0, step=1.0)
            tuning_drift = st.slider("Tuning Drift / Error (Hz):", -2500.0, 2500.0, 0.0, step=100.0)

        with col3:
            st.markdown("##### 🎛 Demodulator, Audio & ADC")
            demod_method = st.radio("AM Demodulator:", ["envelope", "coherent"], index=0)
            enable_agc = st.checkbox("Enable Automatic Gain Control (AGC)", value=True)
            enable_bpf = st.checkbox("Enable 300–3400 Hz Voice Filter", value=True)
            adc_bits = st.selectbox("ADC Resolution (Bit Depth):", [8, 10, 12, 16, 24], index=3)

    # Re-run simulation dynamically with chosen parameters
    params = ReceiverParameters(
        signal_type=sig_type,
        tone_freq_hz=tone_freq,
        carrier_freq_mhz=carrier_freq,
        modulation_index=mod_index,
        channel_snr_db=chan_snr,
        rf_filter_bandwidth_khz=rf_bw,
        lna_gain_db=lna_gain,
        tuning_error_hz=tuning_drift,
        demodulator_type=demod_method,
        enable_agc=enable_agc,
        enable_audio_bpf=enable_bpf,
        adc_resolution_bits=adc_bits,
        duration_sec=0.5
    )

    results = engine.run_simulation(params=params)

    stages = results["stages"]
    t = results["time_sim"]
    fs = results["fs_sim"]
    s5_m = stages["stage5_demodulator"]["metrics"]
    s7_m = stages["stage7_adc"]["metrics"]

    # Real-Time Measurement Badges
    st.markdown("#### 📊 Dynamic Real-Time Receiver KPIs")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Recovered SNR", f"{s5_m.get('fidelity_snr_db', 0):.1f} dB", delta=f"{chan_snr:.0f} dB In")
    with kpi2:
        st.metric("Demod Correlation (ρ)", f"{s5_m.get('cross_correlation', 0):.4f}")
    with kpi3:
        st.metric("Demod MSE", f"{s5_m.get('mean_squared_error', 0):.5f}")
    with kpi4:
        st.metric("Quantization Step", f"{s7_m.get('quantization_step_mv', 0):.3f} mV", delta=f"{adc_bits}-bit")
    with kpi5:
        st.metric("Empirical SQNR", f"{s7_m.get('measured_sqnr_db', 0):.1f} dB", delta=f"Theory: {s7_m.get('theoretical_sqnr_db', 0):.1f} dB")

    st.markdown("---")

    # Multi-Trace Dynamic Visualizers
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.markdown("**1. Transmitted AM Modulated RF Waveform s_AM(t)**")
        fig_am = plot_time_waveform(t, stages["stage1_rf_input"]["am_transmitted"], title="Transmitted AM DSB-FC Signal")
        st.plotly_chart(fig_am, use_container_width=True)
    with row1_right:
        st.markdown("**2. RF Spectrum: Carrier & Sideband Distribution**")
        fig_spec = plot_frequency_spectrum(stages["stage1_rf_input"]["am_transmitted"], fs=fs, title="AM RF Spectrum")
        st.plotly_chart(fig_spec, use_container_width=True)

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.markdown("**3. Demodulated vs Original Audio Waveform Comparison**")
        fig_comp = plot_dual_waveform_comparison(
            t, stages["stage1_rf_input"]["modulating_audio"], stages["stage5_demodulator"]["output_audio"],
            name_a="Original Voice", name_b="Recovered Demod Audio",
            title="Demodulated Audio vs Original Audio Overlay"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    with row2_right:
        st.markdown("**4. ADC Quantization Staircase (Analog vs Quantized)**")
        fig_adc = plot_dual_waveform_comparison(
            t, stages["stage7_adc"]["analog_input"], stages["stage7_adc"]["quantized_signal"],
            name_a="Analog Audio", name_b="Quantized Steps",
            title=f"ADC Quantization ({adc_bits}-bit)"
        )
        st.plotly_chart(fig_adc, use_container_width=True)

    # Audio Playback
    st.markdown("#### 🔊 Real-Time Demodulated Audio Playback")
    dig_audio = stages["stage8_digital_output"]["audio_samples"]
    wav_bytes = WAVHandler.to_in_memory_wav_bytes(dig_audio, sample_rate=16000)
    col_play, col_dl = st.columns([2, 1])
    with col_play:
        st.audio(wav_bytes, format="audio/wav")
    with col_dl:
        st.download_button("⬇ Download Demodulated WAV", data=wav_bytes, file_name="simulated_demod_audio.wav", mime="audio/wav")

    return results
