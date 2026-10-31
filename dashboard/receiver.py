"""
Tab 2: Receiver Operation & Stage-by-Stage Signal Chain Inspection.
Allows probing and visualizing the signal transformation at every individual block
of the VHF aviation receiver receiving chain.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any
from utils.plotting import plot_time_waveform, plot_dual_waveform_comparison, plot_frequency_spectrum


def render_receiver_tab(simulation_results: Dict[str, Any]):
    """
    Renders Tab 2: Receiver Operation.
    """
    st.subheader("📡 Receiver Receiving Chain – Stage Inspector")
    st.markdown(
        "Inspect the transformation of the signal as it traverses each physical hardware stage: "
        "`Antenna` ➔ `RF Filter` ➔ `LNA` ➔ `Tuning/IF` ➔ `AM Demodulator` ➔ `Audio Conditioning` ➔ `ADC` ➔ `Digital WAV`."
    )

    stages = simulation_results["stages"]
    t = simulation_results["time_sim"]
    fs = simulation_results["fs_sim"]

    stage_options = {
        "Stage 1: RF Front-End Input": "stage1_rf_input",
        "Stage 2: RF Band-Pass Filter": "stage2_rf_filter",
        "Stage 3: Low Noise Amplifier (LNA)": "stage3_lna",
        "Stage 4: Frequency Tuning & Downconversion": "stage4_tuning",
        "Stage 5: AM Demodulator": "stage5_demodulator",
        "Stage 6: Audio Conditioning & AGC": "stage6_audio_proc",
        "Stage 7: Analog-to-Digital Converter (ADC)": "stage7_adc",
        "Stage 8: Digital Audio Output (WAV 16 kHz)": "stage8_digital_output"
    }

    selected_stage_label = st.selectbox("Select Receiver Stage to Probe:", list(stage_options.keys()), index=4)
    stage_key = stage_options[selected_stage_label]
    stage_data = stages[stage_key]

    st.markdown("---")

    # Render specific stage waveforms and spectra
    col_left, col_right = st.columns([1, 1])

    if stage_key == "stage1_rf_input":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**RF Input Waveform (AM DSB-FC + Channel Noise)**  \n`SNR: {m.get('measured_snr_db', 0):.1f} dB | Attenuation: {m.get('attenuation_db', 0):.1f} dB`")
            fig_time = plot_time_waveform(t, stage_data["received_rf"], title="Stage 1: Received RF Signal Waveform", y_label="Voltage (V)")
            st.plotly_chart(fig_time, use_container_width=True)
        with col_right:
            st.markdown("**Transmitted vs Received Modulating Envelope**")
            fig_env = plot_dual_waveform_comparison(
                t, stage_data["modulating_audio"], stage_data["am_transmitted"],
                name_a="Modulating Speech x(t)", name_b="Transmitted AM s(t)",
                title="Modulating Baseband vs AM Modulated Waveform"
            )
            st.plotly_chart(fig_env, use_container_width=True)

    elif stage_key == "stage2_rf_filter":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**Input vs Filtered Output Waveform**  \n`Passband: {m['low_cutoff_hz']:.0f} Hz - {m['high_cutoff_hz']:.0f} Hz | BW: {m['bandwidth_khz']} kHz`")
            fig_comp = plot_dual_waveform_comparison(
                t, stage_data["input_signal"], stage_data["output_signal"],
                name_a="Unfiltered Input (Noise)", name_b="Filtered Output",
                title="Stage 2: Filter Input vs Output Waveform"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        with col_right:
            st.markdown("**RF Filtered Frequency Spectrum (FFT)**")
            fig_fft = plot_frequency_spectrum(stage_data["output_signal"], fs=fs, title="Stage 2: Filtered RF Spectrum")
            st.plotly_chart(fig_fft, use_container_width=True)

    elif stage_key == "stage3_lna":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**LNA Small-Signal Amplification**  \n`Gain: {m['actual_gain_db']:.1f} dB ({m['linear_gain']:.1f}x) | NF: {m['noise_figure_db']} dB | P1dB: {m['p1db_limit_v']} V`")
            fig_comp = plot_dual_waveform_comparison(
                t, stage_data["input_signal"], stage_data["output_signal"],
                name_a="Pre-LNA Signal (mV)", name_b="Post-LNA Amplified (V)",
                title="Stage 3: Before vs After LNA Amplification"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        with col_right:
            st.markdown("**Amplified Signal Spectrum**")
            fig_fft = plot_frequency_spectrum(stage_data["output_signal"], fs=fs, title="Stage 3: LNA Output Spectrum")
            st.plotly_chart(fig_fft, use_container_width=True)

    elif stage_key == "stage4_tuning":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**Tuned Channel Downconversion**  \n`Channel: {m['selected_channel_mhz']:.3f} MHz | LO Offset: {m['tuning_error_hz']} Hz | IF: {m['if_frequency_hz']/1000:.1f} kHz`")
            fig_if = plot_time_waveform(t, stage_data["output_if"], title="Stage 4: Downconverted IF Signal Waveform", y_label="IF Voltage (V)")
            st.plotly_chart(fig_if, use_container_width=True)
        with col_right:
            st.markdown("**Baseband In-Phase (I) & Quadrature (Q) Signals**")
            fig_iq = plot_dual_waveform_comparison(
                t, stage_data["i_component"], stage_data["q_component"],
                name_a="In-Phase I(t)", name_b="Quadrature Q(t)",
                title="Stage 4: Baseband I and Q Signal Components"
            )
            st.plotly_chart(fig_iq, use_container_width=True)

    elif stage_key == "stage5_demodulator":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**Demodulated Audio Waveform**  \n`Correlation: {m.get('cross_correlation', 0):.4f} | MSE: {m.get('mean_squared_error', 0):.5f} | Method: {m['demod_method'].upper()}`")
            fig_demod = plot_time_waveform(t, stage_data["output_audio"], title="Stage 5: Recovered Audio Waveform", y_label="Normalized Voltage")
            st.plotly_chart(fig_demod, use_container_width=True)
        with col_right:
            st.markdown("**Demodulated Audio Spectrum (300 Hz - 3.4 kHz)**")
            fig_fft = plot_frequency_spectrum(stage_data["output_audio"], fs=fs, title="Stage 5: Recovered Baseband Spectrum", max_freq_hz=8000)
            st.plotly_chart(fig_fft, use_container_width=True)

    elif stage_key == "stage6_audio_proc":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**Post-Demodulation Conditioned Audio**  \n`BPF: {m.get('low_cutoff_hz', 300):.0f}-{m.get('high_cutoff_hz', 3400):.0f} Hz | AGC Applied: {m.get('mean_gain_db', 0):.1f} dB`")
            fig_proc = plot_dual_waveform_comparison(
                t, stage_data["input_signal"], stage_data["output_audio"],
                name_a="Pre-Conditioned Audio", name_b="Filtered & AGC Conditioned",
                title="Stage 6: Pre vs Post Audio Conditioning"
            )
            st.plotly_chart(fig_proc, use_container_width=True)
        with col_right:
            st.markdown("**Conditioned Audio Spectrum**")
            fig_fft = plot_frequency_spectrum(stage_data["output_audio"], fs=fs, title="Stage 6: Voice Band Spectrum (Clean)", max_freq_hz=6000)
            st.plotly_chart(fig_fft, use_container_width=True)

    elif stage_key == "stage7_adc":
        m = stage_data["metrics"]
        with col_left:
            st.markdown(f"**Analog Input vs Quantized Staircase Output**  \n`Resolution: {m['resolution_bits']}-bit ({m['num_quantization_levels']} steps) | Step: {m['quantization_step_mv']:.3f} mV`")
            fig_adc = plot_dual_waveform_comparison(
                t, stage_data["analog_input"], stage_data["quantized_signal"],
                name_a="Analog Input v(t)", name_b="Quantized Steps v_q[n]",
                title="Stage 7: Mid-Tread ADC Quantization"
            )
            st.plotly_chart(fig_adc, use_container_width=True)
        with col_right:
            st.markdown(f"**Instantaneous Quantization Noise Waveform (e_q = v_q - v)**  \n`Empirical SQNR: {m['measured_sqnr_db']:.1f} dB (Theory: {m['theoretical_sqnr_db']:.1f} dB)`")
            fig_err = plot_time_waveform(t, stage_data["quantization_error"], title="Stage 7: Quantization Error e_q[n]", y_label="Error (V)")
            st.plotly_chart(fig_err, use_container_width=True)

    elif stage_key == "stage8_digital_output":
        st.markdown(f"**Final Digital Speech Stream (16.0 kHz PCM Linear Audio)**  \n`Duration: {stage_data['duration_sec']:.2f} s | Total Samples: {len(stage_data['audio_samples'])}`")
        from acquisition.wav_handler import WAVHandler
        wav_bytes = WAVHandler.to_in_memory_wav_bytes(stage_data["audio_samples"], sample_rate=16000)
        st.audio(wav_bytes, format="audio/wav")

        t_16k = np.linspace(0, stage_data["duration_sec"], len(stage_data["audio_samples"]), endpoint=False)
        fig_speech = plot_time_waveform(t_16k, stage_data["audio_samples"], title="Stage 8: Final 16 kHz Audio Stream", y_label="Normalized PCM")
        st.plotly_chart(fig_speech, use_container_width=True)

    # Detailed metrics table for probed stage
    st.markdown("#### 📋 Stage Diagnostics & Key Parameters")
    metrics_display = {k: v for k, v in stage_data.get("metrics", {}).items() if not isinstance(v, (np.ndarray, list))}
    st.json(metrics_display)
