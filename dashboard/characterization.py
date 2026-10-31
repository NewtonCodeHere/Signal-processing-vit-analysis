"""
Tab 4: Receiver Characterization Module.
Provides interactive parametric testbench sweeps:
1. Frequency Response & Bode Plot (Passband, -3dB Bandwidth, Selectivity)
2. Linearity & Gain (Pin vs Pout, P1dB Compression)
3. Noise Sensitivity & Threshold Effect (Input SNR vs Output SNR)
4. Modulation Linearity & Harmonic Distortion (THD vs Modulation Index)
5. ADC Quantization & SQNR Sweep (8-bit to 24-bit)
6. Overall Performance KPI Table
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from characterization.frequency_response import FrequencyResponseAnalyzer
from characterization.gain_analysis import GainAnalyzer
from characterization.noise_analysis import NoiseAnalyzer
from characterization.modulation_analysis import ModulationAnalyzer
from characterization.adc_analysis import ADCAnalyzer
from characterization.performance_metrics import ReceiverPerformanceEvaluator
from utils.plotting import (
    plot_bode_response,
    plot_compression_curve,
    apply_dark_theme,
    PLOT_THEME
)


def render_characterization_tab(simulation_results: Dict[str, Any], is_hardware_mode: bool = False):
    """
    Renders Tab 4: Receiver Characterization.
    """
    st.subheader("🔬 Receiver Characterization & Parametric Sweeps")
    st.caption("Perform rigorous RF, DSP, and ADC parameter sweeps to quantify receiver performance limits and physical characteristics.")

    test_type = st.selectbox(
        "Select Characterization Sweep Testbench:",
        [
            "1. Frequency Response & Filter Selectivity (Bode Plot)",
            "2. Front-End Gain & 1 dB Compression (P1dB)",
            "3. Noise Figure & Sensitivity Sweep (AM Threshold Effect)",
            "4. Modulation Index Linearity & THD Distortion",
            "5. ADC Quantization & Empirical vs Theoretical SQNR",
            "6. Overall Receiver Performance KPI Summary"
        ],
        index=0
    )

    st.markdown("---")

    if "1. Frequency Response" in test_type:
        st.markdown("#### 1. RF Filter Selectivity & Frequency Response")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            cf_khz = st.number_input("Center Frequency (kHz):", value=40.0, step=5.0)
        with col_c2:
            bw_khz = st.selectbox("Configured Bandwidth (kHz):", [8.33, 12.5, 25.0, 50.0], index=2)
        with col_c3:
            f_order = st.slider("Filter Order:", 2, 8, 4, step=2)

        fra = FrequencyResponseAnalyzer(fs=200000)
        res = fra.analyze_filter(center_freq_hz=cf_khz * 1000.0, bandwidth_khz=bw_khz, filter_order=f_order)

        bode_fig = plot_bode_response(
            res["frequencies_hz"], res["magnitude_db"], res["phase_deg"],
            title=f"RF Front-End Bandpass Filter (BW = {bw_khz} kHz, Order = {f_order})"
        )
        st.plotly_chart(bode_fig, use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Measured -3dB BW", f"{res['measured_bw_3db_hz']/1000.0:.2f} kHz")
        k2.metric("Lower -3dB Cutoff", f"{res['measured_f_low_3db_hz']/1000.0:.2f} kHz")
        k3.metric("Upper -3dB Cutoff", f"{res['measured_f_high_3db_hz']/1000.0:.2f} kHz")
        k4.metric("Selectivity Shape Factor (20/3 dB)", f"{res['shape_factor_20_3']:.2f}")

    elif "2. Front-End Gain" in test_type:
        st.markdown("#### 2. LNA Linearity & 1 dB Compression Point (P1dB)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            cfg_gain = st.slider("Small-Signal Gain (dB):", 10.0, 35.0, 20.0, step=1.0)
        with col_g2:
            p1db_limit = st.slider("Saturation Ceiling (V):", 1.0, 5.0, 2.5, step=0.5)

        ga = GainAnalyzer()
        with st.spinner("Executing RF power sweep..."):
            res_gain = ga.sweep_input_power(gain_db=cfg_gain, p1db_v=p1db_limit, pin_dbm_range=(-70.0, 10.0), step_db=2.0)

        comp_fig = plot_compression_curve(
            res_gain["pin_dbm"], res_gain["pout_dbm"],
            small_signal_gain_db=cfg_gain,
            p1db_in=res_gain.get("p1db_input_dbm")
        )
        st.plotly_chart(comp_fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Small-Signal Gain", f"{cfg_gain:.1f} dB")
        m2.metric("Input P1dB", f"{res_gain.get('p1db_input_dbm', 'N/A')} dBm" if res_gain.get('p1db_input_dbm') else "> 10 dBm")
        m3.metric("Output P1dB", f"{res_gain.get('p1db_output_dbm', 'N/A')} dBm" if res_gain.get('p1db_output_dbm') else "> 30 dBm")

    elif "3. Noise Figure" in test_type:
        st.markdown("#### 3. Channel SNR vs Recovered Audio SNR (AM Threshold Effect)")
        st.caption("Observe the sharp cliff around 8–10 dB input SNR where envelope detection breaks down due to noise dominating carrier vector.")

        na = NoiseAnalyzer()
        with st.spinner("Running parametric SNR sweep across [-10 dB, +40 dB]..."):
            res_noise = na.sweep_input_snr(snr_min_db=-10.0, snr_max_db=40.0, step_db=2.5)

        fig_snr = go.Figure()
        fig_snr.add_trace(go.Scatter(
            x=res_noise["input_snr_db"],
            y=res_noise["output_snr_db"],
            mode="lines+markers",
            line={"color": PLOT_THEME["primary_cyan"], "width": 2.5},
            name="Recovered Audio SNR"
        ))
        if res_noise.get("detected_threshold_snr_db") is not None:
            fig_snr.add_vline(
                x=res_noise["detected_threshold_snr_db"],
                line_dash="dot",
                line_color=PLOT_THEME["warning_yellow"],
                annotation_text=f"Threshold Knee: ~{res_noise['detected_threshold_snr_db']:.1f} dB"
            )

        fig_snr.update_xaxes(title_text="Input Channel RF SNR (dB)")
        fig_snr.update_yaxes(title_text="Output Audio SNR (dB)")
        st.plotly_chart(apply_dark_theme(fig_snr, title="Receiver Noise Performance & Threshold Knee"), use_container_width=True)

    elif "4. Modulation Index" in test_type:
        st.markdown("#### 4. Modulation Index Sweep & Total Harmonic Distortion (THD)")
        ma = ModulationAnalyzer(fs=200000)
        with st.spinner("Computing modulation sweep..."):
            res_mod = ma.sweep_modulation_index(m_range=(0.1, 1.5), step=0.1)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_pow = go.Figure()
            fig_pow.add_trace(go.Scatter(x=res_mod["modulation_indices"], y=res_mod["carrier_powers_w"], name="Carrier Power (W)"))
            fig_pow.add_trace(go.Scatter(x=res_mod["modulation_indices"], y=res_mod["sideband_powers_w"], name="Sideband Power (W)"))
            fig_pow.update_xaxes(title_text="Modulation Index (m)")
            fig_pow.update_yaxes(title_text="Power (Watts)")
            st.plotly_chart(apply_dark_theme(fig_pow, title="Carrier vs Sideband Power Distribution"), use_container_width=True)

        with col_m2:
            fig_thd = go.Figure()
            fig_thd.add_trace(go.Scatter(
                x=res_mod["modulation_indices"], y=res_mod["thd_percent"],
                mode="lines+markers", line={"color": PLOT_THEME["danger_red"], "width": 2.0}, name="THD (%)"
            ))
            fig_thd.add_vline(x=1.0, line_dash="dash", line_color=PLOT_THEME["warning_yellow"], annotation_text="Overmodulation Threshold (m=1.0)")
            fig_thd.update_xaxes(title_text="Modulation Index (m)")
            fig_thd.update_yaxes(title_text="Total Harmonic Distortion (%)")
            st.plotly_chart(apply_dark_theme(fig_thd, title="Audio THD vs Modulation Index (Overmodulation Distortion)"), use_container_width=True)

    elif "5. ADC Quantization" in test_type:
        st.markdown("#### 5. ADC Quantization & Empirical vs Theoretical SQNR")
        aa = ADCAnalyzer(fs=200000)
        res_adc = aa.sweep_bit_depths(bit_depths=[8, 10, 12, 14, 16, 20, 24])

        fig_sqnr = go.Figure()
        fig_sqnr.add_trace(go.Scatter(
            x=res_adc["bit_depths"], y=res_adc["theoretical_sqnr_db"],
            mode="lines", line={"color": PLOT_THEME["accent_orange"], "dash": "dash"}, name="Theoretical SQNR (6.02N + 1.76 dB)"
        ))
        fig_sqnr.add_trace(go.Scatter(
            x=res_adc["bit_depths"], y=res_adc["measured_sqnr_db"],
            mode="lines+markers", line={"color": PLOT_THEME["primary_cyan"], "width": 2.0}, name="Measured Empirical SQNR"
        ))
        fig_sqnr.update_xaxes(title_text="ADC Resolution (Bits)")
        fig_sqnr.update_yaxes(title_text="Signal-to-Quantization-Noise Ratio (dB)")
        st.plotly_chart(apply_dark_theme(fig_sqnr, title="ADC Resolution vs SQNR Performance"), use_container_width=True)

    elif "6. Overall Receiver Performance" in test_type:
        st.markdown("#### 6. Overall Receiver Performance Summary")
        summary_obj = ReceiverPerformanceEvaluator.generate_performance_summary(simulation_results, is_hardware_mode=is_hardware_mode)
        st.dataframe(summary_obj["summary_df"], use_container_width=True)
