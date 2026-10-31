"""
Tab 8: Results, Comparative Analysis & Report Export.
Provides side-by-side scenario comparison, complete KPI measurement tables,
and downloads for CSV logs, demodulated WAV audio, and comprehensive Markdown/PDF reports.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from characterization.performance_metrics import ReceiverPerformanceEvaluator
from utils.export import DataExporter
from acquisition.wav_handler import WAVHandler


def render_reports_tab(
    simulation_results: Dict[str, Any],
    ser_results: Dict[str, Any],
    is_hardware_mode: bool = False
):
    """
    Renders Tab 8: Results and Report.
    """
    st.subheader("📑 Experimental Results & Examination Report")
    st.caption("Review overall receiver metrics, perform comparative analysis across simulation runs, and export verified project logs.")

    summary_obj = ReceiverPerformanceEvaluator.generate_performance_summary(
        simulation_results,
        is_hardware_mode=is_hardware_mode
    )

    # 1. Performance Summary Table
    st.markdown("#### 1. Receiver Performance Evaluation Table")
    df_summary = summary_obj["summary_df"]
    st.dataframe(df_summary, use_container_width=True)

    # Export Buttons
    col_csv, col_wav, col_rep = st.columns(3)

    with col_csv:
        csv_bytes = DataExporter.to_csv_bytes(df_summary)
        st.download_button(
            label="⬇ Export Measurements (CSV)",
            data=csv_bytes,
            file_name="receiver_characterization_metrics.csv",
            mime="text/csv"
        )

    with col_wav:
        dig_audio = simulation_results["stages"]["stage8_digital_output"]["audio_samples"]
        wav_bytes = WAVHandler.to_in_memory_wav_bytes(dig_audio, sample_rate=16000)
        st.download_button(
            label="⬇ Download Demodulated Audio (WAV)",
            data=wav_bytes,
            file_name="demodulated_aviation_speech.wav",
            mime="audio/wav"
        )

    with col_rep:
        md_report = DataExporter.generate_experiment_markdown_report(
            simulation_results, ser_results, is_hardware_mode=is_hardware_mode
        )
        st.download_button(
            label="📄 Export Examination Report (MD)",
            data=md_report.encode("utf-8"),
            file_name="viva_examination_report.md",
            mime="text/markdown"
        )

    st.markdown("---")

    # 2. Comparative Analysis Section
    st.markdown("#### 2. Comparative Analysis: Scenario A vs Scenario B")
    st.caption("Benchmark how varying parameters (e.g. 16-bit vs 8-bit ADC, or 30 dB SNR vs 5 dB SNR) impacts audio fidelity and SER confidence.")

    # Save current run into session state if requested
    if "saved_scenario" not in st.session_state:
        st.session_state["saved_scenario"] = None

    col_btn_save, col_btn_clear = st.columns([1, 4])
    with col_btn_save:
        if st.button("💾 Save Current Run as Scenario A"):
            st.session_state["saved_scenario"] = {
                "params": simulation_results["params"].to_dict(),
                "snr_out": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("fidelity_snr_db", 0),
                "corr": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("cross_correlation", 0),
                "mse": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("mean_squared_error", 0),
                "sqnr": simulation_results["stages"]["stage7_adc"]["metrics"].get("measured_sqnr_db", 0),
                "emotion": ser_results.get("dominant_emotion", "Neutral"),
                "confidence": ser_results.get("confidence", 0) * 100.0
            }
            st.success("Saved current configuration as Scenario A!")

    if st.session_state["saved_scenario"] is not None:
        sc_a = st.session_state["saved_scenario"]
        sc_b = {
            "snr_out": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("fidelity_snr_db", 0),
            "corr": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("cross_correlation", 0),
            "mse": simulation_results["stages"]["stage5_demodulator"]["metrics"].get("mean_squared_error", 0),
            "sqnr": simulation_results["stages"]["stage7_adc"]["metrics"].get("measured_sqnr_db", 0),
            "emotion": ser_results.get("dominant_emotion", "Neutral"),
            "confidence": ser_results.get("confidence", 0) * 100.0
        }

        comp_df = pd.DataFrame([
            {"Metric": "Modulation Index (m)", "Scenario A (Saved)": f"{sc_a['params'].get('modulation_index', 0):.2f}", "Scenario B (Current)": f"{simulation_results['params'].modulation_index:.2f}"},
            {"Metric": "Channel SNR (dB)", "Scenario A (Saved)": f"{sc_a['params'].get('channel_snr_db', 0):.1f} dB", "Scenario B (Current)": f"{simulation_results['params'].channel_snr_db:.1f} dB"},
            {"Metric": "ADC Resolution", "Scenario A (Saved)": f"{sc_a['params'].get('adc_resolution_bits', 16)}-bit", "Scenario B (Current)": f"{simulation_results['params'].adc_resolution_bits}-bit"},
            {"Metric": "Recovered Audio SNR", "Scenario A (Saved)": f"{sc_a['snr_out']:.2f} dB", "Scenario B (Current)": f"{sc_b['snr_out']:.2f} dB"},
            {"Metric": "Demod Correlation (rho)", "Scenario A (Saved)": f"{sc_a['corr']:.4f}", "Scenario B (Current)": f"{sc_b['corr']:.4f}"},
            {"Metric": "Demod MSE", "Scenario A (Saved)": f"{sc_a['mse']:.6f}", "Scenario B (Current)": f"{sc_b['mse']:.6f}"},
            {"Metric": "ADC Measured SQNR", "Scenario A (Saved)": f"{sc_a['sqnr']:.1f} dB", "Scenario B (Current)": f"{sc_b['sqnr']:.1f} dB"},
            {"Metric": "Predicted SER Emotion", "Scenario A (Saved)": sc_a["emotion"], "Scenario B (Current)": sc_b["emotion"]},
            {"Metric": "SER Confidence", "Scenario A (Saved)": f"{sc_a['confidence']:.1f}%", "Scenario B (Current)": f"{sc_b['confidence']:.1f}%"}
        ])
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Click 'Save Current Run as Scenario A' above, then adjust any parameter in the Simulation Lab to view real-time comparative deltas.")

    st.markdown("---")

    # 3. Live Report Preview
    with st.expander("👁 View Formatted Examination Report Preview"):
        st.markdown(md_report)
