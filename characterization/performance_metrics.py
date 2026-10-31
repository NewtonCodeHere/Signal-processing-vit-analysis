"""
Overall Receiver Performance Summary Module.
Aggregates performance metrics across RF, IF, demodulation, audio conditioning, and ADC stages.
Explicitly distinguishes between physically measured hardware values and software-simulated parameters.
"""

from typing import Dict, Any, List
import pandas as pd


class ReceiverPerformanceEvaluator:
    """
    Summarizes end-to-end receiver performance indicators.
    """

    @staticmethod
    def generate_performance_summary(
        simulation_results: Dict[str, Any],
        is_hardware_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extracts key KPIs and returns a structured summary table.

        Args:
            simulation_results: Output dictionary from ReceiverSimulationEngine.
            is_hardware_mode: If True, flags values as measured from physical receiver hardware.

        Returns:
            Dictionary containing KPI summary and comparison table.
        """
        stages = simulation_results["stages"]
        p = simulation_results["params"]

        s1_m = stages["stage1_rf_input"]["metrics"]
        s2_m = stages["stage2_rf_filter"]["metrics"]
        s3_m = stages["stage3_lna"]["metrics"]
        s5_m = stages["stage5_demodulator"]["metrics"]
        s7_m = stages["stage7_adc"]["metrics"]

        source_tag = "Measured (Hardware)" if is_hardware_mode else "Simulated (Mathematical Model)"

        summary_rows = [
            {"Parameter": "Operating Band", "Value": "118.000 - 136.975 MHz", "Unit": "MHz", "Category": "VHF Airband Spec", "Data Source": "Verified Specification"},
            {"Parameter": "Modulation Scheme", "Value": "AM (DSB-FC)", "Unit": "Type", "Category": "RF Standards", "Data Source": "Verified Specification"},
            {"Parameter": "Tuned Channel Frequency", "Value": f"{p.carrier_freq_mhz:.3f}", "Unit": "MHz", "Category": "Tuning", "Data Source": source_tag},
            {"Parameter": "Modulation Index (m)", "Value": f"{p.modulation_index:.2f}", "Unit": "-", "Category": "Modulation", "Data Source": source_tag},
            {"Parameter": "RF Filter Bandwidth", "Value": f"{s2_m.get('bandwidth_khz', 25.0):.1f}", "Unit": "kHz", "Category": "RF Selectivity", "Data Source": source_tag},
            {"Parameter": "LNA Gain", "Value": f"{s3_m.get('actual_gain_db', 20.0):.1f}", "Unit": "dB", "Category": "RF Amplification", "Data Source": source_tag},
            {"Parameter": "LNA Noise Figure", "Value": f"{s3_m.get('noise_figure_db', 2.5):.1f}", "Unit": "dB", "Category": "RF Noise", "Data Source": source_tag},
            {"Parameter": "Input Channel SNR", "Value": f"{s1_m.get('measured_snr_db', p.channel_snr_db):.1f}", "Unit": "dB", "Category": "RF Channel", "Data Source": source_tag},
            {"Parameter": "Recovered Audio SNR", "Value": f"{s5_m.get('fidelity_snr_db', 0.0):.1f}", "Unit": "dB", "Category": "Audio Fidelity", "Data Source": source_tag},
            {"Parameter": "Demodulation Cross-Correlation (rho)", "Value": f"{s5_m.get('cross_correlation', 0.0):.4f}", "Unit": "-", "Category": "Fidelity", "Data Source": source_tag},
            {"Parameter": "Demodulation Error (MSE)", "Value": f"{s5_m.get('mean_squared_error', 0.0):.5f}", "Unit": "-", "Category": "Fidelity", "Data Source": source_tag},
            {"Parameter": "ADC Resolution", "Value": f"{s7_m.get('resolution_bits', 16)}", "Unit": "bits", "Category": "Digitization", "Data Source": source_tag},
            {"Parameter": "ADC Quantization Step", "Value": f"{s7_m.get('quantization_step_mv', 0.0):.3f}", "Unit": "mV", "Category": "Digitization", "Data Source": source_tag},
            {"Parameter": "ADC SQNR (Empirical)", "Value": f"{s7_m.get('measured_sqnr_db', 0.0):.1f}", "Unit": "dB", "Category": "Digitization", "Data Source": source_tag},
            {"Parameter": "Digital Audio Sample Rate", "Value": f"{s7_m.get('output_fs', 16000)}", "Unit": "Hz", "Category": "SER Standards", "Data Source": "Verified Specification"}
        ]

        df_summary = pd.DataFrame(summary_rows)

        return {
            "is_hardware_mode": is_hardware_mode,
            "overall_snr_improvement_db": float(s5_m.get("fidelity_snr_db", 0.0) - s1_m.get("measured_snr_db", 0.0)),
            "reconstruction_correlation": float(s5_m.get("cross_correlation", 0.0)),
            "summary_df": df_summary,
            "summary_records": summary_rows
        }
