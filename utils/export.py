"""
Report and Data Export Utilities Module.
Generates CSV formatted measurement logs and structured experiment summary text.
"""

import io
import pandas as pd
from typing import Dict, Any


class DataExporter:
    """
    Handles CSV and text data exports for project review panels and laboratory logs.
    """

    @staticmethod
    def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
        """
        Converts pandas DataFrame into CSV byte buffer for browser download.
        """
        return dataframe.to_csv(index=False).encode("utf-8")

    @staticmethod
    def generate_experiment_markdown_report(
        simulation_results: Dict[str, Any],
        ser_results: Dict[str, Any],
        is_hardware_mode: bool = False
    ) -> str:
        """
        Builds a comprehensive technical markdown report for the faculty review panel.
        """
        p = simulation_results["params"]
        stages = simulation_results["stages"]
        s1 = stages["stage1_rf_input"]["metrics"]
        s3 = stages["stage3_lna"]["metrics"]
        s5 = stages["stage5_demodulator"]["metrics"]
        s7 = stages["stage7_adc"]["metrics"]

        mode_str = "Physical Receiver Hardware Mode" if is_hardware_mode else "Digital Twin Simulation Mode"

        report = f"""# AVIATION RECEIVER CHARACTERIZATION & SPEECH EMOTION RECOGNITION REPORT
**Institution:** Thiagarajar College of Engineering, Madurai  
**Department:** Electronics and Communication Engineering  
**Project:** 22EC790 - Real-Time Aviation Speech Acquisition & SER  
**Operating Mode:** {mode_str}  

---

## 1. RF FRONT-END & RECEIVER CONFIGURATION
- **Civil Aviation Frequency:** {p.carrier_freq_mhz:.3f} MHz (VHF Airband: 118.000 - 136.975 MHz)
- **Modulation Standard:** AM Double Sideband Full Carrier (DSB-FC)
- **Modulation Index (m):** {p.modulation_index:.2f} ({'OVER-MODULATION' if p.modulation_index > 1.0 else 'Linear'})
- **Channel Additive SNR:** {s1.get('measured_snr_db', p.channel_snr_db):.1f} dB
- **RF Filter Bandwidth:** {p.rf_filter_bandwidth_khz:.1f} kHz
- **LNA Amplification Gain:** {s3.get('actual_gain_db', p.lna_gain_db):.1f} dB
- **LNA Noise Figure:** {s3.get('noise_figure_db', p.lna_noise_figure_db):.1f} dB

---

## 2. DEMODULATION & AUDIO CONDITIONING
- **Demodulation Technique:** {p.demodulator_type.upper()} Detector
- **Recovered Audio Fidelity SNR:** {s5.get('fidelity_snr_db', 0.0):.2f} dB
- **Cross-Correlation with Original:** {s5.get('cross_correlation', 0.0):.4f}
- **Demodulation Mean Squared Error (MSE):** {s5.get('mean_squared_error', 0.0):.6f}
- **Voice Bandpass Conditioning:** {p.audio_bpf_low_hz:.0f} Hz - {p.audio_bpf_high_hz:.0f} Hz
- **Automatic Gain Control (AGC):** {'ENABLED' if p.enable_agc else 'DISABLED'}

---

## 3. ANALOG-TO-DIGITAL CONVERTER (ADC)
- **Quantization Bit Depth:** {s7.get('resolution_bits', 16)} bits ({s7.get('num_quantization_levels', 65536)} levels)
- **Quantization Step (LSB):** {s7.get('quantization_step_mv', 0.0):.4f} mV
- **Theoretical SQNR (6.02N + 1.76):** {s7.get('theoretical_sqnr_db', 0.0):.2f} dB
- **Empirical Measured SQNR:** {s7.get('measured_sqnr_db', 0.0):.2f} dB
- **Digital Output Rate:** {s7.get('output_fs', 16000)} Hz (Aviation Speech Standard)

---

## 4. SPEECH EMOTION RECOGNITION (SER) INFERENCE
- **Classification Model:** {ser_results.get('model_name', 'CNN-BiLSTM')}
- **Inference Provenance:** {ser_results.get('inference_mode', 'Model Checkpoint')}
- **Predicted Emotion:** **{ser_results.get('dominant_emotion', 'Neutral')}**
- **Classification Confidence:** {ser_results.get('confidence', 0.0) * 100.0:.2f}%
- **Status:** {'Model Checkpoint Active' if ser_results.get('weights_loaded') else 'Acoustic Baseline Mode'}
"""
        return report
