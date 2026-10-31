"""
Simulation Pipeline Module.
Executes the end-to-end receiver chain and logs the exact waveform, spectrum,
and performance metrics at every single stage:
Stage 1: RF Input (Modulated + Channel)
Stage 2: RF Front-End Bandpass Filter
Stage 3: Low Noise Amplifier (LNA)
Stage 4: Frequency Tuning & Channel Selection
Stage 5: AM Demodulation
Stage 6: Audio Processing (BPF + AGC + Denoiser)
Stage 7: ADC Quantization & Digitization
Stage 8: Final Digital WAV Audio
"""

import numpy as np
from typing import Dict, Any, Optional
from config.settings import FS_SIM, FS_AUDIO
from config.receiver_config import ReceiverParameters
from receiver.signal_generator import SignalGenerator
from receiver.am_modulator import AMModulator
from receiver.channel_model import VHFChannelModel
from receiver.rf_filter import RFFilter
from receiver.lna_model import LNAModel
from receiver.frequency_tuner import FrequencyTuner
from receiver.am_demodulator import AMDemodulator
from receiver.audio_processor import AudioProcessor
from receiver.agc import AGC
from receiver.adc_simulator import ADCSimulator


class ReceiverSimulationEngine:
    """
    Coordinator engine that executes the full digital twin simulation of the physical VHF receiver.
    """

    def __init__(self, fs_sim: int = FS_SIM, fs_audio: int = FS_AUDIO):
        self.fs_sim = fs_sim
        self.fs_audio = fs_audio

        self.sig_gen = SignalGenerator(fs=self.fs_sim)
        self.modulator = AMModulator(fs=self.fs_sim)
        self.channel = VHFChannelModel()
        self.rf_filter = RFFilter(fs=self.fs_sim)
        self.lna = LNAModel()
        self.tuner = FrequencyTuner(fs=self.fs_sim)
        self.demodulator = AMDemodulator(fs=self.fs_sim)
        self.audio_proc = AudioProcessor(fs=self.fs_sim)
        self.agc = AGC(fs=self.fs_sim)
        self.adc = ADCSimulator(sim_fs=self.fs_sim)

    def run_simulation(
        self,
        params: Optional[ReceiverParameters] = None,
        custom_audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs complete receiver simulation from antenna RF to digital WAV audio.

        Returns:
            Dictionary containing stage-by-stage waveforms, spectra, and metrics.
        """
        if params is None:
            params = ReceiverParameters()

        # --- Baseband Modulating Signal Generation ---
        if custom_audio_path:
            t, m_signal = self.sig_gen.load_wav_file(custom_audio_path, duration_sec=params.duration_sec)
        elif params.signal_type == "tone":
            t, m_signal = self.sig_gen.generate_single_tone(
                freq_hz=params.tone_freq_hz,
                duration_sec=params.duration_sec
            )
        elif params.signal_type == "multitone":
            t, m_signal = self.sig_gen.generate_multi_tone(duration_sec=params.duration_sec)
        else:
            t, m_signal = self.sig_gen.generate_synthetic_atc_speech(duration_sec=params.duration_sec)

        # --- Stage 1: RF Input & AM Modulation ---
        s_am, mod_metrics = self.modulator.modulate(
            t=t,
            modulating_signal=m_signal,
            carrier_freq_hz=params.if_frequency_hz,
            carrier_amplitude=params.carrier_amplitude,
            modulation_index=params.modulation_index
        )
        s_rf_received, chan_metrics = self.channel.apply_channel(
            signal=s_am,
            snr_db=params.channel_snr_db,
            attenuation_db=params.channel_attenuation_db
        )

        # --- Stage 2: RF Band-Pass Filter ---
        s_filtered, rf_filt_metrics = self.rf_filter.filter_signal(
            input_signal=s_rf_received,
            center_freq_hz=params.if_frequency_hz,
            bandwidth_khz=params.rf_filter_bandwidth_khz,
            filter_order=params.rf_filter_order
        )

        # --- Stage 3: Low Noise Amplifier (LNA) ---
        s_lna, lna_metrics = self.lna.amplify(
            input_signal=s_filtered,
            gain_db=params.lna_gain_db,
            noise_figure_db=params.lna_noise_figure_db,
            p1db_v=params.lna_p1db_v,
            bandwidth_hz=params.rf_filter_bandwidth_khz * 1000.0
        )

        # --- Stage 4: Frequency Tuning & Downconversion ---
        i_comp, q_comp, s_tuned_if, tune_metrics = self.tuner.tune_and_downconvert(
            t=t,
            rf_signal=s_lna,
            selected_freq_mhz=params.carrier_freq_mhz,
            carrier_freq_mhz=params.carrier_freq_mhz,
            if_center_hz=params.if_frequency_hz,
            tuning_error_hz=params.tuning_error_hz,
            channel_bandwidth_khz=params.rf_filter_bandwidth_khz
        )

        # --- Stage 5: AM Demodulation ---
        s_demod, demod_metrics = self.demodulator.demodulate(
            t=t,
            modulated_signal=s_tuned_if,
            method=params.demodulator_type,
            rc_us=params.envelope_rc_us,
            carrier_freq_hz=params.if_frequency_hz,
            original_audio=m_signal
        )

        # --- Stage 6: Audio Processing & AGC ---
        s_audio_proc, audio_metrics = self.audio_proc.process(
            audio_signal=s_demod,
            enable_bpf=params.enable_audio_bpf,
            low_cutoff_hz=params.audio_bpf_low_hz,
            high_cutoff_hz=params.audio_bpf_high_hz,
            enable_dc_blocker=params.enable_dc_blocker,
            enable_denoiser=params.enable_noise_reduction,
            enable_normalize=True
        )

        if params.enable_agc:
            s_agc, agc_gain_curve, agc_metrics = self.agc.process(
                signal=s_audio_proc,
                target_level=params.agc_target_level
            )
        else:
            s_agc = s_audio_proc
            agc_gain_curve = np.zeros_like(s_agc)
            agc_metrics = {"agc_enabled": False}

        # --- Stage 7: ADC Digitization ---
        t_adc, s_quantized, q_error, digital_audio_16k, adc_metrics = self.adc.quantize_and_sample(
            analog_signal=s_agc,
            resolution_bits=params.adc_resolution_bits,
            v_ref=params.adc_vref,
            target_fs=params.output_sampling_rate
        )

        # --- Pack Result Object ---
        results = {
            "time_sim": t,
            "fs_sim": self.fs_sim,
            "fs_audio": self.fs_audio,
            "params": params,
            "stages": {
                "stage1_rf_input": {
                    "name": "RF Front-End Input",
                    "modulating_audio": m_signal,
                    "am_transmitted": s_am,
                    "received_rf": s_rf_received,
                    "metrics": {**mod_metrics, **chan_metrics}
                },
                "stage2_rf_filter": {
                    "name": "RF Band-Pass Filter",
                    "input_signal": s_rf_received,
                    "output_signal": s_filtered,
                    "metrics": rf_filt_metrics
                },
                "stage3_lna": {
                    "name": "Low Noise Amplifier (LNA)",
                    "input_signal": s_filtered,
                    "output_signal": s_lna,
                    "metrics": lna_metrics
                },
                "stage4_tuning": {
                    "name": "Frequency Tuning & Downconversion",
                    "input_signal": s_lna,
                    "output_if": s_tuned_if,
                    "i_component": i_comp,
                    "q_component": q_comp,
                    "metrics": tune_metrics
                },
                "stage5_demodulator": {
                    "name": "AM Demodulator",
                    "input_signal": s_tuned_if,
                    "output_audio": s_demod,
                    "metrics": demod_metrics
                },
                "stage6_audio_proc": {
                    "name": "Audio Conditioning & AGC",
                    "input_signal": s_demod,
                    "output_audio": s_agc,
                    "gain_curve_db": agc_gain_curve,
                    "metrics": {**audio_metrics, **agc_metrics}
                },
                "stage7_adc": {
                    "name": "Analog-to-Digital Converter",
                    "analog_input": s_agc,
                    "quantized_signal": s_quantized,
                    "quantization_error": q_error,
                    "metrics": adc_metrics
                },
                "stage8_digital_output": {
                    "name": "Digital Audio Stream (WAV 16 kHz)",
                    "audio_samples": digital_audio_16k,
                    "sample_rate": self.fs_audio,
                    "duration_sec": len(digital_audio_16k) / self.fs_audio
                }
            }
        }

        return results
