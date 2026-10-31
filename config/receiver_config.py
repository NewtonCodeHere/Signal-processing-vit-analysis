"""
Hardware presets and receiver configuration dataclasses.
Provides both default physical receiver values and user-configurable simulation parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# Common Civil Aviation VHF Channel Presets
VHF_CHANNEL_PRESETS = {
    "121.500 MHz - International Emergency Guard": 121.500,
    "118.100 MHz - Primary Control Tower (TWR)": 118.100,
    "119.300 MHz - Approach / Radar Control (APP)": 119.300,
    "121.900 MHz - Ground Movement Control (GND)": 121.900,
    "127.850 MHz - Automated Terminal Info (ATIS)": 127.850,
    "123.450 MHz - Air-to-Air Operational Intercom": 123.450,
    "131.250 MHz - Company Airline Dispatch / Ops": 131.250
}

@dataclass
class ReceiverParameters:
    """
    Encapsulates all parameters for simulating the end-to-end VHF Aviation Receiver chain.
    """
    # Signal & Modulation
    signal_type: str = "tone"              # 'tone', 'multitone', 'atc_synthetic', 'wav_file'
    tone_freq_hz: float = 1000.0           # Modulating tone frequency (Hz)
    carrier_freq_mhz: float = 121.500      # VHF carrier frequency (MHz)
    carrier_amplitude: float = 1.0         # Carrier peak amplitude (V)
    modulation_index: float = 0.70         # AM modulation index m (0.0 to 1.5)
    duration_sec: float = 1.0              # Simulation time span (seconds)

    # Channel & Antenna
    channel_snr_db: float = 25.0           # Additive White Gaussian Noise SNR (dB)
    channel_attenuation_db: float = 12.0   # Path loss attenuation (dB)

    # RF Front-End (Filter + LNA)
    rf_filter_bandwidth_khz: float = 25.0  # RF front-end bandpass filter BW (kHz)
    rf_filter_order: int = 4               # Filter order (Chebyshev / Butterworth)
    lna_gain_db: float = 20.0              # LNA gain in decibels
    lna_noise_figure_db: float = 2.5       # LNA noise figure (dB)
    lna_p1db_v: float = 2.5                # LNA 1dB compression input/output limit (V)

    # Tuning & IF Stage
    tuning_error_hz: float = 0.0           # Tuning frequency offset / oscillator drift (Hz)
    if_frequency_hz: float = 40000.0       # Simulated Intermediate Frequency (Hz)

    # Demodulator
    demodulator_type: str = "envelope"     # 'envelope' (diode/RC) or 'coherent' (product)
    envelope_rc_us: float = 45.0           # RC discharge time constant (microseconds)

    # Audio Conditioning
    enable_audio_bpf: bool = True          # Enable voice 300-3400 Hz BPF
    audio_bpf_low_hz: float = 300.0        # Audio BPF lower cutoff
    audio_bpf_high_hz: float = 3400.0      # Audio BPF upper cutoff
    enable_agc: bool = True                # Enable Automatic Gain Control
    agc_target_level: float = 0.7          # AGC target amplitude level
    enable_dc_blocker: bool = True         # DC offset rejection
    enable_noise_reduction: bool = True    # Spectral denoiser

    # ADC Digitization
    adc_resolution_bits: int = 16          # Quantization bit depth (8, 10, 12, 16, 24)
    adc_vref: float = 1.0                  # ADC full-scale reference voltage (V)
    output_sampling_rate: int = 16000      # Final output audio sampling rate (Hz)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
