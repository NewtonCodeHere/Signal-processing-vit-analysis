"""
Global system configurations and constant definitions for the
Aviation Speech Processing & Receiver Characterization Platform.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
INPUT_AUDIO_DIR = DATA_DIR / "input"
GENERATED_AUDIO_DIR = DATA_DIR / "generated"
MODELS_DIR = DATA_DIR / "models"

# Audio and DSP Sampling Standards
FS_AUDIO = 16000          # 16 kHz: Standard aviation voice & SER input sampling rate
FS_SIM = 200000           # 200 kHz: Intermediate/Simulation sampling rate for RF/IF processing
WINDOW_DURATION = 4.0     # 4 seconds: Fixed classification window matching Project I & II
N_SAMPLES_WINDOW = int(FS_AUDIO * WINDOW_DURATION)  # 64,000 samples

# VHF Civil Aviation Frequency Band (ICAO Annex 10)
VHF_MIN_FREQ_MHZ = 118.000
VHF_MAX_FREQ_MHZ = 136.975
VHF_CHANNEL_SPACING_KHZ = 25.0       # 25 kHz channelization
VHF_NARROW_SPACING_KHZ = 8.33        # 8.33 kHz split channelization

# Intermediate Frequency (IF) for Simulation
DEFAULT_IF_FREQ_HZ = 40000.0         # 40 kHz IF carrier for stable numerical representation

# Audio Voice Frequency Bounds (Standard Airband Voice Channel)
VOICE_BPF_LOW_HZ = 300.0             # 300 Hz lower speech cutoff
VOICE_BPF_HIGH_HZ = 3400.0           # 3400 Hz upper speech cutoff

# Standard Emotions (CREMA-D, RAVDESS, TESS taxonomy)
EMOTION_LABELS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]

# Color codes for dark-themed Plotly charts
PLOT_THEME = {
    "bg_color": "#0E1117",
    "paper_color": "#161B22",
    "text_color": "#E6EDF3",
    "grid_color": "#30363D",
    "primary_cyan": "#00E5FF",
    "accent_orange": "#FF9100",
    "success_green": "#00E676",
    "warning_yellow": "#FFD600",
    "danger_red": "#FF1744",
    "purple": "#D500F9"
}
