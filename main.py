"""
Main Streamlit Application.
Aviation Speech Processing – Interactive Receiver Simulation, Characterization and SER Platform.
Thiagarajar College of Engineering | Final Year ECE Project.
"""

import streamlit as st
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Aviation Receiver Simulation & SER Platform",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config.receiver_config import ReceiverParameters, VHF_CHANNEL_PRESETS
from receiver.simulation_pipeline import ReceiverSimulationEngine
from acquisition.audio_input import LiveAudioAcquisition
from acquisition.serial_input import SerialADCInterface
from acquisition.wav_handler import WAVHandler
from dashboard.overview import render_overview_tab
from dashboard.receiver import render_receiver_tab
from dashboard.simulation import render_simulation_tab
from dashboard.characterization import render_characterization_tab
from dashboard.audio_analysis import render_audio_analysis_tab
from dashboard.speech_processing import render_speech_processing_tab
from dashboard.emotion_recognition import render_emotion_recognition_tab
from dashboard.reports import render_reports_tab


# Initialize Simulation Engine in Session State
if "simulation_engine" not in st.session_state:
    st.session_state["simulation_engine"] = ReceiverSimulationEngine()

if "current_results" not in st.session_state:
    default_params = ReceiverParameters(
        signal_type="synthetic_atc",
        carrier_freq_mhz=121.500,
        modulation_index=0.75,
        channel_snr_db=25.0,
        adc_resolution_bits=16,
        duration_sec=0.5
    )
    st.session_state["current_results"] = st.session_state["simulation_engine"].run_simulation(params=default_params)

if "latest_ser_results" not in st.session_state:
    st.session_state["latest_ser_results"] = {
        "dominant_emotion": "Neutral",
        "confidence": 0.85,
        "inference_mode": "Model Checkpoint",
        "probabilities": {"Neutral": 0.85, "Calm": 0.10, "Angry": 0.05}
    }


# ==============================================================================
# SIDEBAR CONTROLS & HARDWARE INTEGRATION MODE
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <h3 style="margin: 0; color: #00E5FF;">✈ AVIATION RECEIVER</h3>
        <p style="margin: 0; font-size: 12px; color: #8B949E;">Simulation & SER Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛 Operating Mode")
    operating_mode = st.radio(
        "Select Active Mode:",
        ["Mode A: Digital Twin Simulation", "Mode B: Live Hardware Input"],
        index=0,
        help="Mode A runs the software DSP model. Mode B captures real demodulated audio from the physical receiver module."
    )
    is_hw_mode = (operating_mode == "Mode B: Live Hardware Input")

    if is_hw_mode:
        st.markdown("---")
        st.markdown("### 🔌 Hardware Connection")
        hw_interface = st.selectbox("Hardware Interface:", ["USB Sound Card / Line-In", "Serial UART Stream"])

        if hw_interface == "USB Sound Card / Line-In":
            in_devs = LiveAudioAcquisition.list_input_devices()
            if in_devs:
                dev_names = [f"[{d['device_id']}] {d['name']}" for d in in_devs]
                selected_dev = st.selectbox("Select Audio Device:", dev_names)
                dev_id = int(selected_dev.split("]")[0].replace("[", ""))
            else:
                st.warning("No audio input devices detected.")
                dev_id = None

            if st.button("🔴 Record Physical Receiver Audio (4s)"):
                with st.spinner("Recording 4s from physical receiver..."):
                    rec_audio, rec_metrics = LiveAudioAcquisition().record_audio_block(duration_sec=4.0, device_id=dev_id)
                    st.success("Acquired physical audio!")
                    # Inject hardware audio into simulation results stage 8
                    st.session_state["current_results"]["stages"]["stage8_digital_output"]["audio_samples"] = rec_audio
                    st.session_state["current_results"]["stages"]["stage8_digital_output"]["duration_sec"] = 4.0

        else:
            com_ports = SerialADCInterface.list_available_ports()
            if com_ports:
                port_names = [f"{p['device']} - {p['description']}" for p in com_ports]
                sel_port = st.selectbox("Select COM Port:", port_names)
                baud = st.selectbox("Baud Rate:", [115200, 9600, 57600, 230400], index=0)
                if st.button("🔴 Stream UART ADC Samples"):
                    with st.spinner("Reading ADC samples over UART..."):
                        p_name = sel_port.split(" - ")[0]
                        u_audio, u_m = SerialADCInterface.read_samples_block(p_name, baud_rate=baud)
                        st.info(u_m["status"])
                        if len(u_audio) > 0 and np.max(np.abs(u_audio)) > 0:
                            st.session_state["current_results"]["stages"]["stage8_digital_output"]["audio_samples"] = u_audio
            else:
                st.info("No active COM ports detected. Connect receiver microcontroller via USB.")

    st.markdown("---")
    st.markdown("### 🔊 Live Audio Monitor")
    dig_audio = st.session_state["current_results"]["stages"]["stage8_digital_output"]["audio_samples"]
    if len(dig_audio) > 0:
        wav_bytes = WAVHandler.to_in_memory_wav_bytes(dig_audio, sample_rate=16000)
        st.audio(wav_bytes, format="audio/wav")

    st.markdown("---")
    st.markdown("### 🟢 Subsystem Status")
    st.markdown(f"""
    - **Simulation Engine:** `ONLINE`
    - **Physical Line-In:** `{'ACTIVE' if is_hw_mode else 'STANDBY'}`
    - **Speech DSP Engine:** `16.0 kHz Ready`
    - **Deep Learning SER:** `ViT & BiLSTM Ready`
    """)


# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tabs = st.tabs([
    "1. Project Overview",
    "2. Receiver Operation",
    "3. Simulation Lab",
    "4. Receiver Characterization",
    "5. Audio Analysis",
    "6. Speech Processing",
    "7. Emotion Recognition",
    "8. Results & Report"
])

# Tab 1: Overview
with tabs[0]:
    render_overview_tab(is_hardware_mode=is_hw_mode)

# Tab 2: Receiver Operation (Stage Chain)
with tabs[1]:
    render_receiver_tab(st.session_state["current_results"])

# Tab 3: Simulation Lab
with tabs[2]:
    updated_results = render_simulation_tab(st.session_state["simulation_engine"])
    st.session_state["current_results"] = updated_results

# Tab 4: Receiver Characterization
with tabs[3]:
    render_characterization_tab(st.session_state["current_results"], is_hardware_mode=is_hw_mode)

# Tab 5: Audio Analysis
with tabs[4]:
    render_audio_analysis_tab(st.session_state["current_results"])

# Tab 6: Speech Processing Pipeline
with tabs[5]:
    render_speech_processing_tab(st.session_state["current_results"])

# Tab 7: Emotion Recognition
with tabs[6]:
    ser_out = render_emotion_recognition_tab(st.session_state["current_results"])
    if ser_out:
        st.session_state["latest_ser_results"] = ser_out

# Tab 8: Results & Report
with tabs[7]:
    render_reports_tab(
        st.session_state["current_results"],
        st.session_state["latest_ser_results"],
        is_hardware_mode=is_hw_mode
    )
