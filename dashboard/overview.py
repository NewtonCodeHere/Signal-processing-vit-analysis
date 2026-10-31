"""
Tab 1: Project Overview & Architecture Reference.
Renders academic project background, system objectives, and the primary
Pilot-ATC VHF Communication Framework architecture diagram.
"""

import streamlit as st
from pathlib import Path
from config.settings import ASSETS_DIR


def render_overview_tab(is_hardware_mode: bool = False):
    """
    Renders Tab 1: Project Overview.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #161B22 0%, #0D1117 100%); padding: 22px; border-radius: 10px; border-left: 5px solid #00E5FF; margin-bottom: 25px;">
        <h2 style="margin: 0; color: #FFFFFF; font-size: 24px;">✈ Real-Time Aviation Speech Dataset Acquisition & Speech Emotion Recognition</h2>
        <p style="margin: 6px 0 0 0; color: #8B949E; font-size: 14px;">
            <b>B.E. Final Year Project | Department of Electronics and Communication Engineering</b><br>
            Thiagarajar College of Engineering, Madurai | Anna University
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Target Band", value="118.0 - 137.0 MHz", delta="VHF Airband (AM DSB-FC)")
    with col2:
        st.metric(label="Speech Sample Rate", value="16.0 kHz", delta="16-bit PCM Linear")
    with col3:
        st.metric(label="SER Best Model", value="CNN-BiLSTM (87%)", delta="RAVDESS+CREMAD+TESS")
    with col4:
        mode_label = "Hardware Line-In" if is_hardware_mode else "Digital Simulation"
        st.metric(label="Platform Mode", value=mode_label, delta="Active & Calibrated")

    st.markdown("---")

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("📌 Project Motivation & Objectives")
        st.markdown("""
        **1. Overcoming Dataset Limitations:**  
        Standard Speech Emotion Recognition (SER) datasets (*RAVDESS, CREMA-D, TESS, SAVEE*) are recorded by actors in quiet acoustic studios. They lack:
        - Real aviation phraseology and standard calls.
        - Aircraft cockpit high-decibel environmental noise.
        - VHF bandpass channel distortion and fading effects.

        **2. The Pilot–ATC Radio Focus:**  
        Cockpit voice recorder (CVR) audio is classified and strictly inaccessible under aviation safety laws. Therefore, this project captures **Pilot–ATC VHF ground-to-air radio communications** (118.000–136.975 MHz AM DSB-FC), which realistically embodies operational aviation speech.

        **3. Dual Operating Modes:**  
        - **Subsystem A (Simulation Lab):** Full software digital twin recreation of the completed physical receiver for parametric characterization and faculty review.
        - **Subsystem B (Hardware Line-In):** Captures real-time demodulated audio from the physical receiver module via USB line-in, microphone, or UART ADC stream.
        """)

        st.subheader("🏛 Academic Credits")
        st.markdown("""
        - **Faculty Guide:** Dr. M.N. Suresh, Associate Professor, Dept. of ECE, TCE Madurai
        - **Project Investigators:** Nishaanth R (2302917710621091), Hari Hara Sudhan C (2303917710621032), Prem Kumar P (2303917710621317)
        """)

    with right_col:
        st.subheader("🖼 Pilot–ATC VHF Communication Framework")
        framework_img_path = Path(ASSETS_DIR) / "receiver_framework.png"
        if framework_img_path.exists():
            st.image(str(framework_img_path), caption="Figure 4: Pilot–ATC VHF Communication Framework & Receiving System", use_container_width=True)
        else:
            st.warning("Framework diagram image not found in assets/ directory.")

    st.markdown("---")
    st.subheader("🧩 End-to-End System Architecture")
    st.markdown("""
    ```
    [TRANSMITTER]: Pilot Speech -> Microphone -> Audio System -> AM Modulator -> VHF Transmitter (118-137 MHz) -> Antenna
                                                                                                            │
    [CHANNEL]: VHF Free-Space Airband Channel (Path Loss + AWGN Noise) ◄────────────────────────────────────┘
                                     │
    [RECEIVING SYSTEM]: Receiving Antenna -> RF BPF -> LNA -> VHF Tuning -> AM Demodulator -> Audio Conditioning -> ADC
                                                                                                            │
    [DIGITAL SPEECH & SER]: Digital Audio (WAV) -> Voice Activity Detection (VAD) -> 128x128 Log-Mel -> ViT / CNN-BiLSTM -> Emotion Probabilities
    ```
    """)
