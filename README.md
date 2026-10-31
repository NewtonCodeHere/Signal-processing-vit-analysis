# Aviation Speech Processing & Receiver Characterization Platform

**Speech Emotion Recognition & VHF Receiver Digital Twin Platform**  
**Lead Contributor & Developer:** NewtonCodeHere (Newton)  
**Academic Reference:** Department of Electronics and Communication Engineering  

---

## 1. Overview & System Mission

This platform is a high-fidelity software digital twin, characterization suite, and Speech Emotion Recognition (SER) deployment platform designed to represent the completed physical aviation VHF receiver module.

### Core Problems Addressed:
1. **Domain-Specific Speech Dataset:** Standard SER datasets (*RAVDESS, CREMA-D, TESS, SAVEE*) rely on studio-recorded actors without aviation phraseology, cockpit noise, or RF channel degradation.
2. **Accessible Aviation Audio:** Cockpit voice recorder (CVR) audio is classified and strictly inaccessible; this project captures operational **Pilot–ATC VHF voice communication** (118.000–136.975 MHz AM DSB-FC).
3. **Physical Receiver Characterization:** Simulates and measures the electrical characteristics of every physical receiving block (Antenna $\rightarrow$ RF Filter $\rightarrow$ LNA $\rightarrow$ Tuning $\rightarrow$ AM Demodulator $\rightarrow$ Audio Conditioning $\rightarrow$ ADC $\rightarrow$ 16 kHz WAV Audio).

---

## 2. System Architecture

```
[TRANSMITTER SIDE (AIRCRAFT)]:
Pilot Speaks -> Cockpit Mic -> Aircraft Audio System -> AM Modulator (DSB-FC) -> VHF Transmitter (118-137 MHz) -> Antenna
                                                                                                                │
[PROPAGATION CHANNEL]: VHF Airband Channel (Free-Space Path Loss + AWGN Noise) ◄────────────────────────────────┘
                                  │
[GROUND DATA ACQUISITION RECEIVER]:
Receiving Antenna -> RF Front-End (Filter + LNA) -> Tuning / Channel Selection (118-137 MHz) -> AM Demodulator
                                                                                                │
Audio Conditioning (Voice BPF 300-3400 Hz, AGC, Denoiser) ◄─────────────────────────────────────┘
       │
       ▼
Analog-to-Digital Converter (ADC: 8 to 24-bit configurable, 16 kHz linear PCM)
       │
       ▼
[DIGITAL SPEECH & SER PIPELINE]:
Voice Activity Detection (VAD) -> 4-Second Windowing -> 128x128 Log-Mel Spectrogram -> ViT / CNN-BiLSTM -> Softmax Emotion Probabilities
```

---

## 3. Installation & Setup Instructions

### Prerequisites
- Windows 10/11 Laptop or Desktop PC
- Python 3.10 to 3.14 installed with pip added to PATH

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\Users\luffy\Downloads\AirCraftSignalProcessingProject
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Complete Pytest Verification Suite
```powershell
pytest tests/
```
*(All 16 unit tests verify modulation physics, ADC SQNR, filter selectivity, and neural network forward passes).*

### Step 4: Launch the Interactive Dashboard
```powershell
streamlit run main.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 4. Navigation & Demonstration Guide for Review Panel

| Tab # | Title | What to Demonstrate to Faculty Guide / Panel |
| :--- | :--- | :--- |
| **Tab 1** | **Project Overview** | Shows original Anna University bonafide context, objectives, and the uploaded Pilot-ATC communication framework diagram (Figure 4). |
| **Tab 2** | **Receiver Operation** | Select any hardware stage (1 to 8) to inspect input vs. output waveforms, FFT spectra, gain in dB, bandwidth, and insertion loss. |
| **Tab 3** | **Simulation Lab** | Adjust sliders for carrier frequency, modulation index $m$, channel SNR, LNA gain, and ADC bit depth. Show that outputs recalculate dynamically in real-time. Demonstrate overmodulation clipping ($m > 1.0$). |
| **Tab 4** | **Receiver Characterization** | Execute automated parametric sweeps: Bode plot (-3dB BW and shape factor), 1 dB compression curve ($P_{1\text{dB}}$), AM threshold effect knee, and ADC SQNR vs. $6.02N + 1.76\text{ dB}$. |
| **Tab 5** | **Audio Analysis** | Listen to recovered demodulated speech via the audio player, inspect peak/RMS levels, crest factor, and 3D time-frequency spectrograms. |
| **Tab 6** | **Speech Processing** | Demonstrate standard telecommunication conditioning: 16.0 kHz resampling, pre-emphasis, 4-second fixed windowing, and Voice Activity Detection (VAD). |
| **Tab 7** | **Emotion Recognition** | Select between **Model 6: CNN-BiLSTM** (87% benchmark) and **Model 4: Vertically Patched ViT**. Display dominant emotion, confidence gauge, and 8-class probability bars. |
| **Tab 8** | **Results & Report** | Save Scenario A, adjust parameters, and demonstrate side-by-side comparative deltas. Download CSV measurements, clean WAV audio, and Markdown/PDF examination report. |

---

## 5. Connecting the Physical Receiver (Hardware Input Mode)

The platform supports two modes via the sidebar toggle:

### Mode A: Digital Twin Simulation
The software generates calibrated RF/IF signals and simulates the full hardware receiving chain.

### Mode B: Physical Hardware Input
1. Connect the analog audio output of the physical receiver module (e.g. from LM386 / audio pre-amplifier) to the PC via:
   - **USB Sound Card / 3.5mm Microphone Line-In**, OR
   - **Microcontroller UART Stream** (e.g., STM32/ESP32 ADC streaming samples at 115200 baud).
2. In the Sidebar, flip the toggle to **Mode B: Live Hardware Input**.
3. Select your audio device or COM port and click **🔴 Record Physical Receiver Audio (4s)**.
4. The live acquired audio is fed into the Voice Activity Detection, Log-Mel spectrogram generator, and SER neural networks!

---

## 6. Directory Structure
```
AirCraftSignalProcessingProject/
├── main.py                          # Streamlit application entry point
├── requirements.txt                 # Dependencies specification
├── README.md                        # Documentation & laboratory manual
├── config/                          # Global settings & VHF channel presets
├── receiver/                        # Subsystem A: Digital twin receiver engine (Stages 1 to 8)
├── characterization/                # Subsystem B: Automated parametric testbenches & sweeps
├── acquisition/                     # Hardware integration: USB audio, serial UART, WAV handling
├── dsp/                             # Speech conditioning, VAD segmentation & Log-Mel matrices
├── features/                        # 40 MFCC extraction with Delta/Delta-Delta dynamics
├── models/                          # Subsystem C: Vertically Patched ViT & CNN-BiLSTM SER models
├── dashboard/                       # Streamlit UI tabs 1 through 8
├── utils/                           # Dark-themed Plotly charts & report export engine
├── assets/                          # Original Pilot-ATC communication framework diagram
├── data/                            # Input audio, generated audio, and model checkpoints
└── tests/                           # Pytest validation testbench (16 passing tests)
```
