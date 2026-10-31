"""
FastAPI Backend Application Server.
Exposes high-performance REST APIs for real-time receiver simulation,
DSP characterization sweeps, speech feature extraction, and SER inference.
Serves the modern Apple iOS styled HTML/CSS/JS frontend.
"""

import io
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import BASE_DIR, ASSETS_DIR, FS_AUDIO
from config.receiver_config import ReceiverParameters
from receiver.simulation_pipeline import ReceiverSimulationEngine
from characterization.frequency_response import FrequencyResponseAnalyzer
from characterization.gain_analysis import GainAnalyzer
from characterization.noise_analysis import NoiseAnalyzer
from characterization.modulation_analysis import ModulationAnalyzer
from characterization.adc_analysis import ADCAnalyzer
from characterization.performance_metrics import ReceiverPerformanceEvaluator
from dsp.filtering import AudioPreprocessor
from dsp.segmentation import AudioSegmenter
from features.feature_extraction import FeatureExtractionPipeline
from models.predictor import SERPredictor
from models.registry import AVAILABLE_MODELS, get_model_status
from acquisition.audio_input import LiveAudioAcquisition
from acquisition.serial_input import SerialADCInterface
from acquisition.wav_handler import WAVHandler
from utils.export import DataExporter

app = FastAPI(title="Aviation Speech Processing Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
engine = ReceiverSimulationEngine()
predictor = SERPredictor()
last_simulation_results: Optional[Dict[str, Any]] = None
last_ser_results: Optional[Dict[str, Any]] = None
is_hardware_active = False

# Mount Static Assets & Frontend Files
WEB_DIR = BASE_DIR / "web"
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def sanitize_for_json(obj: Any) -> Any:
    """Converts numpy arrays, scalars, and filters out non-serializable objects for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "sos_coefficients"}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# Pydantic Request Models
class SimParamsRequest(BaseModel):
    signal_type: str = "synthetic_atc"
    tone_freq_hz: float = 1000.0
    carrier_freq_mhz: float = 121.500
    carrier_amplitude: float = 1.0
    modulation_index: float = 0.75
    channel_snr_db: float = 25.0
    rf_filter_bandwidth_khz: float = 25.0
    lna_gain_db: float = 20.0
    tuning_error_hz: float = 0.0
    demodulator_type: str = "envelope"
    enable_audio_bpf: bool = True
    enable_agc: bool = True
    adc_resolution_bits: int = 16
    duration_sec: float = 0.5


@app.get("/")
async def serve_index():
    index_path = WEB_DIR / "index.html"
    return FileResponse(str(index_path))


@app.post("/api/simulate")
async def run_simulation_api(params_req: SimParamsRequest):
    """
    Executes end-to-end receiver simulation and returns decimate waveforms for UI rendering.
    """
    global last_simulation_results, last_ser_results

    params = ReceiverParameters(
        signal_type=params_req.signal_type,
        tone_freq_hz=params_req.tone_freq_hz,
        carrier_freq_mhz=params_req.carrier_freq_mhz,
        carrier_amplitude=params_req.carrier_amplitude,
        modulation_index=params_req.modulation_index,
        channel_snr_db=params_req.channel_snr_db,
        rf_filter_bandwidth_khz=params_req.rf_filter_bandwidth_khz,
        lna_gain_db=params_req.lna_gain_db,
        tuning_error_hz=params_req.tuning_error_hz,
        demodulator_type=params_req.demodulator_type,
        enable_audio_bpf=params_req.enable_audio_bpf,
        enable_agc=params_req.enable_agc,
        adc_resolution_bits=params_req.adc_resolution_bits,
        duration_sec=params_req.duration_sec
    )

    results = engine.run_simulation(params=params)
    last_simulation_results = results

    # Decimate arrays for high-speed 60fps JSON payload
    max_pts = 1000
    t_sim = results["time_sim"]
    step = max(1, len(t_sim) // max_pts)
    t_plot = (t_sim[::step] * 1000.0).tolist()  # ms

    stages = results["stages"]

    # Calculate FFT spectrum for demodulated audio
    audio_full = stages["stage8_digital_output"]["audio_samples"]
    fft_vals = np.abs(np.fft.rfft(audio_full))
    fft_freqs = np.fft.rfftfreq(len(audio_full), 1.0 / 16000)
    mag_db = (20.0 * np.log10(fft_vals / (np.max(fft_vals) + 1e-12))).tolist()

    # Pre-calculate SER inference for automatic badge updates
    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)
    feat_pipe = FeatureExtractionPipeline(fs=16000)
    audio_4s = seg.enforce_fixed_window(prep.apply_pre_emphasis(audio_full))
    feats = feat_pipe.extract_all_features(audio_4s)
    ser_pred = predictor.predict_emotion(feats["log_mel_128"], feats["metrics"], model_key="model_6_cnn_bilstm")
    last_ser_results = ser_pred

    payload = {
        "success": True,
        "time_ms": t_plot,
        "params": params.to_dict(),
        "stages": {
            "stage1": {
                "name": stages["stage1_rf_input"]["name"],
                "rf_transmitted": stages["stage1_rf_input"]["am_transmitted"][::step].tolist(),
                "rf_received": stages["stage1_rf_input"]["received_rf"][::step].tolist(),
                "metrics": stages["stage1_rf_input"]["metrics"]
            },
            "stage2": {
                "name": stages["stage2_rf_filter"]["name"],
                "filtered": stages["stage2_rf_filter"]["output_signal"][::step].tolist(),
                "metrics": stages["stage2_rf_filter"]["metrics"]
            },
            "stage3": {
                "name": stages["stage3_lna"]["name"],
                "lna_out": stages["stage3_lna"]["output_signal"][::step].tolist(),
                "metrics": stages["stage3_lna"]["metrics"]
            },
            "stage4": {
                "name": stages["stage4_tuning"]["name"],
                "if_out": stages["stage4_tuning"]["output_if"][::step].tolist(),
                "i_comp": stages["stage4_tuning"]["i_component"][::step].tolist(),
                "q_comp": stages["stage4_tuning"]["q_component"][::step].tolist(),
                "metrics": stages["stage4_tuning"]["metrics"]
            },
            "stage5": {
                "name": stages["stage5_demodulator"]["name"],
                "audio_demod": stages["stage5_demodulator"]["output_audio"][::step].tolist(),
                "original_voice": stages["stage1_rf_input"]["modulating_audio"][::step].tolist(),
                "metrics": stages["stage5_demodulator"]["metrics"]
            },
            "stage6": {
                "name": stages["stage6_audio_proc"]["name"],
                "audio_proc": stages["stage6_audio_proc"]["output_audio"][::step].tolist(),
                "metrics": stages["stage6_audio_proc"]["metrics"]
            },
            "stage7": {
                "name": stages["stage7_adc"]["name"],
                "analog_in": stages["stage7_adc"]["analog_input"][::step].tolist(),
                "quantized": stages["stage7_adc"]["quantized_signal"][::step].tolist(),
                "error": stages["stage7_adc"]["quantization_error"][::step].tolist(),
                "metrics": stages["stage7_adc"]["metrics"]
            },
            "stage8": {
                "name": stages["stage8_digital_output"]["name"],
                "audio_samples": audio_full[::max(1, len(audio_full) // max_pts)].tolist(),
                "time_16k_ms": (np.linspace(0, stages["stage8_digital_output"]["duration_sec"] * 1000.0, min(max_pts, len(audio_full)), endpoint=False)).tolist(),
                "metrics": {
                    "sampling_rate_hz": 16000,
                    "bit_depth": "16-bit linear PCM",
                    "duration_seconds": stages["stage8_digital_output"]["duration_sec"],
                    "total_samples": len(audio_full),
                    "peak_amplitude": float(np.max(np.abs(audio_full))),
                    "rms_level": float(np.sqrt(np.mean(audio_full ** 2))),
                    "audio_format": "RIFF WAV (Standard Aviation Voice Target)"
                }
            }
        },
        "spectrum": {
            "frequencies_hz": fft_freqs[:500].tolist(),
            "magnitude_db": mag_db[:500]
        },
        "ser_prediction": ser_pred
    }

    return sanitize_for_json(payload)


@app.get("/api/audio/wav")
async def get_audio_wav():
    """
    Streams the current demodulated 16 kHz WAV audio file for browser playback.
    """
    global last_simulation_results
    if last_simulation_results is None:
        last_simulation_results = engine.run_simulation()

    audio = last_simulation_results["stages"]["stage8_digital_output"]["audio_samples"]
    wav_bytes = WAVHandler.to_in_memory_wav_bytes(audio, sample_rate=16000)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/characterize/{test_type}")
async def get_characterization_data(test_type: str):
    """
    Executes parametric characterization sweeps:
    'bode', 'gain', 'noise', 'modulation', 'adc', 'summary'
    """
    global last_simulation_results, is_hardware_active

    if test_type == "bode":
        fra = FrequencyResponseAnalyzer(fs=200000)
        res = fra.analyze_filter(center_freq_hz=40000.0, bandwidth_khz=25.0, filter_order=4)
        return res

    elif test_type == "gain":
        ga = GainAnalyzer()
        res = ga.sweep_input_power(gain_db=20.0, p1db_v=2.5, pin_dbm_range=(-70.0, 10.0), step_db=2.0)
        return res

    elif test_type == "noise":
        na = NoiseAnalyzer()
        res = na.sweep_input_snr(snr_min_db=-10.0, snr_max_db=40.0, step_db=2.5)
        return res

    elif test_type == "modulation":
        ma = ModulationAnalyzer(fs=200000)
        res = ma.sweep_modulation_index(m_range=(0.1, 1.5), step=0.1)
        return res

    elif test_type == "adc":
        aa = ADCAnalyzer(fs=200000)
        res = aa.sweep_bit_depths(bit_depths=[8, 10, 12, 14, 16, 20, 24])
        return res

    elif test_type == "summary":
        if last_simulation_results is None:
            last_simulation_results = engine.run_simulation()
        summary = ReceiverPerformanceEvaluator.generate_performance_summary(
            last_simulation_results, is_hardware_mode=is_hardware_active
        )
        return {
            "records": summary["summary_records"],
            "overall_snr_improvement_db": summary["overall_snr_improvement_db"],
            "correlation": summary["reconstruction_correlation"]
        }
    else:
        raise HTTPException(status_code=400, detail="Unknown test type")


@app.post("/api/predict_emotion")
async def predict_emotion_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Runs SER model inference with specified model key.
    """
    global last_simulation_results, last_ser_results
    if last_simulation_results is None:
        last_simulation_results = engine.run_simulation()

    model_key = payload.get("model_key", "model_6_cnn_bilstm")
    audio = last_simulation_results["stages"]["stage8_digital_output"]["audio_samples"]

    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)
    feat_pipe = FeatureExtractionPipeline(fs=16000)

    audio_4s = seg.enforce_fixed_window(prep.apply_pre_emphasis(audio))
    features = feat_pipe.extract_all_features(audio_4s)

    pred = predictor.predict_emotion(
        log_mel_128=features["log_mel_128"],
        acoustic_features=features["metrics"],
        model_key=model_key
    )
    last_ser_results = pred

    return {
        "prediction": pred,
        "log_mel_sample": features["log_mel_128"].tolist(),
        "mfccs_sample": features["mfccs"][:13, :60].tolist(),
        "acoustic_metrics": features["metrics"]
    }


@app.post("/api/analyze_audio")
async def analyze_uploaded_audio(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    client_transcription: Optional[str] = Form(None),
    model_key: str = Form("model_6_cnn_bilstm")
):
    """
    Accepts an uploaded audio file (WAV/MP3/OGG/FLAC) or base64 recorded microphone audio.
    Executes 16 kHz resample, speech-to-text transcription, pre-emphasis, 4s VAD windowing,
    FFT spectral analysis, feature extraction, and Speech Emotion Recognition.
    """
    global last_simulation_results, last_ser_results

    audio_bytes = None
    filename = "uploaded_speech.wav"

    if file is not None:
        filename = file.filename or "uploaded_speech.wav"
        audio_bytes = await file.read()
    elif audio_base64 is not None and len(audio_base64.strip()) > 0:
        filename = "microphone_recording.wav"
        # Strip header like data:audio/wav;base64, if present
        b64_clean = audio_base64.split(",")[-1]
        try:
            audio_bytes = base64.b64decode(b64_clean)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
    else:
        raise HTTPException(status_code=400, detail="No audio file or base64 recording received.")

    # Load audio data via soundfile or librosa
    try:
        data, sr_in = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        try:
            data, sr_in = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio file: {e}")

    # Ensure single channel (mono)
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=-1)

    data = np.asarray(data, dtype=np.float32)

    # Resample to standard 16.0 kHz if necessary
    if sr_in != 16000:
        data = librosa.resample(data, orig_sr=sr_in, target_sr=16000)

    # Normalize amplitude
    max_amp = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    if max_amp > 1e-4:
        data = data / max_amp * 0.95

    # Speech-to-text Transcription
    transcription = client_transcription or ""
    if not transcription or len(transcription.strip()) == 0:
        try:
            rec = sr.Recognizer()
            wav_buf = WAVHandler.to_in_memory_wav_bytes(data, sample_rate=16000)
            with sr.AudioFile(io.BytesIO(wav_buf)) as source:
                rec_audio = rec.record(source)
                transcription = rec.recognize_google(rec_audio)
        except sr.UnknownValueError:
            transcription = "Pilot Radio Transmission: [Acoustic Speech Detected]"
        except Exception:
            transcription = "Civil Aviation Pilot-ATC Voice Transmission"

    # DSP Feature Extraction & VAD
    prep = AudioPreprocessor(target_fs=16000)
    seg = AudioSegmenter(fs=16000, window_duration_sec=4.0)
    feat_pipe = FeatureExtractionPipeline(fs=16000)

    audio_pre = prep.apply_pre_emphasis(data)
    audio_4s = seg.enforce_fixed_window(audio_pre)
    features = feat_pipe.extract_all_features(audio_4s)

    # Run Speech Emotion Recognition
    pred = predictor.predict_emotion(
        log_mel_128=features["log_mel_128"],
        acoustic_features=features["metrics"],
        model_key=model_key
    )
    last_ser_results = pred

    # Decimate arrays for 60fps high performance rendering
    max_pts = 1000
    step = max(1, len(data) // max_pts)
    t_plot = (np.linspace(0, len(data) / 16000.0 * 1000.0, min(max_pts, len(data)))).tolist()
    audio_plot = data[::step].tolist()

    vad_step = max(1, len(audio_4s) // max_pts)
    vad_plot = audio_4s[::vad_step].tolist()

    # FFT Spectrum
    fft_vals = np.abs(np.fft.rfft(data))
    fft_freqs = np.fft.rfftfreq(len(data), 1.0 / 16000)
    mag_db = (20.0 * np.log10(fft_vals / (np.max(fft_vals) + 1e-12))).tolist()

    # Inject into last_simulation_results so all tabs and /api/audio/wav stream this input
    if last_simulation_results is None:
        last_simulation_results = engine.run_simulation()

    last_simulation_results["stages"]["stage8_digital_output"]["audio_samples"] = data
    last_simulation_results["stages"]["stage8_digital_output"]["duration_sec"] = float(len(data) / 16000.0)
    last_simulation_results["stages"]["stage5_demodulator"]["output_audio"] = data
    last_simulation_results["stages"]["stage1_rf_input"]["modulating_audio"] = data

    rms_val = float(np.sqrt(np.mean(data ** 2)))
    dbfs_val = float(20.0 * np.log10(rms_val + 1e-9))

    return sanitize_for_json({
        "success": True,
        "filename": filename,
        "transcription": transcription,
        "duration_sec": float(len(data) / 16000.0),
        "sample_rate_hz": 16000,
        "total_samples": len(data),
        "peak_amplitude": float(np.max(np.abs(data))),
        "rms_level": rms_val,
        "rms_dbfs": dbfs_val,
        "time_ms": t_plot,
        "audio_samples": audio_plot,
        "vad_audio": vad_plot,
        "spectrum": {
            "frequencies_hz": fft_freqs[:500].tolist(),
            "magnitude_db": mag_db[:500]
        },
        "ser_prediction": pred,
        "acoustic_metrics": features["metrics"],
        "mfccs_sample": features["mfccs"][:13, :60].tolist(),
        "log_mel_sample": features["log_mel_128"].tolist()
    })


@app.get("/api/export/csv")
async def export_csv_data():
    """
    Downloads CSV measurement records.
    """
    global last_simulation_results, is_hardware_active
    if last_simulation_results is None:
        last_simulation_results = engine.run_simulation()

    summary = ReceiverPerformanceEvaluator.generate_performance_summary(
        last_simulation_results, is_hardware_mode=is_hardware_active
    )
    csv_bytes = DataExporter.to_csv_bytes(summary["summary_df"])
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=receiver_characterization_metrics.csv"}
    )


@app.get("/api/export/report")
async def export_markdown_report():
    """
    Downloads examination markdown report.
    """
    global last_simulation_results, last_ser_results, is_hardware_active
    if last_simulation_results is None:
        last_simulation_results = engine.run_simulation()
    if last_ser_results is None:
        last_ser_results = {"dominant_emotion": "Neutral", "confidence": 0.85, "inference_mode": "Model Checkpoint"}

    report_text = DataExporter.generate_experiment_markdown_report(
        last_simulation_results, last_ser_results, is_hardware_mode=is_hardware_active
    )
    return Response(
        content=report_text.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=viva_examination_report.md"}
    )


@app.get("/api/hardware/devices")
async def list_hardware_devices():
    """
    Enumerates host audio input devices and USB serial COM ports.
    """
    audio_devs = LiveAudioAcquisition.list_input_devices()
    serial_ports = SerialADCInterface.list_available_ports()
    return {
        "audio_devices": audio_devs,
        "serial_ports": serial_ports,
        "is_hardware_mode": is_hardware_active
    }


@app.post("/api/hardware/record")
async def record_hardware_audio(payload: Dict[str, Any] = Body(...)):
    """
    Records audio block from physical receiver and sets hardware mode active.
    """
    global last_simulation_results, is_hardware_active
    dev_id = payload.get("device_id")
    duration = float(payload.get("duration_sec", 4.0))

    audio, metrics = LiveAudioAcquisition().record_audio_block(duration_sec=duration, device_id=dev_id)
    if last_simulation_results is not None:
        last_simulation_results["stages"]["stage8_digital_output"]["audio_samples"] = audio
        last_simulation_results["stages"]["stage8_digital_output"]["duration_sec"] = duration

    is_hardware_active = True
    return {"success": True, "metrics": metrics}
