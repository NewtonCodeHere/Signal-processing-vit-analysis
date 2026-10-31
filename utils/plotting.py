"""
Plotly Visualization Utilities Module.
Provides interactive, dark-themed, publication-ready Plotly charts
for all receiver waveforms, spectra, Bode responses, and emotion distributions.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
from config.settings import PLOT_THEME


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """
    Applies consistent dark-themed styling to any Plotly figure.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": PLOT_THEME["text_color"]}
        },
        paper_bgcolor=PLOT_THEME["paper_color"],
        plot_bgcolor=PLOT_THEME["bg_color"],
        font={"color": PLOT_THEME["text_color"], "family": "Inter, Roboto, sans-serif"},
        margin={"l": 50, "r": 30, "t": 45, "b": 45},
        height=height,
        xaxis={
            "gridcolor": PLOT_THEME["grid_color"],
            "zerolinecolor": PLOT_THEME["grid_color"],
            "tickfont": {"color": PLOT_THEME["text_color"]}
        },
        yaxis={
            "gridcolor": PLOT_THEME["grid_color"],
            "zerolinecolor": PLOT_THEME["grid_color"],
            "tickfont": {"color": PLOT_THEME["text_color"]}
        },
        legend={
            "bgcolor": "rgba(22, 27, 34, 0.7)",
            "bordercolor": PLOT_THEME["grid_color"],
            "font": {"color": PLOT_THEME["text_color"]}
        }
    )
    return fig


def plot_time_waveform(
    time_arr: np.ndarray,
    signal_arr: np.ndarray,
    title: str = "Time-Domain Waveform",
    y_label: str = "Amplitude (V)",
    max_points: int = 2500,
    line_color: str = PLOT_THEME["primary_cyan"]
) -> go.Figure:
    """
    Renders an oscilloscope time-domain waveform.
    """
    step = max(1, len(time_arr) // max_points)
    t_plot = time_arr[::step] * 1000.0  # Convert to milliseconds
    s_plot = signal_arr[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=s_plot,
        mode="lines",
        line={"color": line_color, "width": 1.5},
        name="Signal"
    ))
    fig.update_xaxes(title_text="Time (ms)")
    fig.update_yaxes(title_text=y_label)
    return apply_dark_theme(fig, title=title)


def plot_dual_waveform_comparison(
    time_arr: np.ndarray,
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    name_a: str = "Signal A",
    name_b: str = "Signal B",
    title: str = "Signal Comparison",
    max_points: int = 2500
) -> go.Figure:
    """
    Overlays two time-domain signals for comparative analysis.
    """
    step = max(1, len(time_arr) // max_points)
    t_plot = time_arr[::step] * 1000.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=signal_a[::step],
        mode="lines",
        line={"color": PLOT_THEME["primary_cyan"], "width": 1.5},
        name=name_a
    ))
    fig.add_trace(go.Scatter(
        x=t_plot,
        y=signal_b[::step],
        mode="lines",
        line={"color": PLOT_THEME["accent_orange"], "width": 1.5, "dash": "dot"},
        name=name_b
    ))
    fig.update_xaxes(title_text="Time (ms)")
    fig.update_yaxes(title_text="Normalized Amplitude")
    return apply_dark_theme(fig, title=title)


def plot_frequency_spectrum(
    signal_arr: np.ndarray,
    fs: int = 200000,
    title: str = "Frequency Spectrum (FFT)",
    max_freq_hz: Optional[float] = None
) -> go.Figure:
    """
    Renders FFT power spectrum in decibels.
    """
    n = len(signal_arr)
    fft_vals = np.abs(np.fft.rfft(signal_arr))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    # Normalize to dBFS
    max_fft = np.max(fft_vals) + 1e-12
    mag_db = 20.0 * np.log10(fft_vals / max_fft)

    if max_freq_hz:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        mag_db = mag_db[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs / 1000.0,
        y=mag_db,
        mode="lines",
        line={"color": PLOT_THEME["success_green"], "width": 1.2},
        name="Magnitude"
    ))
    fig.update_xaxes(title_text="Frequency (kHz)")
    fig.update_yaxes(title_text="Magnitude (dBFS)", range=[-80, 5])
    return apply_dark_theme(fig, title=title)


def plot_bode_response(
    freqs_hz: List[float],
    magnitude_db: List[float],
    phase_deg: List[float],
    title: str = "RF Filter Frequency Response (Bode Plot)"
) -> go.Figure:
    """
    Renders 2-panel Bode Plot with Magnitude and Phase.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("<b>Magnitude Response (dB)</b>", "<b>Phase Response (degrees)</b>")
    )

    f_khz = np.array(freqs_hz) / 1000.0

    fig.add_trace(go.Scatter(
        x=f_khz,
        y=magnitude_db,
        mode="lines",
        line={"color": PLOT_THEME["primary_cyan"], "width": 2.0},
        name="Magnitude (dB)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=f_khz,
        y=phase_deg,
        mode="lines",
        line={"color": PLOT_THEME["accent_orange"], "width": 1.5},
        name="Phase (°)"
    ), row=2, col=1)

    fig.update_xaxes(title_text="Frequency (kHz)", row=2, col=1)
    fig.update_yaxes(title_text="Gain (dB)", range=[-60, 5], row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", row=2, col=1)

    return apply_dark_theme(fig, title=title, height=480)


def plot_compression_curve(
    pin_dbm: List[float],
    pout_dbm: List[float],
    small_signal_gain_db: float,
    p1db_in: Optional[float] = None
) -> go.Figure:
    """
    Plots Pin vs Pout with ideal linear gain reference and P1dB marker.
    """
    fig = go.Figure()

    # Ideal linear line: Pout = Pin + Gain
    linear_pout = np.array(pin_dbm) + small_signal_gain_db
    fig.add_trace(go.Scatter(
        x=pin_dbm,
        y=linear_pout,
        mode="lines",
        line={"color": PLOT_THEME["grid_color"], "dash": "dash", "width": 1.5},
        name="Ideal Linear Gain"
    ))

    # Actual amplifier curve
    fig.add_trace(go.Scatter(
        x=pin_dbm,
        y=pout_dbm,
        mode="lines+markers",
        line={"color": PLOT_THEME["primary_cyan"], "width": 2.0},
        marker={"size": 4},
        name="Measured Pout"
    ))

    if p1db_in is not None:
        fig.add_vline(
            x=p1db_in,
            line_dash="dot",
            line_color=PLOT_THEME["danger_red"],
            annotation_text=f"P1dB In: {p1db_in:.1f} dBm",
            annotation_position="top left"
        )

    fig.update_xaxes(title_text="Input Power Pin (dBm)")
    fig.update_yaxes(title_text="Output Power Pout (dBm)")
    return apply_dark_theme(fig, title="Amplifier Linearity & 1 dB Compression (P1dB)")


def plot_spectrogram_heatmap(
    spec_matrix: np.ndarray,
    title: str = "Log-Mel Spectrogram (128x128)",
    y_label: str = "Mel Frequency Bin"
) -> go.Figure:
    """
    Renders Log-Mel Spectrogram matrix as a colored heatmap.
    """
    fig = go.Figure(data=go.Heatmap(
        z=spec_matrix,
        colorscale="Viridis",
        colorbar={"title": "Norm Energy"}
    ))
    fig.update_xaxes(title_text="Time Frame (Patch)")
    fig.update_yaxes(title_text=y_label)
    return apply_dark_theme(fig, title=title)


def plot_emotion_probabilities(
    prob_dict: Dict[str, float],
    dominant_emotion: str
) -> go.Figure:
    """
    Renders horizontal bar chart of predicted emotion probabilities.
    """
    labels = list(prob_dict.keys())
    scores = [prob_dict[k] * 100.0 for k in labels]

    colors = [
        PLOT_THEME["danger_red"] if k == dominant_emotion and k in ["Angry", "Fearful"]
        else PLOT_THEME["success_green"] if k == dominant_emotion
        else PLOT_THEME["primary_cyan"] if k == dominant_emotion
        else "#4A5568"
        for k in labels
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="auto"
    ))
    fig.update_xaxes(title_text="Confidence Probability (%)", range=[0, 100])
    fig.update_yaxes(title_text="Emotion Class", autorange="reversed")
    return apply_dark_theme(fig, title=f"Predicted Emotion Distribution: {dominant_emotion}")
