"""
Plotly interactive chart components for the condition-monitoring dashboard.
Implements industrial dark theme aesthetics (graphite, cyan, amber, crimson).
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go


DARK_THEME_LAYOUT = dict(
    paper_bgcolor="#12151c",
    plot_bgcolor="#181d28",
    font=dict(color="#d1d5db", family="Inter, -apple-system, sans-serif"),
    xaxis=dict(gridcolor="#242b3d", zerolinecolor="#374151"),
    yaxis=dict(gridcolor="#242b3d", zerolinecolor="#374151"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def create_waveform_chart(
    signal: np.ndarray,
    sampling_rate_hz: float = 5000.0,
    title: str = "Vibration Acceleration Waveform",
    peak_threshold: Optional[float] = None,
) -> go.Figure:
    """Creates an interactive time-series plot of raw or preprocessed vibration samples."""
    signal = np.asarray(signal)
    dt_ms = (1.0 / sampling_rate_hz) * 1000.0
    time_ms = np.arange(len(signal)) * dt_ms

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=signal,
            mode="lines",
            name="Acceleration (g)",
            line=dict(color="#00d4ff", width=1.5),
            hovertemplate="Time: %{x:.2f} ms<br>Amplitude: %{y:.3f} g<extra></extra>",
        )
    )

    if peak_threshold is not None:
        fig.add_hline(
            y=peak_threshold,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Threshold (+{peak_threshold:.1f}g)",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-peak_threshold,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Threshold (-{peak_threshold:.1f}g)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f3f4f6")),
        xaxis_title="Time (ms)",
        yaxis_title="Acceleration (g)",
        **DARK_THEME_LAYOUT,
    )
    return fig


def create_fft_spectrum_chart(
    signal: np.ndarray,
    sampling_rate_hz: float = 5000.0,
    title: str = "Fast Fourier Transform (FFT) Power Spectrum",
    max_freq_hz: Optional[float] = None,
) -> go.Figure:
    """Creates a frequency spectrum plot showing dominant meshing frequencies and sidebands."""
    signal = np.asarray(signal)
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) / n
    fft_vals[1:] *= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)

    if max_freq_hz:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        fft_vals = fft_vals[mask]

    peak_idx = int(np.argmax(fft_vals))
    dom_freq = freqs[peak_idx]
    peak_mag = fft_vals[peak_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fft_vals,
            mode="lines",
            name="FFT Magnitude",
            line=dict(color="#38bdf8", width=1.2),
            hovertemplate="Frequency: %{x:.1f} Hz<br>Magnitude: %{y:.4f}<extra></extra>",
        )
    )

    # Highlight dominant frequency peak
    fig.add_annotation(
        x=dom_freq,
        y=peak_mag,
        text=f"Peak: {dom_freq:.1f} Hz ({peak_mag:.3f})",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#f59e0b",
        font=dict(color="#f59e0b", size=11),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f3f4f6")),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Spectral Magnitude",
        **DARK_THEME_LAYOUT,
    )
    return fig


def create_confusion_matrix_chart(
    cm: List[List[int]],
    classes: List[str],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Creates an annotated confusion matrix heatmap."""
    z = np.array(cm)
    text = [[f"{val}" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#181d28"], [0.5, "#0284c7"], [1.0, "#00d4ff"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f3f4f6")),
        xaxis_title="Predicted Condition",
        yaxis_title="True Ground Truth",
        **DARK_THEME_LAYOUT,
    )
    return fig


def create_model_comparison_bar(
    model_names: List[str],
    f1_scores: List[float],
    latencies_ms: List[float],
) -> go.Figure:
    """Creates a dual-axis comparison of Macro-F1 score vs Inference Latency."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=f1_scores,
            name="Macro-F1 Score",
            marker_color="#00d4ff",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=latencies_ms,
            name="Latency (ms)",
            marker_color="#f59e0b",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=dict(text="Model Performance & Inference Latency Comparison", font=dict(size=14, color="#f3f4f6")),
        barmode="group",
        yaxis=dict(title="Macro-F1 Score", range=[0.0, 1.05], gridcolor="#242b3d"),
        yaxis2=dict(
            title="Latency (ms)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        **DARK_THEME_LAYOUT,
    )
    return fig
