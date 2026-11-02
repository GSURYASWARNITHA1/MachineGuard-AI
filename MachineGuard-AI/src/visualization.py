"""Plotly visualization components with dark industrial theme for MachineGuard AI."""
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import COLORS, DECISION_THRESHOLD, MEDIUM_RISK_THRESHOLD

# High-contrast color tokens
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#CBD5E1"
TEXT_MUTED = "#94A3B8"
ACCENT_CYAN = "#00D9FF"
CARD_BG = "#161B22"
BORDER_COLOR = "#30363D"
GRID_COLOR = "#21262D"


def make_dark_layout(**overrides) -> dict:
    """Safely combine dark layout defaults with caller overrides without duplicate keyword arguments."""
    layout = {
        "paper_bgcolor": CARD_BG,
        "plot_bgcolor": CARD_BG,
        "font": dict(color=TEXT_PRIMARY, family="Segoe UI, sans-serif"),
        "margin": dict(l=40, r=40, t=55, b=40),
        "hoverlabel": dict(bgcolor=CARD_BG, font_color=TEXT_PRIMARY, font_size=12, bordercolor=ACCENT_CYAN),
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in layout and isinstance(layout[key], dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout


def plot_risk_gauge(probability: float, threshold: float = DECISION_THRESHOLD) -> go.Figure:
    """Create a sleek industrial speedometer gauge for machine failure risk."""
    pct = probability * 100
    thresh_pct = threshold * 100
    med_pct = MEDIUM_RISK_THRESHOLD * 100

    # Determine bar color
    if probability >= threshold:
        bar_color = "#FF5C5C"  # High risk crimson
    elif probability >= MEDIUM_RISK_THRESHOLD:
        bar_color = "#FFB020"  # Medium risk amber
    else:
        bar_color = "#00D084"  # Low risk emerald

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 46, "color": bar_color, "family": "Segoe UI, sans-serif"}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Machine Failure Risk Score", "font": {"size": 18, "color": ACCENT_CYAN, "family": "Segoe UI, sans-serif"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": TEXT_MUTED,
                "tickvals": [0, 20, 40, 60, 80, 100],
                "tickfont": {"size": 12, "color": TEXT_SECONDARY}
            },
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": GRID_COLOR,
            "borderwidth": 1,
            "bordercolor": BORDER_COLOR,
            "steps": [
                {"range": [0, med_pct], "color": "rgba(0, 208, 132, 0.22)"},
                {"range": [med_pct, thresh_pct], "color": "rgba(255, 176, 32, 0.22)"},
                {"range": [thresh_pct, 100], "color": "rgba(255, 92, 92, 0.25)"}
            ],
            "threshold": {
                "line": {"color": "#FF0055", "width": 4},
                "thickness": 0.82,
                "value": thresh_pct
            }
        }
    ))

    fig.update_layout(**make_dark_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=25, r=25, t=50, b=20)
    ))
    return fig


def plot_confusion_matrix(cm_data: Dict[str, Any]) -> go.Figure:
    """Generate an annotated confusion matrix with industrial cost implications."""
    tp, fp, fn, tn = cm_data["tp"], cm_data["fp"], cm_data["fn"], cm_data["tn"]
    
    matrix = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>{tn:,}</b><br><span style='font-size:11px;color:#94a3b8'>True Negative<br>Healthy Machine (Penalty: 0)</span>",
         f"<b>{fp:,}</b><br><span style='font-size:11px;color:#fcd34d'>False Positive<br>False Alarm (Penalty: 1 unit)</span>"],
        [f"<b>{fn:,}</b><br><span style='font-size:11px;color:#fca5a5'>False Negative<br>Missed Failure (Penalty: 10 units)</span>",
         f"<b>{tp:,}</b><br><span style='font-size:11px;color:#6ee7b7'>True Positive<br>Failure Caught</span>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Predicted Healthy", "Predicted Failure"],
        y=["Actual Healthy", "Actual Failure"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#FFFFFF", "family": "Segoe UI, sans-serif"},
        colorscale=[
            [0.0, "#111827"],
            [0.08, "#1f2937"],
            [0.25, "#0369a1"],
            [1.0, "#0284c7"]
        ],
        showscale=False
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text=f"Test Set Confusion Matrix (Threshold = {cm_data['threshold']:.2f})", font=dict(color=ACCENT_CYAN, size=16)),
        xaxis=dict(tickfont=dict(size=12, color=TEXT_PRIMARY), side="bottom"),
        yaxis=dict(tickfont=dict(size=12, color=TEXT_PRIMARY), autorange="reversed"),
        height=340
    ))
    return fig


def plot_roc_curve(roc_data: Dict[str, Any], current_point: Dict[str, Any]) -> go.Figure:
    """Plot interactive ROC curve with optimal threshold marker."""
    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#4b5563", dash="dash", width=1.5),
        name="Random Chance (AUC = 0.50)"
    ))

    # ROC curve
    fig.add_trace(go.Scatter(
        x=roc_data["fpr"],
        y=roc_data["tpr"],
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=3),
        name=f"Calibrated Model (AUC = {roc_data['auc']:.4f})"
    ))

    # Operating point
    fig.add_trace(go.Scatter(
        x=[current_point["fpr"]],
        y=[current_point["recall"]],
        mode="markers",
        marker=dict(size=13, color="#FF5C5C", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
        name=f"Operating Point (Thr = {current_point['threshold']:.2f})"
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text="Receiver Operating Characteristic (ROC) Curve", font=dict(color=ACCENT_CYAN, size=16)),
        xaxis=dict(
            title=dict(text="False Positive Rate (FPR)", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            range=[-0.02, 1.02]
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate / Recall (TPR)", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            range=[-0.02, 1.02]
        ),
        legend=dict(x=0.45, y=0.15, bgcolor="rgba(22,27,34,0.85)", font=dict(color=TEXT_PRIMARY, size=11)),
        height=380
    ))
    return fig


def plot_pr_curve(pr_data: Dict[str, Any], current_point: Dict[str, Any]) -> go.Figure:
    """Plot Precision-Recall curve with imbalance baseline."""
    fig = go.Figure()

    # Baseline (prevalence)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[pr_data["baseline"], pr_data["baseline"]],
        mode="lines",
        line=dict(color="#64748b", dash="dash", width=1.5),
        name=f"Baseline Failure Prevalence ({pr_data['baseline']*100:.1f}%)"
    ))

    # PR curve
    fig.add_trace(go.Scatter(
        x=pr_data["recall"],
        y=pr_data["precision"],
        mode="lines",
        line=dict(color="#38bdf8", width=3),
        name=f"Calibrated Model (PR-AUC = {pr_data['auc']:.4f})"
    ))

    # Operating point
    fig.add_trace(go.Scatter(
        x=[current_point["recall"]],
        y=[current_point["precision"]],
        mode="markers",
        marker=dict(size=13, color="#FF5C5C", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
        name=f"Operating Point (Thr = {current_point['threshold']:.2f})"
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text="Precision-Recall Curve (Primary Imbalance Metric)", font=dict(color=ACCENT_CYAN, size=16)),
        xaxis=dict(
            title=dict(text="Recall (Failure Detection Rate)", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            range=[-0.02, 1.02]
        ),
        yaxis=dict(
            title=dict(text="Precision (Positive Predictive Value)", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            range=[-0.02, 1.05]
        ),
        legend=dict(x=0.05, y=0.15, bgcolor="rgba(22,27,34,0.85)", font=dict(color=TEXT_PRIMARY, size=11)),
        height=380
    ))
    return fig


def plot_feature_importance(importance_list: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart for permutation feature importance."""
    df = pd.DataFrame(importance_list).sort_values("drop", ascending=True)

    name_clean = {
        "torque": "Torque [Nm]",
        "rpm": "Rotational Speed [rpm]",
        "air_temp": "Air Temperature [K]",
        "tool_wear": "Tool Wear [min]",
        "proc_temp": "Process Temperature [K]",
        "type_L": "Product Type: L (Low)",
        "type_H": "Product Type: H (High)",
        "type_M": "Product Type: M (Medium)"
    }
    df["display_name"] = df["feature"].map(lambda x: name_clean.get(x, x))

    fig = go.Figure(go.Bar(
        x=df["drop"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["drop"],
            colorscale=[[0, "#0369a1"], [0.5, "#0284c7"], [1.0, ACCENT_CYAN]],
            line=dict(color=BORDER_COLOR, width=1)
        ),
        error_x=dict(type="data", array=df["std"], color="#94A3B8")
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text="Permutation Feature Importance (PR-AUC Drop on Test)", font=dict(color=ACCENT_CYAN, size=16)),
        xaxis=dict(
            title=dict(text="Mean PR-AUC Drop When Shuffled", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11)
        ),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_PRIMARY, size=12)),
        height=380
    ))
    return fig


def plot_model_comparison(cv_rows: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing candidate models on raw vs engineered features."""
    df = pd.DataFrame(cv_rows)
    df["Label"] = df["model"] + " (" + df["features"] + ")"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Label"],
        y=df["pr_auc"],
        name="PR-AUC (Primary Metric)",
        marker=dict(color=ACCENT_CYAN),
        error_y=dict(type="data", array=df["pr_auc_std"], color="#94A3B8")
    ))
    fig.add_trace(go.Bar(
        x=df["Label"],
        y=df["roc_auc"],
        name="ROC-AUC",
        marker=dict(color="#818CF8"),
        error_y=dict(type="data", array=df["roc_auc_std"], color="#94A3B8")
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text="Model Family & Feature Engineering Comparison (5-Fold CV on Train)", font=dict(color=ACCENT_CYAN, size=16)),
        barmode="group",
        xaxis=dict(tickangle=-20, gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_PRIMARY, size=11)),
        yaxis=dict(
            title=dict(text="Score", font=dict(color=TEXT_PRIMARY, size=13)),
            range=[0.3, 1.02],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11)
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(22,27,34,0.85)", font=dict(color=TEXT_PRIMARY, size=11)),
        height=380
    ))
    return fig


def plot_noise_robustness(noise_data: List[Dict[str, Any]]) -> go.Figure:
    """Line chart showing model resilience against sensor telemetry noise."""
    df = pd.DataFrame(noise_data)
    df["Noise_Pct"] = df["noise_frac_of_std"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Noise_Pct"],
        y=df["pr_auc"],
        mode="lines+markers",
        line=dict(color=ACCENT_CYAN, width=3),
        marker=dict(size=8),
        name="PR-AUC"
    ))
    fig.add_trace(go.Scatter(
        x=df["Noise_Pct"],
        y=df["recall"],
        mode="lines+markers",
        line=dict(color="#FFB020", width=3),
        marker=dict(size=8),
        name="Recall @ Chosen Threshold"
    ))

    fig.update_layout(**make_dark_layout(
        title=dict(text="Sensor Noise Robustness (Gaussian Perturbation on Test Telemetry)", font=dict(color=ACCENT_CYAN, size=16)),
        xaxis=dict(
            title=dict(text="Added Sensor Noise (% of Feature Std)", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11)
        ),
        yaxis=dict(
            title=dict(text="Performance Score", font=dict(color=TEXT_PRIMARY, size=13)),
            range=[0.4, 1.0],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_SECONDARY, size=11)
        ),
        legend=dict(x=0.6, y=0.95, bgcolor="rgba(22,27,34,0.85)", font=dict(color=TEXT_PRIMARY, size=11)),
        height=340
    ))
    return fig


def plot_batch_risk_donut(summary: Dict[str, Any]) -> go.Figure:
    """Donut chart for fleet inspection breakdown with safe layout overrides."""
    labels = ["Low Risk (Healthy)", "Medium Risk (Advisory)", "High Risk (Critical)"]
    values = [summary["low_risk_count"], summary["medium_risk_count"], summary["high_risk_count"]]
    colors = ["#00D084", "#FFB020", "#FF5C5C"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BORDER_COLOR, width=1.5)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#FFFFFF", family="Segoe UI, sans-serif")
    )])

    # Use make_dark_layout to cleanly merge without duplicate margin keyword arguments
    fig.update_layout(**make_dark_layout(
        title=dict(text="Fleet Risk Distribution Overview", font=dict(color=ACCENT_CYAN, size=16)),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    ))
    return fig


def plot_sensor_radar(
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float
) -> go.Figure:
    """Radar chart comparing input sensor levels against normal operating envelope."""
    # Normalize sensors to 0 - 100% scale based on expected operating range
    def norm(val, lo, hi):
        return max(0, min(100, ((val - lo) / (hi - lo)) * 100))

    categories = [
        "Air Temp",
        "Process Temp",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ]

    # Current machine readings
    machine_vals = [
        norm(air_temp, 295, 305),
        norm(proc_temp, 305, 315),
        norm(rpm, 1100, 2900),
        norm(torque, 5, 80),
        norm(tool_wear, 0, 250)
    ]

    # Safe operating zone benchmark (approx 35% to 65% of max envelope)
    safe_low = [30, 30, 20, 20, 0]
    safe_high = [70, 70, 60, 60, 75]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=safe_high,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 208, 132, 0.16)',
        line=dict(color="rgba(0, 208, 132, 0.5)", width=1.5),
        name="Safe Operating Envelope"
    ))

    fig.add_trace(go.Scatterpolar(
        r=machine_vals,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 217, 255, 0.32)',
        line=dict(color=ACCENT_CYAN, width=2.5),
        name="Current Machine Telemetry"
    ))

    fig.update_layout(**make_dark_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_MUTED, size=9)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_PRIMARY, size=11, family="Segoe UI, sans-serif")),
            bgcolor=CARD_BG
        ),
        title=dict(text="Sensor Telemetry Profile vs Safe Envelope", font=dict(color=ACCENT_CYAN, size=16)),
        showlegend=True,
        legend=dict(x=0.0, y=-0.2, orientation="h", font=dict(color=TEXT_PRIMARY, size=11)),
        height=340
    ))
    return fig
