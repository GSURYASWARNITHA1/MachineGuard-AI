"""MachineGuard AI — Industrial Predictive Maintenance System.
Neural Storm 2026 Hackathon Edition.
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from config import (
    ROOT_DIR,
    DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    DECISION_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    COST_FN,
    COST_FP,
    SENSOR_SPECS,
    PRODUCT_TYPES,
    COLORS
)
from src.prediction import (
    ModelManager,
    predict_single,
    predict_batch,
    get_risk_category,
    PRESETS,
    NOMINAL_BASELINES
)
from src.preprocessing import kelvin_to_celsius, celsius_to_kelvin
from src.evaluation import EvaluationManager
from src.visualization import (
    plot_risk_gauge,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance,
    plot_model_comparison,
    plot_noise_robustness,
    plot_batch_risk_donut,
    plot_sensor_radar
)

# Page configuration
st.set_page_config(
    page_title="MachineGuard AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Dark CSS — Global Readability & Contrast
CUSTOM_CSS = """
<style>
    /* =========================================================================
       1. Global Background, Typography & Contrast
       ========================================================================= */
    .stApp {
        background-color: #0E1117;
        color: #F1F5F9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Headings */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #00D9FF !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    h4, .stMarkdown h4 {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }
    h5, h6, .stMarkdown h5, .stMarkdown h6 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }

    /* Body text, Paragraphs, Lists */
    p, .stMarkdown p, li, .stMarkdown li {
        color: #CBD5E1 !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }
    strong, b, .stMarkdown strong, .stMarkdown b {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    code {
        color: #00D9FF !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        padding: 2px 7px !important;
        border-radius: 4px !important;
        font-size: 0.88rem !important;
        font-family: Consolas, monospace !important;
    }

    /* =========================================================================
       2. Top Header & Toolbar Styling (Eliminate white bar)
       ========================================================================= */
    header[data-testid="stHeader"] {
        background-color: #0E1117 !important;
        background: #0E1117 !important;
        color: #F1F5F9 !important;
        border-bottom: 1px solid #1F2937 !important;
    }
    header[data-testid="stHeader"] * {
        color: #CBD5E1 !important;
    }
    div[data-testid="stToolbar"] {
        color: #CBD5E1 !important;
    }
    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, #00D9FF, #0284C7) !important;
        height: 3px !important;
    }

    /* =========================================================================
       3. High-Contrast Sidebar Styling
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li {
        color: #CBD5E1 !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #94A3B8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1F2937 !important;
    }

    /* Sidebar Navigation Tiles */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #27354F !important;
        border-color: #00D9FF !important;
        transform: translateX(4px) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(0, 217, 255, 0.22) 0%, rgba(30, 41, 59, 0.85) 100%) !important;
        border-color: #00D9FF !important;
        border-left: 4px solid #00D9FF !important;
        box-shadow: 0 0 14px rgba(0, 217, 255, 0.2) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label div {
        color: #F1F5F9 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span {
        color: #00D9FF !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
        accent-color: #00D9FF !important;
    }

    /* =========================================================================
       4. Inputs, Selectboxes & Sliders
       ========================================================================= */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F1F5F9 !important;
    }
    div[data-baseweb="select"] * {
        color: #F1F5F9 !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] label,
    label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
    }
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #CBD5E1 !important;
    }

    /* =========================================================================
       5. Cards, Metrics & Badges
       ========================================================================= */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00D9FF 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .brand-sub {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    
    .industrial-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .industrial-card-highlight {
        background-color: #161B22;
        border: 1px solid #0284C7;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 16px rgba(2, 132, 199, 0.2);
    }

    /* Metric Callout Boxes */
    .metric-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #00D9FF;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 0.82rem;
        color: #00D084;
        font-weight: 600;
    }

    /* Streamlit native st.metric override */
    div[data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* Status Badges */
    .badge-critical {
        background-color: rgba(255, 92, 92, 0.22);
        border: 1px solid #FF5C5C;
        color: #FFB3B3;
        padding: 5px 15px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
        display: inline-block;
    }
    .badge-advisory {
        background-color: rgba(255, 176, 32, 0.22);
        border: 1px solid #FFB020;
        color: #FFDF99;
        padding: 5px 15px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
        display: inline-block;
    }
    .badge-nominal {
        background-color: rgba(0, 208, 132, 0.22);
        border: 1px solid #00D084;
        color: #99F2D1;
        padding: 5px 15px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
        display: inline-block;
    }

    /* Transparency Disclosure Banner */
    .disclosure-banner {
        background-color: #1E293B;
        border-left: 4px solid #00D9FF;
        padding: 0.95rem 1.35rem;
        border-radius: 0 8px 8px 0;
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    /* =========================================================================
       6. Buttons, Tabs, Expanders & Dataframe
       ========================================================================= */
    div.stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.3rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%) !important;
        border-color: #00D9FF !important;
        box-shadow: 0 0 14px rgba(0, 217, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stExpander"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }
    div[data-testid="stExpander"] details summary:hover span {
        color: #00D9FF !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: #161B22 !important;
        border: 1px dashed #38BDF8 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #CBD5E1 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 6px 6px 0 0;
        border: 1px solid transparent;
        color: #94A3B8;
        padding: 9px 18px;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        border-color: #38BDF8 #30363D transparent #30363D !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #00D9FF !important;
        font-weight: 700 !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] p, small {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
    }

    /* Alerts */
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #F8FAFC !important;
        font-size: 0.92rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Navigation Pages definition
NAV_PAGES = [
    "🏠 Home Dashboard",
    "⚡ Single Machine Diagnostic",
    "📁 Batch Fleet Inspection",
    "📊 Model Analytics & Validation",
    "ℹ️ System Architecture & About"
]

if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = NAV_PAGES[0]

def on_radio_change():
    st.session_state.nav_choice = st.session_state.nav_radio

# Calculate index safely
current_index = NAV_PAGES.index(st.session_state.nav_choice) if st.session_state.nav_choice in NAV_PAGES else 0

# Sidebar Navigation & System Diagnostics
with st.sidebar:
    st.markdown("### ⚙️ MACHINEGUARD AI")
    st.caption("Industrial Predictive Maintenance System")
    st.markdown("---")
    
    st.radio(
        "Navigation Menu",
        options=NAV_PAGES,
        index=current_index,
        key="nav_radio",
        on_change=on_radio_change
    )
    nav_option = st.session_state.nav_choice
    
    st.markdown("---")
    st.markdown("#### 🛡️ System Telemetry")
    st.markdown(f"""
    - **Model Family:** Calibrated HistGradientBoosting
    - **Features:** 8 Raw + 4 Physical Engineered
    - **Cost Decision Cutoff:** `{DECISION_THRESHOLD:.2f}`
    - **Cost Penalty Ratio:** 10 : 1 (FN : FP)
    - **Prevalence in Fleet:** 3.39% Failures
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color:#1E293B; border:1px solid #334155; padding:12px; border-radius:8px; font-size:0.82rem; color:#CBD5E1; line-height:1.55;'>
        <strong style="color:#F8FAFC;">⚠️ Benchmark Disclosure:</strong><br>
        Trained on the synthetic UCI AI4I 2020 predictive maintenance benchmark. Designed for industrial hackathon demonstration. Not validated on physical plant machinery.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION A: HOME DASHBOARD
# ==============================================================================
if nav_option == "🏠 Home Dashboard":
    st.markdown('<div class="brand-title">MachineGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Intelligent Predictive Maintenance System for Industrial Machinery — Neural Storm 2026</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclosure-banner">
        <strong style="color:#00D9FF;">ℹ️ Transparency & Benchmark Disclosure:</strong> MachineGuard AI is evaluated on the standard UCI AI4I 2020 Predictive Maintenance dataset (10,000 milling machines). The dataset is a synthetic benchmark reflecting thermodynamic, electrical, and mechanical milling failure dynamics. No real-world factory trial is claimed.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Performance Callouts
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">PR-AUC (Test)</div>
            <div class="metric-value">0.9155</div>
            <div class="metric-delta">↑ 22.4% vs Raw</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">0.9853</div>
            <div class="metric-delta">Near-perfect separation</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Failure Recall</div>
            <div class="metric-value">85.3%</div>
            <div class="metric-delta">58 of 68 failures caught</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Precision</div>
            <div class="metric-value">95.1%</div>
            <div class="metric-delta">Only 3 false alarms</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Cost Cutoff</div>
            <div class="metric-value">{DECISION_THRESHOLD:.2f}</div>
            <div class="metric-delta">10x FN Cost-Optimal</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Industrial problem statement & value overview
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("""
        <div class="industrial-card">
            <h3 style="color:#00D9FF; margin-top:0;">🏭 Predictive Maintenance Motivation</h3>
            <p style="color:#CBD5E1; line-height:1.65;">
                In discrete manufacturing and mechanical machining plants, unplanned equipment breakdown interrupts production cycles, 
                causes tool damage, and forces emergency technician intervention.
            </p>
            <p style="color:#CBD5E1; line-height:1.65;">
                Traditional maintenance relies either on <em>reactive repairs</em> (running components until failure) or rigid 
                <em>calendar schedules</em> (replacing tooling prematurely). 
                <strong>MachineGuard AI investigates data-driven risk estimation</strong>: evaluating sensor telemetry to estimate failure risk 
                and guide maintenance inspection prioritization.
            </p>
            <h4 style="color:#38BDF8;">Key Engineering Features:</h4>
            <ul style="color:#CBD5E1; line-height:1.75;">
                <li><strong>Physics-Informed Features:</strong> Derives temperature difference (ΔT), rotational drive power (Watts), and cutting overstrain (Tool Wear × Torque).</li>
                <li><strong>Asymmetric Cost Optimization:</strong> Operates at an out-of-fold cost-optimal cutoff (0.34) reflecting a 10:1 penalty ratio for missed breakdowns vs routine checks.</li>
                <li><strong>No Data Leakage:</strong> Strict separation of target failure flags during training; verified held-out test split.</li>
                <li><strong>Interpretable Root Triggers:</strong> Analyzes telemetry against physical failure modes (HDF, PWF, OSF, TWF).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""<div class="industrial-card-highlight">
<h4 style="color:#00D9FF; margin-top:0; font-size:1.2rem; font-weight:700;">🚀 Quick Navigation</h4>
<p style="color:#CBD5E1; font-size:0.9rem; margin-bottom:14px;">Select an action to jump directly to that module:</p>
</div>""", unsafe_allow_html=True)
        
        # Action 1: Single Machine Diagnostic
        if st.button("⚡ Run Single Machine Diagnostic", key="qnav_single", use_container_width=True):
            st.session_state.nav_choice = "⚡ Single Machine Diagnostic"
            st.rerun()
        st.caption("Simulate real-time sensor telemetry or test physical failure presets (HDF, PWF, OSF, TWF).")
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        
        # Action 2: Batch Fleet Inspection
        if st.button("📁 Inspect Batch Fleet (CSV Upload)", key="qnav_batch", use_container_width=True):
            st.session_state.nav_choice = "📁 Batch Fleet Inspection"
            st.rerun()
        st.caption("Upload multi-machine CSV logs, audit fleet health index, and download work orders.")
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        
        # Action 3: Model Analytics & Validation
        if st.button("📊 View Model Analytics & Validation", key="qnav_analytics", use_container_width=True):
            st.session_state.nav_choice = "📊 Model Analytics & Validation"
            st.rerun()
        st.caption("Inspect ROC/PR curves, dynamic decision thresholding, and cost matrix tradeoffs.")
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        
        # Action 4: System Architecture & About
        if st.button("ℹ️ System Architecture & About", key="qnav_about", use_container_width=True):
            st.session_state.nav_choice = "ℹ️ System Architecture & About"
            st.rerun()
        st.caption("Review system architecture, synthetic benchmark disclosures, and edge IoT roadmap.")

    # Machine health & dataset summary
    st.markdown("### 📊 Benchmark Dataset Health Overview (10,000 Machines)")
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        st.metric(label="Total Recorded Machines", value="10,000", delta="UCI AI4I 2020")
    with dcol2:
        st.metric(label="Healthy Machines", value="9,661 (96.61%)", delta="Normal Fleet")
    with dcol3:
        st.metric(label="Failed Machines", value="339 (3.39%)", delta="Extreme Imbalance", delta_color="inverse")
    with dcol4:
        st.metric(label="Product Quality Tiers", value="L: 60% | M: 30% | H: 10%", delta="Industrial Distribution")


# ==============================================================================
# SECTION B: SINGLE MACHINE PREDICTION
# ==============================================================================
elif nav_option == "⚡ Single Machine Diagnostic":
    st.markdown('<div class="brand-title">Single Machine Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Input machine telemetry readings to calculate failure probability, risk classification, and root physical triggers.</div>', unsafe_allow_html=True)
    
    # Presets selector
    st.markdown("#### ⚡ Quick Presets (Click to Test Real Failure Conditions)")
    preset_cols = st.columns(5)
    
    selected_preset_key = None
    with preset_cols[0]:
        if st.button("🟢 Nominal Machine", use_container_width=True):
            selected_preset_key = "Healthy Nominal Machine"
    with preset_cols[1]:
        if st.button("🔥 Heat Dissipation (HDF)", use_container_width=True):
            selected_preset_key = "Heat Dissipation Failure (HDF)"
    with preset_cols[2]:
        if st.button("⚡ Power Overload (PWF)", use_container_width=True):
            selected_preset_key = "Power Failure - Overload (PWF)"
    with preset_cols[3]:
        if st.button("💥 Overstrain (OSF)", use_container_width=True):
            selected_preset_key = "Overstrain Failure (OSF)"
    with preset_cols[4]:
        if st.button("⚙️ Critical Tool Wear (TWF)", use_container_width=True):
            selected_preset_key = "Critical Tool Wear Failure (TWF)"
            
    # Session state for sensor values
    if "sensor_state" not in st.session_state:
        st.session_state.sensor_state = PRESETS["Healthy Nominal Machine"].copy()
        st.session_state.temp_unit = "K"
        
    if selected_preset_key:
        st.session_state.sensor_state = PRESETS[selected_preset_key].copy()
        st.info(f"Loaded Preset: **{selected_preset_key}** — {st.session_state.sensor_state['description']}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sensor Input Controls
    with st.expander("🛠️ Sensor Telemetry Controls & Configuration", expanded=True):
        u_col1, u_col2 = st.columns([1, 4])
        with u_col1:
            temp_unit = st.radio("Temperature Display Unit", ["Kelvin (K)", "Celsius (°C)"], horizontal=False)
            is_celsius = "Celsius" in temp_unit
        with u_col2:
            st.caption("Adjust sliders or enter numbers directly. Safe baseline values are shown for reference.")
            
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🌡️ Thermal & Material Parameters")
            
            # Air temp
            air_k_default = float(st.session_state.sensor_state["air_temp"])
            if is_celsius:
                air_c_default = round(kelvin_to_celsius(air_k_default), 1)
                air_input = st.slider("Air Temperature (°C)", min_value=20.0, max_value=35.0, value=air_c_default, step=0.1)
                air_temp_k = celsius_to_kelvin(air_input)
            else:
                air_temp_k = st.slider("Air Temperature (K)", min_value=295.0, max_value=305.0, value=air_k_default, step=0.1)
                
            # Process temp
            proc_k_default = float(st.session_state.sensor_state["proc_temp"])
            if is_celsius:
                proc_c_default = round(kelvin_to_celsius(proc_k_default), 1)
                proc_input = st.slider("Process Temperature (°C)", min_value=30.0, max_value=45.0, value=proc_c_default, step=0.1)
                proc_temp_k = celsius_to_kelvin(proc_input)
            else:
                proc_temp_k = st.slider("Process Temperature (K)", min_value=305.0, max_value=315.0, value=proc_k_default, step=0.1)
                
            # Product Type
            curr_type = st.session_state.sensor_state.get("product_type", "L")
            type_idx = 0 if curr_type == "L" else (1 if curr_type == "M" else 2)
            prod_type = st.selectbox(
                "Product Quality Variant",
                options=["L (Low Quality / High Volume - 60%)", "M (Medium Quality - 30%)", "H (High Quality / Premium - 10%)"],
                index=type_idx
            )[0]
            
        with c2:
            st.markdown("##### ⚙️ Mechanical & Spindle Parameters")
            rpm_default = int(st.session_state.sensor_state["rpm"])
            rpm_val = st.slider("Rotational Speed (rpm)", min_value=1100, max_value=2900, value=rpm_default, step=10)
            
            torque_default = float(st.session_state.sensor_state["torque"])
            torque_val = st.slider("Torque (Nm)", min_value=3.0, max_value=80.0, value=torque_default, step=0.5)
            
            tool_default = int(st.session_state.sensor_state["tool_wear"])
            tool_val = st.slider("Tool Wear Duration (min)", min_value=0, max_value=260, value=tool_default, step=1)

    # Run Prediction
    prediction = predict_single(
        air_temp=air_temp_k,
        proc_temp=proc_temp_k,
        rpm=rpm_val,
        torque=torque_val,
        tool_wear=tool_val,
        product_type=prod_type,
        temp_unit="K"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display Diagnostic Output
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        # Gauge chart
        gauge_fig = plot_risk_gauge(prediction["probability"], prediction["threshold"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Risk Badge and classification
        r_level = prediction["risk_level"]
        badge_class = "badge-critical" if r_level == "High Risk" else ("badge-advisory" if r_level == "Medium Risk" else "badge-nominal")
        
        st.markdown(f"""
        <div style="text-align:center; margin-top:-15px; margin-bottom:15px;">
            <span class="{badge_class}">{prediction['icon']} {r_level.upper()}</span>
            <div style="color:#CBD5E1; font-size:0.9rem; margin-top:6px;">
                Priority: <strong style="color:#F8FAFC;">{prediction['priority']}</strong> | Operating Status: <strong style="color:#F8FAFC;">{prediction['status']}</strong>
            </div>
            <div style="color:#94A3B8; font-size:0.84rem; margin-top:4px;">
                Calibrated Probability: <strong style="color:#00D9FF;">{prediction['probability']*100:.2f}%</strong> (Cutoff: {prediction['threshold']*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h5 style='color:#00D9FF;'>📋 Recommended Maintenance Actions:</h5>", unsafe_allow_html=True)
        for rec in prediction["recommendations"]:
            st.markdown(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_col2:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        # Radar chart
        radar_fig = plot_sensor_radar(air_temp_k, proc_temp_k, rpm_val, torque_val, tool_val)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Physical failure trigger alerts
        st.markdown("<h5 style='color:#00D9FF;'>🔬 Root Physical Triggers Analysis:</h5>", unsafe_allow_html=True)
        triggers = prediction["triggers"]
        if triggers:
            for tr in triggers:
                st.markdown(f"""
                <div style="background:#2A1418; border:1px solid #FF5C5C; padding:10px 14px; border-radius:8px; margin-bottom:8px;">
                    <strong style="color:#FFB3B3; font-size:0.95rem;">🚨 {tr['name']}</strong>
                    <div style="color:#F1F5F9; font-size:0.88rem; margin-top:3px;">{tr['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0F291E; border:1px solid #00D084; padding:10px 14px; border-radius:8px; margin-bottom:8px;">
                <strong style="color:#99F2D1; font-size:0.95rem;">✅ No Physical Failure Triggers Active</strong>
                <div style="color:#F1F5F9; font-size:0.88rem; margin-top:3px;">All thermodynamic and mechanical strain criteria are within certified design tolerances.</div>
            </div>
            """, unsafe_allow_html=True)
            
        ind = prediction["indicators"]
        st.markdown(f"""
        <div style="font-size:0.88rem; color:#CBD5E1; margin-top:12px; background:#0B0F15; padding:12px; border-radius:8px; border:1px solid #1F2937;">
            <strong style="color:#00D9FF;">Derived Engineering Parameters:</strong><br>
            • Heat Dissipation ΔT: <code>{ind['temp_diff']:.1f} K</code> (Safe: > 8.6 K)<br>
            • Spindle Power: <code>{ind['power_w']:.0f} W</code> (Safe: 3,500 – 9,000 W)<br>
            • Tool Overstrain: <code>{ind['strain']:.0f} min·Nm</code> (Limit: {ind['strain_limit']:.0f} min·Nm)<br>
            • Torque/Speed Ratio: <code>{ind['torque_per_rpm']:.4f} Nm/rpm</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Telemetry baseline comparison table
    with st.expander("📊 Detailed Telemetry Comparison vs Industrial Nominal Baselines", expanded=False):
        dev_df = pd.DataFrame(prediction["deviations"])
        st.dataframe(dev_df, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION C: BATCH FLEET INSPECTION
# ==============================================================================
elif nav_option == "📁 Batch Fleet Inspection":
    st.markdown('<div class="brand-title">Batch Fleet Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Upload CSV telemetry logs to perform simultaneous predictive screening across entire machine fleets.</div>', unsafe_allow_html=True)
    
    # Download sample files
    col_d1, col_d2 = st.columns(2)
    sample_fleet_path = ROOT_DIR / "data" / "sample_fleet.csv"
    with col_d1:
        if sample_fleet_path.exists():
            with open(sample_fleet_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample Test Fleet CSV (25 Machines)",
                    data=f.read(),
                    file_name="sample_fleet.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    with col_d2:
        template_csv = "Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Type\n298.1,308.6,1551,42.8,0,M\n298.2,308.7,1408,46.3,3,L\n"
        st.download_button(
            label="⬇️ Download Blank Template CSV",
            data=template_csv,
            file_name="machineguard_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Machine Telemetry CSV", type=["csv"])
    
    # Default to sample fleet if user hasn't uploaded a file yet for instant evaluation!
    df_to_process = None
    source_label = ""
    
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    else:
        st.info("💡 No file uploaded yet. Showing live evaluation of the pre-loaded **25-machine sample fleet** below. Upload your own CSV anytime!")
        if sample_fleet_path.exists():
            df_to_process = pd.read_csv(sample_fleet_path)
            source_label = "Pre-loaded Test Fleet (25 Industrial Machines)"
            
    if df_to_process is not None:
        st.caption(f"Inspecting: **{source_label}** ({len(df_to_process)} records)")
        
        batch_results = predict_batch(df_to_process)
        
        if not batch_results["success"]:
            st.error("Failed to process batch CSV:")
            for err in batch_results["errors"]:
                st.error(f"• {err}")
        else:
            if batch_results["warnings"]:
                with st.expander("⚠️ Data Cleansing & Validation Notes", expanded=False):
                    for w in batch_results["warnings"]:
                        st.warning(f"• {w}")
                        
            summary = batch_results["summary"]
            res_df = batch_results["results_df"]
            
            # Fleet KPI summary cards
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Machines</div>
                    <div class="metric-value">{summary['total_machines']}</div>
                    <div class="metric-delta">Scanned 100%</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="metric-box" style="border-left-color:#FF5C5C;">
                    <div class="metric-label">Critical (High Risk)</div>
                    <div class="metric-value" style="color:#FF5C5C;">{summary['high_risk_count']}</div>
                    <div class="metric-delta" style="color:#FF5C5C;">Immediate Inspection</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="metric-box" style="border-left-color:#FFB020;">
                    <div class="metric-label">Advisory (Medium)</div>
                    <div class="metric-value" style="color:#FFB020;">{summary['medium_risk_count']}</div>
                    <div class="metric-delta" style="color:#FFB020;">Schedule Check</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="metric-box" style="border-left-color:#00D084;">
                    <div class="metric-label">Healthy (Low Risk)</div>
                    <div class="metric-value" style="color:#00D084;">{summary['low_risk_count']}</div>
                    <div class="metric-delta" style="color:#00D084;">Nominal Operation</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi5:
                st.markdown(f"""
                <div class="metric-box" style="border-left-color:#00D9FF;">
                    <div class="metric-label">Fleet Health Index</div>
                    <div class="metric-value" style="color:#00D9FF;">{summary['healthy_percentage']}%</div>
                    <div class="metric-delta">Healthy Proportion</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Distribution Donut & Filter controls
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                donut_fig = plot_batch_risk_donut(summary)
                st.plotly_chart(donut_fig, use_container_width=True)
            with f_col2:
                st.markdown("#### 🔍 Filter & Prioritize Fleet Inspection")
                filter_choice = st.selectbox(
                    "Filter by Risk Category",
                    ["All Machines", "🚨 High Risk (Action Required)", "⚠️ Medium Risk (Advisory)", "✅ Low Risk (Healthy)"]
                )
                
                filtered_df = res_df.copy()
                if "High Risk" in filter_choice:
                    filtered_df = filtered_df[filtered_df["Risk_Category"] == "High Risk"]
                elif "Medium Risk" in filter_choice:
                    filtered_df = filtered_df[filtered_df["Risk_Category"] == "Medium Risk"]
                elif "Low Risk" in filter_choice:
                    filtered_df = filtered_df[filtered_df["Risk_Category"] == "Low Risk"]
                    
                st.caption(f"Showing **{len(filtered_df)}** of **{len(res_df)}** machines")
                
                # Download results button
                csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Filtered Inspection Report (CSV)",
                    data=csv_bytes,
                    file_name="machineguard_inspection_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            # Table display
            st.markdown("### 📋 Machine Telemetry & Risk Register")
            
            # Format dataframe for clean display
            display_cols = [c for c in ["UDI", "Product ID", "Type", "Risk_Score_Pct", "Risk_Category", "Recommended_Action", "Suspected_Modes", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"] if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_cols],
                use_container_width=True,
                hide_index=True
            )


# ==============================================================================
# SECTION D: MODEL ANALYTICS & VALIDATION
# ==============================================================================
elif nav_option == "📊 Model Analytics & Validation":
    st.markdown('<div class="brand-title">Model Analytics & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Comprehensive, reproducible machine learning evaluation based on the 2,000 held-out test records.</div>', unsafe_allow_html=True)
    
    analytics_tabs = st.tabs([
        "📈 ROC & PR Curves",
        "🎯 Cost Matrix & Confusion Matrix",
        "⚔️ Model Family Comparison",
        "💡 Feature Importance",
        "🔬 Stress Testing & Robustness"
    ])
    
    # Tab 1: Curves
    with analytics_tabs[0]:
        st.markdown("#### Dynamic Operating Point on Test Curves")
        st.markdown("Use the slider to see how the decision threshold changes precision, recall, and detection tradeoff on the 2,000 held-out test machines.")
        
        thr_slider = st.slider("Decision Cutoff Threshold", min_value=0.05, max_value=0.95, value=DECISION_THRESHOLD, step=0.01)
        point_eval = EvaluationManager.evaluate_at_threshold(thr_slider)
        
        roc_data = EvaluationManager.get_roc_data()
        pr_data = EvaluationManager.get_pr_data()
        
        c1, c2 = st.columns(2)
        with c1:
            roc_fig = plot_roc_curve(roc_data, point_eval)
            st.plotly_chart(roc_fig, use_container_width=True)
            st.caption(f"ROC-AUC: **{roc_data['auc']:.4f}** | At Cutoff {thr_slider:.2f}: Recall (TPR) = **{point_eval['recall']*100:.1f}%**, FPR = **{point_eval['fpr']*100:.2f}%**")
        with c2:
            pr_fig = plot_pr_curve(pr_data, point_eval)
            st.plotly_chart(pr_fig, use_container_width=True)
            st.caption(f"PR-AUC: **{pr_data['auc']:.4f}** (vs 3.4% baseline) | Precision = **{point_eval['precision']*100:.1f}%**, Recall = **{point_eval['recall']*100:.1f}%**")
            
    # Tab 2: Confusion Matrix & Cost
    with analytics_tabs[1]:
        st.markdown("#### Industrial Cost Matrix & Confusion Matrix")
        st.markdown("""
        In industrial predictive maintenance, errors are **asymmetric**:
        - **False Negative (Missed Breakdown):** Unpredicted failure risk; penalty weight = 10 units.
        - **False Positive (False Alarm):** Routine inspection verification; penalty weight = 1 unit.
        """)
        
        cm_col1, cm_col2 = st.columns([1, 1])
        with cm_col1:
            cm_eval = EvaluationManager.evaluate_at_threshold(thr_slider)
            cm_fig = plot_confusion_matrix(cm_eval)
            st.plotly_chart(cm_fig, use_container_width=True)
        with cm_col2:
            st.markdown(f"""
            <div class="industrial-card">
                <h4 style="color:#00D9FF; margin-top:0;">Cost Tradeoff Comparison</h4>
                <p style="color:#CBD5E1;">Evaluated on 2,000 held-out test machines (68 true failures):</p>
                <table style="width:100%; color:#CBD5E1; font-size:0.92rem; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #30363D;">
                        <th style="padding:8px; text-align:left; color:#00D9FF;">Metric</th>
                        <th style="padding:8px; text-align:center; color:#CBD5E1;">Threshold 0.50</th>
                        <th style="padding:8px; text-align:center; color:#00D9FF;">Optimal 0.34</th>
                    </tr>
                    <tr style="border-bottom:1px solid #21262D;">
                        <td style="padding:8px; color:#F1F5F9;">Failures Caught (TP)</td>
                        <td style="padding:8px; text-align:center;">57 / 68 (83.8%)</td>
                        <td style="padding:8px; text-align:center; color:#00D084; font-weight:bold;">58 / 68 (85.3%)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262D;">
                        <td style="padding:8px; color:#F1F5F9;">Missed Failures (FN)</td>
                        <td style="padding:8px; text-align:center; color:#FF5C5C;">11 missed</td>
                        <td style="padding:8px; text-align:center; color:#00D084; font-weight:bold;">10 missed (-10% FN)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262D;">
                        <td style="padding:8px; color:#F1F5F9;">False Alarms (FP)</td>
                        <td style="padding:8px; text-align:center;">1</td>
                        <td style="padding:8px; text-align:center; color:#CBD5E1;">3</td>
                    </tr>
                    <tr style="border-bottom:1px solid #21262D;">
                        <td style="padding:8px; color:#F1F5F9;">Total Normalized Cost</td>
                        <td style="padding:8px; text-align:center;">111.0 units</td>
                        <td style="padding:8px; text-align:center; color:#00D9FF; font-weight:bold;">103.0 units (-7.2% Cost)</td>
                    </tr>
                    <tr>
                        <td style="padding:8px; color:#F1F5F9;">Brier Calibration Score</td>
                        <td style="padding:8px; text-align:center;" colspan="2"><strong style="color:#00D084;">0.00625</strong> (Calibrated) vs 0.01577 (Raw)</td>
                    </tr>
                </table>
                <div style="font-size:0.82rem; color:#94A3B8; margin-top:14px; line-height:1.5;">
                    <em>Selection Protocol: The 0.34 cutoff was selected entirely on Out-of-Fold training cross-validation to minimize total cost without looking at the test set.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Tab 3: Model Family Comparison
    with analytics_tabs[2]:
        st.markdown("#### Repeated 5-Fold Stratified Cross-Validation on Training Data")
        st.markdown("Three model families evaluated across both raw and physics-engineered feature sets:")
        
        metrics_dict = EvaluationManager.get_metrics()
        comp_fig = plot_model_comparison(metrics_dict["cv_comparison"])
        st.plotly_chart(comp_fig, use_container_width=True)
        
        comp_table = EvaluationManager.get_model_comparison_table()
        st.dataframe(comp_table, use_container_width=True, hide_index=True)
        
    # Tab 4: Feature Importance
    with analytics_tabs[3]:
        st.markdown("#### Permutation Feature Importance on Held-Out Test Set")
        st.markdown("Calculated by shuffling each feature column 10 times and recording the drop in PR-AUC:")
        
        imp_list = metrics_dict["permutation_importance_pr_auc_drop"]
        imp_fig = plot_feature_importance(imp_list)
        st.plotly_chart(imp_fig, use_container_width=True)
        
        st.markdown("""
        <div style="background:#161B22; border:1px solid #30363D; padding:14px 18px; border-radius:8px; margin-top:12px;">
            <strong style="color:#00D9FF; font-size:1rem;">Engineering Insights:</strong><br>
            <ul style="color:#CBD5E1; line-height:1.7; margin-top:6px; margin-bottom:0;">
                <li><strong style="color:#F8FAFC;">Torque (0.7558 drop)</strong> and <strong style="color:#F8FAFC;">Rotational Speed (0.5638 drop)</strong> are by far the dominant drivers of machine failure.</li>
                <li><strong style="color:#F8FAFC;">Air Temperature (0.4833 drop)</strong> and <strong style="color:#F8FAFC;">Tool Wear (0.3673 drop)</strong> represent secondary thermodynamic and wear degradation axes.</li>
                <li>Product type indicators (L, M, H) have small direct permutation drops because quality tier is primarily expressed through operating mechanical load.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    # Tab 5: Stress Testing
    with analytics_tabs[4]:
        st.markdown("#### Robustness Audits: Sensor Noise & Generalization")
        
        st_col1, st_col2 = st.columns(2)
        with st_col1:
            st.markdown("##### 📡 Sensor Noise Resilience")
            noise_fig = plot_noise_robustness(metrics_dict["noise_robustness"])
            st.plotly_chart(noise_fig, use_container_width=True)
            st.caption("Gaussian noise added to test telemetry up to 20% of feature standard deviation. The model maintains high recall even with 10% sensor drift.")
            
        with st_col2:
            st.markdown("##### 🏷️ Failure Mode Recall Breakdown")
            fm_table = EvaluationManager.get_failure_mode_breakdown()
            st.dataframe(fm_table[["Failure Mode", "Total in Test Set", "Correctly Caught", "Recall Rate"]], use_container_width=True, hide_index=True)
            st.caption("Heat Dissipation (HDF), Power Overload (PWF), and Overstrain (OSF) achieve **100% recall**. Tool Wear Failure (TWF) is subtle in single-snapshot telemetry (20% recall).")
            
        st.markdown("##### 📦 Leave-One-Product-Type-Out Generalization (Trained on 2 types, tested on unseen 3rd)")
        lotto_df = EvaluationManager.get_lotto_table()
        st.dataframe(lotto_df, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION E: SYSTEM ARCHITECTURE & ABOUT
# ==============================================================================
elif nav_option == "ℹ️ System Architecture & About":
    st.markdown('<div class="brand-title">System Architecture & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">End-to-end design, data provenance, and production roadmap for MachineGuard AI.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="industrial-card">
        <h3 style="color:#00D9FF; margin-top:0;">🏗️ End-to-End System Pipeline</h3>
        <p style="color:#CBD5E1; line-height:1.65;">
            MachineGuard AI is engineered as a robust, modular predictive maintenance stack designed for deployment in industrial IoT gateways or edge computing clusters:
        </p>
        <div style="background:#0B0F15; padding:16px 20px; border-radius:8px; border:1px solid #1F2937; font-family:Consolas, monospace; color:#38BDF8; font-size:0.9rem; line-height:1.75;">
            [ Industrial Sensors ] → (Air/Proc Temp, RPM, Torque, Tool Wear)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Ingestion & Data Validation ] → (Schema alias mapping, Celsius/Kelvin normalization, null imputation)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Leakage Prevention Guardrail ] → (Drops UDI, Product ID, and all TWF/HDF/PWF/OSF/RNF target flags)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Physics-Informed Feature Engineering ] → (ΔT heat dissipation, Spindle Power W, Tool Overstrain min·Nm)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Calibrated HistGradientBoosting ] → (Platt sigmoid calibration to guarantee well-calibrated probabilities)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Asymmetric Cost Decision Engine ] → (Optimal Cutoff: 0.34 with 10:1 FN:FP penalty)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            [ Actionable UI & Maintenance Recommendations ] → (Root physical trigger alerts & technician checklist)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 Machine Learning Rigor & Protocol")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("""
        <div class="industrial-card">
            <h4 style="color:#00D9FF;">1. Strict Leakage Prevention</h4>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                In the AI4I 2020 dataset, failure mode columns (<code>TWF</code>, <code>HDF</code>, <code>PWF</code>, <code>OSF</code>, <code>RNF</code>) are sub-components of the target <code>Machine failure</code>. 
                Using them as model inputs represents fatal target leakage. 
                Our pipeline strictly purges all failure mode flags and identifier columns (<code>UDI</code>, <code>Product ID</code>) prior to any feature extraction.
            </p>
            <h4 style="color:#00D9FF;">2. Zero Test Contamination</h4>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                A stratified 80/20 train/test split was established at step 0. 
                All model comparison, repeated 5-fold cross-validation, feature engineering evaluation, probability calibration, and cost threshold tuning were performed strictly on the 8,000 training records. The 2,000 test records were scored exactly once.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="industrial-card">
            <h4 style="color:#00D9FF;">3. Calibrated Probabilities</h4>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                Standard gradient boosted trees output uncalibrated margins that tend to bunch near 0 and 1. 
                We apply sigmoid (Platt) calibration via <code>CalibratedClassifierCV</code>. 
                This improved the Brier calibration loss on the test set from <strong style="color:#FFB020;">0.01577</strong> to <strong style="color:#00D084;">0.00625</strong>, ensuring risk scores reflect true statistical failure frequencies.
            </p>
            <h4 style="color:#00D9FF;">4. Cost-Sensitive Optimization</h4>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                With 3.39% failure prevalence, optimizing for default 0.50 threshold yields unacceptable false negatives. 
                By factoring the empirical manufacturing cost ratio (10x penalty for missed breakdowns), the optimal operating point of <strong style="color:#00D9FF;">0.34</strong> reduces economic cost by 7.2% and captures 85.3% of failures.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Dataset Limitations & Industrial Roadmap")
    st.markdown("""
    <div class="industrial-card">
        <h4 style="color:#FFB020; margin-top:0;">Limitations of Synthetic Benchmark:</h4>
        <ul style="color:#CBD5E1; font-size:0.92rem; line-height:1.7;">
            <li><strong>Static Snapshots vs Time-Series Telemetry:</strong> The AI4I dataset contains single-point sensor readings rather than continuous high-frequency vibration streams (FFT spectrums, acoustic emissions).</li>
            <li><strong>Synthetic Failure Formulas:</strong> While grounded in thermodynamic and mechanical laws, real machine tool degradation involves complex chatter vibration, lubrication breakdown, and stochastic cutting tool chipping.</li>
        </ul>
        <h4 style="color:#00D9FF;">Future Scope for Production Deployment:</h4>
        <ol style="color:#CBD5E1; font-size:0.92rem; line-height:1.7;">
            <li><strong>High-Frequency Vibration & Accelerometer Ingestion:</strong> Connect 10 kHz 3-axis accelerometer sensors via MQTT/OPC-UA for early bearing defect detection.</li>
            <li><strong>Edge Inference on Industrial Gateways:</strong> Quantize model into ONNX / TensorRT for sub-millisecond inference on Siemens or Advantech edge controllers.</li>
            <li><strong>Automated Work Order Dispatch:</strong> Integrate with enterprise CMMS (Computerized Maintenance Management Systems) such as SAP PM or IBM Maximo to auto-generate technician work orders.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
