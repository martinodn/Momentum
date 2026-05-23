"""
Momentum — Tracciamento progressione corsa post-infortunio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk

import math
import os
import json
from collections import Counter
from datetime import datetime
import garmin_client
from segment_stats import get_weekly_segment_bests
from data_loader import (
    load_running_activities,
    get_weekly_summary,
    get_zone_labels,
    _format_pace,
    _format_duration,
    load_overrides,
    save_overrides,
    load_all_activities,
    load_polylines,
    TRACKING_START,
    UPLOADS_DIR,
)

def _map_deck_kwargs():
    """Mapbox light se c'è un token, altrimenti Carto Positron (no API key)."""
    token = os.environ.get("MAPBOX_API_KEY")
    try:
        token = token or st.secrets.get("mapbox", {}).get("token")
    except Exception:
        pass
    if token:
        return {
            "map_provider": "mapbox",
            "map_style": "light",
            "api_keys": {"mapbox": token},
        }
    return {"map_provider": "carto", "map_style": "light"}


def _pace_axis_ticks(paces: pd.Series, step_min: float = 0.5) -> tuple[list[float], list[str]]:
    """Tick asse Y per passo: valori float min/km, etichette M:SS."""
    if paces.empty:
        return [], []
    lo, hi = float(paces.min()), float(paces.max())
    start = np.floor(lo / step_min) * step_min
    end = np.ceil(hi / step_min) * step_min
    ticks = list(np.arange(start, end + 1e-9, step_min))
    labels = [_format_pace(t).replace(" /km", "") for t in ticks]
    return ticks, labels


def _last_run_start_lat_lon(polylines: list, runs_df: pd.DataFrame) -> tuple[float, float] | None:
    """Primo punto GPS della corsa più recente che ha traccia (ordine per data in df)."""
    if runs_df.empty:
        return None
    paths = {p["activityId"]: p["path"] for p in polylines if p.get("path")}
    for activity_id in reversed(runs_df["activity_id"].tolist()):
        try:
            aid = int(activity_id)
        except (TypeError, ValueError):
            continue
        path = paths.get(aid)
        if path:
            lon, lat = path[0]
            return lat, lon
    return None


def _zoom_for_radius_km(lat: float, radius_km: float = 12.5, viewport_px: int = 1100) -> float:
    """Livello zoom pydeck per mostrare circa 2×radius_km di larghezza."""
    lat_rad = math.radians(lat)
    mpp_at_z0 = 156543.03 * math.cos(lat_rad)
    width_m = radius_km * 2 * 1000
    return math.log2(viewport_px * mpp_at_z0 / width_m)


def _heatmap_from_polylines(polylines: list, grid_precision: int = 4) -> pd.DataFrame:
    """Aggrega i punti GPS per cella; weight = quante volte sei passato lì."""
    counts: Counter = Counter()
    for track in polylines:
        for lon, lat in track["path"]:
            counts[(round(lon, grid_precision), round(lat, grid_precision))] += 1
    if not counts:
        return pd.DataFrame(columns=["lon", "lat", "weight"])
    return pd.DataFrame(
        [{"lon": lon, "lat": lat, "weight": w} for (lon, lat), w in counts.items()]
    )


# Blu chiaro (poche passate) → arancione → rosso scuro (molte passate)
_HEATMAP_COLORS = [
    [198, 219, 239],
    [102, 194, 214],
    [49, 163, 189],
    [253, 174, 97],
    [244, 109, 67],
    [215, 48, 39],
    [165, 0, 38],
]

st.set_page_config(page_title="Momentum", page_icon="🏃", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Full-width layout */
section.main > div.block-container {
    max-width: 100%;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Global Reset and Scrollbar */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6, .section-title, .progress-title, .metric-label, .stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background-color: #080a13;
    color: #e2e8f0;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080a13;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4f46e5;
}

/* Hero Section */
.hero {
    background: radial-gradient(circle at top right, rgba(79, 70, 229, 0.15), transparent 60%),
                radial-gradient(circle at bottom left, rgba(6, 182, 212, 0.08), transparent 50%),
                linear-gradient(185deg, #111422 0%, #080a13 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 30%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: #94a3b8;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    font-weight: 500;
}

/* Metric Cards Grid */
.metric-card {
    background: rgba(17, 22, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 4px solid #6366f1;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.25);
}
.metric-card.card-purple { border-left-color: #a855f7; }
.metric-card.card-purple:hover { box-shadow: 0 12px 30px -10px rgba(168, 85, 247, 0.25); }

.metric-card.card-blue { border-left-color: #3b82f6; }
.metric-card.card-blue:hover { box-shadow: 0 12px 30px -10px rgba(59, 130, 246, 0.25); }

.metric-card.card-amber { border-left-color: #f59e0b; }
.metric-card.card-amber:hover { box-shadow: 0 12px 30px -10px rgba(245, 158, 11, 0.25); }

.metric-card.card-rose { border-left-color: #f43f5e; }
.metric-card.card-rose:hover { box-shadow: 0 12px 30px -10px rgba(244, 63, 94, 0.25); }

.metric-card.card-emerald { border-left-color: #10b981; }
.metric-card.card-emerald:hover { box-shadow: 0 12px 30px -10px rgba(16, 185, 129, 0.25); }

.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .4rem;
}
.metric-value {
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.metric-sub {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: .3rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17, 22, 39, 0.8) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    display: flex !important;
    width: 100% !important;
}
.stTabs [data-baseweb="tab"] {
    flex: 1 !important;
    justify-content: center !important;
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
}

/* Tables / DataFrames */
.stDataFrame, [data-testid="stTable"] {
    background-color: rgba(17, 22, 39, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    overflow: hidden;
}

/* Custom st.expander style */
div[data-testid="stExpander"] {
    background: rgba(17, 22, 39, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    margin-bottom: 0.8rem !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] details {
    border: none !important;
}
div[data-testid="stExpander"] summary {
    background: rgba(20, 28, 47, 0.8) !important;
    padding: 1rem 1.2rem !important;
    border-radius: 16px !important;
    color: #f8fafc !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stExpander"] summary:hover {
    background: rgba(30, 41, 69, 0.9) !important;
    color: #60a5fa !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: rgba(10, 13, 23, 0.95) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 1.5rem !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4) !important;
    background: linear-gradient(135deg, #5b54f3, #7275f8) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Form fields (text inputs, selects, dates) */
.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
    background-color: rgba(17, 22, 39, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within, .stSelectbox > div > div:focus-within, .stDateInput > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: rgba(17, 22, 39, 0.4) !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
    background-color: rgba(17, 22, 39, 0.6) !important;
}

/* Section titles */
.section-title {
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 2rem 0 1rem 0;
    background: linear-gradient(90deg, #f8fafc, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* PR badge */
.pr-badge {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 8px;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
    display: inline-block;
}

/* Recovery Progress Widget */
.recovery-progress-card {
    background: linear-gradient(135deg, rgba(20, 28, 47, 0.8) 0%, rgba(13, 17, 29, 0.8) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}
.progress-title {
    font-weight: 700;
    font-size: 1.2rem;
    color: #f8fafc;
    margin-bottom: 0.8rem;
    letter-spacing: -0.01em;
}
.progress-stats {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}
.progress-val {
    font-weight: 700;
    color: #60a5fa;
}
.progress-pct {
    color: #a78bfa;
    font-weight: 600;
}
.progress-bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.progress-bar-fill {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    height: 100%;
    border-radius: 999px;
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.4);
}
.progress-desc {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_running_activities()
    weekly = get_weekly_summary(df)
    weekly_segments = get_weekly_segment_bests(df)
    return df, weekly, weekly_segments


# ── Auto‑fetch Garmin data (on startup, silent) ──────────────────────────────
st.cache_data.clear()
garmin_client.fetch_garmin_activities(force=True)
df, weekly, weekly_segments = get_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("🏃 Momentum")

if df.empty:
    st.error("No running activities found. Check the JSON files in the activities folder.")
    st.stop()

# ── Recovery Progress Widget ──────────────────────────────────────────────────
if not weekly.empty:
    latest_week_km = weekly.iloc[-1]["total_km"]
    if len(weekly) >= 2:
        prev_week_km = weekly.iloc[-2]["total_km"]
        weekly_target = round(prev_week_km * 1.05, 1)  # +5% rispetto alla settimana precedente
    else:
        weekly_target = round(latest_week_km * 1.05, 1)
else:
    latest_week_km = 0.0
    weekly_target = 20.0

progress_pct = min(latest_week_km / weekly_target, 1.0) if weekly_target > 0 else 0.0

st.markdown(f"""
<div class="recovery-progress-card">
  <div class="progress-title">📈 Weekly Target — +5% vs previous week</div>
  <div class="progress-stats">
    <span class="progress-val">{latest_week_km:.1f} / {weekly_target:.1f} km</span>
    <span class="progress-pct">{progress_pct * 100:.0f}% of target</span>
  </div>
  <div class="progress-bar-container">
    <div class="progress-bar-fill" style="width: {progress_pct * 100}%;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
total_km = df["distance_km"].sum()
total_runs = len(df)
last_run = df.iloc[-1]
best_distance = df["distance_km"].max()

col1, col2, col3, col4 = st.columns(4)
metrics = [
    (col1, "🏃 Total Runs", str(total_runs), "since March 2026", "card-purple"),
    (col2, "📏 Total Distance", f"{total_km:.0f} km", f"longest: {best_distance:.1f} km", "card-blue"),
    (col3, "❤️ Avg Heart Rate", f"{df['avg_hr'].mean():.0f} bpm" if df["avg_hr"].notna().any() else "—", "heart rate", "card-rose"),
    (col4, "📅 Last Run", last_run["date_str"], f"{last_run['distance_km']:.1f} km · {last_run['avg_pace_str']}", "card-emerald"),
]
for col, label, val, sub, card_class in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card {card_class}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Progress", "📅 Weekly", "👟 Activities", "❤️ HR Zones", "🗺️ Map", "⚙️ Settings"])

_AXIS = dict(
    gridcolor="rgba(255, 255, 255, 0.05)",
    zeroline=False,
    tickfont=dict(color="#64748b", size=10, family="Inter, sans-serif"),
)
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=0, r=0, t=50, b=10),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#111625",
        bordercolor="rgba(255, 255, 255, 0.08)",
        font=dict(color="#f8fafc", family="Inter, sans-serif", size=12)
    ),
)

def layout(**overrides):
    """Merge DARK_LAYOUT with default axes and any overrides."""
    base = {**DARK_LAYOUT, "xaxis": _AXIS, "yaxis": _AXIS}
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PROGRESSIONE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Performance over time</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Passo
    with c1:
        fig = go.Figure()
        df_p = df.dropna(subset=["avg_pace"])
        df_p = df_p[df_p["distance_km"] >= 2.5]
        # Trendline
        if len(df_p) > 2:
            z = np.polyfit(range(len(df_p)), df_p["avg_pace"], 1)
            trend = np.polyval(z, range(len(df_p)))
            fig.add_trace(go.Scatter(
                x=df_p["date"], y=trend,
                mode="lines", name="Trend",
                line=dict(color="#f43f5e", width=2, dash="dash"),
                customdata=[_format_pace(t) for t in trend],
                hovertemplate="Trend: %{customdata}<extra></extra>",
                showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=df_p["date"], y=df_p["avg_pace"],
            mode="lines+markers", name="Avg Pace",
            line=dict(color="#818cf8", width=3),
            marker=dict(size=8, color="#818cf8", symbol="circle", line=dict(color="#080a13", width=1)),
            hovertemplate="Pace: %{customdata}<extra></extra>",
            customdata=df_p["avg_pace_str"],
        ))
        pace_yaxis = dict(**_AXIS, autorange="reversed")
        if not df_p.empty:
            tickvals, ticktext = _pace_axis_ticks(df_p["avg_pace"])
            pace_yaxis["tickvals"] = tickvals
            pace_yaxis["ticktext"] = ticktext
        fig.update_layout(**layout(
            title=dict(text="⏱️ Avg Pace (min/km)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
            yaxis=pace_yaxis,
            height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Distanza (aggregata per giorno)
    with c2:
        df_day_dist = (
            df.assign(day=df["date"].dt.normalize())
            .groupby("day", as_index=False)
            .agg(
                distance_km=("distance_km", "sum"),
                runs=("distance_km", "count"),
                is_pr=("is_pr", "any"),
            )
            .rename(columns={"day": "date"})
            .sort_values("date")
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_day_dist["date"], y=df_day_dist["distance_km"],
            name="Distance",
            marker=dict(
                color=df_day_dist["distance_km"],
                colorscale=[[0, "#4f46e5"], [1, "#06b6d4"]],
                showscale=False,
            ),
            customdata=df_day_dist["runs"],
            hovertemplate="%{y:.2f} km · %{customdata} run<extra></extra>",
        ))
        prs = df_day_dist[df_day_dist["is_pr"] == True]
        if not prs.empty:
            fig2.add_trace(go.Scatter(
                x=prs["date"], y=prs["distance_km"] + 0.3,
                mode="markers+text", text=["🏆"] * len(prs),
                textposition="top center", marker=dict(size=1, opacity=0),
                name="Personal Record", showlegend=True,
            ))
        fig2.update_layout(**layout(
            title=dict(text="📏 Distance per day (km)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
            height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    # Frequenza Cardiaca
    with c3:
        df_hr = df.dropna(subset=["avg_hr"])
        df_hr = df_hr[df_hr["distance_km"] >= 2.5]
        fig3 = go.Figure()
        if not df_hr.empty:
            hr_hover = list(zip(
                df_hr["distance_km"],
                df_hr["duration_min"].apply(
                    lambda m: _format_duration(m) if pd.notna(m) else "—"
                ),
            ))
            hr_hover_tpl = (
                "%{y:.0f} bpm<br>"
                "Distance: %{customdata[0]:.2f} km<br>"
                "Duration: %{customdata[1]}"
                "<extra></extra>"
            )
            fig3.add_trace(go.Scatter(
                x=df_hr["date"], y=df_hr["max_hr"],
                name="Max HR", mode="lines",
                line=dict(color="#fda4af", width=1.5, dash="dot"),
                fill=None,
                customdata=hr_hover,
                hovertemplate="Max HR: " + hr_hover_tpl,
            ))
            fig3.add_trace(go.Scatter(
                x=df_hr["date"], y=df_hr["avg_hr"],
                name="Avg HR", mode="lines+markers",
                line=dict(color="#f43f5e", width=3),
                marker=dict(size=7, color="#f43f5e", line=dict(color="#080a13", width=1)),
                fill="tonexty", fillcolor="rgba(244, 63, 94, 0.04)",
                customdata=hr_hover,
                hovertemplate="Avg HR: " + hr_hover_tpl,
            ))
        fig3.update_layout(**layout(
            title=dict(text="❤️ Heart Rate (bpm)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
            height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    # Training Load
    with c4:
        df_tl = df.dropna(subset=["training_load"])
        fig4 = go.Figure()
        if not df_tl.empty:
            fig4.add_trace(go.Bar(
                x=df_tl["date"], y=df_tl["training_load"],
                marker=dict(
                    color=df_tl["training_load"],
                    colorscale=[[0, "#f59e0b"], [1, "#d97706"]],
                ),
                hovertemplate="Load: %{y:.0f}<extra></extra>",
                name="Training Load",
            ))
        fig4.update_layout(**layout(
            title=dict(text="⚡ Training Load", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
            height=320,
        ))
        st.plotly_chart(fig4, use_container_width=True)

    # Cadenza + Falcata
    c5, c6 = st.columns(2)
    df_cad = df.dropna(subset=["avg_cadence"])
    df_cad = df_cad[df_cad["distance_km"] >= 3.0]
    if not df_cad.empty:
        with c5:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=df_cad["date"], y=df_cad["avg_cadence_spm"],
                mode="lines+markers", name="Cadence",
                line=dict(color="#10b981", width=3),
                marker=dict(size=7, color="#10b981", line=dict(color="#080a13", width=1)),
                hovertemplate="Cadence: %{y:.0f} spm<extra></extra>",
            ))
            # 180 spm reference line
            fig5.add_hline(y=180, line_dash="dash", line_color="#f43f5e",
                           annotation_text="Target 180 spm", annotation_position="top left",
                           annotation_font=dict(color="#f43f5e", size=10))
            fig5.update_layout(**layout(
                title=dict(text="👟 Cadence (steps/min)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=300,
            ))
            st.plotly_chart(fig5, use_container_width=True)

    df_str = df.dropna(subset=["avg_stride_length"])
    df_str = df_str[df_str["distance_km"] >= 3.0]
    if not df_str.empty:
        with c6:
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=df_str["date"], y=df_str["avg_stride_length"],
                mode="lines+markers", name="Stride Length",
                line=dict(color="#06b6d4", width=3),
                marker=dict(size=7, color="#06b6d4", line=dict(color="#080a13", width=1)),
                hovertemplate="Stride: %{y:.0f} cm<extra></extra>",
            ))
            fig6.update_layout(**layout(
                title=dict(text="📐 Stride Length (cm)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=300,
            ))
            st.plotly_chart(fig6, use_container_width=True)

    # ── Scatter: Distance vs Avg Speed ────────────────────────────────────────
    st.markdown('<div class="section-title">🔵 Activity Map — Distance vs Speed</div>', unsafe_allow_html=True)

    df_scatter = df.dropna(subset=["avg_pace", "distance_km"]).copy()
    # Compute avg speed in km/h from pace (min/km)
    df_scatter["avg_speed_kmh"] = 60.0 / df_scatter["avg_pace"]

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=df_scatter["distance_km"],
        y=df_scatter["avg_speed_kmh"],
        mode="markers",
        name="Activity",
        marker=dict(
            size=12,
            color=df_scatter["avg_speed_kmh"],
            colorscale=[[0, "#4f46e5"], [0.5, "#818cf8"], [1, "#06b6d4"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="km/h", font=dict(color="#94a3b8", size=11, family="Inter, sans-serif")),
                tickfont=dict(color="#94a3b8", size=10, family="Inter, sans-serif"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.05)",
                thickness=12,
                len=0.7,
            ),
            line=dict(color="#080a13", width=1.5),
            opacity=0.9,
        ),
        customdata=list(zip(
            df_scatter["date_str"],
            df_scatter["distance_km"],
            df_scatter["avg_speed_kmh"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Distance: %{customdata[1]:.2f} km<br>"
            "Speed: %{customdata[2]:.2f} km/h"
            "<extra></extra>"
        ),
    ))
    fig_sc.update_layout(**layout(
        title=dict(
            text="🔵 Distance vs Avg Speed per activity",
            font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15),
            x=0,
        ),
        xaxis=dict(
            **_AXIS,
            title=dict(text="Distance (km)", font=dict(color="#94a3b8", size=12, family="Inter, sans-serif")),
        ),
        yaxis=dict(
            **_AXIS,
            title=dict(text="Avg Speed (km/h)", font=dict(color="#94a3b8", size=12, family="Inter, sans-serif")),
        ),
        hovermode="closest",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
    ))
    st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SETTIMANALE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if weekly.empty:
        st.info("No weekly data available.")
    else:
        st.markdown('<div class="section-title">Weekly Summary</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly["week_label"], y=weekly["total_km"],
                marker=dict(
                    color=weekly["total_km"],
                    colorscale=[[0, "#6366f1"], [1, "#a855f7"]],
                    showscale=False,
                ),
                text=weekly["total_km"].apply(lambda x: f"{x:.1f}"),
                textposition="outside",
                textfont=dict(color="#f8fafc", family="Inter, sans-serif", size=10),
                hovertemplate="<b>%{x}</b><br>%{y:.1f} km<extra></extra>",
            ))
            fig.update_layout(**layout(
                title=dict(text="📏 Weekly kilometres", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=320,
            ))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=weekly["week_label"], y=weekly["n_runs"],
                marker=dict(color="#06b6d4"),
                text=weekly["n_runs"],
                textposition="outside",
                textfont=dict(color="#f8fafc", family="Inter, sans-serif", size=10),
            ))
            fig2.update_layout(**layout(
                title=dict(text="🏃 Runs per week", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=320,
                yaxis=dict(**_AXIS, dtick=1),
            ))
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig3 = go.Figure()
            w_pace = weekly.dropna(subset=["avg_pace"])
            fig3.add_trace(go.Scatter(
                x=w_pace["week_label"], y=w_pace["avg_pace"],
                mode="lines+markers+text",
                line=dict(color="#a855f7", width=3),
                marker=dict(size=8, color="#a855f7", line=dict(color="#080a13", width=1)),
                text=w_pace["avg_pace_str"],
                textposition="top center",
                textfont=dict(size=10, color="#f8fafc", family="Inter, sans-serif"),
            ))
            fig3.update_layout(**layout(
                title=dict(text="⏱️ Weekly avg pace", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                yaxis=dict(**_AXIS, autorange="reversed"),
                height=300,
            ))
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            w_load = weekly.dropna(subset=["total_load"])
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=w_load["week_label"], y=w_load["total_load"],
                marker=dict(
                    color=w_load["total_load"],
                    colorscale=[[0, "#f59e0b"], [1, "#f43f5e"]],
                    showscale=False,
                ),
            ))
            fig4.update_layout(**layout(
                title=dict(text="⚡ Weekly Training Load", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=300,
            ))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown(
            '<div class="section-title">🚀 Fastest consecutive segments (Garmin splits)</div>',
            unsafe_allow_html=True,
        )
        if weekly_segments.empty:
            st.info(
                "No split data yet. Click **Refresh Garmin data** in Settings to download "
                "km-by-km splits (needed for best 1 / 5 / 10 km per week)."
            )
        else:
            fig_seg = go.Figure()
            seg_styles = {
                1: {"color": "#06b6d4", "name": "Best 1 km"},
                5: {"color": "#a855f7", "name": "Best 5 km"},
                10: {"color": "#f43f5e", "name": "Best 10 km"},
            }
            for km, style in seg_styles.items():
                w_seg = weekly_segments[weekly_segments["segment_km"] == km]
                if w_seg.empty:
                    continue
                fig_seg.add_trace(
                    go.Scatter(
                        x=w_seg["week_label"],
                        y=w_seg["pace_min_km"],
                        mode="lines+markers+text",
                        name=style["name"],
                        line=dict(color=style["color"], width=3),
                        marker=dict(size=8, color=style["color"], line=dict(color="#080a13", width=1)),
                        text=w_seg["pace_str"].str.replace(" /km", "", regex=False),
                        textposition="top center",
                        textfont=dict(size=9, color="#f8fafc", family="Inter, sans-serif"),
                        customdata=list(zip(w_seg["pace_str"], w_seg["time_str"])),
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>"
                            f"{style['name']}: %{{customdata[0]}}<br>"
                            f"Time: %{{customdata[1]}}"
                            "<extra></extra>"
                        ),
                    )
                )
            fig_seg.update_layout(
                **layout(
                    title=dict(
                        text="⏱️ Weekly best pace — consecutive 1 / 5 / 10 km",
                        font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15),
                        x=0,
                    ),
                    yaxis=dict(**_AXIS, autorange="reversed", title=dict(text="Pace (min/km)")),
                    height=380,
                    legend=dict(
                        bgcolor="rgba(0,0,0,0)",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        # ── Delta charts ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Weekly volume & fastest splits</div>', unsafe_allow_html=True)

        def _delta_bar(series_pct, label, color_pos, color_neg, title, x_labels=None):
            """Barre Δ%: (valore corrente − settimana precedente) / settimana precedente × 100."""
            x = x_labels if x_labels is not None else weekly["week_label"].iloc[1:]
            colors = [color_pos if v >= 0 else color_neg for v in series_pct]
            text_vals = [f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%" for v in series_pct]
            fig = go.Figure(go.Bar(
                x=x,
                y=series_pct,
                marker=dict(color=colors, opacity=0.85),
                text=text_vals,
                textposition="outside",
                textfont=dict(color="#f8fafc", size=10, family="Inter, sans-serif"),
                hovertemplate="<b>%{x}</b><br>" + label + ": %{text}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
            fig.update_layout(**layout(
                title=dict(text=title, font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0),
                height=300,
                yaxis=dict(**_AXIS, ticksuffix="%"),
            ))
            return fig

        def _segment_pace_delta_bar(seg_df: pd.DataFrame, km: int, title: str):
            w = (
                seg_df[seg_df["segment_km"] == km]
                .sort_values("week_start")
                .dropna(subset=["pct_delta"])
            )
            if w.empty:
                return None
            # pct_delta: negativo = più veloce; invertiamo solo il segno per colori (↑ verde = migliorato)
            display = -w["pct_delta"]
            return _delta_bar(
                display,
                "Δ Pace",
                "#10b981",
                "#f43f5e",
                title,
                x_labels=w["week_label"].tolist(),
            )

        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            if "pct_delta_total_km" in weekly.columns:
                delta_km = weekly["pct_delta_total_km"].iloc[1:].fillna(0)
                st.plotly_chart(_delta_bar(
                    delta_km, "Δ Volume",
                    "#10b981", "#f43f5e",
                    "📏 Δ% Weekly volume",
                ), use_container_width=True)

        with cd2:
            fig_1k = _segment_pace_delta_bar(
                weekly_segments, 1, "⏱️ Δ% Fastest 1 km (↑ = faster)"
            )
            if fig_1k:
                st.plotly_chart(fig_1k, use_container_width=True)
            else:
                st.caption("Need at least 2 weeks of 1 km split data.")

        with cd3:
            fig_5k = _segment_pace_delta_bar(
                weekly_segments, 5, "⏱️ Δ% Fastest 5 km (↑ = faster)"
            )
            if fig_5k:
                st.plotly_chart(fig_5k, use_container_width=True)
            else:
                st.caption("Need at least 2 weeks of 5 km split data.")

        # Summary table
        st.markdown('<div class="section-title">Weekly progression & load table</div>', unsafe_allow_html=True)
        
        display_weekly = weekly[[
            "week_label", "n_runs", 
            "total_km", "delta_total_km_str", 
            "max_distance", "delta_max_dist_str",
            "top2_pace_str", "delta_top2_pace_str",
            "avg_pace_str", "avg_hr", "total_duration"
        ]].copy()
        
        display_weekly.columns = [
            "Week", "Runs", 
            "Total km", "Δ% Vol", 
            "Longest run", "Δ% Longest",
            "Top-2 Pace (≥5k)", "Δ% Top-2 Pace",
            "Avg pace", "Avg HR", "Total time"
        ]
        
        display_weekly["Total km"] = display_weekly["Total km"].apply(
            lambda x: f"{x:.1f} km" if pd.notna(x) else "—"
        )
        display_weekly["Longest run"] = display_weekly["Longest run"].apply(
            lambda x: f"{x:.2f} km" if pd.notna(x) else "—"
        )
        display_weekly["Total time"] = display_weekly["Total time"].apply(
            lambda x: _format_duration(x) if pd.notna(x) else "—"
        )
        display_weekly["Avg HR"] = display_weekly["Avg HR"].apply(
            lambda x: f"{x:.0f} bpm" if pd.notna(x) else "—"
        )
        st.dataframe(display_weekly, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ATTIVITÀ
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">All runs · most recent first</div>', unsafe_allow_html=True)

    df_disp = df.sort_values("date", ascending=False).copy()

    for _, row in df_disp.iterrows():
        pr_badge = '<span class="pr-badge">🏆 PR</span>' if row.get("is_pr") else ""
        location = f" · {row['location']}" if row.get("location") else ""
        dist = f"{row['distance_km']:.2f} km" if pd.notna(row.get("distance_km")) else "—"
        dur = _format_duration(row.get("duration_min"))
        pace = row.get("avg_pace_str", "—")
        avg_hr_str = f"{int(row['avg_hr'])} bpm" if pd.notna(row.get("avg_hr")) else "—"
        elev = f"+{int(row['elevation_gain'])}m" if pd.notna(row.get("elevation_gain")) else ""

        with st.expander(f"**{row['date_str']}** — {row['name']}{pr_badge}  ·  {dist}  ·  {pace}"):
            ca, cb, cc, cd, ce = st.columns(5)
            ca.metric("Distance", dist)
            cb.metric("Duration", dur)
            cc.metric("Pace", pace)
            cd.metric("Avg HR", avg_hr_str)
            ce.metric("Elevation", elev if elev else "—")

            if pd.notna(row.get("training_load")):
                cf, cg, ch = st.columns(3)
                cf.metric("Training Load", f"{row['training_load']:.0f}")
                cg.metric("Aerobic TE", f"{row['aerobic_te']:.1f}" if pd.notna(row.get("aerobic_te")) else "—")
                ch.metric("Cadence", f"{int(row['avg_cadence_spm'])} spm" if pd.notna(row.get("avg_cadence_spm")) and row.get("avg_cadence_spm") else "—")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ZONE HR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    zone_labels = get_zone_labels()
    zone_cols = [f"hr_zone_{i}" for i in range(7)]
    zone_names = [zone_labels[c] for c in zone_cols]

    # Totale per zona (ore)
    zone_totals = {zone_labels[c]: df[c].sum() / 60 for c in zone_cols if c in df.columns}

    df_zones = pd.DataFrame(list(zone_totals.items()), columns=["Zone", "Hours"])
    df_zones = df_zones[df_zones["Hours"] > 0]

    ZONE_COLORS = ["#64748b", "#3b82f6", "#10b981", "#eab308", "#f97316", "#ef4444", "#a855f7"]

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=df_zones["Zone"],
            values=df_zones["Hours"],
            hole=0.6,
            marker=dict(colors=ZONE_COLORS[:len(df_zones)], line=dict(color="#080a13", width=2)),
            textinfo="percent",
            textfont=dict(color="#f8fafc", size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>%{value:.2f} h (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            title=dict(text="⏱️ Total HR zone distribution", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0, y=0.95),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=9, family="Inter, sans-serif"),
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            height=400,
            margin=dict(l=0, r=0, t=55, b=40),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Stacked bar per corsa
        df_z = df.copy()
        fig_stack = go.Figure()
        for i, (col, name, color) in enumerate(zip(zone_cols, zone_names, ZONE_COLORS)):
            if col in df_z.columns:
                vals = df_z[col].fillna(0) / 60  # in minuti
                fig_stack.add_trace(go.Bar(
                    name=name,
                    x=df_z["date_str"],
                    y=vals,
                    marker=dict(color=color),
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}} min<extra></extra>",
                ))
        fig_stack.update_layout(**layout(
            barmode="stack",
            title=dict(text="❤️ HR zones per run (min)", font=dict(family="Outfit, sans-serif", color="#f8fafc", size=15), x=0, y=0.95),
            height=400,
            margin=dict(l=0, r=0, t=55, b=20),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, family="Inter, sans-serif"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(**_AXIS, tickangle=-45),
        ))
        st.plotly_chart(fig_stack, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MAPPA (HEATMAP)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">🗺️ Geographic Heatmap</div>', unsafe_allow_html=True)
    
    polylines = load_polylines()
    
    if not polylines:
        st.info("No GPS tracks available. Click 'Refresh Garmin data' below to download them.")
    else:
        df_heat = _heatmap_from_polylines(polylines)

        start = _last_run_start_lat_lon(polylines, df)
        if start:
            center_lat, center_lon = start
        else:
            center_lon = df_heat["lon"].mean()
            center_lat = df_heat["lat"].mean()

        layers = [
            pdk.Layer(
                "HeatmapLayer",
                df_heat,
                get_position=["lon", "lat"],
                get_weight="weight",
                radius_pixels=28,
                intensity=1.2,
                threshold=0.12,
                color_range=_HEATMAP_COLORS,
            ),
        ]
        if start:
            df_start = pd.DataFrame([{"lon": center_lon, "lat": center_lat}])
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    df_start,
                    get_position=["lon", "lat"],
                    get_fill_color=[34, 197, 94, 220],
                    get_radius=60,
                    pickable=True,
                )
            )

        map_zoom = _zoom_for_radius_km(center_lat, radius_km=20)

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=map_zoom,
            pitch=0,
        )

        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            **_map_deck_kwargs(),
        )

        st.pydeck_chart(r, height=560)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — GESTIONE (ESCLUSIONI E ATTIVITÀ MANUALI)
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">Manage Activities (Exclude / Add)</div>', unsafe_allow_html=True)
    
    col_esc, col_man = st.columns(2)
    
    overrides = load_overrides()
    
    with col_esc:
        st.subheader("🚫 Exclude Garmin Activities")
        
        # Load all running activities without exclusion filters
        all_raw_acts = load_all_activities()
        # Filter only running from March 2026
        start_ms = TRACKING_START.timestamp() * 1000
        raw_runs = [
            a for a in all_raw_acts
            if a.get("activityType") == "running"
            and a.get("startTimeLocal", 0) >= start_ms
        ]
        
        # Filter out already excluded ones
        excluded_set = set(overrides["excluded_activity_ids"])
        available_to_exclude = [r for r in raw_runs if int(r.get("activityId", 0)) not in excluded_set]
        
        if available_to_exclude:
            # Sort by date descending
            available_to_exclude = sorted(available_to_exclude, key=lambda x: x.get("startTimeLocal", 0), reverse=True)
            
            # Build readable options
            options = []
            for r in available_to_exclude:
                r_id = int(r.get("activityId"))
                r_ts = r.get("startTimeLocal")
                r_date = datetime.fromtimestamp(r_ts / 1000).strftime("%d/%m/%Y")
                r_dist = r.get("distance", 0) / 100_000
                r_name = r.get("name", "Run")
                options.append((r_id, f"{r_date} — {r_name} ({r_dist:.2f} km) [ID: {r_id}]"))
                
            selected_act = st.selectbox(
                "Select a Garmin run to hide:",
                options=options,
                format_func=lambda x: x[1]
            )
            
            if st.button("Exclude selected activity", type="secondary"):
                overrides["excluded_activity_ids"].append(selected_act[0])
                save_overrides(overrides)
                st.cache_data.clear()
                st.success("Activity excluded successfully!")
                st.rerun()
        else:
            st.info("No Garmin activities available to exclude.")
            
        # Show currently excluded activities
        st.markdown('<div class="section-title">Currently excluded activities</div>', unsafe_allow_html=True)
        if overrides["excluded_activity_ids"]:
            for esc_id in overrides["excluded_activity_ids"]:
                # Look up details in raw list
                details = next((r for r in raw_runs if int(r.get("activityId", 0)) == esc_id), None)
                if details:
                    r_ts = details.get("startTimeLocal")
                    r_date = datetime.fromtimestamp(r_ts / 1000).strftime("%d/%m/%Y")
                    r_dist = details.get("distance", 0) / 100_000
                    label = f"❌ {r_date} — {details.get('name', 'Run')} ({r_dist:.2f} km)"
                else:
                    label = f"❌ Activity ID: {esc_id}"
                    
                c_lbl, c_btn = st.columns([4, 1])
                c_lbl.write(label)
                if c_btn.button("Restore", key=f"restore_{esc_id}"):
                    overrides["excluded_activity_ids"].remove(esc_id)
                    save_overrides(overrides)
                    st.cache_data.clear()
                    st.success("Activity restored!")
                    st.rerun()
        else:
            st.write("No activities currently excluded.")
            
    with col_man:
        st.subheader("➕ Add Manual Activity")
        with st.form("manual_activity_form", clear_on_submit=True):
            form_date = st.date_input("Run date:", value=datetime.today())
            form_name = st.text_input("Name:", value="Manual run")
            form_dist = st.number_input("Distance (km):", min_value=0.0, step=0.1, format="%.2f")
            form_dur = st.number_input("Duration (minutes):", min_value=0.0, step=1.0, format="%.1f")
            
            form_hr = st.number_input("Avg HR (bpm, optional):", min_value=0, max_value=250, value=0)
            form_elev = st.number_input("Elevation gain (m, optional):", min_value=0, step=5)
            form_cal = st.number_input("Calories (optional):", min_value=0, step=10)
            form_rpe = st.slider("RPE (perceived effort 1-10, optional):", 1, 10, 5)
            
            submitted = st.form_submit_button("Add Activity")
            if submitted:
                if form_dist <= 0 or form_dur <= 0:
                    st.error("Distance and duration must be greater than 0.")
                else:
                    # Calcola passo (min/km)
                    avg_pace = form_dur / form_dist
                    
                    new_act = {
                        "date": form_date.strftime("%Y-%m-%d"),
                        "name": form_name,
                        "distance_km": form_dist,
                        "duration_min": form_dur,
                        "avg_pace": avg_pace,
                        "avg_hr": form_hr if form_hr > 0 else None,
                        "elevation_gain": form_elev if form_elev > 0 else None,
                        "calories": form_cal if form_cal > 0 else None,
                        "rpe": form_rpe
                    }
                    
                    overrides["manual_activities"].append(new_act)
                    save_overrides(overrides)
                    st.cache_data.clear()
                    st.success("Manual activity added!")
                    st.rerun()
                    
        # Show active manual runs
        st.markdown('<div class="section-title">Manual activities added</div>', unsafe_allow_html=True)
        if overrides["manual_activities"]:
            for idx, m in enumerate(overrides["manual_activities"]):
                c_lbl, c_btn = st.columns([4, 1])
                c_lbl.write(f"🏃 {m['date']} — {m['name']} ({m['distance_km']} km in {m['duration_min']} min)")
                if c_btn.button("Remove", key=f"del_manual_{idx}"):
                    overrides["manual_activities"].pop(idx)
                    save_overrides(overrides)
                    st.cache_data.clear()
                    st.success("Manual activity removed!")
                    st.rerun()
        else:
            st.write("No manual activities added yet.")

    # ── Footer / Refresh ──────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
c_empty, c_btn = st.columns([4, 1])
with c_btn:
    if st.button('🔄 Refresh Garmin data', use_container_width=True):
        with st.spinner('🔄 Fetching activities from Garmin...'):
            garmin_client.fetch_garmin_activities(force=True)
            st.cache_data.clear()
            st.success('✅ Garmin data updated')
            st.rerun()
