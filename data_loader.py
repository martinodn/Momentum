"""
data_loader.py — Carica e processa le attività di corsa da Garmin Connect export.
Filtra solo le corse da marzo 2026 in poi.
"""

import json
import os
from datetime import datetime
from typing import Optional
import pandas as pd

# Path relativo ai file JSON di Garmin
ACTIVITIES_DIR = os.path.join(
    os.path.dirname(__file__),
    "activities", "DI_CONNECT", "DI-Connect-Fitness"
)

UPLOADS_DIR = os.path.join(
    os.path.dirname(__file__),
    "activities", "uploads"
)
os.makedirs(UPLOADS_DIR, exist_ok=True)

POLYLINES_DIR = os.path.join(
    os.path.dirname(__file__),
    "activities", "polylines"
)
os.makedirs(POLYLINES_DIR, exist_ok=True)

ACTIVITY_FILES = [
    "martinodn_0_summarizedActivities.json",
    "martinodn_1001_summarizedActivities.json",
    "martinodn_2002_summarizedActivities.json",
    "martinodn_3003_summarizedActivities.json",
]

# Inizio del periodo di tracciamento (rientro dall'infortunio)
TRACKING_START = datetime(2026, 3, 1)

# File di override manuale
OVERRIDES_FILE = os.path.join(os.path.dirname(__file__), "overrides.json")


def load_overrides() -> dict:
    """Carica il file overrides.json con esclusioni e attività manuali."""
    if not os.path.exists(OVERRIDES_FILE):
        return {"excluded_activity_ids": [], "manual_activities": []}
    with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "excluded_activity_ids": [int(x) for x in data.get("excluded_activity_ids", [])],
        "manual_activities": data.get("manual_activities", []),
    }


def save_overrides(overrides: dict) -> None:
    """Salva il file overrides.json."""
    existing = load_overrides()
    existing.update(overrides)
    with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def load_polylines() -> list:
    """Carica tutte le tracce GPS salvate nella cartella polylines.
    Ritorna una lista di dizionari con { 'activityId': ..., 'path': [[lon, lat], ...] }
    """
    polylines = []
    if not os.path.exists(POLYLINES_DIR):
        return polylines
        
    for fname in os.listdir(POLYLINES_DIR):
        if fname.endswith(".json"):
            try:
                activity_id = int(fname.split(".")[0])
                with open(os.path.join(POLYLINES_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and isinstance(data, list) and len(data) > 0:
                    polylines.append({
                        "activityId": activity_id,
                        "path": data
                    })
            except Exception:
                continue
    return polylines


def _ms_to_minutes(ms: Optional[float]) -> Optional[float]:
    """Converte millisecondi in minuti."""
    if ms is None:
        return None
    return ms / 60_000


def _cm_to_km(cm: Optional[float]) -> Optional[float]:
    """Converte centimetri in chilometri (il formato nativo di Garmin export)."""
    if cm is None:
        return None
    return cm / 100_000


def _cmps_to_pace(cmps: Optional[float]) -> Optional[float]:
    """
    Converte velocità Garmin (cm/ms) in passo min/km.
    La conversione è: cm/ms × 10 = m/s
    Restituisce None se la velocità è 0 o None.
    """
    if not cmps or cmps <= 0:
        return None
    mps = cmps * 10  # cm/ms × 10 = m/s
    return 1000 / (mps * 60)  # min/km


def _format_pace(pace_min_km: Optional[float]) -> str:
    """Formatta un passo float (min/km) come stringa 'MM:SS /km'."""
    if pace_min_km is None:
        return "—"
    minutes = int(pace_min_km)
    seconds = int((pace_min_km - minutes) * 60)
    return f"{minutes}:{seconds:02d} /km"


def pct_change_vs_previous(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Δ% settimana su settimana: (corrente − precedente) / precedente × 100."""
    if current is None or previous is None:
        return None
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return None
    return (current - previous) / previous * 100


def _format_duration(minutes: Optional[float]) -> str:
    """Formatta durata in minuti come stringa 'Xh YYm' o 'YYm'."""
    if minutes is None:
        return "—"
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def load_all_activities() -> list[dict]:
    """Carica tutte le attività dai file JSON di Garmin e dalla cartella uploads."""
    all_activities = []
    
    # 1. Carica solo i file caricati dall'utente (Garmin)
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            if filename.endswith('.json'):
                path = os.path.join(UPLOADS_DIR, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and "summarizedActivitiesExport" in data[0]:
                        activities = data[0].get("summarizedActivitiesExport", [])
                        all_activities.extend(activities)
                    elif isinstance(data, dict) and "summarizedActivitiesExport" in data:
                        activities = data.get("summarizedActivitiesExport", [])
                        all_activities.extend(activities)
                    elif isinstance(data, list):
                        all_activities.extend(data)
                except Exception as e:
                    print(f"Errore nel caricamento del file {filename}: {e}")
                    
    return all_activities


def _parse_fit_file(file_path: str) -> Optional[dict]:
    """Decodifica un file .fit e restituisce un dizionario compatibile con la riga del DataFrame."""
    try:
        from fitparse import FitFile
        fitfile = FitFile(file_path)
        
        # Cerca il record sessione (contiene i riepiloghi)
        sessions = list(fitfile.get_messages("session"))
        if not sessions:
            return None
        
        session = sessions[0]
        data = session.get_values()
        
        # Controlla se lo sport è running
        sport = data.get("sport")
        if sport != "running":
            return None
            
        start_time_utc = data.get("start_time")
        if not start_time_utc:
            return None
            
        # Converti UTC in locale (assumendo il fuso locale della macchina)
        from datetime import timezone
        if start_time_utc.tzinfo is None:
            ts = start_time_utc.replace(tzinfo=timezone.utc).timestamp()
        else:
            ts = start_time_utc.timestamp()
            
        date = datetime.fromtimestamp(ts)
        
        # Filtra per data inizio tracking (marzo 2026)
        if date < TRACKING_START:
            return None
            
        total_dist = data.get("total_distance")  # in metri
        dist_km = (total_dist / 1000.0) if total_dist else 0.0
        
        total_timer = data.get("total_timer_time")  # in secondi
        if not total_timer:
            total_timer = data.get("total_elapsed_time", 0.0)
            
        dur_min = total_timer / 60.0
        
        avg_speed = data.get("avg_speed")  # in m/s
        avg_pace = (1000.0 / (avg_speed * 60.0)) if (avg_speed and avg_speed > 0) else (dur_min / dist_km if dist_km > 0 else None)
        
        max_speed = data.get("max_speed")  # in m/s
        max_pace = (1000.0 / (max_speed * 60.0)) if (max_speed and max_speed > 0) else None
        
        avg_hr = data.get("avg_heart_rate")
        max_hr = data.get("max_heart_rate")
        
        elev_gain = data.get("total_ascent")  # in metri
        elev_loss = data.get("total_descent")  # in metri
        
        avg_cadence = data.get("avg_cadence")
        avg_cadence_spm = (avg_cadence * 2) if avg_cadence else None
        
        # Calcolo approssimativo stride length in cm se non presente
        avg_stride_length = None
        if avg_speed and avg_cadence:
            avg_stride_length = round((avg_speed * 3000.0) / avg_cadence, 1)
            
        calories = data.get("total_calories")
        training_load = data.get("total_training_effect")
        
        # Training Effect
        aerobic_te = data.get("total_training_effect")
        anaerobic_te = data.get("total_anaerobic_training_effect")
        
        hr_zones = {f"hr_zone_{i}": None for i in range(7)}
        
        # Nome dell'attività basato sul momento della giornata
        hour = date.hour
        if 5 <= hour < 12:
            time_of_day = "Corsa mattutina"
        elif 12 <= hour < 17:
            time_of_day = "Corsa pomeridiana"
        elif 17 <= hour < 22:
            time_of_day = "Corsa serale"
        else:
            time_of_day = "Corsa notturna"
            
        activity_id = int(ts)
        
        row = {
            "activity_id": activity_id,
            "name": time_of_day,
            "date": date,
            "date_str": date.strftime("%d/%m/%Y"),
            "week": date.isocalendar()[1],
            "week_label": f"W{date.isocalendar()[1]}",
            "month": date.strftime("%b %Y"),
            "location": "",
            
            "distance_km": round(dist_km, 2),
            "duration_min": round(dur_min, 1),
            "moving_min": round(dur_min, 1),
            "avg_pace": round(avg_pace, 3) if avg_pace else None,
            "avg_pace_str": _format_pace(avg_pace),
            "max_pace": round(max_pace, 3) if max_pace else None,
            
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "min_hr": None,
            
            "elevation_gain": round(elev_gain) if elev_gain is not None else None,
            "elevation_loss": round(elev_loss) if elev_loss is not None else None,
            
            "avg_cadence": avg_cadence,
            "avg_cadence_spm": avg_cadence_spm,
            "avg_stride_length": avg_stride_length,
            
            "calories": round(calories) if calories is not None else None,
            "training_load": training_load,
            "aerobic_te": aerobic_te,
            "anaerobic_te": anaerobic_te,
            "moderate_intensity_min": None,
            "vigorous_intensity_min": None,
            "vo2max": None,
            "is_pr": False,
            "workout_rpe": None,
            "workout_feel": None,
            **hr_zones
        }
        return row
    except Exception as e:
        print(f"Errore nel parsing del file FIT {file_path}: {e}")
        return None


def load_running_activities() -> pd.DataFrame:
    """
    Carica, filtra e processa le attività di corsa da marzo 2026 in poi.
    Restituisce un DataFrame pandas pronto all'uso.
    """
    all_activities = load_all_activities()

    # Timestamp di inizio tracking in ms
    start_ms = TRACKING_START.timestamp() * 1000

    overrides = load_overrides()
    excluded_ids = set(overrides["excluded_activity_ids"])

    # Filtra solo corse dal periodo di rientro, escludendo quelle eliminate
    runs = [
        a for a in all_activities
        if a.get("activityType") == "running"
        and a.get("startTimeLocal", 0) >= start_ms
        and int(a.get("activityId", 0)) not in excluded_ids
    ]

    rows = []
    
    # 1. Carica attività da file .fit in uploads
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            if filename.lower().endswith(".fit"):
                fit_path = os.path.join(UPLOADS_DIR, filename)
                fit_row = _parse_fit_file(fit_path)
                if fit_row and int(fit_row["activity_id"]) not in excluded_ids:
                    rows.append(fit_row)

    if not runs and not rows and not overrides["manual_activities"]:
        return pd.DataFrame()

    for a in runs:
        ts = a.get("startTimeLocal")
        date = datetime.fromtimestamp(ts / 1000) if ts else None

        duration_min = _ms_to_minutes(a.get("duration"))
        elapsed_min = _ms_to_minutes(a.get("elapsedDuration"))
        moving_min = _ms_to_minutes(a.get("movingDuration"))

        dist_km = _cm_to_km(a.get("distance"))
        avg_pace = _cmps_to_pace(a.get("avgSpeed"))
        max_pace = _cmps_to_pace(a.get("maxSpeed"))

        # HR zones in minuti
        hr_zones = {
            f"hr_zone_{i}": _ms_to_minutes(a.get(f"hrTimeInZone_{i}"))
            for i in range(7)
        }

        row = {
            # Identificazione
            "activity_id": a.get("activityId"),
            "name": a.get("name", "Corsa"),
            "date": date,
            "date_str": date.strftime("%d/%m/%Y") if date else "",
            "week": date.isocalendar()[1] if date else None,
            "week_label": f"W{date.isocalendar()[1]}" if date else "",
            "month": date.strftime("%b %Y") if date else "",
            "location": a.get("locationName", ""),

            # Metriche principali
            "distance_km": round(dist_km, 2) if dist_km else None,
            "duration_min": round(duration_min, 1) if duration_min else None,
            "moving_min": round(moving_min, 1) if moving_min else None,
            "avg_pace": round(avg_pace, 3) if avg_pace else None,
            "avg_pace_str": _format_pace(avg_pace),
            "max_pace": round(max_pace, 3) if max_pace else None,

            # Frequenza cardiaca
            "avg_hr": a.get("avgHr"),
            "max_hr": a.get("maxHr"),
            "min_hr": a.get("minHr"),

            # Elevazione: cm → m
            "elevation_gain": round(a.get("elevationGain", 0) / 100) if a.get("elevationGain") else None,
            "elevation_loss": round(a.get("elevationLoss", 0) / 100) if a.get("elevationLoss") else None,

            # Dati biomeccanici
            "avg_cadence": a.get("avgRunCadence"),  # passi/min (un piede)
            "avg_cadence_spm": (a.get("avgRunCadence") or 0) * 2 or None,  # passi/min totali
            "avg_stride_length": round(a.get("avgStrideLength", 0), 1) if a.get("avgStrideLength") else None,  # già in cm

            # Training
            "calories": round(a.get("calories", 0)) if a.get("calories") else None,
            "training_load": a.get("activityTrainingLoad"),
            "aerobic_te": a.get("aerobicTrainingEffect"),
            "anaerobic_te": a.get("anaerobicTrainingEffect"),
            "moderate_intensity_min": a.get("moderateIntensityMinutes"),
            "vigorous_intensity_min": a.get("vigorousIntensityMinutes"),
            "vo2max": a.get("vO2MaxValue"),
            "is_pr": a.get("pr", False),

            # RPE e feeling (se disponibili dai workout Garmin)
            "workout_rpe": a.get("workoutRpe"),
            "workout_feel": a.get("workoutFeel"),

            # HR zones (min)
            **hr_zones,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Aggiunge attività manuali da overrides.json
    manual = overrides.get("manual_activities", [])
    if manual:
        manual_rows = []
        for m in manual:
            try:
                date = datetime.strptime(m["date"], "%Y-%m-%d")
            except Exception:
                continue
            avg_pace_val = m.get("avg_pace")  # già in min/km float
            manual_rows.append({
                "activity_id": f"manual_{m['date']}",
                "name": m.get("name", "Corsa manuale"),
                "date": date,
                "date_str": date.strftime("%d/%m/%Y"),
                "week": date.isocalendar()[1],
                "week_label": f"W{date.isocalendar()[1]}",
                "month": date.strftime("%b %Y"),
                "location": m.get("location", ""),
                "distance_km": m.get("distance_km"),
                "duration_min": m.get("duration_min"),
                "moving_min": m.get("duration_min"),
                "avg_pace": avg_pace_val,
                "avg_pace_str": _format_pace(avg_pace_val),
                "max_pace": None,
                "avg_hr": m.get("avg_hr"),
                "max_hr": m.get("max_hr"),
                "min_hr": None,
                "elevation_gain": m.get("elevation_gain"),
                "elevation_loss": None,
                "avg_cadence": None,
                "avg_cadence_spm": None,
                "avg_stride_length": None,
                "calories": m.get("calories"),
                "training_load": None,
                "aerobic_te": None,
                "anaerobic_te": None,
                "moderate_intensity_min": None,
                "vigorous_intensity_min": None,
                "vo2max": None,
                "is_pr": False,
                "workout_rpe": m.get("rpe"),
                "workout_feel": None,
                **{f"hr_zone_{i}": None for i in range(7)},
            })
        if manual_rows:
            df = pd.concat([df, pd.DataFrame(manual_rows)], ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)
    df["run_number"] = range(1, len(df) + 1)
    return df


def get_weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggrega le statistiche settimanali."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Normalizza la data a inizio settimana (lunedì a mezzanotte) per raggruppare correttamente
    df["week_start"] = df["date"].apply(
        lambda d: pd.Timestamp(d).normalize() - pd.Timedelta(days=pd.Timestamp(d).weekday()) if pd.notna(d) else None
    )

    # 1. Calcola il passo medio delle 2 attività più veloci (almeno 5 km) per settimana
    runs_5k = df[(df["distance_km"] >= 5.0) & (df["avg_pace"].notna())].copy()
    top2_series = runs_5k.groupby("week_start")["avg_pace"].apply(lambda x: x.nsmallest(2).mean())

    # 2. Aggregazione settimanale principale
    weekly = df.groupby("week_start").agg(
        n_runs=("activity_id", "count"),
        total_km=("distance_km", "sum"),
        max_distance=("distance_km", "max"),
        avg_hr=("avg_hr", "mean"),
        total_duration=("duration_min", "sum"),
        total_load=("training_load", "sum"),
        total_elevation=("elevation_gain", "sum"),
    ).reset_index()

    # Unisci il passo top 2
    weekly = weekly.merge(top2_series.rename("top2_pace"), on="week_start", how="left")

    # Calcola il passo medio settimanale reale (minuti totali / km totali)
    weekly["avg_pace"] = weekly.apply(
        lambda r: r["total_duration"] / r["total_km"] if r["total_km"] > 0 else None,
        axis=1
    )

    # Ordina per data per garantire shift coerenti di settimana in settimana
    weekly = weekly.sort_values("week_start").reset_index(drop=True)

    # 3. Calcolo del Delta in % di settimana in settimana
    weekly["prev_total_km"] = weekly["total_km"].shift(1)
    weekly["pct_delta_total_km"] = weekly.apply(
        lambda r: pct_change_vs_previous(r["total_km"], r["prev_total_km"]),
        axis=1,
    )

    weekly["prev_max_dist"] = weekly["max_distance"].shift(1)
    weekly["pct_delta_max_dist"] = weekly.apply(
        lambda r: pct_change_vs_previous(r["max_distance"], r["prev_max_dist"]),
        axis=1,
    )

    weekly["prev_top2_pace"] = weekly["top2_pace"].shift(1)
    weekly["pct_delta_top2_pace"] = weekly.apply(
        lambda r: pct_change_vs_previous(r["top2_pace"], r["prev_top2_pace"]),
        axis=1,
    )

    # Formattazione stringhe per display
    weekly["avg_pace_str"] = weekly["avg_pace"].apply(_format_pace)
    weekly["top2_pace_str"] = weekly["top2_pace"].apply(lambda x: _format_pace(x) if pd.notna(x) else "—")

    # Formattatori emoji per i delta
    def fmt_delta(val, is_pace=False):
        if pd.isna(val) or val is None:
            return "—"
        sign = "+" if val >= 0 else ""
        if is_pace:
            # Per il ritmo, negativo significa velocizzato (minore tempo al km)
            emoji = "🚀" if val < 0 else "🐌"
            return f"{sign}{val:.1f}% {emoji}"
        else:
            emoji = "📈" if val > 0 else "📉"
            return f"{sign}{val:.1f}% {emoji}"

    weekly["delta_total_km_str"] = weekly["pct_delta_total_km"].apply(lambda x: fmt_delta(x, False))
    weekly["delta_max_dist_str"] = weekly["pct_delta_max_dist"].apply(lambda x: fmt_delta(x, False))
    weekly["delta_top2_pace_str"] = weekly["pct_delta_top2_pace"].apply(lambda x: fmt_delta(x, True))

    weekly["week_label"] = weekly["week_start"].apply(
        lambda d: d.strftime("%d %b") if pd.notna(d) else ""
    )
    weekly["total_km"] = weekly["total_km"].round(1)
    weekly["max_distance"] = weekly["max_distance"].round(2)
    weekly["avg_hr"] = weekly["avg_hr"].round(0)

    return weekly


def get_zone_labels() -> dict:
    """Etichette delle zone HR."""
    return {
        "hr_zone_0": "Z0 – Recupero",
        "hr_zone_1": "Z1 – Facile",
        "hr_zone_2": "Z2 – Aerobico",
        "hr_zone_3": "Z3 – Moderato",
        "hr_zone_4": "Z4 – Soglia",
        "hr_zone_5": "Z5 – VO2max",
        "hr_zone_6": "Z6 – Anaerobico",
    }


# Limiti superiori (% FCmax), stile predefinito Garmin Connect — 7 zone
_HR_ZONE_PCT_MAX = [0.52, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]


def resolve_max_hr(df: pd.DataFrame, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    if not df.empty and "max_hr" in df.columns and df["max_hr"].notna().any():
        return int(df["max_hr"].max())
    return 192


def get_hr_zone_ranges(
    max_hr: int,
    custom_upper_bpm: Optional[list[int]] = None,
) -> list[dict]:
    """Intervalli BPM per zona 0-6 (in ordine)."""
    labels = get_zone_labels()
    if custom_upper_bpm and len(custom_upper_bpm) >= 7:
        uppers = [int(v) for v in custom_upper_bpm[:7]]
    else:
        uppers = [min(max_hr, int(round(max_hr * pct))) for pct in _HR_ZONE_PCT_MAX]

    zones = []
    lo_bpm = 0
    for i in range(7):
        hi_bpm = min(max_hr, uppers[i])
        if i == 0:
            range_str = f"< {hi_bpm + 1} bpm" if hi_bpm < max_hr else f"≤ {max_hr} bpm"
        elif i == 6 or hi_bpm >= max_hr:
            range_str = f"≥ {lo_bpm} bpm"
            hi_bpm = max_hr
        else:
            range_str = f"{lo_bpm}–{hi_bpm} bpm"
        zones.append(
            {
                "zone_index": i,
                "col": f"hr_zone_{i}",
                "short_label": labels[f"hr_zone_{i}"],
                "range_str": range_str,
                "display_label": f"{labels[f'hr_zone_{i}']} ({range_str})",
            }
        )
        lo_bpm = hi_bpm + 1
    return zones


def build_hr_zone_totals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ore per zona in ordine Z0→Z6 (anche zone a zero)."""
    rows = []
    for i in range(7):
        col = f"hr_zone_{i}"
        hours = float(df[col].sum() / 60) if col in df.columns else 0.0
        rows.append({"zone_index": i, "Hours": hours})
    return pd.DataFrame(rows)
