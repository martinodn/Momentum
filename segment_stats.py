"""
Calcolo dei migliori tempi su segmenti consecutivi (1 / 5 / 10 km) dagli split Garmin.
"""

import json
import os
from typing import Optional

import pandas as pd

from data_loader import _format_pace

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "activities", "splits")
SEGMENT_KMS = (1, 5, 10)


def load_activity_splits() -> dict[int, list]:
    """activity_id -> lista lapDTO Garmin."""
    splits: dict[int, list] = {}
    if not os.path.exists(SPLITS_DIR):
        return splits
    for fname in os.listdir(SPLITS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            activity_id = int(fname.split(".")[0])
            with open(os.path.join(SPLITS_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            laps = data.get("lapDTOs") if isinstance(data, dict) else data
            if laps:
                splits[activity_id] = laps
        except Exception:
            continue
    return splits


def _parse_lap_dtos(lap_dtos: list) -> list[dict]:
    laps = []
    for lap in lap_dtos:
        dist = lap.get("distance") or 0
        dur = lap.get("movingDuration") or lap.get("duration") or 0
        if dist <= 0 or dur <= 0:
            continue
        laps.append({"distance_m": float(dist), "duration_s": float(dur)})
    return laps


def fastest_segment_seconds(laps: list[dict], target_m: float) -> Optional[float]:
    """Tempo minimo (secondi) su target_m metri consecutivi lungo gli split."""
    if not laps:
        return None
    best = None
    n = len(laps)
    for i in range(n):
        dist = 0.0
        time = 0.0
        for j in range(i, n):
            dist += laps[j]["distance_m"]
            time += laps[j]["duration_s"]
            if dist >= target_m * 0.995:
                if dist > target_m:
                    time = time * (target_m / dist)
                if best is None or time < best:
                    best = time
                break
    return best


def _format_segment_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    if secs == 60:
        minutes += 1
        secs = 0
    return f"{minutes}:{secs:02d}"


def get_weekly_segment_bests(df: pd.DataFrame) -> pd.DataFrame:
    """Miglior passo settimanale su 1 / 5 / 10 km consecutivi (split Garmin)."""
    if df.empty:
        return pd.DataFrame()

    splits_map = load_activity_splits()
    if not splits_map:
        return pd.DataFrame()

    df = df.copy()
    df["week_start"] = df["date"].apply(
        lambda d: pd.Timestamp(d).normalize() - pd.Timedelta(days=pd.Timestamp(d).weekday())
        if pd.notna(d)
        else None
    )

    rows = []
    for _, run in df.iterrows():
        try:
            aid = int(run["activity_id"])
        except (TypeError, ValueError):
            continue
        lap_dtos = splits_map.get(aid)
        if not lap_dtos:
            continue
        laps = _parse_lap_dtos(lap_dtos)
        for km in SEGMENT_KMS:
            secs = fastest_segment_seconds(laps, km * 1000)
            if secs is None:
                continue
            rows.append(
                {
                    "week_start": run["week_start"],
                    "segment_km": km,
                    "pace_min_km": (secs / 60.0) / km,
                    "segment_time_s": secs,
                }
            )

    if not rows:
        return pd.DataFrame()

    weekly = (
        pd.DataFrame(rows)
        .groupby(["week_start", "segment_km"], as_index=False)
        .agg(pace_min_km=("pace_min_km", "min"), segment_time_s=("segment_time_s", "min"))
    )
    weekly["pace_str"] = weekly["pace_min_km"].apply(_format_pace)
    weekly["time_str"] = weekly["segment_time_s"].apply(_format_segment_time)
    weekly["week_label"] = weekly["week_start"].apply(
        lambda d: d.strftime("%d %b") if pd.notna(d) else ""
    )

    weekly = weekly.sort_values(["segment_km", "week_start"]).reset_index(drop=True)
    # Δ% = (settimana corrente − precedente) / precedente × 100 (per ogni distanza)
    weekly["pct_delta"] = (
        weekly.groupby("segment_km")["pace_min_km"].pct_change() * 100
    )

    return weekly.sort_values(["week_start", "segment_km"]).reset_index(drop=True)
