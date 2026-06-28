import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict

import streamlit as st
from garminconnect import Garmin

from data_loader import ACTIVITY_FILES, ACTIVITIES_DIR, UPLOADS_DIR, TRACKING_START, POLYLINES_DIR, RUNNING_TYPES
from segment_stats import SPLITS_DIR

# Path where cache timestamp is stored
CACHE_INFO_PATH = os.path.join(ACTIVITIES_DIR, "garmin_fetch_info.json")


def _load_cache_info() -> Dict:
    """Load cache metadata (last fetch timestamp)."""
    if os.path.exists(CACHE_INFO_PATH):
        try:
            with open(CACHE_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache_info(info: Dict) -> None:
    """Persist cache metadata."""
    os.makedirs(ACTIVITIES_DIR, exist_ok=True)
    with open(CACHE_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f)


def _should_fetch(force: bool = False) -> bool:
    """Decide whether we need to hit Garmin's API."""
    if force:
        return True
    ttl = int(st.secrets.get("garmin", {}).get("cache_ttl", 3600))
    info = _load_cache_info()
    last = info.get("last_fetch", 0)
    return (time.time() - last) > ttl


def _normalize_api_activity(a: Dict) -> Dict:
    """
    Convert a Garmin API activity (from get_activities_by_date) into the
    format expected by data_loader.load_running_activities().

    API units:           Loader units:
      distance: metres   centimetres  (x100)
      duration: seconds  milliseconds (x1000)
      speed:    m/s      cm/ms        (x0.1)
      elevation: metres  centimetres  (x100)
      hr zones: seconds  milliseconds (x1000)
      activityType: dict string
      startTimeLocal: string  milliseconds int
    """
    # activity type
    act_type = a.get("activityType", {})
    type_key = act_type.get("typeKey", "") if isinstance(act_type, dict) else str(act_type)

    # start time: string -> ms int
    start_str = a.get("startTimeLocal", "")
    try:
        dt = datetime.fromisoformat(start_str.replace(" ", "T"))
        start_time_ms = int(dt.timestamp() * 1000)
    except Exception:
        start_time_ms = 0

    # speed: m/s -> cm/ms (multiply by 0.1)
    avg_speed_cmms = (a.get("averageSpeed") or 0) * 0.1
    max_speed_cmms = (a.get("maxSpeed") or 0) * 0.1

    # HR zones: seconds -> milliseconds
    hr_zones = {}
    for i in range(7):
        val = a.get(f"hrTimeInZone_{i}")
        hr_zones[f"hrTimeInZone_{i}"] = (val * 1000) if val is not None else None

    return {
        "activityId":               a.get("activityId"),
        "name":                     a.get("activityName", "Corsa"),
        "startTimeLocal":           start_time_ms,
        "activityType":             type_key,
        "distance":                 (a.get("distance") or 0) * 100,         # m -> cm
        "duration":                 (a.get("duration") or 0) * 1000,        # s -> ms
        "elapsedDuration":          (a.get("elapsedDuration") or 0) * 1000,
        "movingDuration":           (a.get("movingDuration") or 0) * 1000,
        "avgSpeed":                 avg_speed_cmms,
        "maxSpeed":                 max_speed_cmms,
        "avgHr":                    a.get("averageHR"),
        "maxHr":                    a.get("maxHR"),
        "minHr":                    None,
        "avgRunCadence":            a.get("averageRunningCadenceInStepsPerMinute"),
        "avgStrideLength":          a.get("avgStrideLength"),               # already cm
        "elevationGain":            (a.get("elevationGain") or 0) * 100,   # m -> cm
        "elevationLoss":            (a.get("elevationLoss") or 0) * 100,
        "calories":                 a.get("calories"),
        "activityTrainingLoad":     a.get("activityTrainingLoad"),
        "aerobicTrainingEffect":    a.get("aerobicTrainingEffect"),
        "anaerobicTrainingEffect":  a.get("anaerobicTrainingEffect"),
        "moderateIntensityMinutes": a.get("moderateIntensityMinutes"),
        "vigorousIntensityMinutes": a.get("vigorousIntensityMinutes"),
        "vO2MaxValue":              a.get("vO2MaxValue"),
        "pr":                       a.get("pr", False),
        "locationName":             a.get("locationName", ""),
        **hr_zones,
    }


def fetch_activity_polylines(client: Garmin, activities: List[Dict]) -> None:
    """Fetch and cache GPS polylines for each activity if not already cached."""
    os.makedirs(POLYLINES_DIR, exist_ok=True)
    for act in activities:
        activity_id = act.get("activityId")
        if not activity_id:
            continue
            
        poly_path = os.path.join(POLYLINES_DIR, f"{activity_id}.json")
        if os.path.exists(poly_path):
            continue
            
        try:
            details = client.get_activity_details(activity_id)
            polyline = details.get("geoPolylineDTO", {}).get("polyline", [])
            # Extract [lon, lat] pairs
            path = []
            for point in polyline:
                if "lat" in point and "lon" in point:
                    path.append([point["lon"], point["lat"]])
                    
            if path:
                with open(poly_path, "w", encoding="utf-8") as f:
                    json.dump(path, f)
                # Sleep briefly to avoid hammering the Garmin API
                time.sleep(1)
        except Exception as e:
            print(f"Error fetching polyline for {activity_id}: {e}")


def fetch_activity_splits(client: Garmin, activities: List[Dict]) -> None:
    """Scarica e cache gli split km-by-km (lapDTOs) per ogni attività."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for act in activities:
        activity_id = act.get("activityId")
        if not activity_id:
            continue

        split_path = os.path.join(SPLITS_DIR, f"{activity_id}.json")
        if os.path.exists(split_path):
            continue

        try:
            data = client.get_activity_splits(activity_id)
            lap_dtos = data.get("lapDTOs") if isinstance(data, dict) else None
            if lap_dtos:
                with open(split_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching splits for {activity_id}: {e}")


def fetch_garmin_activities(force: bool = False) -> List[Dict]:
    """Login to Garmin Connect, fetch running activities since TRACKING_START,
    normalise them, save to uploads/, and return the list.
    """
    if not _should_fetch(force):
        return []

    # Clear previous uploads to avoid stale data
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    for filename in os.listdir(UPLOADS_DIR):
        if filename.lower().endswith(".json"):
            try:
                os.remove(os.path.join(UPLOADS_DIR, filename))
            except Exception as e:
                print(f"Errore rimozione {filename}: {e}")

    garmin_cfg = st.secrets.get("garmin", {})
    username = garmin_cfg.get("username")
    password = garmin_cfg.get("password")
    if not username or not password:
        raise RuntimeError("Credenziali Garmin mancanti in secrets.toml")

    client = Garmin(username, password)
    client.login()

    # Date range
    start_date_str = (TRACKING_START - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_str = datetime.now().strftime("%Y-%m-%d")

    raw_response = client.get_activities_by_date(start_date_str, end_date_str)

    # API can return a list or a dict
    if isinstance(raw_response, dict) and "summarizedActivitiesExport" in raw_response:
        raw_activities = raw_response["summarizedActivitiesExport"]
    elif isinstance(raw_response, list):
        raw_activities = raw_response
    else:
        raw_activities = []

    # DEBUG: log all raw activity types coming from Garmin before filtering
    from collections import Counter
    raw_type_counts = Counter()
    for a in raw_activities:
        act_type = a.get("activityType", {})
        type_key = act_type.get("typeKey", str(act_type)) if isinstance(act_type, dict) else str(act_type)
        start_str = a.get("startTimeLocal", "")[:10]
        name = a.get("activityName", "")
        raw_type_counts[type_key] += 1
        print(f"[Garmin RAW] {start_str} | typeKey={type_key!r} | name={name!r}")
    print(f"[Garmin RAW] Riepilogo tipi: {dict(raw_type_counts)}")

    # Filter to running-type activities only (running, track_running, treadmill, trail...)
    def _is_running(act):
        act_type = act.get("activityType")
        if isinstance(act_type, dict):
            act_type = act_type.get("typeKey", "")
        return str(act_type) in RUNNING_TYPES

    running = [a for a in raw_activities if _is_running(a)]

    # Normalise to loader-compatible format and filter by TRACKING_START
    start_ms = int(TRACKING_START.timestamp() * 1000)
    activities = []
    for a in running:
        norm = _normalize_api_activity(a)
        if norm.get("startTimeLocal", 0) >= start_ms:
            activities.append(norm)

    # Save
    output = {"summarizedActivitiesExport": activities}
    timestamp = int(time.time())
    dest_path = os.path.join(UPLOADS_DIR, f"garmin_download_{timestamp}.json")
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Garmin] Scaricate {len(activities)} corse ({start_date_str} -> {end_date_str})")
    
    # Fetch polylines and km splits for the new activities
    fetch_activity_polylines(client, activities)
    fetch_activity_splits(client, activities)

    _save_cache_info({"last_fetch": time.time()})
    return activities


def clear_garmin_cache() -> None:
    """Force next fetch by resetting cache timestamp."""
    _save_cache_info({"last_fetch": 0})
