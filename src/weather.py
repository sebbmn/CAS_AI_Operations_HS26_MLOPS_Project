"""Thin client for the Open-Meteo API (no API key required)."""
from datetime import date, timedelta

import pandas as pd
import requests

from config import LATITUDE, LONGITUDE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
]
CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
]


def _get(url, params):
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _hourly_to_frame(payload):
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def fetch_history(days):
    """Historical hourly observations.

    The archive API lags a few days behind, so the most recent days are taken
    from the forecast API (which also serves past days). Both parts are
    concatenated into one continuous hourly series.
    """
    today = date.today()
    archive_end = today - timedelta(days=6)
    archive_start = today - timedelta(days=days)

    parts = []
    if archive_start < archive_end:
        parts.append(
            _hourly_to_frame(
                _get(
                    ARCHIVE_URL,
                    {
                        "latitude": LATITUDE,
                        "longitude": LONGITUDE,
                        "start_date": archive_start.isoformat(),
                        "end_date": archive_end.isoformat(),
                        "hourly": ",".join(HOURLY_VARS),
                        "timezone": "UTC",
                    },
                )
            )
        )
    parts.append(fetch_recent_hours(past_days=7))

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return frame.dropna(subset=HOURLY_VARS)


def fetch_recent_hours(past_days=2):
    """Hourly observations of the last `past_days` days up to the current hour."""
    payload = _get(
        FORECAST_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(HOURLY_VARS),
            "past_days": past_days,
            "forecast_days": 1,
            "timezone": "UTC",
        },
    )
    frame = _hourly_to_frame(payload)
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    return frame[frame["time"] <= now].dropna(subset=HOURLY_VARS).reset_index(drop=True)


def fetch_current():
    """Real-time observation for *now* - the RT features of the inference run."""
    payload = _get(
        FORECAST_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": ",".join(CURRENT_VARS),
            "timezone": "UTC",
        },
    )
    current = payload["current"]
    return {
        "observed_at": pd.to_datetime(current["time"]),
        "temperature_2m": float(current["temperature_2m"]),
        "relative_humidity_2m": float(current["relative_humidity_2m"]),
        "cloud_cover": float(current["cloud_cover"]),
        "wind_speed_10m": float(current["wind_speed_10m"]),
    }
