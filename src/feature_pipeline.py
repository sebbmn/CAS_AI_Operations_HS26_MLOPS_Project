"""F of FTI: raw Open-Meteo data -> features -> Hopsworks Feature Group."""
from config import (
    CITY,
    EVENT_TIME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HISTORY_DAYS,
    PRIMARY_KEY,
)
from features import build_training_frame
from hopsworks_client import ensure_credentials, login
from weather import fetch_history

FEATURE_DESCRIPTIONS = {
    "city": "Location identifier (primary key)",
    "event_time": "Hour of the observation (UTC)",
    "humidity_mean_24h": "AGGREGATED: mean relative humidity over the last 24 hours",
    "precip_sum_24h": "AGGREGATED: total precipitation over the last 24 hours (mm)",
    "pressure_change_3h": "AGGREGATED: sea level pressure change over the last 3 hours (hPa)",
    "cloud_cover": "REAL-TIME: cloud cover at the observed hour (%)",
    "temperature_2m": "REAL-TIME: air temperature at 2m (degC)",
    "relative_humidity_2m": "REAL-TIME: relative humidity at 2m (%)",
    "wind_speed_10m": "REAL-TIME: wind speed at 10m (km/h)",
    "rain_next_1h": "LABEL: 1 if precipitation in the following hour > 0.1 mm",
}


def main():
    ensure_credentials()
    print(f"[1/4] Fetching {HISTORY_DAYS} days of hourly weather for '{CITY}' ...")
    raw = fetch_history(HISTORY_DAYS)
    print(f"      {len(raw)} raw hourly observations "
          f"({raw['time'].min()} .. {raw['time'].max()})")

    print("[2/4] Engineering features ...")
    frame = build_training_frame(raw)
    print(f"      {len(frame)} feature rows, "
          f"rain_next_1h positive rate = {frame['rain_next_1h'].mean():.3f}")

    print("[3/4] Creating / getting the feature group ...")
    project = login()
    fs = project.get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather features for the 'rain in the next hour' model",
        primary_key=[PRIMARY_KEY],
        event_time=EVENT_TIME,
        online_enabled=True,
        time_travel_format="NONE",
    )

    print(f"[4/4] Inserting {len(frame)} rows ...")
    feature_group.insert(frame, wait=True)

    # The online store keeps one row per primary key. Re-inserting the most
    # recent row makes sure the online lookup in the inference pipeline returns
    # the latest aggregated features instead of an arbitrary row of the batch.
    feature_group.insert(frame.tail(1), wait=True)

    for name, description in FEATURE_DESCRIPTIONS.items():
        feature_group.update_feature_description(name, description)

    print(f"Done. Feature group '{FEATURE_GROUP_NAME}' "
          f"v{FEATURE_GROUP_VERSION} is up to date.")


if __name__ == "__main__":
    main()
