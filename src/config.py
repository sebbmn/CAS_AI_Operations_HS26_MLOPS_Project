"""Central configuration, read from environment variables (see .env.example)."""
import os

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITY = os.getenv("CITY", "zurich")
LATITUDE = float(os.getenv("LATITUDE", "47.3769"))
LONGITUDE = float(os.getenv("LONGITUDE", "8.5417"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "365"))

# Hopsworks object names / versions
FEATURE_GROUP_NAME = "weather_hourly"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "weather_rain_next_1h"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "rain_next_1h_model"

# Columns
PRIMARY_KEY = "city"
EVENT_TIME = "event_time"
LABEL = "rain_next_1h"

# Feature order used for training and inference (must stay identical in both).
AGGREGATED_FEATURES = [
    "humidity_mean_24h",     # aggregated over 24 timesteps
    "precip_sum_24h",        # aggregated over 24 timesteps
    "pressure_change_3h",    # aggregated over 3 timesteps
]
REALTIME_FEATURES = [
    "cloud_cover",           # only known at inference time
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]
FEATURES = AGGREGATED_FEATURES + REALTIME_FEATURES

LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/app/artifacts")
