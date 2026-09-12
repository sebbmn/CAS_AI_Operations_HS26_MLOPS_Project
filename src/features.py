"""Feature engineering shared by the feature pipeline and the inference pipeline."""
import pandas as pd

from config import CITY, EVENT_TIME, LABEL, PRIMARY_KEY

RAIN_THRESHOLD_MM = 0.1


def add_aggregated_features(frame):
    """Aggregations over multiple timesteps of the hourly series."""
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["humidity_mean_24h"] = (
        frame["relative_humidity_2m"].rolling(window=24, min_periods=24).mean()
    )
    frame["precip_sum_24h"] = (
        frame["precipitation"].rolling(window=24, min_periods=24).sum()
    )
    frame["pressure_change_3h"] = frame["pressure_msl"] - frame["pressure_msl"].shift(3)
    return frame


def build_training_frame(frame):
    """Raw hourly observations -> feature/label dataframe ready for the feature group.

    The label is the *next* hour's precipitation, so the most recent row has no
    known label yet and is dropped.
    """
    frame = add_aggregated_features(frame)
    frame[LABEL] = (
        frame["precipitation"].shift(-1) > RAIN_THRESHOLD_MM
    ).astype("int32")
    frame.loc[frame.index[-1], LABEL] = pd.NA

    frame[PRIMARY_KEY] = CITY
    frame[EVENT_TIME] = frame["time"]

    columns = [
        PRIMARY_KEY,
        EVENT_TIME,
        "humidity_mean_24h",
        "precip_sum_24h",
        "pressure_change_3h",
        "cloud_cover",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        LABEL,
    ]
    result = frame[columns].dropna().reset_index(drop=True)
    result[LABEL] = result[LABEL].astype("int32")
    for column in columns[2:-1]:
        result[column] = result[column].astype("float64")
    return result
