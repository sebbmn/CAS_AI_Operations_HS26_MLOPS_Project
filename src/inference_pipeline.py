"""I of FTI: aggregated features from the store + live RT features -> prediction."""
import os

import joblib
import pandas as pd

from config import (
    AGGREGATED_FEATURES,
    CITY,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    FEATURES,
    MODEL_NAME,
    PRIMARY_KEY,
    REALTIME_FEATURES,
)
from features import add_aggregated_features
from hopsworks_client import login
from weather import fetch_current, fetch_recent_hours


def aggregated_from_feature_store(feature_view):
    """Latest aggregated features for the city from the online feature store."""
    feature_view.init_serving()
    vector = feature_view.get_feature_vector(
        entry={PRIMARY_KEY: CITY}, return_type="pandas"
    )
    return {name: float(vector[name].iloc[0]) for name in AGGREGATED_FEATURES}


def aggregated_from_api():
    """Fallback: recompute the aggregations from the last 48h of observations."""
    recent = add_aggregated_features(fetch_recent_hours(past_days=2))
    last = recent.dropna(subset=AGGREGATED_FEATURES).iloc[-1]
    return {name: float(last[name]) for name in AGGREGATED_FEATURES}


def main():
    project = login()
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION
    )

    print("[1/4] Reading the aggregated features from the feature store ...")
    try:
        aggregated = aggregated_from_feature_store(feature_view)
        source = "Hopsworks online feature store"
    except Exception as error:  # online store unavailable / key not materialised yet
        print(f"      online lookup failed ({error}); recomputing from the API")
        aggregated = aggregated_from_api()
        source = "recomputed from Open-Meteo (fallback)"
    print(f"      {source}: {aggregated}")

    print("[2/4] Fetching the real-time features from Open-Meteo ...")
    current = fetch_current()
    realtime = {name: current[name] for name in REALTIME_FEATURES}
    print(f"      observed at {current['observed_at']} UTC: {realtime}")

    print("[3/4] Downloading the model from the Hopsworks Model Registry ...")
    model_registry = project.get_model_registry()
    registry_model = max(model_registry.get_models(MODEL_NAME), key=lambda m: m.version)
    model_dir = registry_model.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    print(f"      {MODEL_NAME} v{registry_model.version} "
          f"(metrics: {registry_model.training_metrics})")

    print("[4/4] Predicting ...")
    features = pd.DataFrame([{**aggregated, **realtime}])[FEATURES]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    print("\n--- Prediction -------------------------------------------")
    print(f"City                  : {CITY}")
    print(f"Reference time (UTC)  : {current['observed_at']}")
    print(f"Rain in the next hour : {'YES' if prediction else 'NO'} "
          f"(p = {probability:.3f})")
    print(f"Umbrella?             : {'take one' if prediction else 'not needed'}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
