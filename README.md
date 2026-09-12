# Rain in the next hour – FTI pipelines with Hopsworks

Predicts whether it rains in the next hour in Zurich. Three pipelines (Feature, Training,
Inference) built on the Hopsworks Feature Store and Model Registry. Model quality is not the goal.

## Data and features

Source: [Open-Meteo](https://open-meteo.com/) (free, no API key). Hourly data, UTC.

- History: archive API, last 365 days → training
- Now: forecast API `current=...` → inference

Target `rain_next_1h`: 1 if precipitation in the following hour > 0.1 mm.

| Type | Feature | Definition |
|---|---|---|
| Aggregated | `humidity_mean_24h` | rolling mean of relative humidity, 24 h |
| Aggregated | `precip_sum_24h` | rolling sum of precipitation, 24 h |
| Aggregated | `pressure_change_3h` | `pressure_msl` minus its value 3 h earlier |
| Real-time | `cloud_cover`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m` | current observation, fetched live at inference time |

Model: `RandomForestClassifier(n_estimators=50)`, evaluated with accuracy and F1.

## Pipelines

```
Open-Meteo history ──► feature_pipeline.py ──► Feature Group  weather_hourly  (PK city, event_time, online)
                                                        │
                                               training_pipeline.py ──► Feature View ──► train/test split (in Hopsworks)
                                                        ──► RandomForest ──► Model Registry
                                                                                  │
Open-Meteo current ──► inference_pipeline.py ◄────────────────────────────────────┘
                       aggregated features from the online store + live RT features ──► prediction
```

**feature_pipeline.py** fetch raw data → compute features and label → create/get feature group → insert.

**training_pipeline.py** feature view over features + label → `create_train_test_split(0.2)` → train →
`joblib.dump` → `mr.python.create_model(...).save(...)`.

**inference_pipeline.py** `get_feature_vector({"city": ...})` for the aggregated features → live RT features
from Open-Meteo → download latest model version → predict.

## Run

Requirements: Docker, a free [hopsworks.ai](https://hopsworks.ai) account with an API key
(Account Settings → API keys, scopes `project featurestore job kafka`).

```bash
cp .env.example .env      # set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT
docker compose up --build # runs feature → training → inference in order
```

Without Docker (Python 3.11, Hopsworks needs < 3.14):

```bash
pip install -r requirements.txt
set -a && source .env && set +a
PYTHONPATH=src python src/feature_pipeline.py
PYTHONPATH=src python src/training_pipeline.py
PYTHONPATH=src python src/inference_pipeline.py
```

Example output:

```
Rain in the next hour : NO (p = 0.180)
Umbrella?             : not needed
```

## Layout

```
src/config.py              names, versions, feature order
src/weather.py             Open-Meteo client
src/features.py            feature engineering (shared by F and I)
src/feature_pipeline.py    F
src/training_pipeline.py   T
src/inference_pipeline.py  I
Dockerfile, docker-compose.yml, requirements.txt, .env.example
```

## Limitations

- Docker image is pinned to `linux/amd64`: Hopsworks 5 writes Delta tables from the client and
  needs `hops-deltalake`, which has no `linux/arm64` wheel. On Apple Silicon, Docker Desktop runs
  it via Rosetta. A first attempt with `time_travel_format="NONE"` was rejected by the server.
- The online store holds one row per `city`, so the aggregated features at inference time are up
  to 1 h old (the latest row has no label yet and is dropped). Negligible for 24 h aggregates.
- If the online lookup fails, inference recomputes the aggregates from the API and says so in the
  log. This is a fallback for the demo, not the intended path.
- One feature group, no point-in-time joins, no data validation, no transformation functions.
- The model is weak and the classes are imbalanced; not optimized on purpose.
