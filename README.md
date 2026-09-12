# CAS AI Operations HS26 – MLOps Projektarbeit

"Muss ich einen Schirm einpacken?" – ein kleines FTI-Projekt (**F**eature, **T**raining,
**I**nference) auf Basis des **Hopsworks Feature Store** und der **Hopsworks Model Registry**.

Das Modell sagt voraus, ob es an einem Ort (Default: Zürich) **in der nächsten Stunde regnet**.
Modellgüte ist ausdrücklich nicht das Ziel – im Zentrum stehen die saubere Trennung der drei
Pipelines und die Nutzung des Feature Stores.

---

## 1. Daten

| | |
|---|---|
| Quelle | [Open-Meteo](https://open-meteo.com/) – frei, kein API-Key nötig |
| Historie (Training) | `archive-api.open-meteo.com/v1/archive`, standardmässig die letzten 365 Tage |
| Aktuelle Daten (Inferenz) | `api.open-meteo.com/v1/forecast` mit `current=...` |
| Granularität | stündlich, UTC |
| Ort | Zürich (47.3769 / 8.5417), über `.env` änderbar |

Die Archiv-API hinkt einige Tage hinterher; die Feature-Pipeline füllt die Lücke bis zur
aktuellen Stunde mit der Forecast-API (`past_days=7`) und hängt beide Teile zu einer
lückenlosen Stundenreihe zusammen.

## 2. Target

`rain_next_1h` – binär: `1`, wenn der Niederschlag **in der folgenden Stunde** > 0.1 mm ist,
sonst `0`. Das Label entsteht durch `precipitation.shift(-1)`, die jeweils letzte Zeile hat
daher noch kein Label und wird verworfen.

## 3. Features

Beide vom Aufgabenblatt geforderten Feature-Arten sind enthalten:

### Aggregierte Features (über mehrere Timesteps)

| Feature | Berechnung |
|---|---|
| `humidity_mean_24h` | Mittlere relative Luftfeuchtigkeit der letzten 24 h (Rolling Mean, Window 24) |
| `precip_sum_24h` | Niederschlagssumme der letzten 24 h (Rolling Sum, Window 24) |
| `pressure_change_3h` | Luftdruckänderung über 3 h (`pressure_msl - pressure_msl.shift(3)`) |

### Real-Time-Features (erst zum Zeitpunkt der Inferenz bekannt)

| Feature | Bedeutung |
|---|---|
| `cloud_cover` | Bewölkungsgrad jetzt (%) |
| `temperature_2m` | Temperatur jetzt (°C) |
| `relative_humidity_2m` | Relative Luftfeuchtigkeit jetzt (%) |
| `wind_speed_10m` | Windgeschwindigkeit jetzt (km/h) |

Die RT-Features werden in der Inferenz-Pipeline **live** von Open-Meteo geholt
(`current=...`) und nicht aus dem Feature Store gelesen – genau so, wie sie in einem echten
Betrieb erst zum Vorhersagezeitpunkt vorliegen. Die aggregierten Features kommen dagegen aus
dem **Online Feature Store** (sie benötigen 24 h Historie und werden von der Feature-Pipeline
vorberechnet).

## 4. Modell

`sklearn.ensemble.RandomForestClassifier(n_estimators=50, random_state=42)` – bewusst simpel.
Evaluiert wird mit Accuracy und F1 auf dem Test-Split; beide Werte werden als Metriken in die
Model Registry geschrieben.

## 5. Aufbau der Pipelines (FTI)

```
Open-Meteo Archive ─┐
                    ├─► [F] feature_pipeline.py ─► Feature Group  weather_hourly (v1)
Open-Meteo Forecast ┘                                    │  PK: city, Event Time: event_time
                                                         │  offline (Historie) + online (letzte Zeile)
                                                         ▼
                              [T] training_pipeline.py ─► Feature View weather_rain_next_1h (v1)
                                                         ─► Training Dataset (train/test split in Hopsworks)
                                                         ─► RandomForest ─► joblib ─► Model Registry
                                                                                        │
Open-Meteo current ────────────► [I] inference_pipeline.py ◄─────────────────────────────┘
   (RT-Features)                    + aggregierte Features aus dem Online Feature Store
                                    ─► Prediction "Schirm ja/nein"
```

### `src/feature_pipeline.py`
1. Rohdaten (Archiv + letzte Tage) via Open-Meteo abrufen
2. Rolling-Window-Aggregationen und Label berechnen (`src/features.py`)
3. Feature Group `weather_hourly` v1 anlegen/holen – Primary Key `city`, Event Time `event_time`,
   `online_enabled=True`
4. Dataframe einfügen (`insert(..., wait=True)`)
5. Feature-Beschreibungen setzen

### `src/training_pipeline.py`
1. Feature Group laden, Query über Features + Label bauen
2. Feature View `weather_rain_next_1h` v1 erstellen/holen (`labels=["rain_next_1h"]`)
3. Training Dataset mit `create_train_test_split(test_size=0.2)` **in Hopsworks** erzeugen
   (Ausbaustufe: kein lokales `train_test_split`)
4. RandomForest trainieren, Accuracy/F1 berechnen
5. Modell lokal mit `joblib.dump` speichern und via `mr.python.create_model(...)` +
   `model.save(...)` in die Hopsworks Model Registry hochladen (inkl. Model Schema,
   Input Example, Feature-View-Referenz und Training-Dataset-Version → Model Lineage)

### `src/inference_pipeline.py`
1. Feature View holen, `init_serving()`, aggregierte Features per
   `get_feature_vector({"city": ...})` aus dem **Online Feature Store** lesen
2. RT-Features live von Open-Meteo abrufen
3. Modell aus der Model Registry herunterladen (`mr.get_model(...).download()`) und laden
4. Feature-Vektor in der Trainingsreihenfolge zusammensetzen, Prediction + Wahrscheinlichkeit
   ausgeben

---

## 6. Setup und Ausführung

### Voraussetzungen
* Docker / Docker Compose (empfohlen) **oder** Python **3.11** (Hopsworks verlangt Python < 3.14)
* Ein kostenloser Account auf [hopsworks.ai](https://hopsworks.ai)

### Hopsworks vorbereiten
1. Auf hopsworks.ai registrieren und ein Projekt anlegen (Name merken).
2. **Account Settings → API keys → New API key** mit den Scopes
   `project`, `featurestore`, `job`, `kafka` erstellen und kopieren.

### Konfiguration
```bash
cp .env.example .env
# .env öffnen und HOPSWORKS_API_KEY sowie HOPSWORKS_PROJECT eintragen
```

| Variable | Bedeutung | Default |
|---|---|---|
| `HOPSWORKS_API_KEY` | API-Key (Pflicht) | – |
| `HOPSWORKS_PROJECT` | Projektname in Hopsworks (leer = Default-Projekt des Accounts) | – |
| `CITY` | Primary Key / Ortsbezeichnung | `zurich` |
| `LATITUDE` / `LONGITUDE` | Koordinaten | Zürich |
| `HISTORY_DAYS` | Tage Historie für das Backfill | `365` |

### Variante A – Docker Compose (alle drei Pipelines nacheinander)
```bash
docker compose up --build
```
Die Services sind über `depends_on: service_completed_successfully` verkettet und laufen daher
strikt in der Reihenfolge **feature → training → inference**. Einzeln:
```bash
docker compose run --rm feature-pipeline
docker compose run --rm training-pipeline
docker compose run --rm inference-pipeline
```

### Variante B – lokal ohne Docker
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
set -a && source .env && set +a
PYTHONPATH=src python src/feature_pipeline.py
PYTHONPATH=src python src/training_pipeline.py
PYTHONPATH=src python src/inference_pipeline.py
```

### Beispiel-Output der Inferenz
```
--- Prediction -------------------------------------------
City                  : zurich
Reference time (UTC)  : 2026-09-12 09:00:00
Rain in the next hour : NO (p = 0.180)
Umbrella?             : not needed
----------------------------------------------------------
```

## 7. Projektstruktur

```
.
├── docker-compose.yml        # drei Services, sequentiell verkettet
├── Dockerfile                # Python 3.11 + Dependencies
├── requirements.txt          # gepinnte Abhängigkeiten
├── .env.example              # Vorlage für die Konfiguration
└── src
    ├── config.py             # zentrale Konfiguration / Feature-Reihenfolge
    ├── hopsworks_client.py   # Login in Hopsworks
    ├── weather.py            # Open-Meteo-Client (Archiv, letzte Stunden, current)
    ├── features.py           # Feature Engineering (von F- und I-Pipeline genutzt)
    ├── feature_pipeline.py   # F
    ├── training_pipeline.py  # T
    └── inference_pipeline.py # I
```

## 8. Umgesetzte Ausbaustufen

* Train/Test-Split im Hopsworks Training Dataset statt lokal mit scikit-learn
* Model Registry in Hopsworks inkl. Model Schema, Input Example und Lineage
  (Feature View + Training-Dataset-Version am Modell hinterlegt)
* Online Feature Store für den Feature-Lookup zur Inferenzzeit
* Containerisierung aller drei Pipelines mit Docker Compose

## 9. Limitationen und Schwierigkeiten

* **Ein einzelnes Feature Group.** Aggregierte und RT-Features liegen in derselben Feature
  Group. Ein Join über mehrere Feature Groups (Ausbaustufe) wäre realistischer, bringt hier
  aber keinen inhaltlichen Mehrwert und wurde zugunsten der Einfachheit weggelassen.
* **Online Store enthält nur die letzte Zeile pro Key.** Primary Key ist ausschliesslich
  `city`, damit der Lookup zur Inferenzzeit ohne Zeitstempel funktioniert. Der Offline Store
  behält die volle Historie (Event Time `event_time`), das Training nutzt diese. Das bedeutet
  aber: die aggregierten Features beim Inferenzlauf sind bis zu ~1 Stunde alt, weil die
  jüngste Zeile ohne Label verworfen wird. Für 24-h-Aggregate ist das vernachlässigbar.
* **Kein Point-in-time-korrekter Join / keine Spine Group.** Nicht nötig, da alle Features aus
  derselben Stundenreihe stammen.
* **Fallback in der Inferenz.** Schlägt der Online-Lookup fehl (z.B. weil die Materialisierung
  des Online Stores noch läuft), berechnet die Pipeline die Aggregate ersatzweise direkt aus
  den letzten 48 h der Open-Meteo-Daten und weist im Log darauf hin. Das ist bewusst ein
  Notnagel, damit die Demo nicht abbricht – fachlich korrekt ist der Weg über den Feature Store.
* **`hops-deltalake` ist nicht installiert.** Das Extra `hopsworks[python]` zieht dieses Paket
  nur als Source-Distribution (Rust-Toolchain + Netzwerkzugriff auf `repo.hops.works` beim
  Build). Stattdessen sind die benötigten Teile des Extras einzeln gepinnt und die Feature
  Group wird mit `time_travel_format="NONE"` erstellt, was ohne Delta funktioniert. Damit
  entfällt Time Travel auf der Feature Group.
* **Modellgüte.** Ein RandomForest auf sieben Wetterfeatures sagt Regen in der nächsten Stunde
  nur mässig voraus; die Klassen sind zudem unbalanciert. Das ist laut Aufgabenstellung
  irrelevant und wurde nicht optimiert.
* **Keine Datenvalidierung** (Great Expectations) und keine Transformation Functions – bewusst
  weggelassen, um die Pipelines minimal und nachvollziehbar zu halten.
