"""T of FTI: Feature View -> training dataset -> model -> Hopsworks Model Registry."""
import os

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    FEATURES,
    LABEL,
    LOCAL_MODEL_DIR,
    MODEL_NAME,
)
from hopsworks_client import login


def main():
    project = login()
    fs = project.get_feature_store()

    print("[1/5] Building the feature view ...")
    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
    )
    query = feature_group.select(FEATURES + [LABEL])
    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        description="Aggregated + real-time weather features with the rain_next_1h label",
        query=query,
        labels=[LABEL],
    )

    print("[2/5] Creating a train/test split inside Hopsworks ...")
    td_version, _ = feature_view.create_train_test_split(
        test_size=0.2,
        description="80/20 random split for the rain_next_1h model",
        write_options={"wait_for_job": True},
    )
    X_train, X_test, y_train, y_test = feature_view.get_train_test_split(td_version)
    X_train, X_test = X_train[FEATURES], X_test[FEATURES]
    y_train, y_test = y_train[LABEL], y_test[LABEL]
    print(f"      training dataset v{td_version}: "
          f"{len(X_train)} train rows / {len(X_test)} test rows")

    print("[3/5] Training the model ...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    print(f"      metrics: {metrics}")

    print("[4/5] Saving the model locally ...")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
    joblib.dump(model, local_path)
    print(f"      {local_path}")

    print("[5/5] Registering the model in the Hopsworks Model Registry ...")
    model_registry = project.get_model_registry()
    model_schema = ModelSchema(
        input_schema=Schema(X_train), output_schema=Schema(y_train)
    )
    registered = model_registry.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        model_schema=model_schema,
        input_example=X_train.head(1),
        description="RandomForest predicting rain in the next hour",
        feature_view=feature_view,
        training_dataset_version=td_version,
    )
    registered.save(LOCAL_MODEL_DIR)
    print(f"Done. Model '{MODEL_NAME}' v{registered.version} registered.")


if __name__ == "__main__":
    main()
