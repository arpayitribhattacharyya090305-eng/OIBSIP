from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car data.csv"
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
TARGET = "Selling_Price"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=280,
        random_state=42,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_save() -> dict:
    df = load_data()
    features = ["Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, predictions), 3),
        "r2": round(r2_score(y_test, predictions), 3),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": features,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    saved_metrics = train_and_save()
    print("Model trained and saved")
    print(f"MAE: {saved_metrics['mae']} lakh")
    print(f"R2 score: {saved_metrics['r2']}")
