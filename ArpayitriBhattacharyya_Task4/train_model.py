from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "spam_detector.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"


def find_dataset() -> Path:
    candidates = [
        ROOT / "data" / "spam.csv",
        Path.home() / "Downloads" / "spam.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place spam.csv in the data folder or keep it in Downloads."
    )


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or find_dataset()
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    return df[df["message"].ne("")]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.96,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.18)),
        ]
    )


def train_and_save(csv_path: Path | None = None) -> dict:
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    labels = ["ham", "spam"]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "dataset_size": int(len(df)),
        "spam_count": int((df["label"] == "spam").sum()),
        "ham_count": int((df["label"] == "ham").sum()),
        "test_size": int(len(y_test)),
        "source": str(csv_path or find_dataset()),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return metrics


if __name__ == "__main__":
    saved_metrics = train_and_save()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy: {saved_metrics['accuracy']:.2%}")
