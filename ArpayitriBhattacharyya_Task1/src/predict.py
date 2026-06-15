from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "knn_iris_model.joblib"
FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def parse_args():
    parser = ArgumentParser(description="Predict the Iris species from flower measurements.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = probabilities.max()

    print(f"Predicted species: {prediction}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
