from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
MODEL_PATH = BASE_DIR / "models" / "knn_iris_model.joblib"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
METRICS_PATH = BASE_DIR / "reports" / "metrics.txt"

FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COLUMN = "Species"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Id"], errors="ignore")
    return df


def save_visualizations(df: pd.DataFrame) -> None:
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Distribution of Iris Species")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "species_distribution.png", dpi=150)
    plt.close()

    pairplot = sns.pairplot(df, hue=TARGET_COLUMN, diag_kind="hist", height=2.2)
    pairplot.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    pairplot.savefig(FIGURES_DIR / "pairplot.png", dpi=150)
    plt.close(pairplot.fig)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, cmap="crest", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def train_model(df: pd.DataFrame):
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


def save_results(model, X_test, y_test, predictions) -> None:
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    METRICS_PATH.write_text(
        f"Iris Flower Classification - KNN Model\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{report}\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - KNN Classifier")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURE_COLUMNS,
    )
    predicted_species = model.predict(sample)[0]

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sample prediction: {predicted_species}")


def main() -> None:
    df = load_data()
    save_visualizations(df)
    model, X_test, y_test, predictions = train_model(df)
    save_results(model, X_test, y_test, predictions)


if __name__ == "__main__":
    main()
