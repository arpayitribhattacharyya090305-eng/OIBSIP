from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "knn_iris_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
METRICS_PATH = BASE_DIR / "reports" / "metrics.txt"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

FEATURE_COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
SPECIES_DISPLAY = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
}


st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="I",
    layout="wide",
)


st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #f7f8f5;
            border-right: 1px solid #e4e7df;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #26342c;
        }

        [data-testid="stAppDeployButton"],
        .stDeployButton {
            display: none;
        }

        .project-header {
            border-bottom: 1px solid #e7e9e3;
            padding-bottom: 1.1rem;
            margin-bottom: 1.4rem;
        }

        .project-kicker {
            color: #5f6f64;
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
        }

        .project-title {
            color: #1f2c25;
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 700;
            margin: 0;
        }

        .project-subtitle {
            color: #5b625c;
            font-size: 1rem;
            line-height: 1.55;
            max-width: 760px;
            margin-top: 0.65rem;
        }

        .result-panel {
            background: #fbfbf8;
            border: 1px solid #e4e7df;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            min-height: 178px;
        }

        .result-label {
            color: #6a736b;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .result-value {
            color: #1d3328;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .result-note {
            color: #626b63;
            font-size: 0.95rem;
            line-height: 1.45;
        }

        .small-note {
            color: #6c746e;
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def predict_species(model, values: list[float]) -> tuple[str, float, pd.DataFrame]:
    sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.classes_

    probability_df = pd.DataFrame(
        {
            "Species": [SPECIES_DISPLAY.get(species, species) for species in classes],
            "Probability": probabilities.round(4),
        }
    ).sort_values("Probability", ascending=False)

    return prediction, probabilities.max(), probability_df


def show_metric_file() -> None:
    if METRICS_PATH.exists():
        st.code(METRICS_PATH.read_text(encoding="utf-8"), language="text")
    else:
        st.warning("Metrics file not found. Run `python iris_project.py` first.")


def show_project_visuals() -> None:
    figure_files = [
        ("Species count", "species_distribution.png"),
        ("Feature correlation", "correlation_heatmap.png"),
        ("Model confusion matrix", "confusion_matrix.png"),
        ("Feature pair plot", "pairplot.png"),
    ]

    left_col, right_col = st.columns(2, gap="large")

    for index, (title, filename) in enumerate(figure_files):
        path = FIGURES_DIR / filename
        if path.exists():
            with left_col if index % 2 == 0 else right_col:
                st.subheader(title)
                st.image(str(path), use_container_width=True)


def format_probability_table(probability_df: pd.DataFrame) -> pd.DataFrame:
    formatted = probability_df.copy()
    formatted["Probability"] = formatted["Probability"].map(lambda value: f"{value:.2%}")
    return formatted


def main() -> None:
    st.markdown(
        """
        <div class="project-header">
            <div class="project-kicker"></div>
            <h1 class="project-title">Iris Species Predictor</h1>
            <div class="project-subtitle">
                A K-Nearest Neighbors model trained on sepal and petal measurements
                from the Iris flower dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python iris_project.py` before opening the app.")
        return

    model = load_model()
    df = load_dataset()

    if "submitted_values" not in st.session_state:
        st.session_state.submitted_values = None

    with st.sidebar:
        st.header("Measurements")
        st.caption("Values are in centimeters.")
        sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.5, 0.1)
        petal_length = st.slider("Petal length", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal width", 0.1, 2.5, 0.2, 0.1)
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

        st.divider()
        st.write("Training split: 80% train, 20% test")
        st.write("Algorithm: KNN, k = 7")

    current_values = [sepal_length, sepal_width, petal_length, petal_width]
    if predict_clicked:
        st.session_state.submitted_values = current_values

    submitted_values = st.session_state.submitted_values

    summary_col, chart_col = st.columns([0.95, 1.25], gap="large")

    if submitted_values is None:
        with summary_col:
            st.subheader("Current result")
            st.info("Choose measurements in the sidebar, then click Predict to see the result.")

        with chart_col:
            st.subheader("Species probabilities")
            st.info("Probabilities will appear after you submit a prediction.")
    else:
        prediction, confidence, probability_df = predict_species(model, submitted_values)
        display_prediction = SPECIES_DISPLAY.get(prediction, prediction)

        with summary_col:
            st.subheader("Current result")
            result_note = "Prediction calculated from the last submitted measurements."
            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">Predicted species</div>
                    <div class="result-value">{display_prediction}</div>
                    <div class="result-note">{result_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric("Confidence", f"{confidence:.2%}")
            st.dataframe(
                pd.DataFrame(
                    [submitted_values],
                    columns=["Sepal length", "Sepal width", "Petal length", "Petal width"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        with chart_col:
            st.subheader("Species probabilities")
            st.bar_chart(probability_df, x="Species", y="Probability", use_container_width=True)
            st.dataframe(
                format_probability_table(probability_df),
                use_container_width=True,
                hide_index=True,
            )

    tab_data, tab_metrics, tab_visuals = st.tabs(["Dataset", "Model report", "Charts"])

    with tab_data:
        st.subheader("Iris dataset")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(f'<div class="small-note">Rows: {df.shape[0]} | Columns: {df.shape[1]}</div>', unsafe_allow_html=True)

    with tab_metrics:
        show_metric_file()

    with tab_visuals:
        show_project_visuals()


if __name__ == "__main__":
    main()
