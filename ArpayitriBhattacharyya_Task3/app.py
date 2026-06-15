from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from train_model import MODEL_PATH, train_and_save


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car data.csv"
CURRENT_YEAR = datetime.now().year
FEATURE_COLUMNS = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
]
NUMERIC_FEATURES = ["Year", "Present_Price", "Driven_kms", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]


st.set_page_config(
    page_title="Car Price Predictor",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17202a;
            --muted: #5c6b73;
            --line: #dce5ea;
            --paper: #ffffff;
            --wash: #f4f8fa;
            --teal: #0f766e;
            --teal-dark: #0b4f4a;
            --coral: #e45252;
            --gold: #b7791f;
            --sidebar: #111b24;
        }
        html, body, .stApp, [class*="css"] {
            box-sizing: border-box;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        .stApp {
            background: var(--wash);
            color: var(--ink);
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stMainMenu"],
        #MainMenu,
        header {
            display: none !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="column"],
        .element-container {
            min-width: 0;
        }
        .block-container {
            max-width: 1220px;
            min-width: 0;
            padding: 2rem 2rem 3rem;
            width: 100%;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #f7fbfc;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.45rem;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.15);
            margin: 1.2rem 0;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.38rem 0.45rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.07);
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        p, li, label {
            color: #25333b;
        }
        .hero {
            background:
                linear-gradient(135deg, rgba(17, 27, 36, 0.98), rgba(15, 91, 85, 0.94) 64%, rgba(183, 121, 31, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            box-shadow: 0 22px 60px rgba(21, 34, 56, 0.14);
            color: #f9fdff;
            margin-bottom: 1.4rem;
            min-height: 230px;
            overflow: hidden;
            padding: 2.25rem 2.45rem;
            position: relative;
            width: 100%;
        }
        .hero:after {
            background:
                linear-gradient(90deg, rgba(255, 255, 255, 0.10) 0 1px, transparent 1px 100%),
                linear-gradient(0deg, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px 100%);
            background-size: 44px 44px;
            content: "";
            inset: 0;
            opacity: 0.18;
            position: absolute;
        }
        .hero > * {
            position: relative;
            z-index: 1;
        }
        .eyebrow {
            color: #8be0d8;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            margin-bottom: 0.65rem;
            text-transform: uppercase;
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.05;
            margin: 0;
        }
        .hero p {
            color: #dce9ec;
            font-size: 1.03rem;
            line-height: 1.62;
            margin: 1rem 0 0;
            max-width: 760px;
            overflow-wrap: break-word;
        }
        .hero-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 1.6rem;
        }
        .hero-chip {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.86rem;
            font-weight: 750;
            line-height: 1.25;
            padding: 0.55rem 0.8rem;
        }
        .metric-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 175px), 1fr));
            margin-bottom: 1.2rem;
            width: 100%;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(25, 42, 54, 0.07);
            min-height: 112px;
            padding: 0.95rem 1rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {
            color: var(--muted);
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div {
            color: var(--ink);
            font-weight: 850;
            overflow-wrap: anywhere;
        }
        [data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.13);
            box-shadow: none;
        }
        [data-testid="stSidebar"] div[data-testid="stMetric"] label,
        [data-testid="stSidebar"] div[data-testid="stMetric"] label p {
            color: #b9ccd2;
        }
        [data-testid="stSidebar"] div[data-testid="stMetricValue"],
        [data-testid="stSidebar"] div[data-testid="stMetricValue"] div {
            color: #ffffff;
        }
        .metric-card,
        .panel,
        .note {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(25, 42, 54, 0.08);
        }
        .metric-card {
            min-height: 118px;
            padding: 1rem 1.1rem;
        }
        .metric-card small {
            color: var(--muted);
            display: block;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            line-height: 1.35;
            overflow-wrap: break-word;
            text-transform: uppercase;
        }
        .metric-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.8rem;
            line-height: 1.1;
            margin-top: 0.45rem;
            overflow-wrap: break-word;
        }
        .metric-card span {
            color: var(--muted);
            display: block;
            margin-top: 0.35rem;
            overflow-wrap: break-word;
        }
        .sidebar-caption {
            color: #b9ccd2;
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }
        .sidebar-stat {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 8px;
            margin-bottom: 0.75rem;
            padding: 0.9rem 1rem;
        }
        .sidebar-stat small {
            color: #b9ccd2;
            display: block;
            font-size: 0.78rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .sidebar-stat strong {
            color: #ffffff;
            display: block;
            font-size: 1.35rem;
            line-height: 1.15;
            margin-top: 0.4rem;
            overflow-wrap: anywhere;
        }
        .section-title {
            align-items: center;
            display: flex;
            gap: 0.75rem;
            margin: 1.45rem 0 0.8rem;
            min-width: 0;
            width: 100%;
        }
        .section-title h2 {
            font-size: 1.35rem;
            line-height: 1.25;
            margin: 0;
            min-width: 0;
            overflow-wrap: break-word;
        }
        .section-title span {
            background: var(--teal);
            border-radius: 999px;
            display: inline-block;
            flex: 0 0 0.58rem;
            height: 0.58rem;
            width: 0.58rem;
        }
        .panel {
            color: #2b343a;
            line-height: 1.62;
            overflow-wrap: break-word;
            padding: 1.15rem 1.2rem;
        }
        .note {
            border-left: 4px solid var(--coral);
            color: #2b343a;
            line-height: 1.6;
            overflow-wrap: break-word;
            padding: 1rem 1.1rem;
        }
        .prediction-result {
            background: linear-gradient(135deg, #0f766e, #0b4f4a);
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(15, 118, 110, 0.24);
            color: #ffffff;
            margin-top: 1rem;
            padding: 1.4rem 1.5rem;
        }
        .prediction-result small {
            color: #cdecea;
            display: block;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .prediction-result strong {
            color: #ffffff;
            display: block;
            font-size: clamp(2rem, 5vw, 3rem);
            line-height: 1.05;
            margin: 0.45rem 0;
            overflow-wrap: break-word;
        }
        .prediction-result span {
            color: #e7f6f5;
            overflow-wrap: break-word;
        }
        .project-note {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
            margin-top: 1.2rem;
            width: 100%;
        }
        .project-note h3 {
            font-size: 1.05rem;
            margin: 0 0 0.55rem;
        }
        .project-note p {
            line-height: 1.58;
            margin: 0;
        }
        .project-note ul {
            margin: 0;
            padding-left: 1.1rem;
        }
        .project-note li {
            line-height: 1.55;
            margin-bottom: 0.3rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--teal), #149080);
            border: 0;
            border-radius: 8px;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.22);
            color: #ffffff;
            font-weight: 800;
            min-height: 3rem;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--teal-dark), var(--teal));
            border: 0;
            color: #ffffff;
        }
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSlider"] {
            border-radius: 8px;
        }
        footer {
            visibility: hidden;
        }
        @media (max-width: 980px) {
            .project-note {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 640px) {
            .block-container {
                padding: 1.2rem 0.9rem 2rem;
            }
            .hero {
                padding: 1.45rem;
            }
            .hero-chip {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df["Depreciation"] = df["Present_Price"] - df["Selling_Price"]
    return df


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


def section_title(title: str) -> None:
    st.subheader(title)


def metric_grid(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, caption) in zip(columns, items):
        with column:
            st.metric(label, value)
            st.caption(caption)


def make_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_comparison_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )


@st.cache_data
def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    y = df["Selling_Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=280,
            random_state=42,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
    }

    rows = []
    for name, model in models.items():
        pipeline = build_comparison_pipeline(model)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append(
            {
                "Model": name,
                "MAE (lakh)": round(mean_absolute_error(y_test, predictions), 3),
                "R2 score": round(r2_score(y_test, predictions), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(["MAE (lakh)", "R2 score"], ascending=[True, False])


def get_feature_importance(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")
        for name in feature_names
    ]
    return (
        pd.DataFrame({"Feature": clean_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
    )


def draw_price_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(
        data=df,
        x="Present_Price",
        y="Selling_Price",
        hue="Fuel_Type",
        palette=["#e45252", "#0f766e", "#b7791f"],
        s=70,
        edgecolor="#ffffff",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Present price vs resale price", fontweight="bold", color="#17202a")
    ax.set_xlabel("Present price (lakh)")
    ax.set_ylabel("Selling price (lakh)")
    ax.grid(True, color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_fuel_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df.groupby("Fuel_Type")["Selling_Price"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x="Fuel_Type",
        y="Selling_Price",
        order=order,
        color="#0f766e",
        width=0.55,
        ax=ax,
    )
    ax.set_title("Resale price spread by fuel type", fontweight="bold", color="#17202a")
    ax.set_xlabel("")
    ax.set_ylabel("Selling price (lakh)")
    ax.grid(True, axis="y", color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_age_price_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(
        data=df,
        x="Car_Age",
        y="Selling_Price",
        scatter_kws={"s": 52, "alpha": 0.72, "color": "#0f766e"},
        line_kws={"color": "#e45252", "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Car age vs selling price", fontweight="bold", color="#17202a")
    ax.set_xlabel("Car age (years)")
    ax.set_ylabel("Selling price (lakh)")
    ax.grid(True, color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_transmission_chart(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("Transmission", as_index=False)["Selling_Price"]
        .median()
        .sort_values("Selling_Price", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(
        data=summary,
        x="Transmission",
        y="Selling_Price",
        color="#b7791f",
        ax=ax,
    )
    ax.set_title("Median price by transmission", fontweight="bold", color="#17202a")
    ax.set_xlabel("")
    ax.set_ylabel("Median selling price (lakh)")
    ax.grid(True, axis="y", color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_depreciation_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(
        df["Depreciation"],
        bins=24,
        color="#0f766e",
        edgecolor="#ffffff",
        ax=ax,
    )
    ax.set_title("Depreciation distribution", fontweight="bold", color="#17202a")
    ax.set_xlabel("Present price - selling price (lakh)")
    ax.set_ylabel("Number of cars")
    ax.grid(True, axis="y", color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_correlation_chart(df: pd.DataFrame) -> None:
    numeric_df = df[["Year", "Present_Price", "Driven_kms", "Owner", "Car_Age", "Depreciation", "Selling_Price"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Correlation between numeric fields", fontweight="bold", color="#17202a")
    fig.patch.set_facecolor("#f4f8fa")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_feature_importance_chart(importance_df: pd.DataFrame) -> None:
    if importance_df.empty:
        st.info("Feature importance is available for tree-based models.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        color="#0f766e",
        ax=ax,
    )
    ax.set_title("Top model feature importances", fontweight="bold", color="#17202a")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(True, axis="x", color="#dce5ea", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f4f8fa")
    sns.despine(ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def prediction_form(df: pd.DataFrame, pipeline) -> None:
    section_title("Try a resale estimate")

    with st.form("prediction_form"):
        left, mid, right = st.columns(3)
        with left:
            car_year = st.slider(
                "Manufacturing year",
                int(df["Year"].min()),
                max(CURRENT_YEAR, int(df["Year"].max())),
                2016,
            )
            present_price = st.number_input(
                "Current showroom price (lakh)",
                min_value=0.10,
                max_value=100.0,
                value=7.50,
                step=0.10,
            )
        with mid:
            driven_kms = st.number_input(
                "Kilometres driven",
                min_value=0,
                max_value=500000,
                value=35000,
                step=1000,
            )
            owner = st.selectbox("Previous owners", sorted(df["Owner"].unique()))
        with right:
            fuel = st.selectbox("Fuel type", sorted(df["Fuel_Type"].unique()))
            seller = st.selectbox("Seller type", sorted(df["Selling_type"].unique()))
            transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))

        submitted = st.form_submit_button("Predict price", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "Year": car_year,
                    "Present_Price": present_price,
                    "Driven_kms": driven_kms,
                    "Fuel_Type": fuel,
                    "Selling_type": seller,
                    "Transmission": transmission,
                    "Owner": owner,
                }
            ]
        )
        prediction = float(pipeline.predict(input_df)[0])
        lower = max(prediction * 0.93, 0)
        upper = prediction * 1.07

        with st.container(border=True):
            st.metric("Estimated selling price", f"Rs. {prediction:.2f} lakh")
            st.write(f"Practical negotiation band: Rs. {lower:.2f} lakh to Rs. {upper:.2f} lakh")
            st.caption("The band is a simple 7% cushion around the model result for listing discussions.")


def render_sidebar(df: pd.DataFrame) -> str:
    st.sidebar.title("Used Car Project")
    st.sidebar.caption("A small Streamlit project for checking used car prices from the sample dataset.")
    page = st.sidebar.radio(
        "Choose a section",
        ["Home", "EDA", "Dataset", "Model notes"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Dataset rows", len(df))
    st.sidebar.metric("Model target", "Selling price")
    st.sidebar.metric("Price unit", "Lakh")
    return page


def main() -> None:
    inject_css()
    df = load_data()
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    metrics = bundle["metrics"]
    page = render_sidebar(df)

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Machine learning mini project</div>
            <h1>Car Price Predictor</h1>
            <p>
                This app estimates the resale price of a used car using details such as year,
                kilometres driven, fuel type, seller type, transmission, ownership count, and
                present showroom price.
            </p>
            <div class="hero-stats">
                <div class="hero-chip">Prediction in lakh</div>
                <div class="hero-chip">Random Forest regression</div>
                <div class="hero-chip">301 sample records</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if page == "Home":
        metric_grid(
            [
                ("Average selling price", f"{df['Selling_Price'].mean():.2f} lakh", "Mean resale value"),
                ("Median car age", f"{df['Car_Age'].median():.0f} years", "Typical vehicle age"),
                ("Average kms", f"{df['Driven_kms'].mean():,.0f}", "Average running"),
                ("Model MAE", f"{metrics['mae']:.2f} lakh", "Average prediction error"),
            ]
        )

        prediction_form(df, pipeline)

        st.info(
            "This is a dataset-based estimate. Actual resale value can change because of car "
            "condition, city, insurance, service history, negotiation, and current market demand."
        )

        left_note, right_note = st.columns([1.15, 0.85])
        with left_note:
            with st.container(border=True):
                st.markdown("#### How this project was put together")
                st.write(
                    "I kept the input fields close to the columns available in the dataset and used "
                    "a Random Forest model because the relation between age, mileage and price is not "
                    "perfectly linear. The app is meant to be understandable first, not overly complex."
                )
        with right_note:
            with st.container(border=True):
                st.markdown("#### Things to improve later")
                st.markdown(
                    "- Add more recent car records\n"
                    "- Include brand and model level details\n"
                    "- Compare multiple regression algorithms"
                )

    elif page == "EDA":
        section_title("Exploratory data analysis")
        st.write(
            "These charts give a quick view of how price changes with age, fuel type, "
            "transmission, depreciation, and numeric relationships in the dataset."
        )

        c1, c2 = st.columns(2)
        with c1:
            draw_age_price_chart(df)
        with c2:
            draw_fuel_chart(df)

        c3, c4 = st.columns(2)
        with c3:
            draw_transmission_chart(df)
        with c4:
            draw_depreciation_chart(df)

        draw_correlation_chart(df)

    elif page == "Dataset":
        section_title("Dataset overview")
        c1, c2 = st.columns([1.08, 0.92])
        with c1:
            draw_price_chart(df)
        with c2:
            draw_fuel_chart(df)

        section_title("First few records")
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    else:
        section_title("Model details")
        metric_grid(
            [
                ("R2 score", f"{metrics['r2']:.3f}", "Explained variance"),
                ("Training rows", str(metrics["train_rows"]), "Rows used for fitting"),
                ("Testing rows", str(metrics["test_rows"]), "Rows held out"),
                ("Features", str(len(metrics["features"])), "Model inputs"),
            ]
        )

        with st.container(border=True):
            st.write(
                "The model is trained with a scikit-learn pipeline. Numeric columns are filled with the "
                "median value if needed, and categorical columns are one-hot encoded. The trained pipeline "
                "is saved with Joblib so the Streamlit app can load it without retraining every time."
            )

        section_title("Feature importance")
        importance_df = get_feature_importance(pipeline)
        left_importance, right_importance = st.columns([1.1, 0.9])
        with left_importance:
            draw_feature_importance_chart(importance_df)
        with right_importance:
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
            st.caption(
                "For one-hot encoded categories, each category level appears as a separate model input."
            )

        section_title("Model comparison")
        comparison_df = compare_models(df)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        best_model = comparison_df.iloc[0]
        st.success(
            f"Best result on this split: {best_model['Model']} with MAE "
            f"{best_model['MAE (lakh)']:.3f} lakh and R2 {best_model['R2 score']:.3f}."
        )

        st.write("Features used by the model:")
        st.code(", ".join(metrics["features"]))


if __name__ == "__main__":
    main()
