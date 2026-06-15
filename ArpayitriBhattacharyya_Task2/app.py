from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "Unemployment in India.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_unemployment_india.csv"
SUMMARY_PATH = BASE_DIR / "reports" / "summary.txt"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

COLORS = {
    "page": "#f8faf7",
    "panel": "#ffffff",
    "sidebar": "#edf4f1",
    "sidebar_panel": "#f7fbf9",
    "ink": "#17211f",
    "muted": "#64736f",
    "line": "#dce5e1",
    "teal": "#087a75",
    "teal_soft": "#dff2ef",
    "blue": "#3e63a8",
    "gold": "#b98224",
    "coral": "#d45d4c",
    "sage": "#6f936f",
    "plum": "#765b9a",
}

CHART_PALETTE = [
    COLORS["teal"],
    COLORS["gold"],
    COLORS["blue"],
    COLORS["coral"],
    COLORS["sage"],
    COLORS["plum"],
    "#3f8791",
    "#9a7041",
]


st.set_page_config(
    page_title="Unemployment Analysis",
    page_icon="U",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --page: #f8faf7;
            --panel: #ffffff;
            --ink: #17211f;
            --muted: #64736f;
            --line: #dce5e1;
            --teal: #087a75;
            --teal-soft: #dff2ef;
            --gold: #b98224;
            --coral: #d45d4c;
            --sidebar: #edf4f1;
            --sidebar-panel: #f7fbf9;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(8, 122, 117, 0.10), transparent 360px),
                linear-gradient(180deg, rgba(185, 130, 36, 0.06), rgba(248, 250, 247, 0) 300px),
                var(--page);
            color: var(--ink);
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 2.15rem;
            padding-bottom: 2.25rem;
        }
        #MainMenu,
        [data-testid="stAppDeployButton"], .stDeployButton {
            display: none;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 122, 117, 0.09), rgba(237, 244, 241, 0) 210px),
                var(--sidebar);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] h2 {
            border-bottom: 1px solid rgba(8, 122, 117, 0.18);
            font-size: 1.15rem;
            margin-bottom: 1rem;
            padding-bottom: 0.65rem;
        }
        [data-testid="stSidebar"] label {
            color: #33413e;
            font-weight: 650;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
            line-height: 1.45;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div {
            background-color: var(--sidebar-panel);
            border-color: rgba(8, 122, 117, 0.22);
            box-shadow: 0 8px 20px rgba(23, 33, 31, 0.05);
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(8, 122, 117, 0.16);
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            border-top: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            box-shadow: 0 14px 34px rgba(23, 33, 31, 0.08);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
        }
        .page-title {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 42px rgba(23, 33, 31, 0.07);
            margin-bottom: 1.45rem;
            padding: 1.35rem 1.45rem 1.25rem;
            position: relative;
            overflow: hidden;
        }
        .page-title::before {
            background: linear-gradient(90deg, var(--teal), var(--gold), var(--coral));
            content: "";
            height: 4px;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
        }
        .kicker {
            color: var(--teal);
            font-size: 0.86rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .title {
            color: var(--ink);
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .subtitle {
            color: var(--muted);
            max-width: 780px;
            line-height: 1.55;
            margin-top: 0.55rem;
        }
        .note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="popover"] {
            border-color: var(--line);
        }
        div[data-baseweb="tag"] {
            background: var(--teal-soft);
            color: var(--ink);
        }
        .stButton > button,
        .stDownloadButton > button {
            background: var(--teal);
            border: 1px solid var(--teal);
            border-radius: 8px;
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--muted);
        }
        .stTabs [aria-selected="true"] {
            color: var(--teal);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        a {
            color: var(--teal);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    path = CLEANED_DATA_PATH if CLEANED_DATA_PATH.exists() else RAW_DATA_PATH
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    if "year" not in df:
        df["year"] = df["date"].dt.year
    return df.sort_values("date")


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def polish_figure(fig, height: int | None = None):
    fig.update_layout(
        template="plotly_white",
        colorway=CHART_PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["panel"],
        font={"family": "Arial, sans-serif", "color": COLORS["ink"], "size": 13},
        title={"font": {"size": 18, "color": COLORS["ink"]}, "x": 0, "xanchor": "left"},
        margin={"l": 20, "r": 20, "t": 58, "b": 38},
        hoverlabel={
            "bgcolor": COLORS["ink"],
            "font_color": COLORS["panel"],
            "bordercolor": COLORS["ink"],
        },
        legend={"title": "", "orientation": "h", "y": 1.08, "x": 0},
        height=height,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e9efec",
        zeroline=False,
        linecolor=COLORS["line"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e9efec",
        zeroline=False,
        linecolor=COLORS["line"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    return fig


def main() -> None:
    st.markdown(
        """
        <div class="page-title">
            <div class="kicker">Data analysis project</div>
            <div class="title">Unemployment Analysis in India</div>
            <div class="subtitle">
                A compact dashboard for studying unemployment rate, employment,
                labour participation, regional variation, and the Covid-period shift.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    with st.sidebar:
        st.header("Filters")
        regions = sorted(df["region"].dropna().unique())
        areas = sorted(df["area"].dropna().unique())
        selected_regions = st.multiselect("Region", regions, default=regions)
        selected_areas = st.multiselect("Area", areas, default=areas)
        date_range = st.date_input(
            "Date range",
            value=(df["date"].min().date(), df["date"].max().date()),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
        )

        st.divider()
        st.write("Dataset: unemployment records by region, date, and area.")

    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = df["date"].min(), df["date"].max()

    filtered = df[
        df["region"].isin(selected_regions)
        & df["area"].isin(selected_areas)
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ]

    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        metric_card("Avg unemployment", f"{filtered['estimated_unemployment_rate_percent'].mean():.2f}%")
    with metric_col2:
        metric_card("Avg labour participation", f"{filtered['estimated_labour_participation_rate_percent'].mean():.2f}%")
    with metric_col3:
        metric_card("Total employed", f"{filtered['estimated_employed'].sum():,.0f}")
    with metric_col4:
        metric_card("Regions shown", f"{filtered['region'].nunique()}")

    monthly = (
        filtered.groupby("date", as_index=False)
        .agg(
            unemployment_rate=("estimated_unemployment_rate_percent", "mean"),
            labour_participation=("estimated_labour_participation_rate_percent", "mean"),
            employed=("estimated_employed", "sum"),
        )
        .sort_values("date")
    )

    trend_fig = px.line(
        monthly,
        x="date",
        y="unemployment_rate",
        markers=True,
        title="Average unemployment rate over time",
        labels={"date": "Date", "unemployment_rate": "Unemployment rate (%)"},
        color_discrete_sequence=[COLORS["teal"]],
    )
    trend_fig.update_traces(line={"width": 3}, marker={"size": 7, "color": COLORS["teal"]})
    trend_fig.add_vrect(
        x0="2020-03-01",
        x1="2020-06-30",
        fillcolor=COLORS["coral"],
        opacity=0.12,
        line_width=0,
        annotation_text="Covid impact period",
        annotation_position="top left",
    )
    polish_figure(trend_fig, height=430)
    st.plotly_chart(trend_fig, use_container_width=True)

    left_col, right_col = st.columns(2, gap="large")
    region_avg = (
        filtered.groupby("region", as_index=False)["estimated_unemployment_rate_percent"]
        .mean()
        .sort_values("estimated_unemployment_rate_percent", ascending=False)
    )
    area_avg = (
        filtered.groupby("area", as_index=False)["estimated_unemployment_rate_percent"]
        .mean()
        .sort_values("estimated_unemployment_rate_percent", ascending=False)
    )

    with left_col:
        region_fig = px.bar(
            region_avg.head(15),
            x="estimated_unemployment_rate_percent",
            y="region",
            orientation="h",
            title="Top regions by average unemployment",
            labels={
                "estimated_unemployment_rate_percent": "Unemployment rate (%)",
                "region": "Region",
            },
            color="estimated_unemployment_rate_percent",
            color_continuous_scale=[
                COLORS["teal_soft"],
                COLORS["teal"],
                COLORS["blue"],
            ],
        )
        region_fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        region_fig.update_coloraxes(showscale=False)
        polish_figure(region_fig, height=540)
        st.plotly_chart(region_fig, use_container_width=True)

    with right_col:
        area_fig = px.bar(
            area_avg,
            x="area",
            y="estimated_unemployment_rate_percent",
            color="area",
            title="Rural and urban comparison",
            labels={
                "estimated_unemployment_rate_percent": "Unemployment rate (%)",
                "area": "Area",
            },
            color_discrete_map={
                "Rural": COLORS["sage"],
                "Urban": COLORS["gold"],
            },
        )
        polish_figure(area_fig, height=540)
        st.plotly_chart(area_fig, use_container_width=True)

    tab_summary, tab_data, tab_images = st.tabs(["Summary", "Data", "Saved charts"])

    with tab_summary:
        if SUMMARY_PATH.exists():
            st.code(SUMMARY_PATH.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Run `python src/analyze_unemployment.py` to generate the summary file.")

    with tab_data:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.markdown(
            f'<div class="note">Showing {filtered.shape[0]} records after filters.</div>',
            unsafe_allow_html=True,
        )

    with tab_images:
        image_files = [
            "unemployment_trend.png",
            "region_unemployment_average.png",
            "rural_urban_distribution.png",
            "correlation_heatmap.png",
            "top_regional_peaks.png",
        ]
        for image in image_files:
            path = FIGURES_DIR / image
            if path.exists():
                st.image(str(path), use_container_width=True)


if __name__ == "__main__":
    main()
