"""Unemployment analysis project pipeline.

Run from the project folder:
    python src/analyze_unemployment.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Unemployment in India.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_unemployment_india.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_PATH = REPORTS_DIR / "summary.txt"

COLORS = {
    "page": "#f8faf7",
    "panel": "#ffffff",
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


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
    )
    return df


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)

    df["date"] = pd.to_datetime(df["date"].str.strip(), dayfirst=True)
    df["frequency"] = df["frequency"].str.strip()
    df["region"] = df["region"].str.strip()
    df["area"] = df["area"].str.strip()

    df = df.drop_duplicates()
    df = df.sort_values(["date", "region", "area"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["month_number"] = df["date"].dt.month
    df["period"] = df["date"].dt.to_period("M").astype(str)

    return df


def save_core_visuals(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(
        style="whitegrid",
        palette=CHART_PALETTE,
        rc={
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["muted"],
            "figure.facecolor": COLORS["page"],
            "grid.color": "#e9efec",
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        },
    )

    monthly = (
        df.groupby("date", as_index=False)
        .agg(
            unemployment_rate=("estimated_unemployment_rate_percent", "mean"),
            labour_participation=("estimated_labour_participation_rate_percent", "mean"),
            employed=("estimated_employed", "sum"),
        )
        .sort_values("date")
    )

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="date", y="unemployment_rate", marker="o", color=COLORS["teal"], linewidth=2.5)
    plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color=COLORS["coral"], alpha=0.15)
    plt.title("Average Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unemployment_trend.png", dpi=150)
    plt.close()

    state_avg = (
        df.groupby("region", as_index=False)["estimated_unemployment_rate_percent"]
        .mean()
        .sort_values("estimated_unemployment_rate_percent", ascending=False)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=state_avg,
        x="estimated_unemployment_rate_percent",
        y="region",
        hue="region",
        palette=sns.color_palette("crest", n_colors=state_avg.shape[0]),
        legend=False,
    )
    plt.title("Average Unemployment Rate by Region")
    plt.xlabel("Unemployment rate (%)")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "region_unemployment_average.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="area",
        y="estimated_unemployment_rate_percent",
        hue="area",
        palette={"Rural": COLORS["sage"], "Urban": COLORS["gold"]},
        legend=False,
    )
    plt.title("Rural vs Urban Unemployment Distribution")
    plt.xlabel("Area")
    plt.ylabel("Unemployment rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rural_urban_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr_cols = [
        "estimated_unemployment_rate_percent",
        "estimated_employed",
        "estimated_labour_participation_rate_percent",
    ]
    sns.heatmap(
        df[corr_cols].corr(),
        annot=True,
        cmap="crest",
        fmt=".2f",
        linewidths=1,
        linecolor=COLORS["panel"],
    )
    plt.title("Correlation Between Key Indicators")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    peak_regions = (
        df.loc[df.groupby("region")["estimated_unemployment_rate_percent"].idxmax()]
        .sort_values("estimated_unemployment_rate_percent", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=peak_regions,
        x="estimated_unemployment_rate_percent",
        y="region",
        hue="area",
        palette={"Rural": COLORS["sage"], "Urban": COLORS["gold"]},
        dodge=False,
    )
    plt.title("Top 10 Regional Unemployment Peaks")
    plt.xlabel("Peak unemployment rate (%)")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_regional_peaks.png", dpi=150)
    plt.close()


def build_summary(df: pd.DataFrame) -> str:
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2020-06-30")
    before_covid = df[df["date"] < covid_start]["estimated_unemployment_rate_percent"]
    covid_period = df[(df["date"] >= covid_start) & (df["date"] <= covid_end)][
        "estimated_unemployment_rate_percent"
    ]

    t_stat, p_value = stats.ttest_ind(before_covid, covid_period, equal_var=False, nan_policy="omit")

    highest_row = df.loc[df["estimated_unemployment_rate_percent"].idxmax()]
    lowest_row = df.loc[df["estimated_unemployment_rate_percent"].idxmin()]
    region_avg = df.groupby("region")["estimated_unemployment_rate_percent"].mean().sort_values(ascending=False)
    area_avg = df.groupby("area")["estimated_unemployment_rate_percent"].mean()

    summary = f"""Unemployment Analysis with Python

Dataset size: {df.shape[0]} rows and {df.shape[1]} columns
Date range: {df["date"].min().date()} to {df["date"].max().date()}
Regions covered: {df["region"].nunique()}

Overall average unemployment rate: {df["estimated_unemployment_rate_percent"].mean():.2f}%
Overall average labour participation rate: {df["estimated_labour_participation_rate_percent"].mean():.2f}%

Highest unemployment observation:
- Region: {highest_row["region"]}
- Area: {highest_row["area"]}
- Date: {highest_row["date"].date()}
- Rate: {highest_row["estimated_unemployment_rate_percent"]:.2f}%

Lowest unemployment observation:
- Region: {lowest_row["region"]}
- Area: {lowest_row["area"]}
- Date: {lowest_row["date"].date()}
- Rate: {lowest_row["estimated_unemployment_rate_percent"]:.2f}%

Top 5 regions by average unemployment rate:
{region_avg.head(5).to_string()}

Average unemployment by area:
{area_avg.to_string()}

Covid-period comparison:
- Before March 2020 average: {before_covid.mean():.2f}%
- March to June 2020 average: {covid_period.mean():.2f}%
- Welch t-test statistic: {t_stat:.4f}
- p-value: {p_value:.6f}
"""
    return summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data()
    df.to_csv(CLEANED_DATA_PATH, index=False)
    save_core_visuals(df)

    summary = build_summary(df)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
