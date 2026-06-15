# Unemployment Analysis with Python

This project analyzes unemployment in India using Python. It studies unemployment rate, estimated employment, labour participation rate, regional patterns, rural/urban differences, and the change during the Covid-19 period.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- SciPy
- Streamlit

## Project Structure

```text
Unemployment_Analysis/
├── app.py
├── data/
│   ├── Unemployment in India.csv
│   └── cleaned_unemployment_india.csv
├── reports/
│   ├── summary.txt
│   └── figures/
│       ├── correlation_heatmap.png
│       ├── region_unemployment_average.png
│       ├── rural_urban_distribution.png
│       ├── top_regional_peaks.png
│       └── unemployment_trend.png
├── src/
│   └── analyze_unemployment.py
├── PROJECT_REPORT.md
└── requirements.txt
└── Unemployment_Analysis.ipynb
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Generate cleaned data, summary, and charts:

```bash
python src/analyze_unemployment.py
```

Open the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

## Analysis Covered

- Data cleaning and date formatting
- Monthly unemployment trend
- Average unemployment by region
- Rural vs urban comparison
- Labour participation and employment overview
- Correlation between key indicators
- Statistical comparison between pre-Covid and Covid-period unemployment rates

## Dashboard Features

- Region, area, and date filters
- Key metrics cards
- Interactive Plotly trend chart
- Region-wise unemployment ranking
- Rural and urban comparison
- Raw filtered data table
- Saved analysis charts

## Main Output

The generated analysis summary is saved at:

```text
reports/summary.txt
```
