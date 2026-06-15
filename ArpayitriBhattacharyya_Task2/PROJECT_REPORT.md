# Unemployment Analysis Report

## Objective

The objective of this project is to analyze unemployment trends in India using Python. The project focuses on unemployment rate, employment estimates, labour participation, regional variation, and the period around Covid-19.

## Dataset

The dataset contains unemployment records by region, date, frequency, and area. The main columns are:

- Region
- Date
- Frequency
- Estimated Unemployment Rate (%)
- Estimated Employed
- Estimated Labour Participation Rate (%)
- Area

## Methodology

1. Loaded the dataset using Pandas.
2. Cleaned column names and converted dates into proper datetime format.
3. Removed duplicate records and sorted the data by date, region, and area.
4. Created monthly, regional, and rural/urban summaries.
5. Generated visualizations using Matplotlib and Seaborn.
6. Used SciPy to compare unemployment before Covid-19 with the Covid impact period.
7. Built an interactive Streamlit dashboard using Plotly charts.

## Key Questions

- How did the unemployment rate change over time?
- Which regions had the highest average unemployment?
- How different were rural and urban unemployment patterns?
- Was the Covid-period unemployment rate noticeably different from the earlier period?
- How are unemployment, employment, and labour participation related?

## Visualizations

The project generates the following charts:

- Average unemployment rate over time
- Average unemployment by region
- Rural vs urban unemployment distribution
- Correlation heatmap of key indicators
- Top regional unemployment peaks

## Statistical Analysis

The project compares unemployment before March 2020 with unemployment from March 2020 to June 2020 using Welch's t-test. This helps identify whether the Covid-period unemployment rate was statistically different from the previous period.

## Conclusion

The analysis gives a clear view of unemployment changes across time, region, and area. The Covid-period comparison highlights the sharp shift in unemployment during 2020, while the regional and rural/urban charts show that unemployment impact was not evenly distributed.
