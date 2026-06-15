# OIBSIP Data Science Internship Projects

This repository contains the projects completed for the Oasis Infobyte Data Science Internship. Each task is organized in its own folder with source code, dataset files, trained models or reports, and a task-specific README.

## Projects

| Task 1 | Iris Flower Classification | Classifies Iris flowers into Setosa, Versicolor, and Virginica using sepal and petal measurements. |
| Task 2 | Unemployment Analysis | Analyzes unemployment trends in India with visualizations, regional comparisons, and a Streamlit dashboard. |
| Task 3 | Used Car Price Predictor | Predicts used car selling prices using regression models and a Streamlit prediction interface. |
| Task 4 | Email Spam Detection | Detects whether a message is spam or ham using text classification and a Streamlit app. |

## Repository Structure

```text
OIBSIP/
|-- ArpayitriBhattacharyya_Task1/
|   |-- app.py
|   |-- iris_project.py
|   |-- requirements.txt
|   |-- README.md
|   |-- data/
|   |-- models/
|   |-- reports/
|   `-- src/
|-- ArpayitriBhattacharyya_Task2/
|   |-- app.py
|   |-- Unemployment_Analysis.ipynb
|   |-- requirements.txt
|   |-- README.md
|   |-- PROJECT_REPORT.md
|   |-- data/
|   |-- reports/
|   `-- src/
|-- ArpayitriBhattacharyya_Task3/
|   |-- app.py
|   |-- train_model.py
|   |-- requirements.txt
|   |-- README.md
|   |-- data/
|   `-- models/
`-- ArpayitriBhattacharyya_Task4/
    |-- app.py
    |-- train_model.py
    |-- requirements.txt
    |-- README.md
    |-- data/
    `-- model/
```

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Streamlit

## How to Run

Open the folder for the project you want to run:

```powershell
cd ArpayitriBhattacharyya_Task1
cd ArpayitriBhattacharyya_Task2
cd ArpayitriBhattacharyya_Task3
cd ArpayitriBhattacharyya_Task4
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

Some tasks also include a model training or analysis script:

```powershell
python iris_project.py
python src/analyze_unemployment.py
python train_model.py
```

Check each task's README for the exact commands and project-specific details.

## Task Details

### Task 1: Iris Flower Classification

- Loads and explores the Iris dataset.
- Trains a K-Nearest Neighbors classifier.
- Saves model metrics, figures, and a trained model file.
- Includes a Streamlit app and command-line prediction script.

### Task 2: Unemployment Analysis

- Cleans and analyzes unemployment data for India.
- Studies unemployment rate, employment, labour participation, region, area, and Covid-period changes.
- Generates charts and a summary report.
- Includes an interactive Streamlit dashboard.

### Task 3: Used Car Price Predictor

- Uses car details such as year, present price, kilometres driven, fuel type, seller type, transmission, and owner count.
- Trains a regression model for estimating used car selling price.
- Includes EDA pages, model comparison, feature importance, and prediction UI.

### Task 4: Email Spam Detection

- Classifies messages as spam or ham.
- Uses TF-IDF vectorization with a Naive Bayes classifier.
- Shows prediction confidence and probability scores.
- Includes automatic model loading/training support in the Streamlit app.

## Author

Arpayitri Bhattacharyya

## Acknowledgement

These projects were completed as part of the Oasis Infobyte Data Science Internship.
