# Iris Flower Classification

This project trains a machine learning model that classifies Iris flowers into `setosa`, `versicolor`, and `virginica` using sepal and petal measurements.

## Project Highlights

- Loads and cleans the provided `Iris.csv` dataset.
- Performs exploratory data analysis with Seaborn and Matplotlib.
- Trains a K-Nearest Neighbors classifier using Scikit-learn.
- Uses feature scaling with `StandardScaler` for better KNN performance.
- Saves model metrics, plots, and the trained model for reuse.
- Includes a command-line prediction script for new flower measurements.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Folder Structure

```text
Iris_Classification/
├── data/
│   └── Iris.csv
├── models/
│   └── knn_iris_model.joblib
├── reports/
│   ├── metrics.txt
│   └── figures/
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       ├── pairplot.png
│       └── species_distribution.png
├── src/
│   └── predict.py
├── app.py
├── iris_project.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model and generate reports:

```bash
python iris_project.py
```

Make a prediction with the saved model:

```bash
python src/predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

Open the Streamlit user interface:

```bash
streamlit run app.py
```

## Dataset

The dataset contains 150 Iris flower records with four numeric features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is `Species`.

## Model

The final model is a K-Nearest Neighbors classifier with `k=7`. KNN is a strong beginner-friendly algorithm for this task because the Iris classes form clear groups in the feature space, especially across petal length and petal width.

## Outputs

After running the project, the following files are generated:

- `models/knn_iris_model.joblib`: trained model pipeline
- `reports/metrics.txt`: accuracy and classification report
- `reports/figures/species_distribution.png`: class balance chart
- `reports/figures/pairplot.png`: feature relationship chart
- `reports/figures/correlation_heatmap.png`: feature correlation chart
- `reports/figures/confusion_matrix.png`: model performance chart
