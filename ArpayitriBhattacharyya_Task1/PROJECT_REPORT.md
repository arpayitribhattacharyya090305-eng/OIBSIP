# Iris Flower Classification Report

## Objective

The goal of this project is to classify Iris flowers into three species using machine learning:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The model uses four flower measurements: sepal length, sepal width, petal length, and petal width.

## Dataset Overview

The dataset contains 150 records and 5 useful columns after removing the identifier column:

- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species`

Each species has 50 records, so the dataset is balanced.

## Methodology

1. Loaded the provided CSV file with Pandas.
2. Removed the `Id` column because it does not help prediction.
3. Created exploratory visualizations with Matplotlib and Seaborn.
4. Split the data into training and testing sets using an 80:20 split.
5. Applied `StandardScaler` because KNN depends on distance between points.
6. Trained a K-Nearest Neighbors classifier.
7. Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Saved the trained model using Joblib.

## Model Used

K-Nearest Neighbors was selected because it is simple, interpretable, and performs well on small structured datasets like Iris.

Final model:

- Algorithm: K-Nearest Neighbors
- Number of neighbors: 7
- Test size: 20%
- Random state: 42

## Result

The trained model achieved strong performance on the test data. The detailed classification report is available in:

```text
reports/metrics.txt
```

The generated confusion matrix is available in:

```text
reports/figures/confusion_matrix.png
```

## Conclusion

The KNN model successfully classifies Iris flowers based on their measurements. Petal length and petal width are especially useful features for separating the species. This project demonstrates the complete beginner-friendly machine learning workflow: data loading, exploration, preprocessing, model training, evaluation, and prediction.
