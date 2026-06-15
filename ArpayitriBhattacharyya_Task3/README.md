# Used Car Price Predictor

This is a Streamlit mini project for estimating the selling price of a used car.
The prediction is based on a small car dataset and a scikit-learn regression
pipeline.

## Objective

The aim of this project is to understand how basic car details affect resale
price and to build a simple web app where a user can enter car information and
get an estimated selling price.

## Dataset

The project uses `data/car data.csv`. The main columns used for prediction are:

- `Year`
- `Present_Price`
- `Driven_kms`
- `Fuel_Type`
- `Selling_type`
- `Transmission`
- `Owner`

The target column is `Selling_Price`, measured in lakh.

## Method

The model is trained in `train_model.py`.

I used a `RandomForestRegressor` because car resale price does not always change
in a straight line with age or kilometres driven. For example, two cars with the
same age can still have different prices depending on fuel type, seller type,
transmission, and current showroom price.

The preprocessing is handled inside one scikit-learn pipeline:

- Numeric values are imputed with the median.
- Categorical values are imputed with the most frequent value.
- Categorical columns are converted using one-hot encoding.
- The trained model is saved as `models/car_price_model.joblib`.

## App Features

- Home page with summary statistics
- Prediction form for entering car details
- EDA page with age, fuel type, transmission, depreciation, and correlation charts
- Dataset page with charts and sample rows
- Feature importance chart for the trained Random Forest model
- Model comparison table for Linear Regression, Ridge Regression, Gradient Boosting, and Random Forest
- Model details page with score, train/test rows, and features used

## How To Run

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Train the model:

```powershell
python train_model.py
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

## Current Model Result

After training on the provided dataset, the model gives approximately:

- MAE: `0.72 lakh`
- R2 score: `0.921`

These values may change slightly if the train/test split or model parameters are
changed.

The app also compares a few regression models using the same train/test split.
In the current run, Gradient Boosting gives the lowest MAE, while Random Forest
is kept as the saved model for the main prediction workflow.

## Limitations

This project is based on a limited dataset, so the prediction should be treated
as an estimate only. In real life, resale price also depends on factors that are
not present in the dataset, such as car brand, exact model variant, service
history, accident history, city, insurance status, and market demand.

## Possible Improvements

- Add brand and model name as features.
- Try Linear Regression, XGBoost, or Gradient Boosting and compare results.
- Add more recent car records.
- Add a page for feature importance.
- Validate predictions with real listings from different cities.
