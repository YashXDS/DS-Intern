# Car Price Prediction Project

## Project Overview

This project involves building a machine learning model to predict the **Selling Price** of cars based on various features such as **Year**, **Present Price**, **Driven Kilometers**, **Fuel Type**, **Transmission**, and **Owner**. The project utilizes Random Forest Regressor with hyperparameter tuning to achieve optimal model performance.

## Dataset

The dataset used for this project is `car data.csv`. It contains the following columns:
- **Year**: The year the car was manufactured.
- **Present_Price**: The current ex-showroom price of the car.
- **Driven_kms**: The total kilometers driven by the car.
- **Fuel_Type**: The fuel type of the car (Petrol/Diesel/CNG).
- **Selling_type**: Whether the car is being sold by an individual or a dealership.
- **Transmission**: The type of transmission (Manual/Automatic).
- **Owner**: The number of previous owners of the car.
- **Selling_Price**: The price at which the car was sold (target variable).

## Requirements

- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `sklearn`

You can install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Steps Performed
- **Data Preprocessing**:

Dropped irrelevant columns, such as Car_Name.
Handled categorical variables using one-hot encoding.
Standardized numerical features using StandardScaler.

- **Model Building**:

Split the dataset into training and testing sets.
Built a pipeline that includes both preprocessing steps and a Random Forest Regressor model.
Tuned hyperparameters using RandomizedSearchCV for optimal model performance.

- **Model Evaluation**:

Evaluated the model using metrics like Mean Squared Error (MSE) and R-squared Score (R²).
Visualized the comparison between actual and predicted selling prices.
Prediction Function:

Created an interactive function that allows users to input car details to predict the selling price.

- **Model Pipeline**:
```The model pipeline consists of two main components```:

## Preprocessing:

Numeric features are scaled using StandardScaler.
Categorical features are encoded using OneHotEncoder.

## Random Forest Regressor:

The model is optimized using a randomized search over several hyperparameters, including the number of estimators, maximum depth, and more.
Hyperparameter Tuning
We used RandomizedSearchCV to tune the following hyperparameters:

`n_estimators`: Number of trees in the forest.
`max_features`: Number of features to consider when looking for the best split.
`max_depth`: Maximum depth of the tree.
`min_samples_split`: Minimum number of samples required to split an internal node.
`min_samples_leaf`: Minimum number of samples required to be at a leaf node.
`bootstrap`: Whether bootstrap samples are used when building trees.

The best parameters found were:
```
{'regressor__n_estimators': 400,
 'regressor__max_features': 'auto',
 'regressor__max_depth': 30,
 'regressor__min_samples_split': 2,
 'regressor__min_samples_leaf': 1,
 'regressor__bootstrap': True}
```

## Evaluation Metrics

- **Mean Squared Error (MSE)** : Measures the average squared difference between actual and predicted values.
- **R-squared (R²) Score**: Indicates how well the model explains the variance in the target variable.

## Visualizations
Several visualizations were created to provide insights into the dataset:

## Insights and Future Improvements

Feature Importance: Analyzing which features have the most influence on car price can be explored further.
Model Optimization: Further tuning of hyperparameters or trying different models (e.g., XGBoost) could improve prediction accuracy.
