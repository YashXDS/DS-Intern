# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('S:\\OASIS SUMMER INTERNSHIP\\TASK 3\\car data.csv')

# Inspect column names to ensure they match the expected names
print("Column names:", data.columns)

# Trim any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Data preprocessing
# Drop irrelevant columns like Car_Name if not needed for prediction
data.drop(columns=['Car_Name'], inplace=True)

# Convert categorical variables to numerical using one-hot encoding
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numeric_cols = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Splitting the data into feature and target sets
X = data.drop(columns=['Selling_Price'])
y = data['Selling_Price']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scaling numeric features and one-hot encoding categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the pipeline - preprocessing + model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])

# Model Selection and Training
# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300, 400, 500],
    'regressor__max_features': ['auto', 'sqrt', 'log2'],
    'regressor__max_depth': [None, 10, 20, 30, 40, 50],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print("Best parameters found:")
print(search.best_params_)

# Train the model with best parameters
best_model = search.best_estimator_
best_model.fit(X_train, y_train)

# Model Evaluation
# Predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Function to take user input and predict car price
def predict_car_price(model, numeric_features, categorical_features):
    """
    Predict the selling price of a car based on user inputs.
    
    Parameters:
    model (Pipeline): The trained model pipeline
    numeric_features (list): List of numeric feature names
    categorical_features (list): List of categorical feature names
    """
    # Gather user inputs
    inputs = {}
    print("Enter the following details to predict the car selling price:")
    for feature in numeric_features:
        inputs[feature] = float(input(f"Enter {feature}: "))
    for feature in categorical_features:
        inputs[feature] = input(f"Enter {feature}: ")
    
    # Create a DataFrame from inputs
    input_df = pd.DataFrame([inputs])
    
    # Predict the price using the pipeline
    predicted_price = model.predict(input_df)
    return predicted_price[0]

# Example usage
predicted_price = predict_car_price(best_model, numeric_cols, categorical_cols)
print(f"The predicted selling price of the car is: {predicted_price:.2f}")

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.show()
