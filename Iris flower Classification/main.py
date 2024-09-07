# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, header=None, names=columns)

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Visualize the data
sns.pairplot(data, hue='species')
plt.show()

# Preprocessing
# Convert species to numeric values
data['species'] = data['species'].astype('category').cat.codes

# Split the data into features and target
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
# Using RandomForestClassifier for this example
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Plot feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Function to predict Iris type based on user input
def predict_iris_type():
    # Take user input
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))
    
    # Create a DataFrame with the user input
    user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                             columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Scale the user input data
    user_data = scaler.transform(user_data)
    
    # Predict the Iris type
    prediction = model.predict(user_data)
    
    # Map prediction to species name
    species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_mapping[prediction[0]]
    
    # Output the prediction
    print(f"The predicted Iris type is {predicted_species}.")

# Call the function to take user input and predict Iris type
predict_iris_type()
