import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
# Assuming you have the data in a CSV file named 'iris.csv'
data = pd.read_csv('S:\\OASIS SUMMER INTERNSHIP\\TASK 1\\Iris.csv')

# Separate features and target
X = data.drop(columns=['Species'])
y = data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Check if the accuracy is above 95%
if accuracy >= 0.95:
    print("The model has achieved an accuracy above 95%.")
else:
    print("The model did not achieve an accuracy above 95%.")
