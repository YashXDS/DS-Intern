# Iris Classification Project

## Project Overview

This project aims to classify **Iris flower species** using machine learning techniques. The species included in the dataset are **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica**. The classification is based on features such as **sepal length**, **sepal width**, **petal length**, and **petal width**. The project uses the **Random Forest Classifier** to predict the species, and it also provides a function for users to input flower measurements and predict the species interactively.

## Dataset

The dataset used in this project is the well-known [Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data). It contains the following columns:
- **sepal_length**: Length of the sepal (in cm).
- **sepal_width**: Width of the sepal (in cm).
- **petal_length**: Length of the petal (in cm).
- **petal_width**: Width of the petal (in cm).
- **species**: The species of the flower, which can be one of the following:
  - *Iris-setosa*
  - *Iris-versicolor*
  - *Iris-virginica*

## Requirements

- Python 3.x
- Required Libraries:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`
  - `numpy`

Install the required libraries using:
```bash
pip install pandas seaborn matplotlib scikit-learn numpy
```
## Steps Performed
### Data Loading:

The dataset is loaded directly from the UCI Machine Learning repository.
The column names are added to the dataset for better readability.

### Data Exploration:

Display the first few rows of the dataset.
Generate summary statistics and check for data types using info() and describe().
Visualized relationships between features using pair plots colored by species.

### Data Preprocessing:

Converted the species column into numeric values for model training.
Split the dataset into training and testing sets (70% training, 30% testing).
Scaled the features using StandardScaler to standardize the data for better model performance.

### Model Training:

A Random Forest Classifier was used for classification.
The model was trained using the training dataset.

### Model Evaluation:

Evaluated the model's performance using a confusion matrix, classification report, and accuracy score.
Visualized the feature importances to understand which features are most significant in the classification.

### Prediction Function:

An interactive function allows users to input flower measurements and predicts the Iris species based on the trained model.

## Model Evaluation

`Confusion Matrix`: A matrix showing the number of correct and incorrect predictions for each class.
`Classification Report`: Provides precision, recall, F1-score, and support for each class.
`Accuracy Score`: The overall accuracy of the model on the test data.

## Example model evaluation metrics:

- **Confusion Matrix**:
```
[[16  0  0]
 [ 0 13  1]
 [ 0  0 15]]
```
- **Classification Report**:

``` markdown
             precision    recall  f1-score   support

    Setosa       1.00      1.00      1.00        16
Versicolor       1.00      0.93      0.96        14
 Virginica       0.94      1.00      0.97        15

 Accuracy                            0.98        45
Macro avg        0.98      0.98      0.98        45
Weighted avg     0.98      0.98      0.98        45
```

- **Accuracy Score**: 0.98

## Feature Importance

The Random Forest Classifier allows us to see which features are most important for predicting the species. A bar plot of feature importances shows which features had the most influence.

## Prediction Function

An interactive function is included that allows users to input the following details to predict the Iris species:
- Sepal length (in cm)
- Sepal width (in cm)
- Petal length (in cm)
- Petal width (in cm)

Based on the inputs, the function outputs the predicted Iris species.

## Visualizations

- **Pair Plot**: A pairwise comparison of the features, color-coded by species.
- **Feature Importance Plot**: A bar plot showing the importance of each feature in the Random Forest model.

## Instructions to Run the Project

1. Clone the repository or download the files.
2. Install the required libraries.
3. Run the main script to train the model and make predictions.
4. Use the interactive function to predict Iris species based on your inputs.

## Project Structure

iris_classification_project/
iris_classification.py # Main script for data processing, training, evaluation, and prediction
README.md # Project overview and instructions
markdown
Copy code

## Conclusion

This project demonstrates a practical implementation of an **Iris flower classification model** using the **Random Forest Classifier**. The model achieves high accuracy and allows users to input measurements to predict the Iris species. Visualization of feature importances and pair plots further provides insights into the dataset.

## Future Improvements

- Implement more models like **SVM** or **K-Nearest Neighbors** for comparison.
- Tune hyperparameters of the Random Forest model to further improve accuracy.
