# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('S:\\OASIS SUMMER INTERNSHIP\\TASK 2\\Unemployment in India.csv')

# Inspect column names to ensure they match the expected names
print("Column names:", data.columns)

# Trim any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values if any
# For simplicity, we'll drop missing values. You can choose to fill them based on the context.
data.dropna(inplace=True)

# Convert 'Date' to datetime format if it's not already
data['Date'] = pd.to_datetime(data['Date'])

# Data Visualization

# Unemployment Rate over time for different regions
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend(title='Region')
plt.show()

# Distribution of Unemployment Rate
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Unemployment Rate (%)'], bins=30, kde=True)
plt.title('Distribution of Estimated Unemployment Rate (%)')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
# Ensure all columns used in the correlation matrix are numeric
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Unemployment Rate vs. Labour Participation Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Estimated Labour Participation Rate (%)', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate vs. Labour Participation Rate')
plt.xlabel('Estimated Labour Participation Rate (%)')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend(title='Region')
plt.show()

# Employment over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Estimated Employed', hue='Region')
plt.title('Employment Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Estimated Employed')
plt.legend(title='Region')
plt.show()

# Analysis: Top 5 regions with highest average unemployment rate
avg_unemployment = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(5)
print("Top 5 regions with highest average unemployment rate:")
print(avg_unemployment)

# Analysis: Top 5 regions with highest average labour participation rate
avg_labour_participation = data.groupby('Region')['Estimated Labour Participation Rate (%)'].mean().sort_values(ascending=False).head(5)
print("Top 5 regions with highest average labour participation rate:")
print(avg_labour_participation)
