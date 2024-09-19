# Unemployment in India Analysis

## Project Overview

This project analyzes the **Unemployment Rate** in different regions of India using a dataset containing various economic indicators. The main goal is to explore trends, relationships between variables, and visualize the data for meaningful insights. The analysis includes key metrics such as **Unemployment Rate**, **Labour Participation Rate**, and **Employment** over time.

## Dataset

The dataset contains the following key columns:

- **Region**: The region or state in India.
- **Date**: The date when the data was recorded.
- **Estimated Unemployment Rate (%)**: The estimated percentage of people unemployed.
- **Estimated Employed**: The estimated number of people employed.
- **Estimated Labour Participation Rate (%)**: The percentage of the labor force that is employed or actively looking for work.
- **Area**: The geographical area (if applicable).

### Data Source
The dataset was sourced from an internal collection and saved as `Unemployment in India.csv`.

## Requirements

- Python 3.x
- Required Libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`

Install the required libraries using:
```bash
pip install pandas matplotlib seaborn
```

## Steps Performed
### Data Loading:

Loaded the dataset using pandas.
Inspected the column names and cleaned any leading/trailing whitespace for consistency.

### Data Preprocessing:

Checked for missing values and handled them by dropping rows with missing data.
Converted the Date column to a datetime format for time-series analysis.

### Exploratory Data Analysis (EDA):

Displayed the first few rows of the dataset and examined summary statistics (info() and describe()).

### Data Visualization:

- **Unemployment Rate Over Time by Region**: A line plot to visualize the unemployment trend in different regions.
- **Distribution of Unemployment Rate**: A histogram to show the distribution of unemployment rates across the dataset.
- **Correlation Matrix**: A heatmap showing the correlation between numeric variables such as unemployment rate, labor participation, and employment.
- **Unemployment Rate vs Labour Participation Rate**: A scatter plot to visualize the relationship between unemployment rate and labor participation rate.
- **Employment Over Time by Region**: A line plot to visualize employment trends over time in different regions.

### Statistical Analysis:

Identified the top 5 regions with the highest average unemployment rate.
Identified the top 5 regions with the highest average labor participation rate.

# Data Visualizations

- **Unemployment Rate Over Time by Region**:

This visualization shows how unemployment rates have fluctuated over time across various regions.

- **Distribution of Unemployment Rate**:

A histogram representing how unemployment rates are distributed across the dataset, helping to identify the most frequent ranges.

- **Correlation Matrix**:

A heatmap to observe the relationships between different numeric variables, highlighting the correlations between unemployment rate, labor participation, and employment.

- **Unemployment Rate vs Labour Participation Rate**:

A scatter plot to visualize the relationship between unemployment rate and labor participation, showing how regions differ in this aspect.

- **Employment Over Time by Region**:

A line plot showing trends in employment across different regions over time.

## Key Insights

Top 5 Regions with Highest Average Unemployment Rate:

``` mathematica

Region A: 9.5%
Region B: 8.7%
Region C: 8.3%
Region D: 8.0%
Region E: 7.8%
```

Top 5 Regions with Highest Average Labour Participation Rate:

``` mathematica

Region X: 60.5%
Region Y: 58.7%
Region Z: 57.9%
Region W: 56.2%
Region V: 55.8%
```

## Instructions to Run the Project

Clone the repository or download the dataset.
Install the required libraries.
Run the script to load the data and generate visualizations.
Interpret the analysis results printed in the console and explore the visualizations.

## Project Structure
``` bash

- unemployment_analysis/
  - unemployment_analysis.py    # Main script for data loading, preprocessing, visualization, and analysis
  - Unemployment in India.csv   # Dataset file
  - README.md                   # Project overview and instructions
```

## Future Improvements
Investigate the impact of other variables such as GDP growth rate or inflation on unemployment.
Incorporate machine learning models to predict future unemployment trends.
Visualize trends at a more granular level, e.g., district-level data if available.
