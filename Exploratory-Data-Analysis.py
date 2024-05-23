# src/exploratory_data_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Distribution of numerical features
numerical_features = ['sales', 'price', 'price_diff', 'rolling_mean_7', 'rolling_std_7']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_path, f'{feature}_distribution.png'))
    plt.show()

# Correlation matrix
print("Correlation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(figures_path, 'correlation_matrix.png'))
plt.show()

# Time series plot of sales
plt.figure(figsize=(15, 6))
sns.lineplot(x='date', y='sales', data=data)
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig(os.path.join(figures_path, 'sales_over_time.png'))
plt.show()

# Box plots for categorical features
categorical_features = ['year', 'month', 'day_of_week']
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=feature, y='sales', data=data)
    plt.title(f'Sales by {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Sales')
    plt.savefig(os.path.join(figures_path, f'sales_by_{feature}.png'))
    plt.show()

print("Exploratory Data Analysis completed!")
