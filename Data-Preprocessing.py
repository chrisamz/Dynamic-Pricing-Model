# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime

# Define file paths
raw_data_path = 'data/raw/sales_data.csv'
processed_data_path = 'data/processed/processed_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Data Cleaning
print("Cleaning data...")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Feature Engineering
print("Performing feature engineering...")

# Extract year, month, day, and day of week from date
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Calculate price differences (if applicable)
if 'price' in data.columns:
    data['price_diff'] = data['price'].diff().fillna(0)

# Calculate rolling statistics
data['rolling_mean_7'] = data['sales'].rolling(window=7).mean().fillna(0)
data['rolling_std_7'] = data['sales'].rolling(window=7).std().fillna(0)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'date':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Normalize numerical features
print("Normalizing numerical features...")
scaler = MinMaxScaler()
numerical_features = ['sales', 'price', 'price_diff', 'rolling_mean_7', 'rolling_std_7']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save processed data
print("Saving processed data...")
data.to_csv(processed_data_path, index=False)

print("Data preprocessing completed!")
