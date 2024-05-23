# src/demand_estimation.py

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
results_path = 'results/demand_estimation_results.txt'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Define the features and target variable
X = data[['price']]
y = data['sales']

# Fit a linear regression model
print("Fitting linear regression model...")
model = LinearRegression()
model.fit(X, y)

# Predict demand
y_pred = model.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X, y)

print("Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

with open(results_path, 'w') as f:
    f.write("Demand Estimation Model Performance:\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R-squared: {r2:.4f}\n")

# Calculate demand elasticity
print("Calculating demand elasticity...")
data['price_log'] = np.log(data['price'])
data['sales_log'] = np.log(data['sales'])

elasticity_model = sm.OLS(data['sales_log'], sm.add_constant(data['price_log'])).fit()
elasticity = elasticity_model.params['price_log']

print(f"Demand Elasticity: {elasticity:.4f}")

with open(results_path, 'a') as f:
    f.write(f"Demand Elasticity: {elasticity:.4f}\n")

# Plot the relationship between price and demand
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sales', data=data)
sns.lineplot(x='price', y=y_pred, color='red')
plt.title('Price vs. Sales')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.savefig(os.path.join(figures_path, 'price_vs_sales.png'))
plt.show()

# Plot the log-log relationship between price and sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price_log', y='sales_log', data=data)
sns.lineplot(x='price_log', y=elasticity_model.fittedvalues, color='red')
plt.title('Log-Log Price vs. Sales')
plt.xlabel('Log Price')
plt.ylabel('Log Sales')
plt.savefig(os.path.join(figures_path, 'log_price_vs_log_sales.png'))
plt.show()

print("Demand estimation completed!")
