# src/pricing_strategies.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
results_path = 'results/pricing_strategy_results.txt'
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

# Define pricing strategies
def fixed_pricing(price):
    return price

def dynamic_pricing(demand, base_price):
    return base_price * (1 + demand / 100)

def promotional_pricing(day_of_week, base_price):
    if day_of_week in [5, 6]:  # Assume promotions on weekends
        return base_price * 0.9  # 10% discount
    else:
        return base_price

# Apply pricing strategies and calculate revenue
def apply_pricing_strategy(strategy, data, base_price):
    prices = []
    revenues = []
    for index, row in data.iterrows():
        if strategy == 'fixed':
            price = fixed_pricing(base_price)
        elif strategy == 'dynamic':
            price = dynamic_pricing(row['demand'], base_price)
        elif strategy == 'promotional':
            price = promotional_pricing(row['day_of_week'], base_price)
        else:
            raise ValueError("Unknown strategy")

        prices.append(price)
        revenue = price * row['sales']
        revenues.append(revenue)
    
    data['price'] = prices
    data['revenue'] = revenues

    return data

# Evaluate strategies
strategies = ['fixed', 'dynamic', 'promotional']
base_price = data['price'].mean()
results = {}

for strategy in strategies:
    strategy_data = apply_pricing_strategy(strategy, data.copy(), base_price)
    total_revenue = strategy_data['revenue'].sum()
    avg_price = strategy_data['price'].mean()
    results[strategy] = {'total_revenue': total_revenue, 'avg_price': avg_price}

    print(f"{strategy.capitalize()} Pricing Strategy:")
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Average Price: ${avg_price:.2f}")

    with open(results_path, 'a') as f:
        f.write(f"{strategy.capitalize()} Pricing Strategy:\n")
        f.write(f"Total Revenue: ${total_revenue:.2f}\n")
        f.write(f"Average Price: ${avg_price:.2f}\n")
        f.write('\n')

# Plot strategy comparison
strategies_names = [s.capitalize() for s in strategies]
total_revenues = [results[s]['total_revenue'] for s in strategies]
avg_prices = [results[s]['avg_price'] for s in strategies]

plt.figure(figsize=(12, 6))
plt.bar(strategies_names, total_revenues, color=['blue', 'green', 'red'])
plt.xlabel('Pricing Strategy')
plt.ylabel('Total Revenue')
plt.title('Total Revenue by Pricing Strategy')
plt.savefig(os.path.join(figures_path, 'total_revenue_by_pricing_strategy.png'))
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(strategies_names, avg_prices, color=['blue', 'green', 'red'])
plt.xlabel('Pricing Strategy')
plt.ylabel('Average Price')
plt.title('Average Price by Pricing Strategy')
plt.savefig(os.path.join(figures_path, 'average_price_by_pricing_strategy.png'))
plt.show()

print("Pricing strategy evaluation completed!")
