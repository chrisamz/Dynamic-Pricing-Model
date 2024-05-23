# src/time_series_forecasting.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
arima_model_path = 'models/arima_model.pkl'
prophet_model_path = 'models/prophet_model.pkl'
lstm_model_path = 'models/lstm_model.h5'
results_path = 'results/time_series_forecasting_results.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(arima_model_path), exist_ok=True)
os.makedirs(os.path.dirname(prophet_model_path), exist_ok=True)
os.makedirs(os.path.dirname(lstm_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Ensure date column is in datetime format
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Set date as index
data.set_index('date', inplace=True)

# Split data into training and test sets
train_data = data.iloc[:-30]  # Use all but the last 30 days for training
test_data = data.iloc[-30:]   # Use the last 30 days for testing

# Function to evaluate the model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'{model_name} Performance:')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')

    with open(results_path, 'a') as f:
        f.write(f'{model_name} Performance:\n')
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write('\n')

    return mae, rmse

# ARIMA model
print("Training ARIMA model...")
arima_model = ARIMA(train_data['sales'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=30)

# Evaluate ARIMA model
evaluate_model(test_data['sales'], arima_forecast, 'ARIMA')

# Save ARIMA model
arima_model_fit.save(arima_model_path)

# Prophet model
print("Training Prophet model...")
prophet_data = train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)
prophet_forecast = prophet_forecast.set_index('ds').loc[test_data.index]['yhat']

# Evaluate Prophet model
evaluate_model(test_data['sales'], prophet_forecast, 'Prophet')

# Save Prophet model
prophet_model.stan_backend.logger.setLevel('CRITICAL')
prophet_model_path = 'models/prophet_model.pkl'
with open(prophet_model_path, 'wb') as f:
    pickle.dump(prophet_model, f)

# LSTM model
print("Training LSTM model...")
train_values = train_data['sales'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_values)

X_train = []
y_train = []
for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(lstm_model_path, save_best_only=True, monitor='val_loss')

# Train LSTM model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Prepare test data for LSTM model
total_data = pd.concat((train_data['sales'], test_data['sales']), axis=0)
inputs = total_data[len(total_data) - len(test_data) - 60:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict with LSTM model
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Evaluate LSTM model
evaluate_model(test_data['sales'], lstm_predictions, 'LSTM')

print("Time series forecasting completed!")
