import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
data['date_time'] = pd.to_datetime(data['date_time'])
data.set_index('date_time', inplace=True)

### data to hourly data
traffic_series = data['traffic_volume'].resample('H').mean()

# NaN chcker
print("NaN values before cleaning:", traffic_series.isna().sum())

# replace NaN values with median
traffic_series.fillna(  traffic_series.median(), inplace=True )

# scaler = MinMaxScaler()
scaler = StandardScaler()
traffic_scaled = scaler.fit_transform( traffic_series.values.reshape( -1, 1 ) )

# Function to create sequences for LSTM
def create_sequences(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Define sequence length (24 hours = 1 day of historical data)
time_steps = 24
X, y = create_sequences(traffic_scaled, time_steps)

# Split into training and test sets (80% train, 20% test)
split_idx = int( len( X ) * 0.8 )
X_train, X_test = X[ :split_idx ], X[ split_idx: ]
y_train, y_test = y[ :split_idx ], y[ split_idx: ]

# **Fix: Remove or replace NaN values in training data**
X_train = np.nan_to_num(X_train, nan=np.nanmedian( X_train) )
y_train = np.nan_to_num(y_train, nan=np.nanmedian( y_train) )
X_test = np.nan_to_num(X_test, nan=np.nanmedian( X_test ) )
y_test = np.nan_to_num(y_test, nan=np.nanmedian( y_test ) )

# **Verify NaN removal**
print("NaN values after cleaning:")
print("X_train:", np.isnan(X_train).sum(), "y_train:", np.isnan(y_train).sum())
print("X_test:", np.isnan(X_test).sum(), "y_test:", np.isnan(y_test).sum())

# Reshape X data for LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# **Fix: Reset model before training to avoid bad weights**
tf.keras.backend.clear_session()

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),  
    Dropout(0.2),
    LSTM(50, return_sequences=False),  
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# **Fix: Use Gradient Clipping to prevent NaNs**
optimizer = Adam( learning_rate = 0.0005, clipnorm = 1.0 )
model.compile( optimizer = optimizer, loss = 'mean_squared_error' )

# Train the model
history = model.fit( X_train, y_train, epochs=20, batch_size=32, validation_data=( X_test, y_test ) )

# Predict future traffic volume
y_pred = model.predict( X_test )

# **Fix: Ensure predictions are reshaped correctly**
y_pred_inv = scaler.inverse_transform( y_pred.reshape(-1, 1) )
y_test_inv = scaler.inverse_transform( y_test.reshape(-1, 1) )

# **Fix: Print first few predictions to confirm they are not NaN**
print("Predicted Traffic (First 10 values):", y_pred_inv[:10] )
print("Actual Traffic (First 10 values):", y_test_inv[:10] )

# **Plot results**
plot_range = 200  # Last 200 samples

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[-plot_range:], label="Actual Traffic", color='blue', alpha=0.5)
plt.plot(y_pred_inv[-plot_range:], label="Predicted Traffic", color='red', linewidth=3, linestyle='dashed', marker='o', markersize=5)

plt.xlabel("Time (Last 200 Samples)")
plt.ylabel("Traffic Volume")
plt.title("LSTM Traffic Forecasting (Fixed)")
plt.legend()

# Save the updated figure
plt.savefig("lstm_forecast_fixed.png", dpi=300)

plt.show()
