import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam

# Load the training and testing data from CSV files
df_train = pd.read_csv('/home/ocean/Documents/nepsePreprocessed.csv')
df_test = pd.read_csv('/home/ocean/Documents/nepsefuture.csv')

# Extract the 'close' feature as the target variable for training and testing
train_target_variable = df_train['Close']
test_target_variable = df_test['Close']

# Normalize the target variable to range [0, 1]
scaler = MinMaxScaler()
train_target_variable = scaler.fit_transform(train_target_variable.values.reshape(-1, 1))
test_target_variable = scaler.transform(test_target_variable.values.reshape(-1, 1))

# Define the number of previous time steps to consider for each prediction
time_steps = 10

# Prepare the training data for LSTM training
X_train = []
y_train = []
for i in range(time_steps, len(train_target_variable)):
    X_train.append(train_target_variable[i-time_steps:i, 0])
    y_train.append(train_target_variable[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Prepare the testing data for prediction
X_test = []
y_test = []
for i in range(time_steps, len(test_target_variable)):
    X_test.append(test_target_variable[i-time_steps:i, 0])
    y_test.append(test_target_variable[i, 0])
X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape the input data for LSTM [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Perform predictions on the training data
predicted_values_train = model.predict(X_train)
predicted_values_train = scaler.inverse_transform(predicted_values_train)

# Perform predictions on the testing data
predicted_values_test = model.predict(X_test)
predicted_values_test = scaler.inverse_transform(predicted_values_test)

# Append predicted values to train file
df_train['predicted_close'] = np.nan
df_train['predicted_close'].loc[time_steps:] = predicted_values_train[:, 0]

# Append predicted values to test file
df_test['predicted_close'] = np.nan
df_test['predicted_close'].loc[time_steps:] = predicted_values_test[:, 0]


# Save the updated train and test files
df_train.to_csv('/home/ocean/Documents/nepse_train_file_predicted.csv', index=False)
df_test.to_csv('/home/ocean/Documents/nepse_test_file_predicted.csv', index=False)
