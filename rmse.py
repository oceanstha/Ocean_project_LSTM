import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the CSV file into a pandas DataFrame
df1 = pd.read_csv('/home/ocean/Documents/nepse_test_file_predicted.csv')
df2 = pd.read_csv('/home/ocean/Documents/nepse_test_file_predictedGRU.csv')

# Extract the 'close' and 'predicted_close' columns
actual_values1 = df1['Close'].values
predicted_values1 = df1['predicted_close'].values

actual_values2 = df2['Close'].values
predicted_values2 = df2['predicted_close'].values

# Exclude the first ten rows from the calculations
actual_values1 = actual_values1[10:]
predicted_values1 = predicted_values1[10:]

actual_values2 = actual_values2[10:]
predicted_values2 = predicted_values2[10:]

#calculate average prediction error
mean_actual1=np.mean(actual_values1)
mean_predicted1=np.mean(predicted_values1)

mean_actual2=np.mean(actual_values2)
mean_predicted2=np.mean(predicted_values2)

mean_prediction_error1=np.abs((mean_predicted1-mean_actual1)/mean_actual1)*100
mean_prediction_error2=np.abs((mean_predicted2-mean_actual2)/mean_actual2)*100

# Calculate MSE
mse_lstm = mean_squared_error(actual_values1, predicted_values1)
mse_gru = mean_squared_error(actual_values2, predicted_values2) 
                            
# Calculate RMSE
rmse_lstm = np.sqrt(mean_squared_error(actual_values1, predicted_values1))
rmse_GRU = np.sqrt(mean_squared_error(actual_values2, predicted_values2))


# Print the average prediction error, MSE and RMSE value
print("MEAN_LSTM:", mean_actual1)
print("APE_LSTM:", mean_prediction_error1)
print("APE_GRU:", mean_prediction_error2)
print("MSE_LSTM:", mse_lstm)
print("MSE_GRU:", mse_gru)
print("RMSE_LSTM:", rmse_lstm)
print("RMSE_GRU:", rmse_GRU)
