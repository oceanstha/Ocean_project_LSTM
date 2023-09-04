import pandas as pd
from scipy import stats

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


t_statistic1, p_value1 = stats.ttest_ind(actual_values1, predicted_values1)
t_statistic2, p_value2 = stats.ttest_ind(actual_values2, predicted_values2)


alpha = 0.05  # Example significance level

if p_value1 < alpha:
    print ("P_Value_LSTM:", p_value1)
    print ("T_statistics_LSTM", t_statistic1)
    print("Reject the null hypothesis: 'close' and 'predicted_close' have different means.")
else:
    print ("P_Value_LSTM:", p_value1)
    print ("T_statistics_LSTM", t_statistic1)
    print("Fail to reject the null hypothesis: 'close' and 'predicted_close' have similar means.")


if p_value2 < alpha:
    print ("P_Value_GRU:", p_value2)
    print ("T_statistics_LSTM", t_statistic2)
    print("Reject the null hypothesis: 'close' and 'predicted_close' have different means.")
else:
    print ("P_Value_GRU:", p_value2)
    print ("T_statistics_LSTM", t_statistic2)
    print("Fail to reject the null hypothesis: 'close' and 'predicted_close' have similar means.")