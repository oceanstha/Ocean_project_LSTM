import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('/home/ocean/Documents/nepse_test_file_predicted.csv')

# Assuming 'date', 'close', and 'predicted_close' are columns in your CSV
date = data['Date']
close = data['Close']
predicted_close = data['predicted_close']

plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

# Plot the 'close' curve in blue
plt.plot(date, close, color='blue', label='Actual Close')

# Plot the 'predicted_close' curve in orange
plt.plot(date, predicted_close, color='orange', label='Predicted Close')

# Get the x-axis tick locations and labels
x_ticks = plt.xticks()[0]

# Choose which ticks to display (e.g., every 2nd tick)
display_ticks = x_ticks[::10]  # Adjust the step value as needed

# Update the x-axis ticks
plt.xticks(display_ticks, rotation=45)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual Close vs Predicted Close')
plt.legend()
plt.grid(False)
#plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.tight_layout()
plt.show()
