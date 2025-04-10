# Import necessary libraries
import pandas as pd
import numpy as np
# PyOD is a popular Python library for outlier detection
from pyod.models.knn import KNN
import matplotlib.pyplot as plt

# --- Data Loading and Preparation ---

# Load the wearable data (e.g., Fitbit heart rate)
# index_col='datetime', parse_dates=True: Treats the 'datetime' column as the index and parses it as dates
wearable_data = pd.read_csv("data/P100300/Orig_Fitbit_HR.csv",
                            index_col='datetime',
                            parse_dates=True) # Ensure file path is correct

# Resample data to get median overnight RHR
# '6H': Resamples data into 6-hour intervals
# .median(): Calculates the median heart rate for each interval
# [::4]: Selects every 4th entry, effectively getting one median value per day (assuming 24hr/6hr = 4 intervals)
wd = wearable_data.resample('6H').median()[::4]

# Remove any days with missing data (NaN values)
wd = wd.dropna()

# Preview the prepared data
print("Preview of Median Overnight RHR Data:")
print(wd.head(10))

# --- Anomaly Detection using KNN ---

# Initialize the KNN detector from PyOD
# contamination=0.03: Assumes about 3% of the data are outliers (adjust based on domain knowledge)
# method='mean': Uses the mean distance to the k-neighbors to calculate outlier score
# n_neighbors=5: Considers the 5 nearest neighbors for outlier calculation
knn = KNN(contamination=0.03, method='mean', n_neighbors=5)

# Fit the model to the heart rate data
# Note: KNN for PyOD expects a 2D array, even for univariate time series.
# We usually pass wd[['heartrate']] or wd.values.reshape(-1, 1). Let's use the former for clarity.
knn.fit(wd[['heartrate']])

# Predict outliers (1 indicates outlier, 0 indicates inlier)
predicted_labels = knn.predict(wd[['heartrate']])
# Convert predictions to a pandas Series with the original datetime index for easier handling
predicted = pd.Series(predicted_labels, index=wd.index)

# Filter to get only the data points flagged as outliers
outlier_indices = predicted[predicted == 1].index
outliers = wd.loc[outlier_indices]

# Display the dates and heart rates identified as outliers
print("\n--- Detected Outliers (Anomalies) ---")
print(outliers)

# --- Visualization ---

# Define a function to plot the time series with outliers highlighted
def plot_outliers(outliers_df, data_df, labels=False, title='Wearable RHR (overnight) - KNN'):
    """Plots time series data and highlights outliers."""
    ax = data_df.plot(alpha=0.6, figsize=(15, 5), legend=False) # Start plotting the main data

    # Plot outliers as distinct markers
    outlier_style = 'X' if not labels else 'v'
    outlier_color = 'crimson'
    ax.plot(outliers_df.index, outliers_df['heartrate'], outlier_style,
            color=outlier_color, markersize=9 if not labels else 8,
            label='Outliers', markerfacecolor=outlier_color if labels else 'none',
            markeredgecolor='k' if labels else outlier_color)

    # Add text labels for dates if requested
    if labels:
        for date, row in outliers_df.iterrows():
            hr_value = row['heartrate']
            plt.text(date, hr_value - (hr_value * 0.04), # Position label slightly below the point
                     f'{date.strftime("%m/%d")}', fontsize=8,
                     horizontalalignment='right', verticalalignment='top')

    # Add plot titles and labels
    plt.title(title)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('RHR', fontsize=14)
    # Manually create legend items for clarity
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Wearable RHR', alpha=0.6),
                       Line2D([0], [0], marker=outlier_style, color=outlier_color, label='Outliers',
                              markersize=9 if not labels else 8, linestyle='None',
                              markerfacecolor=outlier_color if labels else 'none',
                              markeredgecolor='k' if labels else outlier_color)]
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the time series highlighting outliers (without date labels)
print("\nPlotting time series with outliers marked...")
plot_outliers(outliers, wd, labels=False)

# Plot the time series highlighting outliers (with date labels)
print("\nPlotting time series with outlier dates labeled...")
plot_outliers(outliers, wd, labels=True)
