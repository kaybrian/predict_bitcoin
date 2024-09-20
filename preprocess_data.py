import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# Create a directory to save preprocessed data
def create_output_directory(output_dir='processed_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Visualize the dataset
def inspect_and_visualize_data(data):
    # General info and first few rows
    print(data.info())
    print(data.head())

    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:\n", missing_values)

    # Visualize price trends over time
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    plt.figure(figsize=(14, 7))
    plt.plot(data['Timestamp'], data['Close'], label='Close Price')
    plt.title('Bitcoin Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Plot distribution of the Close price
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Close'], bins=50, kde=True)
    plt.title('Distribution of Close Prices')
    plt.xlabel('Close Price (USD)')
    plt.show()

    # Visualize the correlation between different features
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Features')
    plt.show()

# Preprocess the dataset
def preprocess_data(data, output_dir):
    # Select relevant columns
    data = data[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)']]

    # Drop rows with missing values
    data = data.dropna()

    # Rescale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']])

    # Create sequences of 24 hours of past data (1440 minutes) to predict the next hour's Close price
    X, y = [], []
    time_window = 60  # predicting one hour ahead
    seq_length = 1440  # 24 hours of past data
    for i in range(seq_length, len(scaled_data) - time_window):
        X.append(scaled_data[i - seq_length:i])
        y.append(scaled_data[i + time_window - 1, 3])  # Index 3 is the Close price

    X, y = np.array(X), np.array(y)

    # Save preprocessed data
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    return X, y, scaler

if __name__ == "__main__":
    # Define file path and output directory
    file_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    output_dir = create_output_directory()

    # Load and inspect the data
    data = load_data(file_path)
    inspect_and_visualize_data(data)

    # Preprocess the data and save it in the output directory
    X, y, scaler = preprocess_data(data, output_dir)
    print(f"Preprocessed data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Preprocessed data saved to {output_dir}")
