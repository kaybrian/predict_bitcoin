import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load preprocessed data
def load_preprocessed_data():
    X = np.load('X.npy')
    y = np.load('y.npy')
    return X, y

# Create the RNN model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer for predicting close price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and validate the model
def train_and_evaluate(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create the model
    model = create_model((X_train.shape[1], X_train.shape[2]))

    # Convert data into a TensorFlow Dataset for efficient training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

    # Train the model
    model.fit(train_dataset, epochs=10, validation_data=test_dataset)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Test MSE: {mse}")

    # Save the trained model
    model.save('btc_forecast_model.h5')

if __name__ == "__main__":
    X, y = load_preprocessed_data()
    train_and_evaluate(X, y)
