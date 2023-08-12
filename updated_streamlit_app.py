import streamlit as st
import numpy as np
from data_preprocessing import load_data, preprocess_data, compute_rsi
from extended_model import build_rnn_model, build_gru_model, build_stacked_lstm_model

# Load data
df = load_data("dataA.csv")

st.title("Stock Price Prediction Web App")

# Select stock
stock_list = df["Symbol"].unique()
selected_stock = st.selectbox("Select a stock for prediction:", stock_list)

# Filter data by stock
df_stock = df[df["Symbol"] == selected_stock]

# Get the most recent 'Open' value for the selected stock
default_open = df_stock.iloc[-1]["Open"]

# Compute 'Moving_Avg' and 'RSI' for the entire dataset
df_stock['Moving_Avg'] = df_stock['Close'].rolling(window=5).mean()
df_stock['RSI'] = compute_rsi(df_stock['Close'], 5)

# Get the most recent values for 'Moving_Avg' and 'RSI'
default_moving_avg = df_stock.iloc[-1]["Moving_Avg"]
default_rsi = df_stock.iloc[-1]["RSI"]

# Preprocess data
x_data, target, scaler_target = preprocess_data(df_stock)

# User input for 'Open', 'Moving_Avg', and 'RSI' with default values
user_input_open = st.number_input("Enter 'Open' price:", min_value=0.0, step=0.01, value=default_open)
user_input_moving_avg = st.number_input("Enter 'Moving Average':", min_value=0.0, step=0.01, value=default_moving_avg)
user_input_rsi = st.number_input("Enter 'RSI' value:", min_value=0.0, max_value=100.0, step=0.01, value=default_rsi)

# Create a feature array from user input
user_input_features = np.array([[user_input_open, user_input_moving_avg, user_input_rsi]])

# Select model
model_type = st.selectbox("Select a model:", ["RNN", "GRU", "LSTM"])

if model_type == "RNN":
    model = build_rnn_model(x_data[0].shape)
elif model_type == "GRU":
    model = build_gru_model(x_data[0].shape)
else:
    model = build_stacked_lstm_model(x_data[0].shape)

# Load trained model weights
model_weights_path = f"models/{selected_stock}_{model_type}.h5"
model.load_weights(model_weights_path)

# Predict using the last sequence from data + user input
input_sequence = np.concatenate((x_data[-1][1:], user_input_features)).reshape(1, x_data[0].shape[0], x_data[0].shape[1])
prediction = model.predict(input_sequence)
predicted_price = scaler_target.inverse_transform(prediction)[0][0]

st.write(f"Predicted Closing Price for {selected_stock} using {model_type}: {predicted_price:.2f}")