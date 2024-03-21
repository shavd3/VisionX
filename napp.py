import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pred_functions


def run():
    # Load the model
    model_path = 'Files/Models/btc_price_prediction_model_4.h5'
    model = load_model(model_path)

    # Streamlit UI
    st.title('Cryptocurrency Price Prediction')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Preprocess data
        df = pred_functions.preprocess_data(uploaded_file)

        # Display original data plot
        st.write("Original Data:")
        pred_functions.plot_data(df)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['close']].values)

        # Prepare the last sequence
        sequence_length = 60
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = np.expand_dims(last_sequence, axis=0)

        # Predict for the next month
        future_dates = pd.date_range(start=df['timestamp'].iloc[-1] + timedelta(hours=1), periods=720,
                                     freq='H')
        future_df = pd.DataFrame(future_dates, columns=['timestamp'])

        predictions = []
        for _ in range(len(future_df)):
            next_step = model.predict(last_sequence)
            predictions.append(next_step[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1] = next_step

        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_transformed = scaler.inverse_transform(predictions_array)
        future_df['predicted_close'] = predictions_transformed

        # Display predicted data plot
        st.write("Prediction:")
        pred_functions.plot_data(df, future_df)
