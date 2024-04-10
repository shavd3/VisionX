import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pred_functions


def run():
    # Assuming the existing dataset path and model path are defined
    dataset_path = 'CSV/Bitcoin_BTCUSDT.csv'
    model_path = 'Files/Models/btc_price_prediction_model_4.h5'


    # Load dataset
    df = pred_functions.load_and_preprocess_dataset(dataset_path)

    # Extract date from timestamp and group by date to get daily closing prices
    df['date'] = df['timestamp'].dt.date  # Extract date from timestamp
    daily_df = df.groupby('date')['close'].agg('last').reset_index()  # last price of each day as the daily closing price
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # Display original data plot
    st.title('Cryptocurrency Price Prediction')
    st.write("Original Data:")
    pred_functions.plot_daily_data(daily_df)

    # Load the model
    model = load_model(model_path)

    # User inputs for prediction frequency and duration
    freq_options = {'Hourly': 'H', 'Daily': 'D', 'Minutely': 'T'}
    pred_freq = st.selectbox('Select prediction frequency:', list(freq_options.keys()))
    duration_options = {
        "1 day": 1,
        "15 days": 15,
        "1 month": 30,
        "2 months": 60,
        "5 months": 150,
        "1 year": 365,
        "5 years": 365 * 5
    }

    # Let the user select the prediction duration
    pred_duration_label = st.selectbox('Select prediction duration:', list(duration_options.keys()))
    pred_duration = duration_options[pred_duration_label]
    # pred_duration = st.number_input('Enter the number of prediction periods:', min_value=1, value=30)

    if st.button('Predict'):
        # Adjusting the prediction range based on user input
        if freq_options[pred_freq] == 'D':
            timedelta_freq = pd.Timedelta(days=1)
            periods = pred_duration
        elif freq_options[pred_freq] == 'H':
            timedelta_freq = pd.Timedelta(hours=1)
            periods = pred_duration * 24  # Assuming 24 hours for each day
        else:  # Minutely ('T')
            timedelta_freq = pd.Timedelta(minutes=1)
            periods = pred_duration * 24 * 60  # Assuming 60 minutes in an hour, 24 hours for each day

        # Adjusting the start date for future_dates based on the selected frequency
        future_dates = pd.date_range(start=df['timestamp'].iloc[-1] + timedelta_freq, periods=periods,
                                     freq=freq_options[pred_freq])
        future_df = pd.DataFrame(future_dates, columns=['timestamp'])

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['close']].values.reshape(-1, 1))

        # Prediction process
        sequence_length = 60
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = np.expand_dims(last_sequence, axis=0)

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
        st.write(f"Prediction for the next {pred_duration} {pred_freq.lower()} periods:")
        lim_df = df[df['timestamp'].dt.year > 2020]
        pred_functions.plot_data_with_plotly(lim_df, future_df)
