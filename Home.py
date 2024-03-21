import streamlit as st
import Pattern_Predictor
import Pattern_Classifier
import Predictor
import napp

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Pattern Predictor", "Pattern Classifier", "Price Predictor", "Input Data Predictor"))


# Define your functions for different functionalities

def pattern_predictor():
    st.title("Cryptocurrency Price Pattern Prediction")
    currencies = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    selected_currency = st.selectbox('Select the currency pair u want to predict:', currencies)

    Pattern_Predictor.run(selected_currency)


def pattern_classifier():
    Pattern_Classifier.run()


def price_predictor():
    Predictor.run()


def input_data_predictor():
    napp.run()


if page == "Pattern Predictor":
    pattern_predictor()
elif page == "Pattern Classifier":
    pattern_classifier()
elif page == "Price Predictor":
    price_predictor()
elif page == "Input Data Predictor":
    input_data_predictor()
