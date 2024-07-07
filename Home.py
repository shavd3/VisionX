import streamlit as st
import Pattern_Predictor
import Pattern_Classifier
import Predictor
import napp

# Sidebar for navigation
st.sidebar.title("CryptoVisionX")
page = st.sidebar.radio("Go to", ("Pattern Predictor", "Pattern Classifier", "Input Data Predictor"))

# Adding a watermark at the bottom
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 0;
        right: 0;
        opacity: 0.5;
        z-index: 9999;
        font-size: 10px;
        padding: 5px;
    }
    </style>
    <div class="watermark">
        Developed by Shavin Fernando
    </div>
    """, unsafe_allow_html=True)

# Define the functions for different functionalities


def pattern_predictor():
    st.title("Cryptocurrency Price Pattern Prediction")
    currencies = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'LTC/USDT', 'DOGE/USDT', 'SOL/USDT',
                  'AVA/USDT', 'DOT/USDT', 'OTHER']
    selected_currency = st.selectbox('Select the cryptocurrency you want to predict:', currencies)

    Pattern_Predictor.run(selected_currency)


def pattern_classifier():
    Pattern_Classifier.run()


# def price_predictor():
#     Predictor.run()


def input_data_predictor():
    napp.run()


if page == "Pattern Predictor":
    pattern_predictor()
elif page == "Pattern Classifier":
    pattern_classifier()
# elif page == "Price Predictor":
#     price_predictor()
elif page == "Input Data Predictor":
    input_data_predictor()
