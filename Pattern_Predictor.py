# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import functions
import Grad_Cam
import Pattern_Classifier


def run(selected_currency):

    if selected_currency == 'BTC/USDT':
        model = load_model('Files/Models/lstm_model_BTCUSDT.h5')
    elif selected_currency == 'ETH/USDT':
        model = load_model('Files/Models/lstm_model_ETHUSDT.h5')
    elif selected_currency == 'BNB/USDT':
        model = load_model('Files/Models/lstm_model_BNBUSDT.h5')
    elif selected_currency == 'XRP/USDT':
        model = load_model('Files/Models/lstm_model_XRPUSDT.h5')
    elif selected_currency == 'ADA/USDT':
        model = load_model('Files/Models/lstm_model_ADAUSDT.h5')
    elif selected_currency == 'LTC/USDT':
        model = load_model('Files/Models/lstm_model_LTCUSDT.h5')
    elif selected_currency == 'DOGE/USDT':
        model = load_model('Files/Models/lstm_model_BTCUSDT.h5')
    elif selected_currency == 'SOL/USDT':
        model = load_model('Files/Models/lstm_model_BTCUSDT.h5')
    elif selected_currency == 'AVA/USDT':
        model = load_model('Files/Models/lstm_model_BTCUSDT.h5')
    elif selected_currency == 'DOT/USDT':
        model = load_model('Files/Models/lstm_model_BTCUSDT.h5')
    else:
        model = load_model('Files/Models/lstm_model_2.h5')

    # Load the pre-trained ResNet50 model without top classification layers
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    fe_model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D()])

    pca_model_path = 'Files/Models/pca_model_2.joblib'
    scaler_model_path = 'Files/Models/scaler_model_2.joblib'

    # Load the PCA model
    with open(pca_model_path, 'rb') as file:
        pca = joblib.load(file)

    # Load the scaler model
    with open(scaler_model_path, 'rb') as file:
        scaler = joblib.load(file)


    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Upload image through Streamlit UI
    uploaded_file = st.file_uploader("Choose a cryptocurrency chart image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Extract features from the uploaded image
        input_features = functions.extract_features(fe_model, uploaded_file)

        # Additional preprocessing steps for user input
        input_features = functions.reduce_features(pca, scaler, input_features)

        # Convert input_features to a NumPy array
        input_features = np.array(input_features)

        # Predict the next month's features
        predicted_next_month_features = functions.predict_next_month(model, input_features)

        # Set common y-axis limits for the plots
        common_y_min = min(np.min(input_features), np.min(predicted_next_month_features))
        common_y_max = max(np.max(input_features), np.max(predicted_next_month_features))

        # Plotting the results using matplotlib
        st.write("## Prediction Results:")

        # Plotting the features of the new input chart image
        functions.plot_features_plotly(input_features, 'Input Features', 'green', common_y_min, common_y_max)

        # Plotting the predicted values for the next month
        functions.plot_features_plotly(predicted_next_month_features, 'Predicted Next Month', 'orange',
                                       common_y_min, common_y_max)

        # Plotting the results using matplotlib
        st.write("## Prediction Results Overall Chart:")

        # Plotting the entire chart with input and predicted features
        functions.plot_all_plotly(input_features, predicted_next_month_features)

        # Explaining the feature involvement for prediction
        st.write("## Involvement of Input image for prediction:")

        img_array = preprocess_input(Grad_Cam.get_img_array(uploaded_file, size=(1024, 1024)))

        last_layer = base_model.layers[-1]
        last_conv_layer_name = last_layer.name

        heatmap = Grad_Cam.make_gradcam_heatmap(img_array, base_model, last_conv_layer_name)

        Grad_Cam.save_and_display_gradcam(uploaded_file, heatmap)

        st.write(" ")
        if st.button('Classify Future Price Direction'):
            # Forecast the future direction of the cryptocurrency price
            st.write("Forecasting the future cryptocurrency price direction...")
            Pattern_Classifier.classify_method(uploaded_file)
    else:
        st.write("Please upload an image to view predictions.")




