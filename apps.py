# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

# Load the trained LSTM model
model = load_model('C:/Users/Dell/Downloads/lstm_model.h5')

# Load the pre-trained ResNet50 model without top classification layers
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
fe_model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D()])

# Load PCA model and other necessary components
pca_model_path = 'C:/Users/Dell/Downloads/pca_model.joblib'
scaler_model_path = 'C:/Users/Dell/Downloads/scaler_model.joblib'

# Load the PCA model
with open(pca_model_path, 'rb') as file:
    pca = joblib.load(file)

# Load the scaler model
with open(scaler_model_path, 'rb') as file:
    scaler = joblib.load(file)


# Other preprocessing functions
def preprocess_image(uploaded_file):
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(uploaded_file)

    # Convert to RGB (if not already in that mode)
    img = img.convert('RGB')

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_features(uploaded_file):
    x = preprocess_image(uploaded_file)
    features_vector = fe_model.predict(x)
    return features_vector


def reduce_features(features):
    # Additional preprocessing like PCA and scaling
    features = pca.transform(scaler.transform(features))
    return features


def predict_next_month(features):
    # Ensure features is a NumPy array
    features = np.array(features)

    # Reshape the features for prediction
    features = np.reshape(features, (1, 1, features.shape[1]))

    # Predict the next month's features
    prediction = model.predict(features)

    return prediction.flatten()


# Define a function to plot the features
def plot_features(features, title, color):
    plt.figure(figsize=(12, 6))
    plt.plot(features.flatten(), label=title, color=color, marker='o')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Values')
    plt.title(title)
    plt.legend()
    plt.ylim(common_y_min, common_y_max)
    st.pyplot()


# Define a function to plot side by side features
def plot_side_by_side(input_features, predicted_features, title1, title2, color1, color2):
    plt.figure(figsize=(35, 10))

    plt.subplot(1, 2, 1)
    plt.plot(input_features.flatten(), label=title1, color=color1, marker='o')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Values')
    plt.title(title1)
    plt.legend()
    plt.ylim(common_y_min, common_y_max)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(predicted_features.flatten(), label=title2, color=color2, marker='o')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Values')
    plt.title(title2)
    plt.legend()
    plt.ylim(common_y_min, common_y_max)
    plt.grid(True)

    plt.tight_layout()
    st.pyplot()


# Streamlit UI
st.title("Cryptocurrency Trend Prediction")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose a cryptocurrency chart image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Extract features from the uploaded image
    input_features = extract_features(uploaded_file)

    # Additional preprocessing steps for user input
    input_features = reduce_features(input_features)

    # Convert input_features to a NumPy array
    input_features = np.array(input_features)

    # Predict the next month's features
    predicted_next_month_features = predict_next_month(input_features)

    # Set common y-axis limits for the plots
    common_y_min = min(np.min(input_features), np.min(predicted_next_month_features))
    common_y_max = max(np.max(input_features), np.max(predicted_next_month_features))

    # Plotting the results using matplotlib
    st.write("## Prediction Results:")

    # Plotting the features of the new input chart image
    plot_features(input_features, 'Input Features', 'green')

    # Plotting the predicted values for the next month
    plot_features(predicted_next_month_features, 'Predicted Next Month', 'orange')

    # Plotting the results using matplotlib
    st.write("## Prediction Results Overall Chart:")

    # Plotting the features side by side
    plot_side_by_side(input_features, predicted_next_month_features, 'Input Features', 'Predicted Next Month', 'green', 'orange')

else:
    st.write("Please upload an image file...")

