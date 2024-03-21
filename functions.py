import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import plotly.graph_objects as go
import cv2


def denoise_image(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def enhance_contrast(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_img)


def detect_edges(img, threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)


def preprocess_image(uploaded_file):
    # Open the image using PIL (Python Imaging Library)
    img = Image.open(uploaded_file)

    # Convert to RGB (if not already in that mode)
    img = img.convert('RGB')
    #img = denoise_image(img)
    #img = enhance_contrast(img)
    #img = detect_edges(img, 100, 200)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_features(fe_model, uploaded_file):
    x = preprocess_image(uploaded_file)
    features_vector = fe_model.predict(x)
    return features_vector


def reduce_features(pca, scaler, features):
    # Additional preprocessing like PCA and scaling
    features = pca.transform(scaler.transform(features))
    return features


def predict_next_month(model, features):
    # Ensure features is a NumPy array
    features = np.array(features)

    # Reshape the features for prediction
    features = np.reshape(features, (1, 1, features.shape[1]))

    # Predict the next month's features
    prediction = model.predict(features)

    return prediction.flatten()


# Define a function to plot the features
def plot_features(features, title, color, common_y_min, common_y_max):
    plt.figure(figsize=(12, 6))
    plt.plot(features.flatten(), label=title, color=color, marker='o')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Values')
    plt.title(title)
    plt.legend()
    plt.ylim(common_y_min, common_y_max)
    st.pyplot()


def plot_features_plotly(features, title, color, common_y_min, common_y_max):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=features.flatten(), mode='lines+markers', name=title, line=dict(color=color)))
    fig.update_layout(title=title, xaxis_title='Feature Index', yaxis_title='Feature Values', yaxis=dict(range=[common_y_min, common_y_max]))
    st.plotly_chart(fig, use_container_width=True)


def plot_all(input_features, predicted_features):
    # Concatenate the new input features and predicted features
    combined_features = np.concatenate([input_features.flatten(), predicted_features.flatten()])

    # Calculate indices for x-axis to separate actual and predicted data
    x_indices = np.arange(len(combined_features))

    # Split the x_indices to actual and predicted for clarity in plotting
    x_actual = x_indices[:len(input_features.flatten())]
    x_predicted = x_indices[len(predicted_features.flatten()):]

    # Plotting
    plt.figure(figsize=(20, 6))

    # Plot actual features
    plt.plot(x_actual, combined_features[:len(x_actual)], label='New Input Chart Features', color='green', marker='o')

    # Plot predicted features
    plt.plot(x_predicted, combined_features[len(x_actual):], label='Predicted Next Month', color='orange', marker='x')

    # Adding legend to differentiate actual and predicted data
    plt.legend()

    # Labels and Title
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Values')
    plt.title('Combined View of Input Chart Features and Predicted Features')

    # Show grid
    plt.grid(True)

    # Display the plot
    st.pyplot()


def plot_all_plotly(input_features, predicted_features):
    combined_features = np.concatenate([input_features.flatten(), predicted_features.flatten()])
    x_indices = np.arange(len(combined_features))
    x_actual = x_indices[:len(input_features.flatten())]
    x_predicted = x_indices[len(input_features.flatten()):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_actual, y=combined_features[:len(x_actual)], mode='lines+markers',
                             name='New Input Chart Features', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_predicted, y=combined_features[len(input_features.flatten()):], mode='lines+markers',
                             name='Predicted Next Month', marker=dict(color='orange')))

    fig.update_layout(title='Combined View of Input Chart Features and Predicted Features', xaxis_title='Feature Index',
                      yaxis_title='Feature Values')
    # fig.update_xaxes(tickvals=np.concatenate([x_actual, x_predicted]),
    #                  ticktext=['Input'] * len(x_actual) + ['Predicted'] * len(x_predicted))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)