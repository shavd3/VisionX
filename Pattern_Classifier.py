# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import Classification
import XAI_lime


def run():
    # Load classification model
    # @st.cache_data()
    def load_model(model_path):
        class_model = tf.keras.models.load_model(model_path)
        return class_model


    class_model = load_model('Files/Models/crypto_chart_pattern_classifier.h5')

    # Streamlit UI
    # Classify the uploaded image
    st.title('Cryptocurrency Chart Pattern Classification')
    st.write('Upload an image of the cryptocurrency chart to classify its pattern.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        class_names = ['Double_Bottom', 'Ascending_Triangle', 'Falling_Wedge', 'Double_Top',
                       'Symmetrical_Triangle', 'Rising_Wedge', 'Descending_Triangle']  # Class names
        predictions = Classification.classify_image(image, class_model, (256, 256))
        class_prediction = class_names[np.argmax(predictions)]
        # st.write(f"Prediction: {class_prediction}")
        # Container to display the results
        container = st.container()
        with container:
            st.markdown(f"<div style='border-style: solid; border-width: 2px; padding: 10px; text-align: center;'>"
                        f"<strong>Classified Pattern Type: {class_prediction}</strong></div>",
                        unsafe_allow_html=True)
            st.write(" ")
            # Additionally, inform the user about the market direction based on the classification
            st.markdown(f"<div style='border-style: solid; border-width: 2px; padding: 10px; text-align: center;'>"
                        f"<div style='text-align: center;'><em>{Classification.pattern_direction[class_prediction]}"
                        f"</em></div>",
                        unsafe_allow_html=True)
        # Display the probabilities for each class
        st.write(" ")
        st.write("Probabilities:")
        for class_name, probability in zip(class_names, predictions):
            st.write(f"{class_name}: {probability:.2%}")

        # # Display LIME explanation
        # explanation_img = XAI_lime.explain_classification(Classification.prep_image(image, (256, 256)),
        #                                                   class_model, class_names)
        # st.image(explanation_img, caption='LIME Explanation', use_column_width=True)
