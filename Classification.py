from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


def classify_image(image, model, image_size):
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add grayscale channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)

    return predictions[0]


def prep_image(image, image_size):
    image = image.convert('L')  # Convert to grayscale ('L' mode)
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add grayscale channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# def preprocess_image(image_path, image_size):
#     image = load_img(image_path, target_size=image_size, color_mode='grayscale')
#     # Convert the image to a numpy array and normalize pixel values
#     image = img_to_array(image) / 255.0
#     image = np.expand_dims(image, axis=-1)
#     image = np.expand_dims(image, axis=0)
#     return image

