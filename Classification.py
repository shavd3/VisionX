import numpy as np


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


# Define a dictionary mapping class names to market directions
pattern_direction = {
    'Double_Bottom': 'A bullish reversal pattern indicating upward movement',
    'Ascending_Triangle': 'A bullish continuation pattern indicating upward movement',
    'Falling_Wedge': 'A bullish reversal pattern indicating upward movement',
    'Double_Top': 'A bearish reversal pattern indicating downward movement',
    'Symmetrical_Triangle': 'A continuation pattern, direction depends on breakout',
    'Rising_Wedge': 'A bearish reversal pattern indicating downward movement',
    'Descending_Triangle': 'A bearish continuation pattern indicating downward movement'
}

