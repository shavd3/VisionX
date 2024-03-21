from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np


def explain_classification(image, model, class_names):
    explainer = lime_image.LimeImageExplainer()

    # Convert the grayscale image to RGB
    image_rgb = np.stack((image,) * 3, axis=-1)

    explanation = explainer.explain_instance(image_rgb,
                                             lambda x: model.predict(x),
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    img_boundry = mark_boundaries(temp / 2 + 0.5, mask)
    return img_boundry
