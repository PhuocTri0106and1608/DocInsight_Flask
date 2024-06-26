import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from helpers_data import process_data
from PIL import Image


IMAGE_RESOLUTION = (250, 250, 1)
BORDER = 30

def shap_model(image):
    img = Image.open(image)
    img.save('input.jpeg')
    img_path = 'input.jpeg'
    data = []
    data.append(process_data(img_path, IMAGE_RESOLUTION, BORDER))
    image = np.array(data)

    model = tf.keras.models.load_model('cnn6_ann3_pow10_adamax_model.h5')

    with open('shap_values.pkl', 'rb') as file:
        value = pickle.load(file)
    # Plot the image explaining the predictions
    shap.image_plot(value, image, show=False)
    f = plt.gcf()
    plt.savefig('result.png')
    predictions = model.predict(image)

    result = predictions[0].tolist()
    return result
