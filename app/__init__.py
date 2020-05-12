
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
import matplotlib.image
from skimage.transform import resize
# from scipy.misc import imread

import base64
import json
from base64 import b64decode
import re

app = Flask(__name__)
json_file = open('../model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('../Model_mnist.h5')
model.compile(loss='sparse_categorical_crossentropy')


@app.route('/')
def index():
    return render_template('sketchpad.html')


@app.route('/predict', methods=["POST"])
def Predict():
    img_data = request.get_data().decode('utf-8')
    # Separate the metadata from the image data
    head, data = img_data.split(',', 1)
    # Decode the image data
    plain_data = base64.b64decode(data)

    # Write the image to a file
    with open('image.png', 'wb') as f:
        f.write(plain_data)

    img = np.array(matplotlib.image.imread('image.png'), np.int)
    # img = matplotlib.image.imread('image.png')
    # img = imread('image.png', mode='L')
    print(img)  
    img = np.invert(img)
    with open('image_inv.png', 'wb') as f:
        f.write(img)
    res = resize(img, (1, 28, 28, 1))
    prediction = model.predict_classes(res)
    print(np.array2string(prediction))
    return np.array2string(prediction[0])
    

if __name__ == '__main__':
    app.run(debug=True)

