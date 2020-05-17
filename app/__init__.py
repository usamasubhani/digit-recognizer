
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
import cv2

app = Flask(__name__)
json_file = open('../model_v0.2.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('../Model_mnist_v0.2.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


@app.route('/')
def index():
    return render_template('sketchpad.html')


@app.route('/predict', methods=["POST"])
def Predict():
    img_data = request.get_data().decode('utf-8')
    # Separate the metadata from the image data
    head, data = img_data.split(',', 1)
    # Write the image to a file
    with open('image.png', 'wb') as f:
        f.write(base64.b64decode(data))

    img = cv2.imread('image.png')[:,:,0]
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = np.invert(np.array([img]))
    print(img.shape)
    prediction = model.predict_classes(img)
    return np.array2string(prediction[0])
    

if __name__ == '__main__':
    app.run(debug=True)

