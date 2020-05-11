# from flask import render_template, request, url_for
from flask import Flask, request, render_template
#from scipy.misc import imsave, imread, imresize
#from app import app
#import numpy as np
#from app.funcs import Data, Train
#from sklearn.metrics.pairwise import cosine_similarity
#import tensorflow as tf
#from tensorflow import get_default_graph
#from tensorflow.keras.models import load_model
#from tensorflow.keras import backend as K
#import time

import base64
import json


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('sketchpad.html')


@app.route('/api/predict', methods=["POST"])
def Predict():
    data = request.get_json()
    img = base64.b64decode(data["image"])
    with open("image.jpg", "wb") as output:
        output.write(img)
     #   output.write(base64.decodebytes(str.encode(data["image"])))
        # output.write( base64.decodebytes(data["image"].encode()) )
        #output.write(data["image"].decode('base64'))
    # json_string = json.loads(data)
    # image_json[""]
    # print(request.get_json())
    
    return str.encode(data["image"])
    

if __name__ == '__main__':
    app.run(debug=True)

