from fastbook import *
from flask import Flask,render_template, request, jsonify
import numpy as np
from fastai.vision import *
import pickle 
import io
import os
from PIL import Image
from flask_cors import CORS

# Initialiazing flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Loading  saved model
model = load_learner('houseplantmodel.pkl')

# Rendering index.html at /
@app.route('/')
def index():
    return "hello" ##return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  print('in predict')
  print(request)
  img_PIL = Image.open(request.files['image'])
  print(img_PIL)
  img = tensor(img_PIL)
  prediction = model.predict(img)[0]
  # Getting Prediction ready to sent it to frontend
  response = {"result": str(prediction)}
  return jsonify(response)

  
# Getting data with POST Method
@app.route('/upload', methods=["POST"])
def upload():
    # try:
        # Getting img from POST
        file = request.files['image']##.read()
        img_PIL = Image.open(file)
        img = tensor(img_PIL)
        # Resizing img to 224 X 224 , This is the size on which model was trained
        #img = open_image(io.BytesIO('sampleplant.jpg'))
        # Prediction using model
        prediction = model.predict(img)[0]

        # Getting Prediction ready to sent it to frontend
        response = {"result": str(prediction)}
        return jsonify(response)

# Getting data with POST Method
##@app.route('/upload', methods=["POST"])
@app.route("/test")
def testPredict():
    # try:
        # Getting img from POST
        file = request.files['user-img'].read()
        img_PIL = Image.open('sampleplant.jpg')
        img = tensor(img_PIL)
        # Resizing img to 224 X 224 , This is the size on which model was trained
        #img = open_image(io.BytesIO('sampleplant.jpg'))
        # Prediction using model
        prediction = model.predict(img)[0]

        # Getting Prediction ready to sent it to frontend
        response = {"result": str(prediction)}
        return jsonify(response)
  

#running app at localhost on port 8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
