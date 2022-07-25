from ast import Import
import os
from joblib import load
from flask import Flask, render_template, url_for_request
from flask_restful import reqparse, abort, Api, Resource
import cv2
import json
import numpy as np



MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Load model
cnn_model = load(model_path)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
class Predict_img(Resource):
    def get(self):
        post = json.loads(raw_data)
        for i in post["data"]:
            sign=cv2.imread(i, 0)
        sign = np.array(sign,dtype=object)
        img = cv2.resize(img,(100,100))
        img=cv2.equalizeHist(img)
        img=img/255.0
        pred = cnn_model.predict(img)
        conf = cnn_model.predict_proba(img)

        payload = {'prediction': pred, 'confidence': conf}
        
        return payload
api.add_resource(Predict_img, '/')

if __name__ == "__main__":
    app.run(debug=True)
