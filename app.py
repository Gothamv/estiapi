from flask import Flask, request, jsonify
import numpy as np
import json
import joblib

app = Flask(__name__)
f = open("carModel.pkl","rb")
model = joblib.load(f)

@app.route('/', methods=['GET'])
def welcome():
    return 'Welcome to the EstiMotor API! 🎉'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data = np.array(data)
    #prediction = np.array2string(model.predict(data))
    a = np.exp(model.predict(data))-1
    return jsonify(np.array2string(a))
