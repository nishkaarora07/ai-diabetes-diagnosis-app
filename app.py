# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow frontend access

# Load trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({'result': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
