from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

loaded_d = joblib.load('/home/muhammad-ahmad-nadeem/Projects/supervised-learning-series/linear_reg/resources/trained_model.pkl')

@app.route('/')
def home():
    return "api to give diabetes prediction"

@app.route('/predict', methods=['POST'])
def prediction():
    try:
      data = request.get_json()
      features = np.array([
              data['Pregnancies'], data['Glucose'], data['BloodPressure'], 
              data['SkinThickness'], data['Insulin'], data['BMI'], 
              data['DiabetesPedigreeFunction'],data['Age']]).reshape(1, -1)
      prediction = loaded_d.predict(features)
      return jsonify({'prediction': prediction[0]})
    except Exception as e:
       return jsonify({'error': str(e)})
    
if __name__ == '__main__':
   app.run(debug=True)