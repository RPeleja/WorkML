from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and transformers
model = joblib.load('WorkML/models/wine_quality_model.pkl')
scaler = joblib.load('WorkML/models/scaler.pkl')
imputer = joblib.load('WorkML/models/imputer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form (match expected model input)
        feature_names = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'density', 'pH', 'sulphates',
            'alcohol', 'type', 'precipitation', 'relativehumidity',
            'solarradiation', 'temperature', 'uvindexmax', 'winddirection',
            'windspeed', 'year', 'month'
        ]

        # Convert form data to float and create feature array
        features = [float(request.form[name]) for name in feature_names]

        # Convert to NumPy array and preprocess
        new_data = np.array([features])
        new_data = imputer.transform(new_data)
        new_data = scaler.transform(new_data)

        # Make prediction
        prediction = model.predict(new_data)[0]
        result = "Bom Vinho" if prediction == 1 else "Vinho Regular"

        return render_template('result.html', result=result)

    except Exception as e:
        return f'Erro ao processar a previsão: {e}'

if __name__ == '__main__':
    app.run(debug=True)
