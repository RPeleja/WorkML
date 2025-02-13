from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and transformers
model = joblib.load('models/wine_quality_model.pkl')
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber os dados do formulário
        features = [float(request.form[f'feature{i}']) for i in range(1, 11)]
        
        # Converter para array numpy e pré-processar
        new_data = np.array([features])
        new_data = imputer.transform(new_data)
        new_data = scaler.transform(new_data)
        
        # Fazer previsão
        prediction = model.predict(new_data)[0]
        result = "Bom" if prediction == 1 else "Regular"
        
        return render_template('result.html', result=result)
    except Exception as e:
        return f'Erro ao processar a previsão: {e}'

if __name__ == '__main__':
    app.run(debug=True)
