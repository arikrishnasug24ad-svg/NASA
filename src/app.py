import os
import joblib
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')

print(f"Looking for model at {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print('WARNING: model file not found. Run training first to create models/model.joblib')
    model = None
else:
    model = joblib.load(MODEL_PATH)
    print('Model loaded')

@app.route('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'web'), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not available. Run training first.'}), 500
    payload = request.get_json()
    # Expect payload to be a dict of features; we will create a single-row DataFrame
    try:
        # Define expected features (must match those used in training)
        expected_numeric = ['Num Stars', 'Num Planets', 'Discovery Year', 'Orbital Period Days',
                            'Orbit Semi-Major Axis', 'Eccentricity', 'Insolation Flux', 'Equilibrium Temperature',
                            'Stellar Effective Temperature', 'Stellar Radius', 'Stellar Mass', 'Stellar Surface Gravity', 'Distance', 'Gaia Magnitude']
        expected_categorical = ['Discovery Method', 'Spectral Type']
        expected = expected_numeric + expected_categorical

        # Build a row containing all expected features; missing ones get None so the pipeline can impute
        row = {k: payload.get(k, None) for k in expected}

        df = pd.DataFrame([row])
        # Convert numeric-looking columns to numeric (coerce errors to NaN for imputation)
        for c in expected_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        preds = model.predict(df)
        return jsonify({'prediction': float(preds[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Serve static files (css/js)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'web'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
