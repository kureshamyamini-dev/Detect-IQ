"""
ClaimWatch AI - Flask Application
Main entry point for the web application.
"""

import os
import random
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = 'models'
model_paths = {
    'scaler': os.path.join(MODEL_DIR, 'scaler.joblib'),
    'RandomForest': os.path.join(MODEL_DIR, 'random_forest.joblib'),
    'DecisionTree': os.path.join(MODEL_DIR, 'decision_tree.joblib'),
    'XGBoost': os.path.join(MODEL_DIR, 'xgboost.joblib'),
}

FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.joblib')
FEATURE_IMPORTANCES_PATH = os.path.join(MODEL_DIR, 'feature_importances.joblib')

# Initialize global variables
models = {}
scaler = None
feature_cols = None
feature_importances = None


def load_models():
    """Load all trained models and artifacts from the models/ directory."""
    global scaler, models, feature_cols, feature_importances
    try:
        # Load Scaler
        if os.path.exists(model_paths['scaler']):
            scaler = joblib.load(model_paths['scaler'])
            print("  [OK] Scaler loaded.")
        else:
            print("  [WARN] Scaler not found.")

        # Load Models
        for name in ['RandomForest', 'DecisionTree', 'XGBoost']:
            path = model_paths.get(name)
            if path and os.path.exists(path):
                models[name] = joblib.load(path)
                print(f"  [OK] {name} model loaded.")
            else:
                print(f"  [WARN] {name} model not found at {path}.")

        # Load Feature Columns
        if os.path.exists(FEATURE_COLS_PATH):
            feature_cols = joblib.load(FEATURE_COLS_PATH)
            print("  [OK] Feature columns loaded.")

        # Load Feature Importances
        if os.path.exists(FEATURE_IMPORTANCES_PATH):
            feature_importances = joblib.load(FEATURE_IMPORTANCES_PATH)
            print("  [OK] Feature importances loaded.")

        print("\nModels and features loaded successfully!\n")

    except Exception as e:
        print(f"Error loading models: {e}")


# --- Load models once when the app starts ---
print("\n--- Loading Models ---")
load_models()


# --- Page Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit')
def submit():
    return render_template('submit.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# --- API Routes ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Receive claim data as JSON and return fraud prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Determine which model to use
        model_name = data.get('model', 'RandomForest')
        if model_name not in models:
            available = list(models.keys())
            if not available:
                return jsonify({'error': 'No models loaded. Run train_models.py first.'}), 500
            model_name = available[0]

        model = models[model_name]

        if scaler is None or feature_cols is None:
            return jsonify({'error': 'Scaler or feature columns not loaded. Run train_models.py first.'}), 500

        # Build feature vector in the correct order
        features = []
        for col in feature_cols:
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing field: {col}'}), 400
            features.append(float(val))

        feature_array = np.array([features])
        scaled_features = scaler.transform(feature_array)

        # Predict
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0]
        fraud_probability = float(proba[1]) * 100  # percentage

        return jsonify({
            'is_fraudulent': bool(prediction),
            'fraud_risk_score': round(fraud_probability, 1),
            'model_used': model_name,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def api_history():
    """Return mock recent claim history for the dashboard."""
    statuses = ['Genuine', 'Genuine', 'Genuine', 'Fraudulent', 'Genuine',
                'Genuine', 'Fraudulent', 'Genuine', 'Genuine', 'Genuine']
    history = []
    base_date = datetime.now()
    for i in range(10):
        status = statuses[i]
        risk = random.randint(55, 95) if status == 'Fraudulent' else random.randint(5, 40)
        history.append({
            'claim_id': f'CLM-{2024000 + i}',
            'date': (base_date - timedelta(days=i * 3)).strftime('%Y-%m-%d'),
            'amount': random.randint(1000, 50000),
            'risk_score': risk,
            'status': status,
        })
    return jsonify(history)


@app.route('/api/metrics')
def api_metrics():
    """Return feature importances for the dashboard chart."""
    if feature_importances:
        return jsonify(feature_importances)
    return jsonify({'error': 'Feature importances not loaded'}), 404


# --- Entry Point ---
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print("\nServer stopped manually.")