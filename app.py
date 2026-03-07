from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Config
model_paths = {
    'RandomForest': 'models/random_forest.joblib',
    'DecisionTree': 'models/decision_tree.joblib',
    'XGBoost': 'models/xgboost.joblib',
    'scaler': 'models/scaler.joblib'
}

models = {}
scaler = None
feature_cols = None
feature_importances = None

def load_models():
    global models, scaler, feature_cols, feature_importances
    try:
        if os.path.exists(model_paths['scaler']):
            scaler = joblib.load(model_paths['scaler'])
        for name in ['RandomForest', 'DecisionTree', 'XGBoost']:
            if os.path.exists(model_paths[name]):
                models[name] = joblib.load(model_paths[name])
        if os.path.exists('models/feature_cols.joblib'):
            feature_cols = joblib.load('models/feature_cols.joblib')
        if os.path.exists('models/feature_importances.joblib'):
            feature_importances = joblib.load('models/feature_importances.joblib')
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        model_name = data.get('model', 'RandomForest')
        model = models.get(model_name)
        if not model:
            return jsonify({'error': f'Model {model_name} not loaded'}), 400

        # Construct feature dictionary from expected input
        input_data = {
            'claim_amount': float(data.get('claim_amount', 0)),
            'incident_severity': int(data.get('incident_severity', 0)),
            'customer_age': int(data.get('customer_age', 35)),
            'months_as_customer': int(data.get('months_as_customer', 12)),
            'policy_deductable': int(data.get('policy_deductable', 1000)),
            'previous_claims': int(data.get('previous_claims', 0))
        }

        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction_probs = model.predict_proba(input_scaled)[0]
        fraud_prob = float(prediction_probs[1])
        prediction = int(fraud_prob > 0.5)

        return jsonify({
            'fraud_risk_score': round(fraud_prob * 100, 2),
            'is_fraudulent': bool(prediction),
            'prediction_label': "Fraudulent" if prediction else "Genuine",
            'model_used': model_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not feature_importances:
        return jsonify({'error': 'Metrics not available'}), 404
        
    safe_importances = {}
    for model_name, feats in feature_importances.items():
        safe_importances[model_name] = {k: float(v) for k, v in feats.items()}
    return jsonify(safe_importances)

@app.route('/api/history', methods=['GET'])
def get_history():
    history = [
        {"claim_id": "CLM-1001", "amount": 4500, "date": "2026-03-01", "status": "Genuine", "risk_score": 12.5},
        {"claim_id": "CLM-1002", "amount": 12500, "date": "2026-03-02", "status": "Genuine", "risk_score": 25.1},
        {"claim_id": "CLM-1003", "amount": 38000, "date": "2026-03-03", "status": "Fraudulent", "risk_score": 89.4},
        {"claim_id": "CLM-1004", "amount": 900, "date": "2026-03-05", "status": "Genuine", "risk_score": 5.2},
        {"claim_id": "CLM-1005", "amount": 28000, "date": "2026-03-06", "status": "Fraudulent", "risk_score": 75.8}
    ]
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
