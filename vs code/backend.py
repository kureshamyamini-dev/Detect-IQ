from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Config
MODEL_PATH = "fraud_model.pkl"
CSV_PATH = "claims.csv"
FEATURES = ["age", "claim_amount", "num_claims", "policy_years"]

# Load model safely
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Prediction endpoint will return an error.")

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=FEATURES + ["fraud_prediction"])
    df_init.to_csv(CSV_PATH, index=False)

@app.route('/')
def home():
    return jsonify({
        "status": "Online",
        "message": "Insurance Fraud Detection API is active"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing on server"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # 1. Extract and Validate Input
        # This ensures all required numbers are present
        input_values = [data.get(f) for f in FEATURES]
        if None in input_values:
            return jsonify({"error": f"Missing one of the required fields: {FEATURES}"}), 400

        # 2. Prepare Data for Model
        # Wrapping in DataFrame prevents "feature names mismatch" errors
        input_df = pd.DataFrame([input_values], columns=FEATURES)

        # 3. Prediction
        prediction = model.predict(input_df)[0]
        result = "Fraudulent Claim" if int(prediction) == 1 else "Legitimate Claim"

        # 4. Log to CSV
        log_entry = input_df.copy()
        log_entry["fraud_prediction"] = result
        log_entry.to_csv(CSV_PATH, mode='a', header=False, index=False)

        return jsonify({
            "status": "success",
            "prediction": result,
            "data_logged": True
        })

    except Exception as e:
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500

@app.route('/claims', methods=['GET'])
def view_claims():
    if not os.path.exists(CSV_PATH):
        return jsonify([])
    
    try:
        df = pd.read_csv(CSV_PATH)
        # Convert to list of dictionaries for easy JSON consumption
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)