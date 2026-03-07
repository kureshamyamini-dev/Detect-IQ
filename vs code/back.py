from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Create CSV if not exists
if not os.path.exists("claims.csv"):
    df = pd.DataFrame(columns=[
        "age","claim_amount","num_claims","policy_years","fraud_prediction"
    ])
    df.to_csv("claims.csv", index=False)


@app.route('/')
def home():
    return "Insurance Fraud Detection Backend Running"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    age = data['age']
    claim_amount = data['claim_amount']
    num_claims = data['num_claims']
    policy_years = data['policy_years']

    input_data = [[age, claim_amount, num_claims, policy_years]]

    prediction = model.predict(input_data)[0]

    result = "Fraudulent Claim" if prediction == 1 else "Legitimate Claim"

    new_claim = pd.DataFrame([{
        "age": age,
        "claim_amount": claim_amount,
        "num_claims": num_claims,
        "policy_years": policy_years,
        "fraud_prediction": result
    }])

    new_claim.to_csv("claims.csv", mode='a', header=False, index=False)

    return jsonify({
        "prediction": result
    })


@app.route('/claims')
def view_claims():

    df = pd.read_csv("claims.csv")

    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True)from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Create CSV if not exists
if not os.path.exists("claims.csv"):
    df = pd.DataFrame(columns=[
        "age","claim_amount","num_claims","policy_years","fraud_prediction"
    ])
    df.to_csv("claims.csv", index=False)


@app.route('/')
def home():
    return "Insurance Fraud Detection Backend Running"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    age = data['age']
    claim_amount = data['claim_amount']
    num_claims = data['num_claims']
    policy_years = data['policy_years']

    input_data = [[age, claim_amount, num_claims, policy_years]]

    prediction = model.predict(input_data)[0]

    result = "Fraudulent Claim" if prediction == 1 else "Legitimate Claim"

    new_claim = pd.DataFrame([{
        "age": age,
        "claim_amount": claim_amount,
        "num_claims": num_claims,
        "policy_years": policy_years,
        "fraud_prediction": result
    }])

    new_claim.to_csv("claims.csv", mode='a', header=False, index=False)

    return jsonify({
        "prediction": result
    })


@app.route('/claims')
def view_claims():

    df = pd.read_csv("claims.csv")

    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True)from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Create CSV if not exists
if not os.path.exists("claims.csv"):
    df = pd.DataFrame(columns=[
        "age","claim_amount","num_claims","policy_years","fraud_prediction"
    ])
    df.to_csv("claims.csv", index=False)


@app.route('/')
def home():
    return "Insurance Fraud Detection Backend Running"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    age = data['age']
    claim_amount = data['claim_amount']
    num_claims = data['num_claims']
    policy_years = data['policy_years']

    input_data = [[age, claim_amount, num_claims, policy_years]]

    prediction = model.predict(input_data)[0]

    result = "Fraudulent Claim" if prediction == 1 else "Legitimate Claim"

    new_claim = pd.DataFrame([{
        "age": age,
        "claim_amount": claim_amount,
        "num_claims": num_claims,
        "policy_years": policy_years,
        "fraud_prediction": result
    }])

    new_claim.to_csv("claims.csv", mode='a', header=False, index=False)

    return jsonify({
        "prediction": result
    })


@app.route('/claims')
def view_claims():

    df = pd.read_csv("claims.csv")

    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True) **