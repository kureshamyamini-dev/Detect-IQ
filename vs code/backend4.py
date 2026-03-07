Create a fully functional backend for an Insurance Fraud Detection Web Application using Python and Flask.

Requirements:

Framework:
Use Flask to create REST APIs.

Machine Learning:
Use scikit-learn to train a fraud detection model using RandomForestClassifier.

Dataset:
Create a sample dataset inside the code with the following features:
- claim_amount
- policy_duration
- previous_claims
- customer_age
- fraud_reported (target variable)

Backend Features:
1. Train a machine learning model to detect fraudulent insurance claims.
2. Save the trained model using joblib.
3. Build Flask APIs with the following routes:

GET /
Return message: "Insurance Fraud Detection API Running"

POST /predict
Accept JSON input:
{
  "claim_amount": number,
  "policy_duration": number,
  "previous_claims": number,
  "customer_age": number
}

Return JSON output:
{
  "prediction": "Fraudulent Claim" or "Genuine Claim",
  "fraud_probability": probability score
}

Additional Requirements:
- Validate user input
- Handle errors properly
- Structure the project into multiple files:
   train_model.py
   model.py
   app.py

Libraries to use:
flask
pandas
numpy
scikit-learn
joblib

The project should run with:
python train_model.py
python app.py

The API should run on http://127.0.0.1:5000
Ensure the code runs without syntax errors and is ready to connect with a frontend.