import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs('models', exist_ok=True)

# Generate synthetic insurance claim data
np.random.seed(42)
n_samples = 2000

# Features
data = {
    'claim_amount': np.random.uniform(500, 50000, n_samples),
    'incident_severity': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
    'customer_age': np.random.randint(18, 90, n_samples),
    'months_as_customer': np.random.randint(1, 240, n_samples),
    'policy_deductable': np.random.choice([500, 1000, 2000], n_samples),
    'previous_claims': np.random.randint(0, 6, n_samples),
    'is_fraud': np.zeros(n_samples)
}
df = pd.DataFrame(data)

# Inject fraud patterns
fraud_mask = (
    ((df['previous_claims'] >= 3) & (df['claim_amount'] > 15000)) |
    ((df['incident_severity'] == 0) & (df['claim_amount'] > 30000)) |
    ((df['months_as_customer'] < 12) & (df['claim_amount'] > 25000))
)

df.loc[fraud_mask, 'is_fraud'] = np.random.choice([0, 1], size=fraud_mask.sum(), p=[0.1, 0.9])
df.loc[~fraud_mask, 'is_fraud'] = np.random.choice([0, 1], size=(~fraud_mask).sum(), p=[0.95, 0.05])

print(f"Total dataset size: {n_samples}")
print(f"Fraud ratio: {df['is_fraud'].mean():.2f}")

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.joblib')

# Initialize and train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

rf_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)

# Save models
joblib.dump(rf_model, 'models/random_forest.joblib')
joblib.dump(dt_model, 'models/decision_tree.joblib')
joblib.dump(xgb_model, 'models/xgboost.joblib')

# Feature columns
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, 'models/feature_cols.joblib')

# Export Feature Importances for Admin Dashboard
importances = {
    'RandomForest': dict(zip(feature_cols, rf_model.feature_importances_)),
    'DecisionTree': dict(zip(feature_cols, dt_model.feature_importances_)),
    'XGBoost': dict(zip(feature_cols, xgb_model.feature_importances_))
}
joblib.dump(importances, 'models/feature_importances.joblib')

print("Models trained and all joblib objects saved successfully to models/.")
