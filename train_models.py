"""
ClaimWatch AI - Model Training Script
Generates synthetic insurance fraud data, trains 3 ML models,
evaluates them, and saves the artifacts to models/ directory.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed. Install it with: pip install xgboost")
    HAS_XGBOOST = False

# -- Configuration -------------------------------------------------
SEED = 42
N_SAMPLES = 5000
TEST_SIZE = 0.2
MODEL_DIR = 'models'
FEATURE_COLS = [
    'claim_amount',
    'incident_severity',
    'customer_age',
    'months_as_customer',
    'policy_deductable',
    'previous_claims'
]

np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


# -- 1. Generate Synthetic Dataset ---------------------------------
def generate_data(n=N_SAMPLES):
    print(f"\n{'='*60}")
    print(f"  Generating {n} synthetic insurance claim records ...")
    print(f"{'='*60}")

    data = {
        'claim_amount':       np.random.exponential(scale=8000, size=n).clip(100, 100000),
        'incident_severity':  np.random.randint(1, 5, size=n),
        'customer_age':       np.random.randint(18, 80, size=n),
        'months_as_customer': np.random.randint(1, 300, size=n),
        'policy_deductable':  np.random.choice([500, 1000, 1500, 2000], size=n),
        'previous_claims':    np.random.randint(0, 10, size=n),
    }
    df = pd.DataFrame(data)

    # Create a realistic fraud label based on feature patterns
    fraud_score = (
        0.3 * (df['claim_amount'] > 15000).astype(int)
      + 0.2 * (df['incident_severity'] >= 4).astype(int)
      + 0.15 * (df['customer_age'] < 25).astype(int)
      + 0.1 * (df['months_as_customer'] < 12).astype(int)
      + 0.15 * (df['previous_claims'] >= 4).astype(int)
      + 0.1 * (df['policy_deductable'] >= 2000).astype(int)
    )
    noise = np.random.uniform(0, 0.3, size=n)
    df['is_fraud'] = ((fraud_score + noise) > 0.55).astype(int)

    fraud_pct = df['is_fraud'].mean() * 100
    print(f"  Dataset shape : {df.shape}")
    print(f"  Fraud ratio   : {fraud_pct:.1f}%")
    return df


# -- 2. Train & Evaluate Models ------------------------------------
def train_and_evaluate():
    df = generate_data()

    X = df[FEATURE_COLS]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Define models
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=8, random_state=SEED
        ),
    }
    if HAS_XGBOOST:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, use_label_encoder=False, eval_metric='logloss'
        )

    feature_importances = {}
    results = {}

    for name, clf in classifiers.items():
        print(f"\n{'-'*60}")
        print(f"  Training {name} ...")
        print(f"{'-'*60}")

        clf.fit(X_train_scaled, y_train)
        y_pred  = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {'accuracy': acc, 'auc': auc}

        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"\n  Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))

        # Feature importances
        importances = clf.feature_importances_
        feat_imp = dict(zip(FEATURE_COLS, importances.tolist()))
        feature_importances[name] = feat_imp

        # Save model
        name_to_file = {
            'RandomForest': 'random_forest.joblib',
            'DecisionTree': 'decision_tree.joblib',
            'XGBoost':      'xgboost.joblib',
        }
        model_path = os.path.join(MODEL_DIR, name_to_file[name])
        joblib.dump(clf, model_path)
        print(f"  [OK] Saved : {model_path}")

    # Save scaler, feature columns, and importances
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, 'feature_cols.joblib'))
    joblib.dump(feature_importances, os.path.join(MODEL_DIR, 'feature_importances.joblib'))

    # -- Summary ----------------------------------------------------
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*40}")
    for name, metrics in results.items():
        print(f"  {name:<20} {metrics['accuracy']:>10.4f} {metrics['auc']:>10.4f}")
    print(f"\n  All models and artifacts saved to '{MODEL_DIR}/' directory.")
    print(f"  You can now run the Flask app:  python app.py\n")


# -- 3. Entry Point -------------------------------------------------
if __name__ == '__main__':
    train_and_evaluate()
