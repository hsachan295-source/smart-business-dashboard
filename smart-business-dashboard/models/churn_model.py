"""
churn_model.py
Trains a churn prediction model (Random Forest + Logistic Regression),
evaluates it, and saves the model + feature importances for the dashboard.
"""

import pandas as pd
import numpy as np
import json, os, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "..", "data")
OUT   = os.path.join(BASE)

# ─── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, "churn_data.csv"))
print(f"Loaded {len(df):,} rows | churn rate: {df['churned'].mean():.1%}")

# ─── Features ────────────────────────────────────────────────────────────────
le_region  = LabelEncoder()
le_segment = LabelEncoder()
df["region_enc"]  = le_region.fit_transform(df["region"])
df["segment_enc"] = le_segment.fit_transform(df["segment"])

FEATURES = [
    "age", "annual_spend", "support_tickets", "nps_score",
    "products_owned", "last_login_days", "region_enc", "segment_enc"
]
TARGET = "churned"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Models ──────────────────────────────────────────────────────────────────
models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    acc   = accuracy_score(y_test, preds)
    cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring="roc_auc").mean()
    results[name] = {"accuracy": round(acc, 4), "auc": round(auc, 4), "cv_auc": round(cv, 4)}
    print(f"\n{name}: Accuracy={acc:.3f} | AUC={auc:.3f} | CV-AUC={cv:.3f}")
    print(classification_report(y_test, preds))

# ─── Best model (RF) ─────────────────────────────────────────────────────────
best_model = models["Random Forest"]
feat_imp = pd.DataFrame({
    "feature":    FEATURES,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nFeature Importances:\n", feat_imp.to_string(index=False))

# ─── Save artefacts ──────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(OUT, "churn_model.pkl"))
joblib.dump(scaler,     os.path.join(OUT, "scaler.pkl"))
joblib.dump(le_region,  os.path.join(OUT, "le_region.pkl"))
joblib.dump(le_segment, os.path.join(OUT, "le_segment.pkl"))

feat_imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

with open(os.path.join(OUT, "model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n✅  Model saved → churn_model.pkl")
print("✅  Results saved → model_results.json")
print("✅  Feature importance → feature_importance.csv")
