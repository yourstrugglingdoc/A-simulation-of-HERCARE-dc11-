import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ── Step 1: Load the dataset we made in Stage 2 ───────────────────────────────
df = pd.read_csv("breast_cancer_dataset.csv", index_col="patient_id")

print("Dataset loaded!")
print(f"Shape: {df.shape[0]} patients × {df.shape[1]} columns\n")

# ── Step 2: Separate features (X) from label (y) ──────────────────────────────
# X = everything the model is allowed to learn from (the 12 features)
# y = what we want it to predict (high_risk: 0 or 1)
feature_cols = [
    "tumour_size_mm", "enhancement_ratio", "adc_value", "margin_irregularity",
    "ki67_index", "oncotype_score", "her2_score", "er_pr_positive",
    "age", "lymph_node_status", "tumour_grade", "family_history"
]

X = df[feature_cols]
y = df["high_risk"]

print(f"Features (X): {X.shape[1]} columns")
print(f"Label   (y): {y.value_counts()[1]} high-risk, {y.value_counts()[0]} low-risk\n")

# ── Step 3: Fill missing data ──────────────────────────────────────────────────
# Replace the 40 NaN values in oncotype_score with the median of all other patients
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

print(f"Missing values before: {X.isnull().sum().sum()}")
print(f"Missing values after:  {pd.DataFrame(X_imputed).isnull().sum().sum()}\n")

# ── Step 4: Split into training and test sets ──────────────────────────────────
# 80% for training (160 patients), 20% for testing (40 patients)
# random_state=42 means we get the same split every time we run it
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} patients")
print(f"Test set:     {X_test.shape[0]} patients\n")

# ── Step 5: Train the model ────────────────────────────────────────────────────
# Random Forest = 100 decision trees voting together
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Done!\n")

# ── Step 6: Evaluate on the test set ──────────────────────────────────────────
# Ask the model to predict probabilities for the 40 patients it never saw
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of high_risk
y_pred       = model.predict(X_test)               # hard prediction: 0 or 1

auroc = roc_auc_score(y_test, y_pred_proba)

print("=" * 45)
print(f"  AUROC score:  {auroc:.3f}  (target > 0.80)")
print("=" * 45)
print()
print(classification_report(y_test, y_pred, target_names=["Low risk", "High risk"]))

# ── Step 7: Feature importance ─────────────────────────────────────────────────
# Which features did the model find most useful?
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("Feature importance (what the model relied on most):")
print("-" * 45)
for feat, score in importances.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<25} {bar}  {score:.3f}")
    