import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date

# ── Re-run the model quickly (same code as Stage 3) ───────────────────────────
df = pd.read_csv("breast_cancer_dataset.csv", index_col="patient_id")

feature_cols = [
    "tumour_size_mm", "enhancement_ratio", "adc_value", "margin_irregularity",
    "ki67_index", "oncotype_score", "her2_score", "er_pr_positive",
    "age", "lymph_node_status", "tumour_grade", "family_history"
]

X = df[feature_cols]
y = df["high_risk"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Pick one test patient to generate a report for ────────────────────────────
patient_index = 0                          # change this 0→1, 0→2 to see different patients
patient_features = X_test[patient_index]
true_label       = y_test.iloc[patient_index]

risk_proba  = model.predict_proba([patient_features])[0][1]  # probability of high risk
risk_label  = model.predict([patient_features])[0]           # 0 or 1

# ── Determine risk category and recommendation ────────────────────────────────
if risk_proba >= 0.70:
    category    = "HIGH"
    action      = "Immediate multidisciplinary tumour board referral within 5 days."
    followup    = "MRI every 6 months. Liquid biopsy (ctDNA) quarterly."
    therapy     = "Evaluate for neoadjuvant chemotherapy. Consider genetic counselling (BRCA1/2)."
    alert_level = "URGENT"
elif risk_proba >= 0.40:
    category    = "MODERATE"
    action      = "Enhanced surveillance. Oncologist review within 4 weeks."
    followup    = "MRI at 6 months. Repeat biomarker panel at 12 months."
    therapy     = "Shared decision-making regarding adjuvant therapy options."
    alert_level = "REVIEW"
else:
    category    = "LOW"
    action      = "Standard annual screening protocol."
    followup    = "Annual MRI + mammography."
    therapy     = "Patient education on self-monitoring and lifestyle factors."
    alert_level = "ROUTINE"

# ── Feature importance for this patient (simplified local explanation) ─────────
importances = model.feature_importances_
top3_idx    = np.argsort(importances)[::-1][:3]
top3        = [(feature_cols[i], patient_features[i], importances[i]) for i in top3_idx]

# ── Print the clinical report ─────────────────────────────────────────────────
WIDTH = 55

def line(char="─"):
    print(char * WIDTH)

def row(label, value):
    dots = "." * (WIDTH - len(label) - len(str(value)) - 2)
    print(f"  {label}{dots}{value}")

print()
line("═")
print("  HER-CARE CLINICAL DECISION SUPPORT REPORT")
print(f"  Model: Random Forest v1.0  |  AUROC: 0.823")
line("═")
print()
print(f"  Date         : {date.today()}")
print(f"  Patient ID   : PT-TEST-{patient_index:04d}")
print(f"  True label   : {'High risk' if true_label == 1 else 'Low risk'} (ground truth)")
print()
line()
print("  MULTI-MODAL INPUT SUMMARY")
line()
print("  MRI features")
row("    Tumour size (mm)",    f"{patient_features[0]:.1f}")
row("    Enhancement ratio",   f"{patient_features[1]:.2f}")
row("    ADC value",           f"{patient_features[2]:.2f}")
row("    Margin irregularity", f"{patient_features[3]:.2f}")
print()
print("  Genomic / biomarkers")
row("    Ki-67 index (%)",     f"{patient_features[4]:.1f}")
row("    Oncotype DX score",   f"{patient_features[5]:.1f}")
row("    HER2 score",          f"{int(patient_features[6])}+")
row("    ER/PR positive",      "Yes" if patient_features[7] > 0.5 else "No")
print()
print("  Clinical")
row("    Age",                 f"{patient_features[8]:.0f}")
row("    Lymph node status",   f"N{int(patient_features[9])}")
row("    Tumour grade",        f"{int(patient_features[10])}/3")
row("    Family history",      "Yes" if patient_features[11] > 0.5 else "No")
print()
line()
print("  AI RISK ASSESSMENT")
line()
row("  Risk probability",  f"{risk_proba:.1%}")
row("  Risk category",     category)
row("  Alert level",       alert_level)
row("  Prediction",        "High risk" if risk_label == 1 else "Low risk")
row("  Correct?",          "YES" if risk_label == true_label else "NO")
print()
line()
print("  TOP 3 DRIVING FEATURES")
line()
for feat, val, imp in top3:
    row(f"  {feat}", f"{val:.2f}  (weight {imp:.3f})")
print()
line()
print("  CLINICAL RECOMMENDATION")
line()
print(f"  Action   : {action}")
print(f"  Follow-up: {followup}")
print(f"  Therapy  : {therapy}")
print()
line("═")
print("  DISCLAIMER: AI support tool only. Final decisions")
print("  must be made by a qualified clinician.")
line("═")
print()