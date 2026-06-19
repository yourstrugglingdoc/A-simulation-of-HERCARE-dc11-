import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date

# ── Re-run model ───────────────────────────────────────────────────────────────
df = pd.read_csv("tcga_brca_dataset.csv")

feature_cols = ["age","stage_numeric","er_status","pr_status",
                "her2_status","lymph_node_ratio","lymph_node_positive"]

X = df[feature_cols]
y = df["died_within_5yr"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200,
    class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ── Pick a test patient ────────────────────────────────────────────────────────
patient_index = 0   # change to 1, 2, 3 to see different patients

patient_features = X_test[patient_index]
true_label       = y_test.iloc[patient_index]
patient_id       = df.iloc[y_test.index[patient_index]]["patient_id"]

risk_proba = model.predict_proba([patient_features])[0][1]
risk_pred  = model.predict([patient_features])[0]

# ── Decode features back to readable values ────────────────────────────────────
stage_map_rev = {1:"Stage I", 2:"Stage II", 3:"Stage III", 4:"Stage IV"}
stage_num = patient_features[1]
stage_str = stage_map_rev.get(round(stage_num), "Unknown") if not np.isnan(stage_num) else "Not recorded"

def yn(val):
    if np.isnan(val): return "Not recorded"
    return "Positive" if val > 0.5 else "Negative"

def node_status(ratio):
    if ratio == 0: return "N0 (no nodes involved)"
    elif ratio < 0.2: return "N1 (low involvement)"
    elif ratio < 0.5: return "N2 (moderate)"
    else: return "N3 (high involvement)"

# ── Risk category and recommendation ──────────────────────────────────────────
if risk_proba >= 0.60:
    category  = "HIGH"
    alert     = "URGENT"
    action    = "Immediate oncology review. Discuss intensified adjuvant therapy."
    followup  = "3-monthly clinical assessment. Consider ctDNA monitoring."
    therapy   = "Evaluate neoadjuvant or adjuvant chemotherapy intensification."
elif risk_proba >= 0.35:
    category  = "MODERATE"
    alert     = "REVIEW"
    action    = "Oncologist review within 4 weeks."
    followup  = "6-monthly imaging. Repeat receptor panel at 12 months."
    therapy   = "Shared decision-making regarding adjuvant endocrine therapy."
else:
    category  = "LOW"
    alert     = "ROUTINE"
    action    = "Standard annual follow-up protocol."
    followup  = "Annual mammography and clinical review."
    therapy   = "Continue current management. Patient education on self-monitoring."

# ── Top 3 driving features ─────────────────────────────────────────────────────
importances = model.feature_importances_
top3_idx    = np.argsort(importances)[::-1][:3]
top3        = [(feature_cols[i], patient_features[i], importances[i])
               for i in top3_idx]

# ── Print report ───────────────────────────────────────────────────────────────
W = 56

def line(c="─"): print(c * W)
def row(k, v):
    dots = "." * (W - len(k) - len(str(v)) - 2)
    print(f"  {k}{dots}{v}")

print()
line("═")
print("  HER-CARE CLINICAL DECISION SUPPORT — TCGA-BRCA")
print(f"  Model: Random Forest v1.0  |  Date: {date.today()}")
line("═")
print()
print(f"  Patient ID  : {patient_id}")
print(f"  Data source : TCGA-BRCA (real clinical data)")
print(f"  True outcome: {'Died within 5 years' if true_label==1 else 'Survived 5+ years'}")
print()
line()
print("  PATIENT PROFILE")
line()
row("  Age at diagnosis",       f"{patient_features[0]:.0f} years")
row("  AJCC pathologic stage",  stage_str)
row("  ER status (ESR1)",       yn(patient_features[2]))
row("  PR status (PGR)",        yn(patient_features[3]))
row("  HER2 status (ERBB2)",    yn(patient_features[4]))
row("  Lymph node ratio",       f"{patient_features[5]:.3f}")
row("  Lymph node status",      node_status(patient_features[5]))
print()
line()
print("  5-YEAR SURVIVAL RISK ASSESSMENT")
line()
row("  Risk probability",       f"{risk_proba:.1%}")
row("  Risk category",          category)
row("  Alert level",            alert)
row("  Predicted outcome",      "High risk" if risk_pred==1 else "Low risk")
row("  Prediction correct",     "YES ✓" if risk_pred==true_label else "NO ✗")
print()
line()
print("  TOP 3 DRIVING FEATURES")
line()
for feat, val, imp in top3:
    row(f"  {feat}", f"{val:.3f}  (model weight {imp:.3f})")
print()
line()
print("  CLINICAL RECOMMENDATION")
line()
print(f"  Action   : {action}")
print(f"  Follow-up: {followup}")
print(f"  Therapy  : {therapy}")
print()
line("═")
print("  DISCLAIMER: AI support tool only. All clinical decisions")
print("  must be made by a qualified clinician. Model trained on")
print("  TCGA-BRCA. Not validated for external deployment.")
line("═")
print()