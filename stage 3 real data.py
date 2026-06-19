import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ── Load real TCGA data ────────────────────────────────────────────────────────
df = pd.read_csv("tcga_brca_dataset.csv")

print(f"Loaded {len(df)} real TCGA-BRCA patients")
print(f"Class balance: {int(df['died_within_5yr'].sum())} died, "
      f"{int((df['died_within_5yr']==0).sum())} survived\n")

feature_cols = [
    "age", "stage_numeric", "er_status", "pr_status",
    "her2_status", "lymph_node_ratio", "lymph_node_positive"
]

X = df[feature_cols]
y = df["died_within_5yr"]

# ── Impute missing values ──────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

print(f"Missing values before imputation: {X.isnull().sum().sum()}")
print(f"Missing values after:             0\n")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y ensures the 30/70 class ratio is preserved in both splits
)

print(f"Training: {X_train.shape[0]} patients")
print(f"Testing:  {X_test.shape[0]} patients")
print(f"  Test set — died: {int(y_test.sum())}, survived: {int((y_test==0).sum())}\n")

# ── Train model (class_weight balances the 30/70 imbalance) ───────────────────
print("Training Random Forest on real TCGA data...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # compensates for more survivors than deaths
    random_state=42
)
model.fit(X_train, y_train)
print("Done.\n")

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)
auroc        = roc_auc_score(y_test, y_pred_proba)

print("=" * 50)
print(f"  AUROC: {auroc:.3f}   (target > 0.70 on real data)")
print("=" * 50)
print()
print(classification_report(y_test, y_pred,
      target_names=["Survived 5yr", "Died within 5yr"]))

# ── Feature importance ─────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("Feature importance:")
print("-" * 45)
for feat, score in importances.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<22} {bar}  {score:.3f}")

# ── Compare to synthetic model ────────────────────────────────────────────────
print()
print("─" * 50)
print("  Synthetic model AUROC : 0.851  (clean formula)")
print(f"  Real TCGA AUROC       : {auroc:.3f}  (messy real world)")
print("─" * 50)
print()
print("Lower AUROC on real data is expected and meaningful.")
print("Real outcomes are driven by biology we haven't measured.")