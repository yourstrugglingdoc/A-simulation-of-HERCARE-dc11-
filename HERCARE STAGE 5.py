import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Re-run model (same as before) ─────────────────────────────────────────────
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

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auroc        = roc_auc_score(y_test, y_pred_proba)

# ── Set up the figure: 2×2 grid of plots ──────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle("HER-CARE Mini-Project — Model Evaluation Report",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Plot 1: ROC Curve ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax1.plot(fpr, tpr, color="#1D9E75", linewidth=2.5,
         label=f"Model (AUROC = {auroc:.3f})")
ax1.plot([0, 1], [0, 1], color="#B4B2A9", linewidth=1,
         linestyle="--", label="Random guess (0.500)")
ax1.fill_between(fpr, tpr, alpha=0.08, color="#1D9E75")
ax1.set_xlabel("False positive rate", fontsize=11)
ax1.set_ylabel("True positive rate", fontsize=11)
ax1.set_title("ROC curve", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.02])
ax1.axhline(y=0.8, color="#D85A30", linewidth=0.8,
            linestyle=":", alpha=0.7, label="0.80 target")
ax1.spines[["top", "right"]].set_visible(False)

# ── Plot 2: Confusion Matrix ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Low risk", "High risk"])
disp.plot(ax=ax2, colorbar=False, cmap="Greens")
ax2.set_title("Confusion matrix (test set)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Predicted label", fontsize=11)
ax2.set_ylabel("True label", fontsize=11)

# ── Plot 3: Feature Importance ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values()
colors = ["#1D9E75" if v >= importances.median() else "#9FE1CB"
          for v in importances]
bars = ax3.barh(importances.index, importances.values,
                color=colors, edgecolor="none", height=0.65)
ax3.set_xlabel("Importance score", fontsize=11)
ax3.set_title("Feature importance", fontsize=12, fontweight="bold")
ax3.spines[["top", "right"]].set_visible(False)
ax3.axvline(x=importances.median(), color="#D85A30", linewidth=0.8,
            linestyle="--", alpha=0.6)
for bar, val in zip(bars, importances.values):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color="#444")

# ── Plot 4: SHAP summary for first 40 test patients ───────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values is [class0, class1] — we want class 1 (high risk)
# Handle both old and new SHAP output formats
if isinstance(shap_values, list):
    shap_high_risk = shap_values[1]
elif shap_values.ndim == 3:
    shap_high_risk = shap_values[:, :, 1]
else:
    shap_high_risk = shap_values

mean_shap = np.abs(shap_high_risk).mean(axis=0)
shap_series = pd.Series(mean_shap, index=feature_cols).sort_values()
shap_colors = ["#534AB7" if v >= shap_series.median() else "#AFA9EC"
               for v in shap_series]

shap_bars = ax4.barh(shap_series.index, shap_series.values,
                      color=shap_colors, edgecolor="none", height=0.65)
ax4.set_xlabel("Mean |SHAP value|", fontsize=11)
ax4.set_title("SHAP explainability", fontsize=12, fontweight="bold")
ax4.spines[["top", "right"]].set_visible(False)
for bar, val in zip(shap_bars, shap_series.values):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color="#444")

# ── Save and show ──────────────────────────────────────────────────────────────
plt.savefig("hercare_evaluation_report.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Report saved to hercare_evaluation_report.png")
print(f"\nFinal AUROC: {auroc:.3f}")
print("\nAll 4 plots generated:")
print("  1. ROC curve          — how well the model separates risk classes")
print("  2. Confusion matrix   — where it gets predictions right and wrong")
print("  3. Feature importance — what the Random Forest relied on")
print("  4. SHAP values        — why each prediction was made (explainability)")
plt.show()