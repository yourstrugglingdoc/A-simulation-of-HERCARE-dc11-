import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # saves to file without opening a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Where to save the output figure ───────────────────────────────────────────
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tcga_evaluation_report.png")

# ── Load and train ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tcga_brca_dataset.csv"))

feature_cols = ["age", "stage_numeric", "er_status", "pr_status",
                "her2_status", "lymph_node_ratio", "lymph_node_positive"]

X = df[feature_cols]
y = df["died_within_5yr"]

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200,
    class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
auroc  = roc_auc_score(y_test, y_prob)

print(f"Model trained. AUROC = {auroc:.3f}")
print("Building evaluation figure...")

# ── Build 2x2 figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle(
    f"HER-CARE — Real TCGA-BRCA Evaluation  |  AUROC: {auroc:.3f}  |  "
    f"n={len(y_test)} test patients",
    fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.34)

# ── Plot 1: ROC curve ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, color="#1D9E75", linewidth=2.5,
         label=f"TCGA model (AUROC = {auroc:.3f})")
ax1.plot([0, 1], [0, 1], color="#B4B2A9", linewidth=1,
         linestyle="--", label="Random guess (0.500)")
ax1.fill_between(fpr, tpr, alpha=0.08, color="#1D9E75")
ax1.axhline(y=0.7, color="#D85A30", linewidth=0.8,
            linestyle=":", alpha=0.7, label="0.70 target")
ax1.set_xlabel("False positive rate", fontsize=11)
ax1.set_ylabel("True positive rate", fontsize=11)
ax1.set_title("ROC curve — real TCGA-BRCA data",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.02])
ax1.spines[["top", "right"]].set_visible(False)

# ── Plot 2: Confusion matrix ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Survived 5yr", "Died <5yr"]
).plot(ax=ax2, colorbar=False, cmap="Greens")
ax2.set_title("Confusion matrix", fontsize=12, fontweight="bold")
ax2.set_xlabel("Predicted", fontsize=11)
ax2.set_ylabel("True", fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)

tn, fp, fn, tp = cm.ravel()
ax2.text(0.5, -0.18,
    f"Sensitivity: {tp/(tp+fn):.0%}  |  "
    f"Specificity: {tn/(tn+fp):.0%}  |  "
    f"Precision: {tp/(tp+fp):.0%}" if (tp+fp) > 0 else "",
    transform=ax2.transAxes,
    ha="center", fontsize=9, color="#555")

# ── Plot 3: Feature importance ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
imp = pd.Series(model.feature_importances_,
                index=feature_cols).sort_values()
colors = ["#1D9E75" if v >= imp.median() else "#9FE1CB" for v in imp]
bars = ax3.barh(imp.index, imp.values,
                color=colors, edgecolor="none", height=0.6)
ax3.set_xlabel("Importance score", fontsize=11)
ax3.set_title("Feature importance (Random Forest)",
              fontsize=12, fontweight="bold")
ax3.axvline(x=imp.median(), color="#D85A30", linewidth=0.8,
            linestyle="--", alpha=0.6, label="Median")
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(fontsize=9)
for bar, val in zip(bars, imp.values):
    ax3.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color="#444")

# ── Plot 4: SHAP ──────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
print("Computing SHAP values (this may take a few seconds)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_hr = shap_values[1]
elif shap_values.ndim == 3:
    shap_hr = shap_values[:, :, 1]
else:
    shap_hr = shap_values

mean_shap = np.abs(shap_hr).mean(axis=0)
ss = pd.Series(mean_shap, index=feature_cols).sort_values()
sc = ["#534AB7" if v >= ss.median() else "#AFA9EC" for v in ss]
sb = ax4.barh(ss.index, ss.values,
              color=sc, edgecolor="none", height=0.6)
ax4.set_xlabel("Mean |SHAP value|", fontsize=11)
ax4.set_title("SHAP explainability (high-risk class)",
              fontsize=12, fontweight="bold")
ax4.spines[["top", "right"]].set_visible(False)
for bar, val in zip(sb, ss.values):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color="#444")

# ── Footer ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
    f"Synthetic AUROC: 0.851 (clean formula, 200 patients)  →  "
    f"Real TCGA AUROC: {auroc:.3f} (real outcomes, {len(df)} patients, "
    f"7 features)  |  Gap motivates multi-modal PhD extension",
    ha="center", fontsize=9, color="#888", style="italic")

# ── Save to disk ───────────────────────────────────────────────────────────────
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── Print results ──────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  AUROC          : {auroc:.3f}")
if (tp + fn) > 0:
    print(f"  Sensitivity    : {tp/(tp+fn):.1%}  (caught {tp} of {tp+fn} deaths)")
if (tn + fp) > 0:
    print(f"  Specificity    : {tn/(tn+fp):.1%}  (cleared {tn} of {tn+fp} survivors)")
if (tp + fp) > 0:
    print(f"  Precision      : {tp/(tp+fp):.1%}")
print("=" * 55)
print(f"  Synthetic AUROC: 0.851  →  Real AUROC: {auroc:.3f}")
print(f"  Gap = {0.851 - auroc:.3f}")
print("=" * 55)
print()

if os.path.exists(SAVE_PATH):
    size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"  Figure saved: {SAVE_PATH}")
    print(f"  File size:    {size_kb:.0f} KB")
    print()
    print("  Open the file in your explorer to see the 4-panel figure.")
else:
    print("  ERROR: file was not saved. Check folder permissions.")
    