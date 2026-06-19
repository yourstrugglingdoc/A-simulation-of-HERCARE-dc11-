import numpy as np
import pandas as pd

# So we get the same "random" numbers every time we run
np.random.seed(42)

N = 200  # 200 synthetic patients

# ── MRI features ──────────────────────────────────────────────────────────────
tumour_size_mm     = np.random.normal(25, 10, N).clip(1, 60)
enhancement_ratio  = np.random.uniform(0.2, 1.0, N)
adc_value          = np.random.normal(0.9, 0.2, N).clip(0.4, 1.5)
margin_irregularity= np.random.uniform(0, 1, N)

# ── Genomic / biomarker features ──────────────────────────────────────────────
ki67_index         = np.random.uniform(0, 100, N)
oncotype_score     = np.random.uniform(0, 100, N)
her2_score         = np.random.choice([1, 2, 3], N, p=[0.5, 0.3, 0.2])
er_pr_positive     = np.random.choice([0, 1], N, p=[0.3, 0.7])

# ── Clinical features ─────────────────────────────────────────────────────────
age                = np.random.normal(55, 12, N).clip(25, 85)
lymph_node_status  = np.random.choice([0, 1, 2], N, p=[0.5, 0.35, 0.15])
tumour_grade       = np.random.choice([1, 2, 3], N, p=[0.2, 0.45, 0.35])
family_history     = np.random.choice([0, 1], N, p=[0.7, 0.3])

# ── Build risk score (this is our "ground truth" label) ───────────────────────
# Higher tumour size, ki67, grade, her2 → higher risk
# Higher ADC and ER/PR positive → lower risk
risk_score = (
    0.25 * (tumour_size_mm / 60) +
    0.20 * (ki67_index / 100) +
    0.15 * (margin_irregularity) +
    0.15 * (her2_score / 3) +
    0.10 * (lymph_node_status / 2) +
    0.10 * (tumour_grade / 3) +
    0.05 * (1 - er_pr_positive) +
    np.random.normal(0, 0.05, N)   # small realistic noise
).clip(0, 1)

high_risk = (risk_score >= 0.5).astype(int)   # 1 = high risk, 0 = low risk

# ── Simulate missing data (real clinical scenario) ────────────────────────────
oncotype_score_missing = oncotype_score.copy().astype(float)
oncotype_score_missing[np.random.choice(N, 40, replace=False)] = np.nan  # 20% missing

# ── Assemble into a DataFrame ─────────────────────────────────────────────────
df = pd.DataFrame({
    # MRI
    "tumour_size_mm":      tumour_size_mm,
    "enhancement_ratio":   enhancement_ratio,
    "adc_value":           adc_value,
    "margin_irregularity": margin_irregularity,
    # Genomic
    "ki67_index":          ki67_index,
    "oncotype_score":      oncotype_score_missing,
    "her2_score":          her2_score,
    "er_pr_positive":      er_pr_positive,
    # Clinical
    "age":                 age,
    "lymph_node_status":   lymph_node_status,
    "tumour_grade":        tumour_grade,
    "family_history":      family_history,
    # Labels
    "risk_score":          risk_score,
    "high_risk":           high_risk,
})

df.index.name = "patient_id"

# ── Save and inspect ──────────────────────────────────────────────────────────
df.to_csv("breast_cancer_dataset.csv")
print("Dataset saved to breast_cancer_dataset.csv")
print(f"\nShape: {df.shape[0]} patients × {df.shape[1]} features")
print(f"\nHigh risk patients : {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")
print(f"Low risk patients  : {(df['high_risk']==0).sum()} ({(df['high_risk']==0).mean()*100:.1f}%)")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print("\nFirst 5 patients:")
print(df.head().to_string())