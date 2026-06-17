import pandas as pd
import numpy as np

# ── Step 1: Load all files ─────────────────────────────────────────────────────
MISSING = "'--"   # TCGA's marker for missing values

clinical  = pd.read_csv("clinical.tsv",        sep="\t", low_memory=False).replace(MISSING, np.nan)
follow    = pd.read_csv("follow_up.tsv",        sep="\t", low_memory=False).replace(MISSING, np.nan)
pathology = pd.read_csv("pathology_detail.tsv", sep="\t", low_memory=False).replace(MISSING, np.nan)

print("Files loaded.")
print(f"  Clinical  : {clinical['cases.submitter_id'].nunique()} unique patients")
print(f"  Follow-up : {follow['cases.submitter_id'].nunique()} unique patients")
print(f"  Pathology : {pathology['cases.submitter_id'].nunique()} unique patients")

# ── Step 2: Build base clinical table (one row per patient) ───────────────────
base = clinical[[
    "cases.submitter_id",
    "demographic.age_at_index",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.days_to_last_follow_up",
]].drop_duplicates(subset="cases.submitter_id").copy()

base.columns = [
    "patient_id", "age", "vital_status",
    "days_to_death", "stage", "days_to_last_follow_up"
]

# ── Step 3: Convert stage → numeric severity (1=early, 4=advanced) ────────────
stage_map = {
    "Stage I":    1, "Stage IA": 1, "Stage IB": 1,
    "Stage IIA":  2, "Stage IIB": 2,
    "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV":   4,
}
base["stage_numeric"] = base["stage"].map(stage_map)

# ── Step 4: Extract ER, PR, HER2 from molecular tests in follow_up ────────────
mol = follow[follow["molecular_tests.gene_symbol"].isin(["ESR1", "PGR", "ERBB2"])].copy()
mol = mol[["cases.submitter_id",
           "molecular_tests.gene_symbol",
           "molecular_tests.test_result"]].copy()

# Keep only clear Positive/Negative calls
mol = mol[mol["molecular_tests.test_result"].isin(["Positive", "Negative"])]

# One row per patient per gene (take first result if duplicates)
mol = mol.drop_duplicates(subset=["cases.submitter_id", "molecular_tests.gene_symbol"])

# Pivot: rows = patients, columns = ESR1 / PGR / ERBB2
mol_wide = mol.pivot(index="cases.submitter_id",
                     columns="molecular_tests.gene_symbol",
                     values="molecular_tests.test_result")
mol_wide.columns.name = None
mol_wide = mol_wide.reset_index().rename(columns={
    "cases.submitter_id": "patient_id",
    "ESR1":  "er_status",
    "PGR":   "pr_status",
    "ERBB2": "her2_status",
})

# Encode as binary: Positive=1, Negative=0
for col in ["er_status", "pr_status", "her2_status"]:
    if col in mol_wide.columns:
        mol_wide[col] = (mol_wide[col] == "Positive").astype(float)
        mol_wide.loc[mol_wide[col].isna(), col] = np.nan

print(f"\nReceptor status extracted for {len(mol_wide)} patients")

# ── Step 5: Extract lymph node data from pathology ────────────────────────────
lymph = pathology[[
    "cases.submitter_id",
    "pathology_details.lymph_nodes_positive",
    "pathology_details.lymph_nodes_tested",
]].copy()
lymph.columns = ["patient_id", "lymph_nodes_positive", "lymph_nodes_tested"]
lymph["lymph_nodes_positive"] = pd.to_numeric(lymph["lymph_nodes_positive"], errors="coerce")
lymph["lymph_nodes_tested"]   = pd.to_numeric(lymph["lymph_nodes_tested"],   errors="coerce")

# Lymph node ratio: fraction of nodes that were positive
lymph["lymph_node_ratio"] = (
    lymph["lymph_nodes_positive"] / lymph["lymph_nodes_tested"]
).clip(0, 1)

# Binary lymph node involvement: any positive node = 1
lymph["lymph_node_positive"] = (lymph["lymph_nodes_positive"] > 0).astype(float)

print(f"Lymph node data for {lymph['patient_id'].nunique()} patients")

# ── Step 6: Build survival outcome label ──────────────────────────────────────
# We'll predict: did the patient die within 5 years (1825 days)?
def make_label(row):
    if row["vital_status"] == "Dead":
        days = pd.to_numeric(row["days_to_death"], errors="coerce")
        if pd.notna(days):
            return 1 if days <= 1825 else 0
    elif row["vital_status"] == "Alive":
        days = pd.to_numeric(row["days_to_last_follow_up"], errors="coerce")
        if pd.notna(days) and days >= 1825:
            return 0   # survived past 5 years = low risk
    return np.nan       # unknown — will be dropped

base["died_within_5yr"] = base.apply(make_label, axis=1)

# ── Step 7: Merge everything together ─────────────────────────────────────────
df = base.merge(mol_wide,  on="patient_id", how="left")
df = df.merge(lymph[["patient_id","lymph_node_ratio","lymph_node_positive"]],
              on="patient_id", how="left")

# ── Step 8: Select final columns ──────────────────────────────────────────────
feature_cols = [
    "age",              # demographic
    "stage_numeric",    # AJCC stage 1–4
    "er_status",        # ER positive (ESR1)
    "pr_status",        # PR positive (PGR)
    "her2_status",      # HER2 positive (ERBB2)
    "lymph_node_ratio", # fraction of positive nodes
    "lymph_node_positive", # any node involved
]

df_final = df[["patient_id"] + feature_cols + ["died_within_5yr"]].copy()

# ── Step 9: Drop rows where label is unknown ──────────────────────────────────
before = len(df_final)
df_final = df_final.dropna(subset=["died_within_5yr"])
after   = len(df_final)
print(f"\nPatients after label filtering: {after} (dropped {before - after} with unknown outcome)")

# ── Step 10: Report and save ──────────────────────────────────────────────────
df_final.to_csv("tcga_brca_dataset.csv", index=False)

print(f"\nDataset saved → tcga_brca_dataset.csv")
print(f"Shape: {df_final.shape[0]} patients × {df_final.shape[1]} columns")
print(f"\nClass balance:")
print(f"  Died within 5 years  : {int(df_final['died_within_5yr'].sum())} patients")
print(f"  Survived 5+ years    : {int((df_final['died_within_5yr']==0).sum())} patients")
print(f"\nMissing values per feature:")
print(df_final[feature_cols].isnull().sum().to_string())
print(f"\nFirst 5 patients:")
print(df_final.head().to_string())
