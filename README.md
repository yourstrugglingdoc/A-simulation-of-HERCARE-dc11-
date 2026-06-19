# HER-CARE DC11 — Breast Cancer AI Pipeline

End-to-end multi-modal breast cancer risk prediction pipeline, built as a proof-of-concept for the HER-CARE doctoral network DC11 position (AI risk model integration into clinical workflows).

## What this project does

- Generates synthetic multi-modal data (MRI, genomic, clinical)
- Trains a Random Forest fusion model with class balancing
- Generates per-patient clinical decision support reports
- Evaluates with AUROC, confusion matrix, and SHAP explainability
- Validated on real TCGA-BRCA data (1,098 patients, 334 labeled)

## Results

| Dataset   | AUROC | Patients | Features |
|-----------|-------|----------|----------|
| Synthetic | 0.851 | 200      | 12       |
| TCGA-BRCA | TBD   | 334      | 7        |

## Data access

Download TCGA-BRCA clinical files from the GDC portal:
https://portal.gdc.cancer.gov/projects/TCGA-BRCA

## Requirements
pip install numpy pandas scikit-learn matplotlib seaborn shap

## Author

Built by Aseel as a self-directed mini-project demonstrating readiness for the HER-CARE DC11 PhD position at the Medical University of Vienna.
