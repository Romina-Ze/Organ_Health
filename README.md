# Organ Health Analysis Pipelines

Code-only repository for organ age, PC1, PLS, PRS, diagnosis, and mediation analyses. Controlled-access UK Biobank data, participant-level outputs, trained models, and generated tables are intentionally excluded from Git.

## Analyses

- `src/organ_age_gap_pipeline.py`: XGBoost biological age and age-gap models.
- `src/pc1_pipeline.py`: organ-level PCA/PC1 scores, protein loadings, and PC-age correlation summaries.
- `src/pls_pipeline.py`: local PLS score models from proteomics and organ phenotype tables.
- `src/diagnosis_pipeline.py`: ICD10 disease-label creation and logistic models for organ metrics.
- `src/prs_pipeline.py`: PRS association models for age-gap, PC1, and PLS metrics.
- `src/mediation_pipeline.py`: bootstrap mediation models for PRS, organ metrics, and incident disease.

Small notebooks in `notebooks/` are wrappers around these scripts. Messy exploratory notebooks should stay private, for example in `notebooks/exploratory/`, which is ignored by Git.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Keep real data local only, for example under `data/raw/`. The `data/` and `results/` folders are ignored.

## Example Commands

```bash
python src/organ_age_gap_pipeline.py --input-dir data/raw --output-dir results/organ_age
python src/pc1_pipeline.py --input-dir data/raw --output-dir results/pc1
python src/diagnosis_pipeline.py --metadata-file data/raw/metadata.csv --metrics-file results/pc1/age_gaps_and_pcs.csv --output-dir results/diagnosis
python src/prs_pipeline.py --input-file data/raw/prs_agegap_pc1_pls.csv --output-dir results/prs
python src/mediation_pipeline.py --input-file data/raw/mediation_master.csv --output-dir results/mediation
```

## Before Pushing To GitHub

Run:

```bash
git status
git ls-files
```

Only code, configs, small notebooks, and documentation should be tracked. Do not push `.csv`, `.txt`, `.xlsx`, `.pkl`, `.joblib`, `data/raw`, `results`, or notebooks with saved participant-level outputs.

If controlled-access data is ever pushed, make the repository private immediately and ask your supervisor or data access team what remediation is required. Removing the file in a later commit is not enough because Git keeps history.
