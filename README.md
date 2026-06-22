# Organ Age Gap XGBoost Pipeline

This repository contains code for training organ age gap models from proteomics data. It is intentionally code-only: UK Biobank data, participant-level outputs, trained model files, and generated result tables must not be committed to GitHub.

## What Is Safe To Commit

Commit:

- pipeline code in `src/`
- clean notebooks in `notebooks/` with no saved outputs
- `README.md`, `.gitignore`, and `requirements.txt`
- small documentation files that do not contain participant-level data

Do not commit:

- UK Biobank raw or processed data
- files with `eid` or participant-level rows
- generated predictions such as age gaps or biological ages
- trained models, SHAP weights, feature importances, or demographic tables unless your data agreement and supervisor explicitly allow public release
- local paths, credentials, or downloaded archives

## Repository Structure

```text
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- data/
|   `-- README.md
|-- notebooks/
|   `-- organ_age_gap_pipeline.ipynb
|-- results/
|   `-- README.md
`-- src/
    `-- organ_age_gap_pipeline.py
```

## Local Setup

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the real data only on your own machine, for example:

```text
data/raw/1.final_merged_protein_organ.csv
data/raw/N.merged_proteomics_mri.csv
data/raw/N.proteomics_preprocessed.csv
```

The `data/` folder is ignored by Git, so these files should stay local.

## Run The Pipeline

From the repository root:

```bash
python src/organ_age_gap_pipeline.py \
  --input-dir data/raw \
  --output-dir results
```

For a broader hyperparameter search similar to the exploratory notebook:

```bash
python src/organ_age_gap_pipeline.py \
  --input-dir data/raw \
  --output-dir results \
  --full-grid
```

## Beginner Git Workflow

Start from this clean folder, not from the whole Downloads project folder:

```bash
git init
git status
git add .gitignore README.md requirements.txt src notebooks data/README.md results/README.md
git commit -m "Add organ age gap pipeline"
git branch -M main
```

Create an empty GitHub repository in the browser, then connect it:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

## Safety Checks Before Every Push

Run these before `git push`:

```bash
git status
git ls-files
git ls-files | rg "(\.csv|\.tsv|\.txt|\.xlsx|\.parquet|\.pkl|\.joblib|\.npy|\.npz|data/|results/)"
```

The last command should only show safe placeholder documentation files, not real data.

If a data file appears in `git status`, do not push. Remove it from the staging area:

```bash
git restore --staged path/to/file
```

If you already committed data but did not push:

```bash
git rm --cached path/to/file
git commit --amend
```

If controlled-access data was pushed to GitHub, delete or make the repository private immediately and ask your supervisor/data access team what remediation is required. Removing the file in a later commit is not enough, because Git keeps history.
