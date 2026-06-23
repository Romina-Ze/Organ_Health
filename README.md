# Organ Health Analysis

This repository contains cleaned Python code and small notebook wrappers for organ-health analyses using UK Biobank data.

The real analysis code is in `src/`.

The notebooks in `notebooks/` are simple readable launchers for the code.

The `configs/` folder stores analysis settings such as ICD10 disease groups and mediation pathways.

UK Biobank data and generated results are not included. Keep them local in `data/` and `results/`.

Included analyses:

- organ age-gap models
- PC1/PCA organ metrics
- PLS organ metrics
- PRS associations
- diagnosis/logistic regression analyses
- mediation analyses

The standalone `Brain_health.ipynb` file provided for refactoring was empty. Brain-related analyses are currently included through Brain age gap, Brain PC1/PLS metrics, diagnosis models, and mediation pathways.
