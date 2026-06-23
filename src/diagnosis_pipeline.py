"""Create diagnosis labels and test organ metrics with logistic regression."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


DEFAULT_CONFIG = Path("configs/diagnosis_icd10.json")


@dataclass
class DiagnosisConfig:
    metadata_file: Path | None
    metrics_file: Path | None
    output_dir: Path
    diagnosis_file: Path | None = None
    config_file: Path = DEFAULT_CONFIG
    participant_id_col: str = "eid"
    year_birth_col: str = "Year_of_birth"
    baseline_age_col: str = "Age_at_recruitment"
    sex_col: str = "Sex"
    followup_col: str = "Follow-up Time"
    metric_suffix: str = "_AgeGap"
    analysis_name: str = "organ_metric_logistic"
    q_threshold: float = 0.05
    z_threshold: float = 1.96


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def numbered_columns(spec: dict) -> list[str]:
    columns = [spec["base"]]
    start = int(spec.get("numbered_start", 1))
    end = int(spec.get("numbered_end", 0))
    prefix = spec["numbered_prefix"]
    columns.extend([f"{prefix}{idx}" for idx in range(start, end + 1)])
    return columns


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing[:10]}")


def create_diagnosis_table(
    metadata: pd.DataFrame,
    icd_config: dict,
    config: DiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [
        config.participant_id_col,
        config.year_birth_col,
        config.baseline_age_col,
    ]
    diagnosis_cols = numbered_columns(icd_config["diagnosis_columns"])
    date_cols = numbered_columns(icd_config["diagnosis_date_columns"])
    require_columns(metadata, base_cols, "metadata")

    diagnosis_cols = [col for col in diagnosis_cols if col in metadata.columns]
    date_cols = [col for col in date_cols if col in metadata.columns]
    if not diagnosis_cols or not date_cols:
        raise ValueError("No diagnosis/date columns found in metadata.")

    diag_long = metadata.melt(
        id_vars=base_cols,
        value_vars=diagnosis_cols,
        var_name="diagnosis_col",
        value_name="ICD10_code",
    )
    date_long = metadata.melt(
        id_vars=[config.participant_id_col],
        value_vars=date_cols,
        var_name="date_col",
        value_name="diagnosis_date",
    )
    diag_long["diag_num"] = diag_long["diagnosis_col"].str.extract(r"(\d+)$")
    date_long["diag_num"] = date_long["date_col"].str.extract(r"(\d+)$")
    diag_long["diag_num"] = diag_long["diag_num"].fillna("0")
    date_long["diag_num"] = date_long["diag_num"].fillna("0")

    long_df = diag_long.merge(
        date_long,
        on=[config.participant_id_col, "diag_num"],
        how="inner",
    )
    long_df = long_df.drop(columns=["diagnosis_col", "date_col", "diag_num"])
    long_df = long_df[long_df["ICD10_code"].notna()].copy()
    long_df["ICD10_code"] = long_df["ICD10_code"].astype(str).str.replace(".", "", regex=False)

    code_to_disease = {
        code: disease
        for disease, codes in icd_config["diseases"].items()
        for code in codes
    }
    long_df["Disease Group"] = long_df["ICD10_code"].map(code_to_disease)
    for disease in icd_config["diseases"]:
        long_df[disease] = (long_df["Disease Group"] == disease).astype(int)

    for aggregate, members in icd_config.get("aggregate_diseases", {}).items():
        present = [member for member in members if member in long_df.columns]
        long_df[aggregate] = long_df[present].max(axis=1) if present else 0

    long_df["diagnosis_year"] = pd.to_datetime(
        long_df["diagnosis_date"],
        errors="coerce",
    ).dt.year
    long_df[config.followup_col] = long_df["diagnosis_year"] - (
        pd.to_numeric(long_df[config.year_birth_col], errors="coerce")
        + pd.to_numeric(long_df[config.baseline_age_col], errors="coerce")
    )
    long_df = long_df[long_df[config.followup_col] >= 0].copy()

    disease_cols = list(icd_config["diseases"].keys()) + list(
        icd_config.get("aggregate_diseases", {}).keys()
    )
    wide = long_df.groupby(config.participant_id_col)[disease_cols].max().reset_index()
    followup = (
        long_df.groupby(config.participant_id_col)[config.followup_col]
        .min()
        .reset_index()
    )
    demographics = metadata[
        [
            config.participant_id_col,
            config.sex_col,
            config.year_birth_col,
            config.baseline_age_col,
        ]
    ].copy()
    diagnosis = wide.merge(demographics, on=config.participant_id_col, how="left")
    diagnosis = diagnosis.merge(followup, on=config.participant_id_col, how="left")

    counts = (
        long_df[disease_cols + [config.participant_id_col]]
        .groupby(config.participant_id_col)
        .max()
        .sum(axis=0)
        .rename("N Participants")
        .reset_index()
        .rename(columns={"index": "Disease"})
    )
    return diagnosis, counts


def metric_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    if suffix == "PC1":
        return [col for col in df.columns if col.endswith(" PC1") or col.endswith("_PC1")]
    return [col for col in df.columns if col.endswith(suffix)]


def fit_logit_model(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[object | None, int, int, int]:
    cols = [outcome] + predictors
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return None, 0, 0, 0

    model_df = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna()
    if model_df.empty:
        return None, 0, 0, 0

    y = model_df[outcome]
    y = y.astype(int) if set(y.unique()) <= {0, 1} else (y > 0).astype(int)
    n_cases = int(y.sum())
    n_total = int(len(y))
    n_controls = n_total - n_cases
    if n_cases == 0 or n_controls == 0:
        return None, n_total, n_cases, n_controls

    X = sm.add_constant(model_df[predictors], has_constant="add")
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception:
        return None, n_total, n_cases, n_controls
    return model, n_total, n_cases, n_controls


def run_logistic_associations(
    diagnosis: pd.DataFrame,
    metrics: pd.DataFrame,
    disease_cols: list[str],
    config: DiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = diagnosis.merge(metrics, on=config.participant_id_col, how="inner")
    metrics_to_test = metric_columns(merged, config.metric_suffix)
    covariates = [config.baseline_age_col, config.sex_col, config.followup_col]
    predictors = metrics_to_test + covariates
    rows = []

    for disease in disease_cols:
        model, n_total, n_cases, n_controls = fit_logit_model(merged, disease, predictors)
        for predictor in predictors:
            if model is not None and predictor in model.params.index:
                ci_low, ci_high = model.conf_int().loc[predictor]
                beta = model.params[predictor]
                se = model.bse[predictor]
                rows.append(
                    {
                        "analysis": config.analysis_name,
                        "Disease": disease,
                        "Predictor": predictor,
                        "Coefficient": beta,
                        "Std Error": se,
                        "z-value": model.tvalues[predictor],
                        "p-value": model.pvalues[predictor],
                        "Odds Ratio": np.exp(beta),
                        "OR_low": np.exp(ci_low),
                        "OR_high": np.exp(ci_high),
                        "n_total": n_total,
                        "n_cases": n_cases,
                        "n_controls": n_controls,
                        "aic": model.aic,
                    }
                )
            else:
                rows.append(
                    {
                        "analysis": config.analysis_name,
                        "Disease": disease,
                        "Predictor": predictor,
                        "Coefficient": np.nan,
                        "Std Error": np.nan,
                        "z-value": np.nan,
                        "p-value": np.nan,
                        "Odds Ratio": np.nan,
                        "OR_low": np.nan,
                        "OR_high": np.nan,
                        "n_total": n_total,
                        "n_cases": n_cases,
                        "n_controls": n_controls,
                        "aic": np.nan,
                    }
                )

    results = pd.DataFrame(rows)
    results["q-value"] = np.nan
    results["fdr_significant"] = False
    test_mask = results["p-value"].notna() & results["Predictor"].isin(metrics_to_test)
    if test_mask.any():
        rejected, q_values, _, _ = multipletests(
            results.loc[test_mask, "p-value"],
            alpha=config.q_threshold,
            method="fdr_bh",
        )
        results.loc[test_mask, "q-value"] = q_values
        results.loc[test_mask, "fdr_significant"] = rejected

    zmat = results.pivot_table(
        index="Disease",
        columns="Predictor",
        values="z-value",
        aggfunc="first",
    )
    zmat = zmat[[col for col in metrics_to_test if col in zmat.columns]]

    thresholded = zmat.copy()
    for disease in thresholded.index:
        for predictor in thresholded.columns:
            row = results[
                (results["Disease"] == disease)
                & (results["Predictor"] == predictor)
            ]
            if row.empty:
                thresholded.loc[disease, predictor] = np.nan
                continue
            q_value = row["q-value"].iloc[0]
            z_value = row["z-value"].iloc[0]
            if pd.isna(q_value) or q_value >= config.q_threshold or abs(z_value) < config.z_threshold:
                thresholded.loc[disease, predictor] = np.nan

    return results, zmat, thresholded


def run_pipeline(config: DiagnosisConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    icd_config = load_config(config.config_file)

    outputs = {}
    if config.diagnosis_file:
        diagnosis = read_csv_checked(config.diagnosis_file)
    elif config.metadata_file:
        metadata = read_csv_checked(config.metadata_file)
        diagnosis, counts = create_diagnosis_table(metadata, icd_config, config)
        outputs["diagnosis"] = output_dir / "diagnosis_labels.csv"
        outputs["diagnosis_counts"] = output_dir / "diagnosis_counts.csv"
        diagnosis.to_csv(outputs["diagnosis"], index=False)
        counts.to_csv(outputs["diagnosis_counts"], index=False)
    else:
        raise ValueError("Provide either --metadata-file or --diagnosis-file.")

    if config.metrics_file:
        metrics = read_csv_checked(config.metrics_file)
        disease_cols = list(icd_config["diseases"].keys()) + list(
            icd_config.get("aggregate_diseases", {}).keys()
        )
        results, zmat, thresholded = run_logistic_associations(
            diagnosis,
            metrics,
            disease_cols,
            config,
        )
        outputs["logistic_results"] = output_dir / f"{config.analysis_name}_results.csv"
        outputs["z_matrix"] = output_dir / f"{config.analysis_name}_z_matrix.csv"
        outputs["thresholded_z_matrix"] = (
            output_dir / f"{config.analysis_name}_z_matrix_thresholded.csv"
        )
        results.to_csv(outputs["logistic_results"], index=False)
        zmat.to_csv(outputs["z_matrix"])
        thresholded.to_csv(outputs["thresholded_z_matrix"])

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--diagnosis-file", type=Path)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnosis"))
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metric-suffix", default="_AgeGap")
    parser.add_argument("--analysis-name", default="organ_metric_logistic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        DiagnosisConfig(
            metadata_file=args.metadata_file,
            diagnosis_file=args.diagnosis_file,
            metrics_file=args.metrics_file,
            output_dir=args.output_dir,
            config_file=args.config_file,
            metric_suffix=args.metric_suffix,
            analysis_name=args.analysis_name,
        )
    )
    print("Diagnosis pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
