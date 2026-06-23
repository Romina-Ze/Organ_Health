"""Run PRS association models for organ metrics.

The input is a local merged table containing participant id, PRS columns,
organ metrics, age, and sex. Participant-level inputs and outputs are ignored
by Git.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


@dataclass
class PRSConfig:
    input_file: Path
    output_dir: Path
    age_col: str = "Age"
    sex_col: str = "Sex"
    prs_columns: dict[str, str] = field(
        default_factory=lambda: {
            "AD": "PRS_AD",
            "Stroke": "PRS_stroke",
            "MS": "PRS_MS",
            "PD": "PRS_PD",
        }
    )
    prs_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "PRS(AD)": "PRS_AD",
            "PRS(stroke)": "PRS_stroke",
            "PRS(MS)": "PRS_MS",
            "PRS(PD)": "PRS_PD",
        }
    )


def read_input(config: PRSConfig) -> pd.DataFrame:
    if not Path(config.input_file).exists():
        raise FileNotFoundError(f"Missing input file: {config.input_file}")
    df = pd.read_csv(config.input_file, low_memory=False)
    df.columns = df.columns.str.strip()
    for old, new in config.prs_aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def add_standardized_prs(df: pd.DataFrame, config: PRSConfig) -> dict[str, str]:
    df = df.copy()
    standardized = {}
    missing = [col for col in config.prs_columns.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing PRS columns: {missing}")
    for label, col in config.prs_columns.items():
        z_col = f"{col}_z"
        values = pd.to_numeric(df[col], errors="coerce")
        df[z_col] = StandardScaler().fit_transform(values.to_frame()).ravel()
        standardized[label] = z_col
    return df, standardized


def discover_metric_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    agegap_metrics = [col for col in df.columns if col.endswith("_AgeGap")]
    pc1_metrics = [
        col
        for col in df.columns
        if col.endswith(" PC1") or col.endswith("_PC1")
    ]
    pls_metrics = [col for col in df.columns if col.endswith("_PLS")]

    metric_sets = {
        "AgeGap": agegap_metrics,
        "PC1": pc1_metrics,
        "PLS": pls_metrics,
    }
    single_metric_sets = {
        "AgeGap": agegap_metrics,
        "PC1": pc1_metrics,
        "PLS": pls_metrics,
    }
    return metric_sets, single_metric_sets


def fit_ols(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    age_col: str,
    sex_col: str,
    standardize_metrics: bool = True,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper | None, int]:
    cols = [outcome] + predictors
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return None, 0

    model_df = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna()
    if model_df.shape[0] < len(predictors) + 10:
        return None, int(model_df.shape[0])

    X = model_df[predictors].copy()
    if standardize_metrics:
        metric_cols = [col for col in predictors if col not in {age_col, sex_col}]
        if metric_cols:
            X[metric_cols] = StandardScaler().fit_transform(X[metric_cols])

    X = sm.add_constant(X, has_constant="add")
    y = model_df[outcome]
    return sm.OLS(y, X).fit(), int(model_df.shape[0])


def model_row(
    model,
    n_used: int,
    analysis: str,
    metric_family: str,
    prs_name: str,
    prs_col: str,
    metric: str,
    covariates: list[str],
) -> dict[str, object]:
    if model is not None and metric in model.params.index:
        ci_low, ci_high = model.conf_int().loc[metric]
        return {
            "analysis": analysis,
            "metric_family": metric_family,
            "PRS": prs_name,
            "PRS_column": prs_col,
            "organ_metric": metric,
            "beta": model.params[metric],
            "se": model.bse[metric],
            "t_value": model.tvalues[metric],
            "p_value": model.pvalues[metric],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n_used,
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "covariates": "; ".join(covariates),
        }
    return {
        "analysis": analysis,
        "metric_family": metric_family,
        "PRS": prs_name,
        "PRS_column": prs_col,
        "organ_metric": metric,
        "beta": np.nan,
        "se": np.nan,
        "t_value": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": n_used,
        "r2": np.nan,
        "adj_r2": np.nan,
        "covariates": "; ".join(covariates),
    }


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["q_value"] = np.nan
    results["fdr_significant"] = False
    mask = results["p_value"].notna()
    if mask.any():
        rejected, q_values, _, _ = multipletests(
            results.loc[mask, "p_value"],
            alpha=0.05,
            method="fdr_bh",
        )
        results.loc[mask, "q_value"] = q_values
        results.loc[mask, "fdr_significant"] = rejected
    return results


def run_pipeline(config: PRSConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_input(config)
    df, prs_outcomes = add_standardized_prs(df, config)
    metric_sets, single_metric_sets = discover_metric_sets(df)
    covariates = [config.age_col, config.sex_col]

    multivariable_rows = []
    for metric_family, metrics in metric_sets.items():
        if not metrics:
            continue
        predictors = metrics + covariates
        for prs_name, prs_col in prs_outcomes.items():
            model, n_used = fit_ols(
                df,
                outcome=prs_col,
                predictors=predictors,
                age_col=config.age_col,
                sex_col=config.sex_col,
            )
            for metric in metrics:
                multivariable_rows.append(
                    model_row(
                        model,
                        n_used,
                        "PRS_multivariate_OLS",
                        metric_family,
                        prs_name,
                        prs_col,
                        metric,
                        covariates,
                    )
                )

    single_rows = []
    for metric_family, metrics in single_metric_sets.items():
        for prs_name, prs_col in prs_outcomes.items():
            for metric in metrics:
                predictors = [metric] + covariates
                model, n_used = fit_ols(
                    df,
                    outcome=prs_col,
                    predictors=predictors,
                    age_col=config.age_col,
                    sex_col=config.sex_col,
                )
                single_rows.append(
                    model_row(
                        model,
                        n_used,
                        "PRS_single_metric_OLS",
                        metric_family,
                        prs_name,
                        prs_col,
                        metric,
                        covariates,
                    )
                )

    multivariable = add_fdr(pd.DataFrame(multivariable_rows))
    single = add_fdr(pd.DataFrame(single_rows))
    outputs = {
        "multivariable": output_dir / "prs_multivariate_ols_results.csv",
        "single_metric": output_dir / "prs_single_metric_ols_results.csv",
    }
    multivariable.to_csv(outputs["multivariable"], index=False)
    single.to_csv(outputs["single_metric"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prs"))
    parser.add_argument("--age-col", default="Age")
    parser.add_argument("--sex-col", default="Sex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        PRSConfig(
            input_file=args.input_file,
            output_dir=args.output_dir,
            age_col=args.age_col,
            sex_col=args.sex_col,
        )
    )
    print("PRS pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
