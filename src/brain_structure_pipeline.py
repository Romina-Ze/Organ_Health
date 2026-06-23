"""Run brain-structure association analyses for organ health metrics.

This is a cleaned version of the final brain-structure notebook analysis. It
keeps UK Biobank data local and writes only generated result tables to the
ignored ``results`` directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


CORTICAL_OUTCOMES = [
    "caudalmiddlefrontal_thickness",
    "lateralorbitofrontal_thickness",
    "medialorbitofrontal_thickness",
    "parsopercularis_thickness",
    "parsorbitalis_thickness",
    "parstriangularis_thickness",
    "rostralmiddlefrontal_thickness",
    "superiorfrontal_thickness",
    "precentral_thickness",
    "inferiorparietal_thickness",
    "postcentral_thickness",
    "precuneus_thickness",
    "superiorparietal_thickness",
    "supramarginal_thickness",
    "paracentral_thickness",
    "entorhinal_thickness",
    "fusiform_thickness",
    "inferiortemporal_thickness",
    "middletemporal_thickness",
    "parahippocampal_thickness",
    "superiortemporal_thickness",
    "transversetemporal_thickness",
    "cuneus_thickness",
    "lateraloccipital_thickness",
    "lingual_thickness",
    "pericalcarine_thickness",
    "caudalanteriorcingulate_thickness",
    "rostralanteriorcingulate_thickness",
    "posteriorcingulate_thickness",
    "insula_thickness",
    "isthmuscingulate_thickness",
]

SUBCORTICAL_OUTCOMES = [
    "accumbens_area_volume",
    "amygdala_volume",
    "caudate_volume",
    "choroid_plexus_volume",
    "hippocampus_volume",
    "inf_lat_vent_volume",
    "lateral_ventricle_volume",
    "pallidum_volume",
    "putamen_volume",
    "subcortical_gm_volume",
    "thalamus_proper_volume",
    "ventraldc_volume",
]

BRAIN_BLOCKS = {
    "Frontal": CORTICAL_OUTCOMES[0:9],
    "Parietal": CORTICAL_OUTCOMES[9:15],
    "Temporal": CORTICAL_OUTCOMES[15:22],
    "Occipital": CORTICAL_OUTCOMES[22:26],
    "Cingulate / Insula": CORTICAL_OUTCOMES[26:31],
    "Subcortical": SUBCORTICAL_OUTCOMES,
}

AGE_CANDIDATES = [
    "Age",
    "age",
    "Age_at_recruitment",
    "Age at recruitment",
    "Age_at_MRI",
    "age_at_mri",
]

SEX_CANDIDATES = [
    "Sex",
    "sex",
    "Genetic_sex",
    "Genetic sex",
    "sex_binary",
]

TIV_CANDIDATES = [
    "TIV",
    "tiv",
    "eTIV",
    "EstimatedTotalIntraCranialVol",
    "intracranial_volume",
]

COMPOSITE_PLS_METRICS = {
    "Metabolic_System_PLS",
    "Immune/Circulatory_PLS",
}

MODEL_RESULT_COLUMNS = [
    "family",
    "outcome",
    "brain_block",
    "metric",
    "beta",
    "std_error",
    "t_value",
    "p_value",
    "n",
    "r_squared",
    "adj_r_squared",
    "aic",
]

SELECTION_SUMMARY_COLUMNS = [
    "family",
    "outcome",
    "brain_block",
    "n",
    "forced_covariates",
    "selected_metrics",
    "n_selected_metrics",
    "r_squared",
    "adj_r_squared",
    "aic",
]


@dataclass
class BrainStructureConfig:
    output_dir: Path
    age_col: str | None = None
    sex_col: str | None = None
    tiv_col: str | None = None
    min_n: int = 30


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def detect_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
    label: str = "column",
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(f"Could not find {label}. Tried: {list(candidates)}")
    return None


def choose_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    required: bool,
    label: str,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{label} not found: {explicit}")
        return explicit
    return detect_column(df, candidates, required=required, label=label)


def normalize_sex(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = series.astype(str).str.strip().str.lower()
        numeric = pd.to_numeric(
            cleaned.replace(
                {
                    "female": 0,
                    "f": 0,
                    "woman": 0,
                    "0": 0,
                    "male": 1,
                    "m": 1,
                    "man": 1,
                    "1": 1,
                    "2": 1,
                }
            ),
            errors="coerce",
        )
    values = set(pd.Series(numeric).dropna().unique())
    if values == {1, 2}:
        numeric = numeric - 1
    return pd.Series(numeric, index=series.index, name=series.name)


def brain_block_for_outcome(outcome: str) -> str:
    for block_name, outcomes in BRAIN_BLOCKS.items():
        if outcome in outcomes:
            return block_name
    return "Other"


def available_outcomes(df: pd.DataFrame, include_subcortical: bool) -> list[str]:
    outcomes = [col for col in CORTICAL_OUTCOMES if col in df.columns]
    if include_subcortical:
        outcomes.extend([col for col in SUBCORTICAL_OUTCOMES if col in df.columns])
    if not outcomes:
        raise ValueError("No expected brain-structure outcome columns were found.")
    return outcomes


def metric_columns(df: pd.DataFrame, family: str) -> list[str]:
    if family == "agegap":
        metrics = [
            col
            for col in df.columns
            if col.endswith("_AgeGap") or col.endswith(" AgeGap")
        ]
    elif family == "pc1":
        metrics = [
            col
            for col in df.columns
            if col.endswith(" PC1") or col.endswith("_PC1")
        ]
    elif family == "pls":
        metrics = [
            col
            for col in df.columns
            if col.endswith("_PLS") and col not in COMPOSITE_PLS_METRICS
        ]
    else:
        raise ValueError(f"Unknown metric family: {family}")

    metrics = [col for col in metrics if col in df.columns]
    if not metrics:
        raise ValueError(f"No {family} metric columns were found.")
    return metrics


def analysis_frame(
    df: pd.DataFrame,
    columns: Iterable[str],
    sex_col: str | None,
) -> pd.DataFrame:
    frame = df[list(dict.fromkeys(columns))].copy()
    for col in frame.columns:
        if col == sex_col:
            frame[col] = normalize_sex(frame[col])
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame


def fit_ols_model(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    sex_col: str | None,
    min_n: int,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper | None, int]:
    frame = analysis_frame(df, [outcome] + predictors, sex_col)
    n = int(len(frame))
    if n < min_n or n <= len(predictors) + 2:
        return None, n

    y = frame[outcome]
    x = sm.add_constant(frame[predictors], has_constant="add")
    try:
        return sm.OLS(y, x).fit(), n
    except (ValueError, np.linalg.LinAlgError):
        return None, n


def bidirectional_aic_selection(
    df: pd.DataFrame,
    outcome: str,
    forced_covariates: list[str],
    candidate_predictors: list[str],
    sex_col: str | None,
    min_n: int,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper | None, list[str], int]:
    selected = list(forced_covariates)
    model, n = fit_ols_model(df, outcome, selected, sex_col, min_n)
    if model is None:
        return None, [], n

    current_aic = model.aic
    improved = True
    while improved:
        improved = False

        add_options = []
        for predictor in candidate_predictors:
            if predictor in selected:
                continue
            add_model, add_n = fit_ols_model(
                df,
                outcome,
                selected + [predictor],
                sex_col,
                min_n,
            )
            if add_model is not None:
                add_options.append((add_model.aic, predictor, add_model, add_n))
        if add_options:
            best_aic, best_predictor, best_model, best_n = min(
                add_options,
                key=lambda item: item[0],
            )
            if best_aic < current_aic:
                selected.append(best_predictor)
                model = best_model
                n = best_n
                current_aic = best_aic
                improved = True

        removable = [col for col in selected if col not in forced_covariates]
        remove_options = []
        for predictor in removable:
            predictors = [col for col in selected if col != predictor]
            remove_model, remove_n = fit_ols_model(
                df,
                outcome,
                predictors,
                sex_col,
                min_n,
            )
            if remove_model is not None:
                remove_options.append((remove_model.aic, predictor, remove_model, remove_n))
        if remove_options:
            best_aic, worst_predictor, best_model, best_n = min(
                remove_options,
                key=lambda item: item[0],
            )
            if best_aic < current_aic:
                selected.remove(worst_predictor)
                model = best_model
                n = best_n
                current_aic = best_aic
                improved = True

    selected_metrics = [col for col in selected if col not in forced_covariates]
    return model, selected_metrics, n


def covariates_for_outcome(
    outcome: str,
    age_col: str,
    sex_col: str,
    tiv_col: str | None,
) -> list[str]:
    covariates = [age_col, sex_col]
    if brain_block_for_outcome(outcome) == "Subcortical" and tiv_col:
        covariates.append(tiv_col)
    return covariates


def add_fdr(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    df = df.copy()
    df["p_fdr_bh"] = np.nan
    if p_col not in df.columns:
        return df
    mask = df[p_col].notna()
    if mask.any():
        df.loc[mask, "p_fdr_bh"] = multipletests(
            df.loc[mask, p_col],
            method="fdr_bh",
        )[1]
    return df


def model_metric_rows(
    family: str,
    outcome: str,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    n: int,
    metrics: list[str],
) -> list[dict[str, object]]:
    rows = []
    for metric in metrics:
        if metric not in model.params.index:
            continue
        rows.append(
            {
                "family": family,
                "outcome": outcome,
                "brain_block": brain_block_for_outcome(outcome),
                "metric": metric,
                "beta": float(model.params[metric]),
                "std_error": float(model.bse[metric]),
                "t_value": float(model.tvalues[metric]),
                "p_value": float(model.pvalues[metric]),
                "n": n,
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
            }
        )
    return rows


def simple_summary(simple_results: pd.DataFrame) -> pd.DataFrame:
    if simple_results.empty:
        return pd.DataFrame()

    summary = (
        simple_results.groupby("metric")
        .agg(
            n_tests=("outcome", "nunique"),
            n_nominal_p_lt_0_05=("p_value", lambda values: int((values < 0.05).sum())),
            n_fdr_p_lt_0_05=("p_fdr_bh", lambda values: int((values < 0.05).sum())),
            min_p=("p_value", "min"),
            min_p_fdr=("p_fdr_bh", "min"),
            max_abs_beta=("beta", lambda values: float(np.nanmax(np.abs(values)))),
        )
        .reset_index()
        .sort_values(["min_p_fdr", "min_p", "metric"], na_position="last")
    )
    return summary


def run_metric_family(
    df: pd.DataFrame,
    family: str,
    config: BrainStructureConfig,
    include_subcortical: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_col = choose_column(df, config.age_col, AGE_CANDIDATES, True, "age column")
    sex_col = choose_column(df, config.sex_col, SEX_CANDIDATES, True, "sex column")
    tiv_col = choose_column(df, config.tiv_col, TIV_CANDIDATES, False, "TIV column")
    if age_col is None or sex_col is None:
        raise ValueError("Age and sex columns are required.")

    outcomes = available_outcomes(df, include_subcortical=include_subcortical)
    metrics = metric_columns(df, family)

    main_rows = []
    selection_rows = []
    simple_rows = []

    for outcome in outcomes:
        covariates = covariates_for_outcome(outcome, age_col, sex_col, tiv_col)
        model, selected_metrics, n = bidirectional_aic_selection(
            df,
            outcome,
            covariates,
            metrics,
            sex_col,
            config.min_n,
        )
        if model is not None:
            main_rows.extend(model_metric_rows(family, outcome, model, n, selected_metrics))
            selection_rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "brain_block": brain_block_for_outcome(outcome),
                    "n": n,
                    "forced_covariates": "; ".join(covariates),
                    "selected_metrics": "; ".join(selected_metrics),
                    "n_selected_metrics": len(selected_metrics),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "aic": float(model.aic),
                }
            )
        else:
            selection_rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "brain_block": brain_block_for_outcome(outcome),
                    "n": n,
                    "forced_covariates": "; ".join(covariates),
                    "selected_metrics": "",
                    "n_selected_metrics": 0,
                    "r_squared": np.nan,
                    "adj_r_squared": np.nan,
                    "aic": np.nan,
                }
            )

        for metric in metrics:
            simple_model, simple_n = fit_ols_model(
                df,
                outcome,
                covariates + [metric],
                sex_col,
                config.min_n,
            )
            if simple_model is None:
                continue
            simple_rows.extend(
                model_metric_rows(family, outcome, simple_model, simple_n, [metric])
            )

    main_results = add_fdr(pd.DataFrame(main_rows, columns=MODEL_RESULT_COLUMNS))
    selection_summary = pd.DataFrame(
        selection_rows,
        columns=SELECTION_SUMMARY_COLUMNS,
    )
    simple_results = add_fdr(pd.DataFrame(simple_rows, columns=MODEL_RESULT_COLUMNS))
    return main_results, selection_summary, simple_results, simple_summary(simple_results)


def write_family_outputs(
    family: str,
    output_dir: Path,
    main_results: pd.DataFrame,
    selection_summary: pd.DataFrame,
    simple_results: pd.DataFrame,
    simple_results_summary: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        f"{family}_bidirectional_main_results": (
            output_dir / f"FINAL_{family}_bidirectional_main_results.csv"
        ),
        f"{family}_bidirectional_selection_summary": (
            output_dir / f"FINAL_{family}_bidirectional_selection_summary.csv"
        ),
        f"{family}_simple_single_organ_OLS_results": (
            output_dir / f"FINAL_{family}_simple_single_organ_OLS_results.csv"
        ),
        f"{family}_simple_single_organ_OLS_summary": (
            output_dir / f"FINAL_{family}_simple_single_organ_OLS_summary.csv"
        ),
    }
    main_results.to_csv(paths[f"{family}_bidirectional_main_results"], index=False)
    selection_summary.to_csv(
        paths[f"{family}_bidirectional_selection_summary"],
        index=False,
    )
    simple_results.to_csv(paths[f"{family}_simple_single_organ_OLS_results"], index=False)
    simple_results_summary.to_csv(
        paths[f"{family}_simple_single_organ_OLS_summary"],
        index=False,
    )
    return paths


def run_pipeline(
    config: BrainStructureConfig,
    agegap_file: Path | None = None,
    pc1_file: Path | None = None,
    pls_file: Path | None = None,
) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {
        "agegap": agegap_file,
        "pc1": pc1_file,
        "pls": pls_file,
    }
    if not any(requested.values()):
        raise ValueError("Provide at least one of agegap_file, pc1_file, or pls_file.")

    outputs: dict[str, Path] = {}
    for family, path in requested.items():
        if path is None:
            continue
        df = read_csv_checked(Path(path))
        include_subcortical = family in {"agegap", "pc1"}
        family_outputs = run_metric_family(
            df,
            family=family,
            config=config,
            include_subcortical=include_subcortical,
        )
        outputs.update(write_family_outputs(family, output_dir, *family_outputs))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agegap-file", type=Path)
    parser.add_argument("--pc1-file", type=Path)
    parser.add_argument("--pls-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/brain_structure"))
    parser.add_argument("--age-col")
    parser.add_argument("--sex-col")
    parser.add_argument("--tiv-col")
    parser.add_argument("--min-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BrainStructureConfig(
        output_dir=args.output_dir,
        age_col=args.age_col,
        sex_col=args.sex_col,
        tiv_col=args.tiv_col,
        min_n=args.min_n,
    )
    outputs = run_pipeline(
        config,
        agegap_file=args.agegap_file,
        pc1_file=args.pc1_file,
        pls_file=args.pls_file,
    )
    print("Brain-structure pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
