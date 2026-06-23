"""Build organ PLS scores from local proteomics and phenotype tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class PLSConfig:
    config_file: Path
    output_dir: Path
    n_splits: int = 5
    random_state: int = 42


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "eid" in df.columns:
        df["eid"] = df["eid"].astype(str)
    return df


def normalize_sex(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = series.astype(str).str.strip().str.lower()
        cleaned = cleaned.replace(
            {
                "female": 0,
                "f": 0,
                "0": 0,
                "male": 1,
                "m": 1,
                "1": 1,
                "2": 1,
            }
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
    unique = set(pd.Series(numeric).dropna().unique())
    if unique == {1, 2}:
        numeric = numeric - 1
    return pd.Series(numeric, index=series.index, name=series.name)


def protein_columns(df: pd.DataFrame, start_col: str, end_col: str) -> list[str]:
    if start_col not in df.columns:
        raise ValueError(f"Protein start column not found: {start_col}")
    columns = list(df.columns)
    start = columns.index(start_col)
    if end_col in df.columns:
        stop = columns.index(end_col) + 1
        return columns[start:stop]
    return columns[start:]


def prepare_covariates(
    df: pd.DataFrame,
    sex_col: str,
    height_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, pd.Series]:
    covariate_cols = [sex_col] + ([height_col] if height_col else [])
    covars = df[covariate_cols].copy()
    covars[sex_col] = normalize_sex(covars[sex_col])
    if height_col:
        covars[height_col] = pd.to_numeric(covars[height_col], errors="coerce")
        covars[height_col] = covars[height_col].fillna(covars[height_col].mean())
    if covars[sex_col].isna().any():
        raise ValueError(f"{sex_col} contains unmapped or missing values.")
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(covars),
        columns=covars.columns,
        index=covars.index,
    )
    return covars, scaled, scaler, covars.mean(numeric_only=True)


def apply_covariates(
    df: pd.DataFrame,
    sex_col: str,
    height_col: str | None,
    scaler: StandardScaler,
    fill_values: pd.Series,
) -> pd.DataFrame:
    covariate_cols = [sex_col] + ([height_col] if height_col else [])
    covars = df[covariate_cols].copy()
    covars[sex_col] = normalize_sex(covars[sex_col])
    if height_col:
        covars[height_col] = pd.to_numeric(covars[height_col], errors="coerce")
    covars = covars.fillna(fill_values)
    return pd.DataFrame(
        scaler.transform(covars),
        columns=covars.columns,
        index=covars.index,
    )


def fit_imputer(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )
    return imputed, imputer


def apply_imputer(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    return pd.DataFrame(
        imputer.transform(df),
        columns=df.columns,
        index=df.index,
    )


def residualize_fit(
    matrix_df: pd.DataFrame,
    covariates_scaled: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LinearRegression]]:
    residuals = pd.DataFrame(index=matrix_df.index, columns=matrix_df.columns, dtype=float)
    models = {}
    for col in matrix_df.columns:
        y = pd.to_numeric(matrix_df[col], errors="coerce")
        if y.isna().any():
            raise ValueError(f"Residualization input still has NA values in {col}.")
        model = LinearRegression().fit(covariates_scaled, y)
        residuals[col] = y - model.predict(covariates_scaled)
        models[col] = model
    return residuals, models


def residualize_apply(
    matrix_df: pd.DataFrame,
    covariates_scaled: pd.DataFrame,
    models: dict[str, LinearRegression],
) -> pd.DataFrame:
    residuals = pd.DataFrame(index=matrix_df.index, columns=matrix_df.columns, dtype=float)
    for col in matrix_df.columns:
        y = pd.to_numeric(matrix_df[col], errors="coerce")
        if y.isna().any():
            raise ValueError(f"Residualization input still has NA values in {col}.")
        residuals[col] = y - models[col].predict(covariates_scaled)
    return residuals


def orient_from_anchors(
    pls: PLSRegression,
    y_train: pd.DataFrame,
    bad_up: list[str],
    good_up: list[str],
) -> tuple[int, float]:
    y_z = pd.DataFrame(
        StandardScaler().fit_transform(y_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    anchor = pd.Series(0.0, index=y_train.index)
    bad_keep = [col for col in bad_up if col in y_z.columns]
    good_keep = [col for col in good_up if col in y_z.columns]
    if bad_keep:
        anchor += y_z[bad_keep].mean(axis=1)
    if good_keep:
        anchor -= y_z[good_keep].mean(axis=1)
    y_scores = pd.Series(pls.y_scores_[:, 0], index=y_train.index)
    corr = y_scores.corr(anchor)
    sign = 1 if pd.notna(corr) and corr >= 0 else -1
    return sign, float(corr) if pd.notna(corr) else np.nan


def cross_validate_pls(
    model_df: pd.DataFrame,
    protein_cols: list[str],
    y_cols: list[str],
    sex_col: str,
    height_col: str | None,
    bad_up: list[str],
    good_up: list[str],
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []
    phenotype_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(model_df), start=1):
        train_df = model_df.iloc[train_idx].copy()
        test_df = model_df.iloc[test_idx].copy()
        y_train = train_df[y_cols].apply(pd.to_numeric, errors="coerce")
        y_test = test_df[y_cols].apply(pd.to_numeric, errors="coerce")
        x_train = train_df[protein_cols].apply(pd.to_numeric, errors="coerce")
        x_test = test_df[protein_cols].apply(pd.to_numeric, errors="coerce")

        _, cov_train_scaled, cov_scaler, fill_values = prepare_covariates(
            train_df,
            sex_col,
            height_col,
        )
        cov_test_scaled = apply_covariates(
            test_df,
            sex_col,
            height_col,
            cov_scaler,
            fill_values,
        )
        x_train_imp, x_imputer = fit_imputer(x_train)
        x_test_imp = apply_imputer(x_test, x_imputer)
        x_train_res, x_models = residualize_fit(x_train_imp, cov_train_scaled)
        y_train_res, y_models = residualize_fit(y_train, cov_train_scaled)
        x_test_res = residualize_apply(x_test_imp, cov_test_scaled, x_models)
        y_test_res = residualize_apply(y_test, cov_test_scaled, y_models)

        pls = PLSRegression(n_components=1).fit(x_train_res, y_train_res)
        sign, anchor_corr = orient_from_anchors(pls, y_train, bad_up, good_up)
        y_pred = pd.DataFrame(
            sign * pls.predict(x_test_res),
            index=x_test_res.index,
            columns=y_cols,
        )
        y_true = sign * y_test_res
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "r2_overall": r2_score(y_true, y_pred, multioutput="uniform_average"),
                "corr_with_worse_index": anchor_corr,
            }
        )
        for col in y_cols:
            phenotype_rows.append(
                {
                    "fold": fold,
                    "phenotype_name": col,
                    "r2": r2_score(y_true[col], y_pred[col]),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(phenotype_rows)


def run_one_organ(
    proteomics: pd.DataFrame,
    phenotypes: pd.DataFrame,
    organ_config: dict,
    global_config: dict,
    output_dir: Path,
    n_splits: int,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    organ_name = organ_config["organ_name"]
    score_name = organ_config.get("score_name", f"{organ_name}_PLS")
    y_cols = organ_config["phenotype_columns"]
    bad_up = organ_config.get("bad_up", [])
    good_up = organ_config.get("good_up", [])
    sex_col = organ_config.get("sex_col", global_config.get("sex_col", "Sex"))
    height_col = organ_config.get("height_col", global_config.get("height_col", "Height"))
    protein_start = organ_config.get(
        "protein_start_col",
        global_config.get("protein_start_col", "A1BG"),
    )
    protein_end = organ_config.get(
        "protein_end_col",
        global_config.get("protein_end_col", "ZPR1"),
    )

    required = ["eid", sex_col] + ([height_col] if height_col else []) + y_cols
    missing = [col for col in required if col not in phenotypes.columns and col not in proteomics.columns]
    if missing:
        raise ValueError(f"Missing columns for {organ_name}: {missing}")

    merged = proteomics.merge(phenotypes[["eid"] + y_cols], on="eid", how="inner")
    protein_cols = protein_columns(merged, protein_start, protein_end)
    complete_cols = y_cols + [sex_col] + ([height_col] if height_col else [])
    model_df = merged.dropna(subset=complete_cols).copy()

    x = model_df[protein_cols].apply(pd.to_numeric, errors="coerce")
    y = model_df[y_cols].apply(pd.to_numeric, errors="coerce")
    x_imp, x_imputer = fit_imputer(x)
    _, cov_scaled, cov_scaler, fill_values = prepare_covariates(model_df, sex_col, height_col)
    x_res, x_models = residualize_fit(x_imp, cov_scaled)
    y_res, _ = residualize_fit(y, cov_scaled)

    fold_metrics, phenotype_r2 = cross_validate_pls(
        model_df,
        protein_cols,
        y_cols,
        sex_col,
        height_col,
        bad_up,
        good_up,
        n_splits=n_splits,
        random_state=random_state,
    )

    pls = PLSRegression(n_components=1).fit(x_res, y_res)
    sign, anchor_corr = orient_from_anchors(pls, y, bad_up, good_up)
    scores = sign * pls.transform(x_res)[:, 0]
    modeling_scores = model_df[["eid"]].copy()
    modeling_scores[score_name] = scores

    x_all = proteomics[protein_cols].apply(pd.to_numeric, errors="coerce")
    x_all_imp = apply_imputer(x_all, x_imputer)
    cov_all_scaled = apply_covariates(proteomics, sex_col, height_col, cov_scaler, fill_values)
    x_all_res = residualize_apply(x_all_imp, cov_all_scaled, x_models)
    all_scores = proteomics[["eid"]].copy()
    all_scores[score_name] = sign * pls.transform(x_all_res)[:, 0]

    protein_weights = pd.DataFrame(
        {
            "Protein": x_res.columns,
            "PLS1_weight": sign * pls.x_weights_.ravel(),
            "PLS1_loading": sign * pls.x_loadings_.ravel(),
        }
    )
    phenotype_weights = pd.DataFrame(
        {
            "phenotype_name": y_res.columns,
            "PLS1_weight": sign * pls.y_weights_.ravel(),
            "PLS1_loading": sign * pls.y_loadings_.ravel(),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "organ_name": organ_name,
                "score_name": score_name,
                "n_modeling": len(model_df),
                "n_all_proteomics": len(all_scores),
                "cv_r2_mean": fold_metrics["r2_overall"].mean(),
                "cv_r2_std": fold_metrics["r2_overall"].std(),
                "corr_with_worse_index": anchor_corr,
            }
        ]
    )

    prefix = organ_name.replace(" ", "_")
    outputs = {
        "all_scores": output_dir / f"{prefix}_pls_all_scores.csv",
        "modeling_scores": output_dir / f"{prefix}_pls_modeling_scores.csv",
        "protein_weights": output_dir / f"{prefix}_pls_protein_weights.csv",
        "phenotype_weights": output_dir / f"{prefix}_pls_phenotype_weights.csv",
        "fold_metrics": output_dir / f"{prefix}_pls_cv_fold_metrics.csv",
        "phenotype_r2": output_dir / f"{prefix}_pls_cv_phenotype_r2.csv",
        "summary": output_dir / f"{prefix}_pls_summary.csv",
    }
    all_scores.to_csv(outputs["all_scores"], index=False)
    modeling_scores.to_csv(outputs["modeling_scores"], index=False)
    protein_weights.to_csv(outputs["protein_weights"], index=False)
    phenotype_weights.to_csv(outputs["phenotype_weights"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    phenotype_r2.to_csv(outputs["phenotype_r2"], index=False)
    summary.to_csv(outputs["summary"], index=False)

    return {
        "all_scores": all_scores,
        "modeling_scores": modeling_scores,
        "protein_weights": protein_weights,
        "phenotype_weights": phenotype_weights,
        "fold_metrics": fold_metrics,
        "phenotype_r2": phenotype_r2,
        "summary": summary,
    }


def run_pipeline(config: PLSConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_json(config.config_file)
    proteomics = read_csv_checked(Path(settings["proteomics_file"]))
    phenotypes = read_csv_checked(Path(settings["phenotype_file"]))

    all_summaries = []
    all_scores = []
    for organ_config in settings.get("organs", []):
        results = run_one_organ(
            proteomics,
            phenotypes,
            organ_config,
            settings,
            output_dir,
            n_splits=config.n_splits,
            random_state=config.random_state,
        )
        all_summaries.append(results["summary"])
        score_col = organ_config.get("score_name", f"{organ_config['organ_name']}_PLS")
        all_scores.append(results["all_scores"][["eid", score_col]])

    outputs = {}
    if all_summaries:
        outputs["summary"] = output_dir / "pls_summary_all_organs.csv"
        pd.concat(all_summaries, ignore_index=True).to_csv(outputs["summary"], index=False)
    if all_scores:
        merged_scores = all_scores[0]
        for score_df in all_scores[1:]:
            merged_scores = merged_scores.merge(score_df, on="eid", how="outer")
        outputs["scores"] = output_dir / "pls_scores_all_organs.csv"
        merged_scores.to_csv(outputs["scores"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=Path("configs/pls_organs.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pls"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        PLSConfig(
            config_file=args.config_file,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
    )
    print("PLS pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
