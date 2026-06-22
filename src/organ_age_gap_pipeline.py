"""Train XGBoost organ age gap models without committing protected data.

This module is a cleaned, scriptable version of the exploratory notebook. It
expects controlled-access data to live outside Git tracking, usually under
``data/raw`` on your local machine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBRegressor


QUICK_PARAM_GRID = {
    "max_depth": [5],
    "n_estimators": [500],
    "learning_rate": [0.1],
}

FULL_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
}


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    mapping_file: str = "1.final_merged_protein_organ.csv"
    merged_file: str = "N.merged_proteomics_mri.csv"
    proteomics_file: str = "N.proteomics_preprocessed.csv"
    participant_id_col: str = "eid"
    age_col: str = "Age_at_recruitment"
    sex_col: str = "Sex"
    imaging_col: str = "Imaging"
    first_protein_col: str = "A1BG"
    last_protein_col: str = "ZPR1"
    organ_col: str = "Organ"
    protein_col: str = "Protein"
    train_indicator_value: int = 0
    test_indicator_value: int = 1
    validation_fraction: float = 0.2
    random_state: int = 42


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def load_inputs(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_df = read_csv_checked(config.input_dir / config.mapping_file)
    main_data = read_csv_checked(config.input_dir / config.merged_file)
    require_columns(
        main_data,
        [
            config.participant_id_col,
            config.age_col,
            config.sex_col,
            config.imaging_col,
            config.first_protein_col,
        ],
        "main data",
    )
    require_columns(
        mapping_df,
        [config.organ_col, config.protein_col],
        "protein-organ mapping",
    )
    return mapping_df, main_data


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def protein_columns(main_data: pd.DataFrame, config: PipelineConfig) -> list[str]:
    columns = list(main_data.columns)
    start = columns.index(config.first_protein_col)
    if config.last_protein_col in main_data.columns:
        stop = columns.index(config.last_protein_col) + 1
        return columns[start:stop]
    return columns[start:]


def build_organ_to_proteins(
    mapping_df: pd.DataFrame,
    main_data: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, list[str]]:
    available_proteins = set(protein_columns(main_data, config))
    usable = mapping_df[mapping_df[config.protein_col].isin(available_proteins)]
    organ_to_proteins = (
        usable.groupby(config.organ_col)[config.protein_col]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    if not organ_to_proteins:
        raise ValueError("No mapped proteins overlap with the main dataset.")
    return organ_to_proteins


def split_train_validation_test(
    main_data: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df = main_data[
        main_data[config.imaging_col] == config.train_indicator_value
    ].copy()
    test_df = main_data[
        main_data[config.imaging_col] == config.test_indicator_value
    ].copy()
    if train_val_df.empty or test_df.empty:
        raise ValueError("Train/validation or test split is empty. Check Imaging coding.")
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.validation_fraction,
        random_state=config.random_state,
    )
    return train_df, val_df, test_df


def fit_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list[float | int]],
    config: PipelineConfig,
    cv: int = 5,
    n_jobs: int = -1,
) -> tuple[XGBRegressor, dict[str, float | int]]:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=1,
    )
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=n_jobs,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def safe_pearsonr(y_true: pd.Series, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    valid = np.isfinite(true) & np.isfinite(pred)
    if valid.sum() < 2 or np.unique(true[valid]).size < 2 or np.unique(pred[valid]).size < 2:
        return float("nan")
    return float(pearsonr(true[valid], pred[valid])[0])


def evaluate_model(
    model: XGBRegressor,
    best_params: dict[str, float | int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: PipelineConfig,
) -> dict[str, float]:
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    cv_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=1,
    )
    cv_scores = cross_val_score(
        cv_model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    age_gap = y_test_pred - y_test
    return {
        "Train R2 CV": float(np.mean(cv_scores)),
        "Validation R2": float(r2_score(y_val, y_val_pred)),
        "Test R2": float(r2_score(y_test, y_test_pred)),
        "Train r": safe_pearsonr(y_train, y_train_pred),
        "Validation r": safe_pearsonr(y_val, y_val_pred),
        "Test r": safe_pearsonr(y_test, y_test_pred),
        "Train MAE": float(mean_absolute_error(y_train, y_train_pred)),
        "Validation MAE": float(mean_absolute_error(y_val, y_val_pred)),
        "Test MAE": float(mean_absolute_error(y_test, y_test_pred)),
        "Train RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "Validation RMSE": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "Test RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "Mean Gap Test": float(age_gap.mean()),
        "STD Gap Test": float(age_gap.std()),
    }


def translate_xgb_feature_key(key: str, feature_names: list[str]) -> str:
    if key in feature_names:
        return key
    if isinstance(key, str) and key.startswith("f") and key[1:].isdigit():
        index = int(key[1:])
        if 0 <= index < len(feature_names):
            return feature_names[index]
    return key


def feature_importance_table(model: XGBRegressor, feature_cols: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    importance_types = ["gain", "weight", "cover", "total_gain", "total_cover"]
    table = pd.DataFrame({"Feature": feature_cols})
    for importance_type in importance_types:
        raw_scores = booster.get_score(importance_type=importance_type)
        mapped_scores = {
            translate_xgb_feature_key(key, feature_cols): value
            for key, value in raw_scores.items()
        }
        table[importance_type] = table["Feature"].map(mapped_scores).fillna(0.0)
    table["Rank gain"] = table["gain"].rank(ascending=False, method="min")
    total_gain = table["gain"].sum()
    table["Gain normalized"] = table["gain"] / total_gain if total_gain else 0.0
    return table.sort_values("Rank gain")


def run_named_model(
    main_data: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    config: PipelineConfig,
    param_grid: dict[str, list[float | int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = split_train_validation_test(main_data, config)
    feature_cols = feature_cols + [config.sex_col]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[config.age_col]
    y_val = val_df[config.age_col]
    y_test = test_df[config.age_col]

    model, best_params = fit_xgb_model(X_train, y_train, param_grid, config)
    metrics = evaluate_model(
        model,
        best_params,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        config,
    )

    X_all = main_data[feature_cols]
    bioage_col = f"{model_name}_BioAge"
    agegap_col = f"{model_name}_AgeGap"
    predictions = main_data[[config.participant_id_col, config.age_col]].copy()
    predictions[bioage_col] = model.predict(X_all)
    predictions[agegap_col] = predictions[bioage_col] - predictions[config.age_col]

    summary = {
        "Model": model_name,
        "Number of features": len(feature_cols),
        "Best Params": json.dumps(best_params, sort_keys=True),
        **metrics,
    }
    importance = feature_importance_table(model, feature_cols)
    importance.insert(0, "Model", model_name)
    return predictions, pd.DataFrame([summary]), importance


def run_organ_specific_models(
    mapping_df: pd.DataFrame,
    main_data: pd.DataFrame,
    config: PipelineConfig,
    param_grid: dict[str, list[float | int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    organ_to_proteins = build_organ_to_proteins(mapping_df, main_data, config)
    train_df, val_df, test_df = split_train_validation_test(main_data, config)

    predictions = main_data[[config.participant_id_col, config.age_col]].copy()
    summaries: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []

    for organ, proteins in sorted(organ_to_proteins.items()):
        feature_cols = proteins + [config.sex_col]
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[config.age_col]
        y_val = val_df[config.age_col]
        y_test = test_df[config.age_col]

        model, best_params = fit_xgb_model(X_train, y_train, param_grid, config)
        metrics = evaluate_model(
            model,
            best_params,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            config,
        )

        bioage_col = f"{organ}_BioAge"
        agegap_col = f"{organ}_AgeGap"
        predictions[bioage_col] = model.predict(main_data[feature_cols])
        predictions[agegap_col] = predictions[bioage_col] - predictions[config.age_col]

        summaries.append(
            {
                "Model": "Organ specific",
                "Organ": organ,
                "Number of proteins": len(proteins),
                "Best Params": json.dumps(best_params, sort_keys=True),
                **metrics,
            }
        )
        importance = feature_importance_table(model, feature_cols)
        importance.insert(0, "Organ", organ)
        importances.append(importance)

    return (
        predictions,
        pd.DataFrame(summaries),
        pd.concat(importances, ignore_index=True),
    )


def residualize_age_gaps(age_gap_df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    corrected = age_gap_df.copy()
    chronological_age = corrected[age_col].to_numpy().reshape(-1, 1)
    age_squared = (corrected[age_col] ** 2).to_numpy().reshape(-1, 1)
    age_squared_orthogonal = age_squared - LinearRegression().fit(
        chronological_age,
        age_squared,
    ).predict(chronological_age)

    for gap_col in [col for col in corrected.columns if col.endswith("_AgeGap")]:
        raw_gap = corrected[gap_col].to_numpy().reshape(-1, 1)
        mask = np.isfinite(chronological_age).ravel() & np.isfinite(raw_gap).ravel()
        residuals = np.full(len(corrected), np.nan, dtype=float)
        if mask.sum() >= 3:
            X_quad = np.hstack(
                [chronological_age[mask], age_squared_orthogonal[mask]]
            )
            model = LinearRegression().fit(X_quad, raw_gap[mask])
            residuals[mask] = (raw_gap[mask] - model.predict(X_quad)).ravel()
        corrected[f"{gap_col}_resid"] = residuals

    return corrected


def demographic_table(main_data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    train_df, val_df, test_df = split_train_validation_test(main_data, config)
    cohorts = {
        "Training": train_df,
        "Validation": val_df,
        "Test": test_df,
        "Total": main_data,
    }

    rows = []
    for cohort, df in cohorts.items():
        age = df[config.age_col]
        sex_counts = df[config.sex_col].value_counts().sort_index()
        rows.append(
            {
                "Cohort": cohort,
                "N": len(df),
                "Age mean": float(age.mean()),
                "Age SD": float(age.std()),
                "Age min": float(age.min()),
                "Age max": float(age.max()),
                "Female N": int(sex_counts.get(0, 0)),
                "Male N": int(sex_counts.get(1, 0)),
                "Female percent": 100 * sex_counts.get(0, 0) / len(df),
                "Male percent": 100 * sex_counts.get(1, 0) / len(df),
            }
        )
    return pd.DataFrame(rows)


def protein_age_correlations(
    config: PipelineConfig,
    main_data: pd.DataFrame,
) -> pd.DataFrame | None:
    proteomics_path = config.input_dir / config.proteomics_file
    if not proteomics_path.exists():
        return None
    proteomics = read_csv_checked(proteomics_path)
    require_columns(proteomics, [config.age_col, config.first_protein_col], "proteomics")
    proteins = protein_columns(proteomics, config)
    rows = []
    for protein in proteins:
        values = proteomics[protein]
        mask = values.notna() & proteomics[config.age_col].notna()
        if mask.sum() < 2:
            rows.append({"Protein": protein, "Corr with age": np.nan, "p value": np.nan})
            continue
        r, p = pearsonr(values[mask], proteomics.loc[mask, config.age_col])
        rows.append({"Protein": protein, "Corr with age": r, "p value": p})
    return pd.DataFrame(rows)


def run_pipeline(
    config: PipelineConfig,
    param_grid: dict[str, list[float | int]] | None = None,
) -> dict[str, Path]:
    config.input_dir = Path(config.input_dir)
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    param_grid = param_grid or QUICK_PARAM_GRID

    mapping_df, main_data = load_inputs(config)

    organ_predictions, organ_summary, organ_importance = run_organ_specific_models(
        mapping_df,
        main_data,
        config,
        param_grid,
    )
    organ_to_proteins = build_organ_to_proteins(mapping_df, main_data, config)
    all_organ_proteins = sorted(
        {protein for proteins in organ_to_proteins.values() for protein in proteins}
    )
    all_proteins = protein_columns(main_data, config)

    organismal_predictions, organismal_summary, organismal_importance = run_named_model(
        main_data,
        all_organ_proteins,
        "Organismal",
        config,
        param_grid,
    )
    all_predictions, all_summary, all_importance = run_named_model(
        main_data,
        all_proteins,
        "All_proteins",
        config,
        param_grid,
    )

    join_cols = [config.participant_id_col, config.age_col]
    combined_predictions = (
        organ_predictions.merge(organismal_predictions, on=join_cols, how="inner")
        .merge(all_predictions, on=join_cols, how="inner")
    )
    residualized = residualize_age_gaps(combined_predictions, config.age_col)
    model_summary = pd.concat(
        [organ_summary, organismal_summary, all_summary],
        ignore_index=True,
        sort=False,
    )
    feature_importances = pd.concat(
        [organ_importance, organismal_importance, all_importance],
        ignore_index=True,
        sort=False,
    )

    outputs = {
        "organ_predictions": config.output_dir / "organ_specific_bioage_predictions.csv",
        "combined_predictions": config.output_dir / "age_gaps_and_bioage.csv",
        "residualized_predictions": config.output_dir / "residualized_age_gaps.csv",
        "model_summary": config.output_dir / "xgboost_model_summary.csv",
        "feature_importances": config.output_dir / "xgb_feature_importances.csv",
        "demographics": config.output_dir / "demographic_table.csv",
    }

    organ_predictions.to_csv(outputs["organ_predictions"], index=False)
    combined_predictions.to_csv(outputs["combined_predictions"], index=False)
    residualized.to_csv(outputs["residualized_predictions"], index=False)
    model_summary.to_csv(outputs["model_summary"], index=False)
    feature_importances.to_csv(outputs["feature_importances"], index=False)
    demographic_table(main_data, config).to_csv(outputs["demographics"], index=False)

    correlations = protein_age_correlations(config, main_data)
    if correlations is not None:
        outputs["protein_age_correlations"] = (
            config.output_dir / "protein_age_correlations.csv"
        )
        correlations.to_csv(outputs["protein_age_correlations"], index=False)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--mapping-file", default="1.final_merged_protein_organ.csv")
    parser.add_argument("--merged-file", default="N.merged_proteomics_mri.csv")
    parser.add_argument("--proteomics-file", default="N.proteomics_preprocessed.csv")
    parser.add_argument("--full-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mapping_file=args.mapping_file,
        merged_file=args.merged_file,
        proteomics_file=args.proteomics_file,
    )
    param_grid = FULL_PARAM_GRID if args.full_grid else QUICK_PARAM_GRID
    outputs = run_pipeline(config, param_grid=param_grid)
    print("Pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
