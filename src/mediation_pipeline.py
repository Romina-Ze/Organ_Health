"""Bootstrap mediation analyses for PRS, organ metrics, and incident disease."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_CONFIG = Path("configs/mediation_pathways.json")


@dataclass
class MediationConfig:
    input_file: Path
    output_dir: Path
    config_file: Path = DEFAULT_CONFIG
    n_boot: int = 1000
    random_state: int = 123
    run_serial: bool = False


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return values * np.nan
    return (values - values.mean()) / std


def design_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="ignore")
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return sm.add_constant(X, has_constant="add")


def bootstrap_p_value(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(2 * min(np.mean(values <= 0), np.mean(values >= 0)))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    return {
        "estimate": float(np.mean(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p_value": bootstrap_p_value(arr),
    }


def fit_simple_once(
    df: pd.DataFrame,
    x: str,
    mediator: str,
    y: str,
    covariates: list[str],
) -> dict[str, float]:
    x_z = f"{x}__z"
    m_z = f"{mediator}__z"
    d = df[[x, mediator, y] + covariates].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        raise ValueError("No complete rows available for mediation model.")
    d[x_z] = standardize(d[x])
    d[m_z] = standardize(d[mediator])
    d[y] = pd.to_numeric(d[y], errors="coerce").astype(int)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    mediator_X = design_matrix(d, [x_z] + covariates)
    mediator_fit = sm.OLS(d[m_z], mediator_X).fit()

    outcome_X = design_matrix(d, [x_z, m_z] + covariates)
    outcome_fit = sm.Logit(d[y], outcome_X).fit(disp=0)

    a_path = mediator_fit.params[x_z]
    b_path = outcome_fit.params[m_z]
    direct = outcome_fit.params[x_z]
    indirect = a_path * b_path
    total = direct + indirect
    proportion = indirect / total if total != 0 else np.nan
    return {
        "n": float(len(d)),
        "a_path_beta": float(a_path),
        "b_path_beta": float(b_path),
        "indirect_effect": float(indirect),
        "direct_effect": float(direct),
        "total_effect": float(total),
        "proportion_mediated": float(proportion) if pd.notna(proportion) else np.nan,
    }


def run_simple_mediation(
    df: pd.DataFrame,
    pathway: dict,
    covariates: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    x = pathway["x"]
    mediator = pathway["mediator"]
    y = pathway["y"]
    label = pathway.get("label", f"{x} -> {mediator} -> {y}")
    needed = [x, mediator, y] + covariates
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {label}: {missing}")

    base = fit_simple_once(df, x, mediator, y, covariates)
    model_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    rng = np.random.default_rng(seed)
    boot = {
        "indirect_effect": [],
        "direct_effect": [],
        "total_effect": [],
        "proportion_mediated": [],
    }

    for _ in range(n_boot):
        sample_idx = rng.choice(model_df.index, size=len(model_df), replace=True)
        boot_df = model_df.loc[sample_idx].copy()
        try:
            fit = fit_simple_once(boot_df, x, mediator, y, covariates)
        except Exception:
            continue
        for key in boot:
            boot[key].append(fit[key])

    indirect = summarize(boot["indirect_effect"])
    direct = summarize(boot["direct_effect"])
    total = summarize(boot["total_effect"])
    proportion = summarize(boot["proportion_mediated"])
    return {
        "pathway": label,
        "x": x,
        "mediator": mediator,
        "y": y,
        "n": int(base["n"]),
        "successful_bootstraps": len(boot["indirect_effect"]),
        "a_path_beta": base["a_path_beta"],
        "b_path_beta": base["b_path_beta"],
        "indirect_effect": base["indirect_effect"],
        "indirect_boot_estimate": indirect["estimate"],
        "indirect_CI_low": indirect["ci_low"],
        "indirect_CI_high": indirect["ci_high"],
        "indirect_p": indirect["p_value"],
        "direct_effect": base["direct_effect"],
        "direct_boot_estimate": direct["estimate"],
        "direct_CI_low": direct["ci_low"],
        "direct_CI_high": direct["ci_high"],
        "direct_p": direct["p_value"],
        "total_effect": base["total_effect"],
        "total_boot_estimate": total["estimate"],
        "total_CI_low": total["ci_low"],
        "total_CI_high": total["ci_high"],
        "total_p": total["p_value"],
        "proportion_mediated": base["proportion_mediated"],
        "proportion_boot_estimate": proportion["estimate"],
        "proportion_CI_low": proportion["ci_low"],
        "proportion_CI_high": proportion["ci_high"],
        "proportion_p": proportion["p_value"],
    }


def fit_serial_once(
    df: pd.DataFrame,
    pathway: dict,
    covariates: list[str],
) -> dict[str, float]:
    x = pathway["x"]
    m1 = pathway["m1"]
    m2 = pathway["m2"]
    y = pathway["y"]
    needed = [x, m1, m2, y] + covariates
    d = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        raise ValueError("No complete rows available for serial mediation.")
    for col in [x, m1, m2]:
        d[col] = standardize(d[col])
    d[y] = pd.to_numeric(d[y], errors="coerce").astype(int)

    m1_fit = sm.OLS(d[m1], design_matrix(d, [x] + covariates)).fit()
    m2_fit = sm.OLS(d[m2], design_matrix(d, [x, m1] + covariates)).fit()
    y_fit = sm.Logit(d[y], design_matrix(d, [x, m1, m2] + covariates)).fit(disp=0)

    a1 = m1_fit.params[x]
    a2 = m2_fit.params[x]
    d21 = m2_fit.params[m1]
    b1 = y_fit.params[m1]
    b2 = y_fit.params[m2]
    direct = y_fit.params[x]
    return {
        "n": float(len(d)),
        "a1_X_to_M1": float(a1),
        "a2_X_to_M2": float(a2),
        "d21_M1_to_M2": float(d21),
        "b1_M1_to_Y": float(b1),
        "b2_M2_to_Y": float(b2),
        "direct_c_prime_X_to_Y": float(direct),
        "indirect_X_M1_Y": float(a1 * b1),
        "indirect_X_M2_Y": float(a2 * b2),
        "serial_indirect_X_M1_M2_Y": float(a1 * d21 * b2),
        "total_indirect": float((a1 * b1) + (a2 * b2) + (a1 * d21 * b2)),
        "outcome_model_aic": float(y_fit.aic),
    }


def run_serial_mediation(
    df: pd.DataFrame,
    pathway: dict,
    covariates: list[str],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    label = pathway.get("label", pathway.get("pathway", "serial_mediation"))
    needed = [pathway["x"], pathway["m1"], pathway["m2"], pathway["y"]] + covariates
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {label}: {missing}")

    base = fit_serial_once(df, pathway, covariates)
    model_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    rng = np.random.default_rng(seed)
    boot_rows = []
    effects = [
        "indirect_X_M1_Y",
        "indirect_X_M2_Y",
        "serial_indirect_X_M1_M2_Y",
        "total_indirect",
        "direct_c_prime_X_to_Y",
    ]

    for _ in range(n_boot):
        sample_idx = rng.choice(model_df.index, size=len(model_df), replace=True)
        boot_df = model_df.loc[sample_idx].copy()
        try:
            fit = fit_serial_once(boot_df, pathway, covariates)
        except Exception:
            continue
        boot_rows.append({effect: fit[effect] for effect in effects})

    boot_df = pd.DataFrame(boot_rows)
    summary = {
        "pathway": label,
        "x": pathway["x"],
        "m1": pathway["m1"],
        "m2": pathway["m2"],
        "y": pathway["y"],
        "n": int(base["n"]),
        "successful_bootstraps": len(boot_df),
        **{key: value for key, value in base.items() if key != "n"},
    }
    for effect in effects:
        stats = summarize(boot_df[effect].tolist() if effect in boot_df else [])
        summary[f"{effect}_boot_estimate"] = stats["estimate"]
        summary[f"{effect}_CI_low"] = stats["ci_low"]
        summary[f"{effect}_CI_high"] = stats["ci_high"]
        summary[f"{effect}_p"] = stats["p_value"]
    boot_df.insert(0, "pathway", label)
    return summary, boot_df


def run_pipeline(config: MediationConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_checked(config.input_file)
    med_config = load_config(config.config_file)
    covariates = med_config.get("covariates", [])

    simple_rows = []
    for pathway in med_config.get("simple_pathways", []):
        simple_rows.append(
            run_simple_mediation(
                df,
                pathway,
                covariates,
                n_boot=config.n_boot,
                seed=config.random_state,
            )
        )

    outputs = {}
    if simple_rows:
        simple_df = pd.DataFrame(simple_rows)
        outputs["simple_mediation"] = output_dir / "simple_mediation_results.csv"
        simple_df.to_csv(outputs["simple_mediation"], index=False)

    if config.run_serial:
        serial_rows = []
        serial_boot = []
        for pathway in med_config.get("serial_pathways", []):
            summary, boot = run_serial_mediation(
                df,
                pathway,
                covariates,
                n_boot=config.n_boot,
                seed=config.random_state,
            )
            serial_rows.append(summary)
            serial_boot.append(boot)
        if serial_rows:
            outputs["serial_mediation"] = output_dir / "serial_mediation_results.csv"
            pd.DataFrame(serial_rows).to_csv(outputs["serial_mediation"], index=False)
        if serial_boot:
            outputs["serial_bootstrap"] = output_dir / "serial_mediation_bootstrap_draws.csv"
            pd.concat(serial_boot, ignore_index=True).to_csv(
                outputs["serial_bootstrap"],
                index=False,
            )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mediation"))
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--run-serial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        MediationConfig(
            input_file=args.input_file,
            output_dir=args.output_dir,
            config_file=args.config_file,
            n_boot=args.n_boot,
            random_state=args.random_state,
            run_serial=args.run_serial,
        )
    )
    print("Mediation pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
