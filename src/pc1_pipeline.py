"""Compute organ PC scores and protein loading summaries.

This is the cleaned version of the final PCA/PC1 analysis. It does not contain
participant data; it expects local input files that are ignored by Git.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


@dataclass
class PC1Config:
    input_dir: Path
    output_dir: Path
    proteomics_file: str = "N.proteomics_preprocessed.csv"
    mapping_file: str = "1.final_merged_protein_organ.csv"
    age_gap_file: str = "resid_Age_Gaps.csv"
    participant_id_col: str = "eid"
    age_col: str = "Age_at_recruitment"
    first_protein_col: str = "A1BG"
    last_protein_col: str = "ZPR1"
    organ_col: str = "Organ"
    protein_col: str = "Protein"
    n_components: int = 3
    scale_proteins: bool = False


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def protein_columns(df: pd.DataFrame, config: PC1Config) -> list[str]:
    require_columns(df, [config.first_protein_col], "proteomics")
    columns = list(df.columns)
    start = columns.index(config.first_protein_col)
    if config.last_protein_col in df.columns:
        stop = columns.index(config.last_protein_col) + 1
        return columns[start:stop]
    return columns[start:]


def build_organ_to_proteins(
    mapping_df: pd.DataFrame,
    proteomics_df: pd.DataFrame,
    config: PC1Config,
) -> dict[str, list[str]]:
    require_columns(mapping_df, [config.organ_col, config.protein_col], "mapping")
    available = set(protein_columns(proteomics_df, config))
    usable = mapping_df[mapping_df[config.protein_col].isin(available)]
    organ_to_proteins = (
        usable.groupby(config.organ_col)[config.protein_col]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    if not organ_to_proteins:
        raise ValueError("No mapped proteins overlap with proteomics data.")
    return organ_to_proteins


def pearson_per_protein(X_df: pd.DataFrame, age_series: pd.Series) -> pd.Series:
    X = X_df.apply(pd.to_numeric, errors="coerce")
    age = pd.to_numeric(age_series, errors="coerce")
    good = age.notna()
    X = X.loc[good]
    age = age.loc[good]

    age_z = (age - age.mean()) / age.std(ddof=0)
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0, ddof=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.mul(age_z, axis=0).mean(axis=0)


def orient_pc1_by_age(
    scores: np.ndarray,
    loadings: np.ndarray,
    age_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    pc1 = pd.Series(scores[:, 0], index=age_series.index)
    corr = pc1.corr(age_series)
    if pd.notna(corr) and corr < 0:
        scores[:, 0] = -scores[:, 0]
        loadings[:, 0] = -loadings[:, 0]
        corr = -corr
    return scores, loadings, float(corr) if pd.notna(corr) else np.nan


def run_pca_block(
    df: pd.DataFrame,
    protein_cols: list[str],
    organ_name: str,
    config: PC1Config,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.Series]:
    use_cols = [config.participant_id_col, config.age_col] + protein_cols
    require_columns(df, use_cols, organ_name)
    sub = df[use_cols].copy()
    age = pd.to_numeric(sub[config.age_col], errors="coerce")
    good = age.notna()
    sub = sub.loc[good].reset_index(drop=True)
    age = age.loc[good].reset_index(drop=True)

    X = sub[protein_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(axis=0))
    X = X.fillna(0.0)
    if config.scale_proteins:
        X = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

    n_components = min(config.n_components, X.shape[1], X.shape[0])
    if n_components < 1:
        raise ValueError(f"Not enough data for PCA block: {organ_name}")

    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    scores, loadings, pc1_age_corr = orient_pc1_by_age(scores, loadings, age)

    pc_names = [f"PC{i}" for i in range(1, n_components + 1)]
    score_cols = [f"{organ_name} {pc}" for pc in pc_names]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    scores_df.insert(0, config.participant_id_col, sub[config.participant_id_col].values)

    loadings_df = pd.DataFrame(loadings, index=protein_cols, columns=pc_names)
    variance_df = loadings_df**2
    contribution_df = variance_df.divide(variance_df.sum(axis=0), axis=1)
    total_variance = variance_df.sum(axis=1)
    age_effect = pearson_per_protein(X, age).reindex(protein_cols)

    loadings_long = (
        loadings_df.reset_index()
        .melt(id_vars="index", var_name="PC", value_name="Loading")
        .rename(columns={"index": "Protein"})
    )
    variance_long = (
        variance_df.reset_index()
        .melt(id_vars="index", var_name="PC", value_name="Variance")
        .rename(columns={"index": "Protein"})
    )
    contribution_long = (
        contribution_df.reset_index()
        .melt(id_vars="index", var_name="PC", value_name="Contribution")
        .rename(columns={"index": "Protein"})
    )

    loadings_long = (
        loadings_long.merge(variance_long, on=["Protein", "PC"])
        .merge(contribution_long, on=["Protein", "PC"])
    )
    loadings_long["Organ"] = organ_name
    loadings_long["TotalVariance"] = loadings_long["Protein"].map(total_variance)
    loadings_long["AgeEffect_r"] = loadings_long["Protein"].map(age_effect)

    summary_rows = []
    for idx, pc in enumerate(pc_names):
        pc_rows = loadings_long[loadings_long["PC"] == pc]
        rho, p_value = spearmanr(
            pc_rows["Loading"],
            pc_rows["AgeEffect_r"],
            nan_policy="omit",
        )
        summary_rows.append(
            {
                "Organ": organ_name,
                "PC": pc,
                "N_proteins": int(pc_rows[["Loading", "AgeEffect_r"]].dropna().shape[0]),
                "ExplainedVariancePct": float(pca.explained_variance_ratio_[idx] * 100),
                "Spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "PC1_age_correlation": pc1_age_corr if pc == "PC1" else np.nan,
            }
        )

    return scores_df, loadings_long, summary_rows, age_effect


def add_fdr(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["p_fdr_bh"] = np.nan
    mask = summary_df["p_value"].notna()
    if mask.any():
        summary_df.loc[mask, "p_fdr_bh"] = multipletests(
            summary_df.loc[mask, "p_value"],
            method="fdr_bh",
        )[1]
    return summary_df


def run_pipeline(config: PC1Config) -> dict[str, Path]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proteomics = read_csv_checked(input_dir / config.proteomics_file)
    mapping = read_csv_checked(input_dir / config.mapping_file)
    age_gaps = read_csv_checked(input_dir / config.age_gap_file)

    require_columns(
        proteomics,
        [config.participant_id_col, config.age_col, config.first_protein_col],
        "proteomics",
    )
    require_columns(age_gaps, [config.participant_id_col], "age gaps")

    organ_to_proteins = build_organ_to_proteins(mapping, proteomics, config)
    all_proteins = protein_columns(proteomics, config)
    mapped_proteins = sorted({p for proteins in organ_to_proteins.values() for p in proteins})

    pc_scores = age_gaps[[config.participant_id_col]].copy()
    loadings = []
    summary_rows = []
    age_effect_rows = []

    blocks = [(organ, proteins) for organ, proteins in sorted(organ_to_proteins.items())]
    blocks.append(("Organismal", mapped_proteins))
    blocks.append(("All Proteins", all_proteins))

    for organ_name, proteins in blocks:
        valid = [protein for protein in proteins if protein in proteomics.columns]
        if len(valid) < 2:
            continue
        scores_df, loadings_long, rows, age_effect = run_pca_block(
            proteomics,
            valid,
            organ_name,
            config,
        )
        pc_scores = pc_scores.merge(scores_df, on=config.participant_id_col, how="left")
        loadings.append(loadings_long)
        summary_rows.extend(rows)
        tmp_age = age_effect.rename("AgeEffect_r").reset_index()
        tmp_age = tmp_age.rename(columns={"index": "Protein"})
        tmp_age.insert(0, "Organ", organ_name)
        age_effect_rows.append(tmp_age)

    age_gaps_and_pcs = age_gaps.merge(pc_scores, on=config.participant_id_col, how="left")
    loadings_df = pd.concat(loadings, ignore_index=True) if loadings else pd.DataFrame()
    summary_df = add_fdr(pd.DataFrame(summary_rows))
    age_effects_df = (
        pd.concat(age_effect_rows, ignore_index=True) if age_effect_rows else pd.DataFrame()
    )

    outputs = {
        "age_gaps_and_pcs": output_dir / "age_gaps_and_pcs.csv",
        "pc_loadings": output_dir / "pc_loadings_long.csv",
        "pc_summary": output_dir / "pc_loading_age_effect_summary.csv",
        "protein_age_effects": output_dir / "protein_age_effects_by_organ.csv",
    }
    age_gaps_and_pcs.to_csv(outputs["age_gaps_and_pcs"], index=False)
    loadings_df.to_csv(outputs["pc_loadings"], index=False)
    summary_df.to_csv(outputs["pc_summary"], index=False)
    age_effects_df.to_csv(outputs["protein_age_effects"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pc1"))
    parser.add_argument("--proteomics-file", default="N.proteomics_preprocessed.csv")
    parser.add_argument("--mapping-file", default="1.final_merged_protein_organ.csv")
    parser.add_argument("--age-gap-file", default="resid_Age_Gaps.csv")
    parser.add_argument("--scale-proteins", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PC1Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        proteomics_file=args.proteomics_file,
        mapping_file=args.mapping_file,
        age_gap_file=args.age_gap_file,
        scale_proteins=args.scale_proteins,
    )
    outputs = run_pipeline(config)
    print("PC1 pipeline complete. Wrote:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
