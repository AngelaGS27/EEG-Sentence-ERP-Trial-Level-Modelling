import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tsv")
    parser.add_argument("--output-prefix", default="language_outputs/predictor_diagnostics")
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")

    diagnostic_predictors = [
        "human_cp",
        "llm_cp",
        "target_surprisal_bits",
        "context_target_similarity",
        "target_zipf_frequency",
        "target_n_letters",
        "target_n_phonemes",
        "target_n_syllables",
        "syntax_mean_dependency_distance",
        "syntax_max_parse_depth",
        "syntax_n_subordinate_clauses",
        "target_valence",
        "target_arousal",
    ]

    available = [
        col for col in diagnostic_predictors
        if col in df.columns
    ]

    if len(available) < 2:
        print("Not enough diagnostic predictors available.")
        print("Available columns were:")
        print(list(df.columns))
        return

    numeric = df[available].copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")

    # Remove columns with no variance, because they break VIF/correlation.
    numeric = numeric.loc[:, numeric.std(skipna=True) != 0]

    if numeric.shape[1] < 2:
        print("Not enough non-constant numeric predictors for diagnostics.")
        return

    numeric = numeric.fillna(numeric.mean())

    numeric = (numeric - numeric.mean()) / numeric.std()
    numeric = sm.add_constant(numeric)

    corr = numeric.corr()
    corr.to_csv(args.output_prefix + "_correlations.tsv", sep="\t")

    vif_rows = []

    for i, col in enumerate(numeric.columns):
        if col == "const":
            continue

        try:
            vif = variance_inflation_factor(numeric.values, i)
        except Exception:
            vif = np.nan

        vif_rows.append(
            {
                "predictor": col,
                "vif": vif,
            }
        )

    vif_df = pd.DataFrame(vif_rows).sort_values(
        "vif",
        ascending=False,
    )

    vif_df.to_csv(args.output_prefix + "_vif.tsv", sep="\t", index=False)

    print("Saved correlation and VIF diagnostics.")
    print("Predictors used:")
    print(list(numeric.columns))