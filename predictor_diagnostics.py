import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output-prefix", default="language_outputs/predictor_diagnostics")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.mean())

    corr = numeric.corr()
    corr.to_csv(args.output_prefix + "_correlations.csv")

    vif_rows = []
    for i, col in enumerate(numeric.columns):
        try:
            vif = variance_inflation_factor(numeric.values, i)
        except Exception:
            vif = np.nan
        vif_rows.append({"predictor": col, "vif": vif})

    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif_df.to_csv(args.output_prefix + "_vif.csv", index=False)

    print("Saved correlation and VIF diagnostics.")


if __name__ == "__main__":
    main()