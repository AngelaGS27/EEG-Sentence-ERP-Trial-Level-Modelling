import argparse
import pandas as pd
import numpy as np


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std() if x.std() not in [0, np.nan] else x * 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--human-cp-col", default="human_cloze_probability")
    parser.add_argument("--llm-cp-col", default="llm_cloze_probability")
    parser.add_argument("--output", default="language_outputs/cloze_metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.human_cp_col in df.columns:
        df["human_cp"] = pd.to_numeric(df[args.human_cp_col], errors="coerce")
        df["z_human_cp"] = zscore(df["human_cp"])
        df["human_unexpectedness"] = 1 - df["human_cp"]

    if args.llm_cp_col in df.columns:
        df["llm_cp"] = pd.to_numeric(df[args.llm_cp_col], errors="coerce")
        df["z_llm_cp"] = zscore(df["llm_cp"])
        df["llm_unexpectedness"] = 1 - df["llm_cp"]

    if "human_cp" in df.columns and "llm_cp" in df.columns:
        df["cp_difference_human_minus_llm"] = df["human_cp"] - df["llm_cp"]
        df["abs_cp_disagreement"] = df["cp_difference_human_minus_llm"].abs()

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()