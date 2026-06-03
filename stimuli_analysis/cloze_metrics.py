import argparse
from typing import Optional

import numpy as np
import pandas as pd


HUMAN_CP_COLUMN_CANDIDATES = [
    "cloze_probability",
    "cloze-probability%_div",
    "human_cloze_probability",
    "human_cp",
    "cloze",
    "cp",
]

LLM_CP_COLUMN_CANDIDATES = [
    "llm_cloze_probability",
    "llm_cp",
    "model_cloze_probability",
    "gpt_cloze_probability",
    "llm_probability",
]


def normalise_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalised_map = {
        normalise_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        candidate_norm = normalise_column_name(candidate)

        if candidate_norm in normalised_map:
            return normalised_map[candidate_norm]

    return None


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)

    if std == 0 or np.isnan(std):
        return values * 0

    return (values - values.mean(skipna=True)) / std


def add_cloze_metrics(out: pd.DataFrame) -> pd.DataFrame:
    human_cp_col = find_column(out, HUMAN_CP_COLUMN_CANDIDATES)
    llm_cp_col = find_column(out, LLM_CP_COLUMN_CANDIDATES)

    if human_cp_col is not None:
        out["human_cp"] = pd.to_numeric(
            out[human_cp_col],
            errors="coerce",
        )

        out["context_constraint_human_cp"] = out["human_cp"]

        if "known_word" in out.columns and "target_word_used" in out.columns:
            out["target_matches_expected_word"] = (
                out["target_word_used"].astype(str).str.lower().str.strip()
                == out["known_word"].astype(str).str.lower().str.strip()
            )

            out["actual_target_human_cp"] = out["human_cp"].where(
                out["target_matches_expected_word"],
                np.nan,
            )
        else:
            out["target_matches_expected_word"] = np.nan
            out["actual_target_human_cp"] = out["human_cp"]

        out["z_human_cp"] = zscore(out["human_cp"])
        out["human_unexpectedness"] = 1 - out["human_cp"]

    if llm_cp_col is not None:
        out["llm_cp"] = pd.to_numeric(
            out[llm_cp_col],
            errors="coerce",
        )

    elif "target_surprisal_bits" in out.columns:
        out["llm_cp"] = 2 ** (
            -pd.to_numeric(out["target_surprisal_bits"], errors="coerce")
        )

    if "llm_cp" in out.columns:
        out["z_llm_cp"] = zscore(out["llm_cp"])
        out["llm_unexpectedness"] = 1 - out["llm_cp"]

    if "human_cp" in out.columns and "llm_cp" in out.columns:
        out["context_constraint_minus_llm_target_probability"] = (
            out["context_constraint_human_cp"] - out["llm_cp"]
        )

        out["abs_context_constraint_llm_disagreement"] = (
            out["context_constraint_minus_llm_target_probability"].abs()
        )

    if "actual_target_human_cp" in out.columns and "llm_cp" in out.columns:
        out["actual_target_cp_minus_llm"] = (
            out["actual_target_human_cp"] - out["llm_cp"]
        )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tsv")
    parser.add_argument("--output", default="language_outputs/cloze_metrics.tsv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    out = add_cloze_metrics(df)

    out.to_csv(args.output, sep="\t", index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()