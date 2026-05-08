"""
Fit component-level mixed linear models for RP, N400, and LPC.

Pipeline position:
    export_erp_long.py
        -> creates eeg_outputs/*_erp_long.csv

    process_stimuli.py
        -> creates language_outputs/ALL_language_metrics.csv

    run_component_lmm.py
        -> merges ERP component averages with language predictors
        -> fits mixed linear models

This script is designed to test whether trial/item-level linguistic predictors explain
ERP component amplitude.

Components:
    RP   ~150-250 ms, posterior/parietal-occipital
    N400 ~300-500 ms, centro-parietal
    LPC  ~500-800 ms, centro-parietal/posterior

Model concept:
    component amplitude ~
        human cloze probability
        LLM cloze probability
        surprisal
        semantic/context-target similarity
        lexical frequency
        phonology
        syntax/sentence complexity
        affective/emotion predictors
        condition/deviation predictors if available

"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------

COMPONENT_WINDOWS = {
    "RP": {
        "time_min": 0.150,
        "time_max": 0.250,
        "roi": ["Pz", "POz", "Oz", "O1", "O2", "PO3", "PO4"],
    },
    "N400": {
        "time_min": 0.300,
        "time_max": 0.500,
        "roi": ["Cz", "CPz", "Pz", "CP1", "CP2", "P3", "P4"],
    },
    "LPC": {
        "time_min": 0.500,
        "time_max": 0.800,
        "roi": ["Pz", "CPz", "Cz", "P3", "P4", "CP1", "CP2"],
    },
}


# ---------------------------------------------------------------------
# Predictor groups
# ---------------------------------------------------------------------

CORE_PREDICTORS = [
    "human_cp",
    "llm_cp",
    "human_unexpectedness",
    "llm_unexpectedness",
    "cp_difference_human_minus_llm",
    "abs_cp_disagreement",
]

LANGUAGE_MODEL_PREDICTORS = [
    "target_surprisal_bits",
    "sentence_mean_surprisal",
    "sentence_perplexity",
    "context_target_similarity",
    "context_shift_index",
    "prior_context_strength",
]

LEXICAL_PREDICTORS = [
    "target_zipf_frequency",
    "target_raw_frequency",
    "target_n_letters",
]

PHONOLOGY_PREDICTORS = [
    "target_n_phonemes",
    "target_n_syllables",
]

SYNTAX_PREDICTORS = [
    "syntax_n_tokens",
    "syntax_mean_dependency_distance",
    "syntax_max_parse_depth",
    "syntax_mean_parse_depth",
    "syntax_n_subordinate_clauses",
]

EMOTION_PREDICTORS = [
    "target_valence",
    "target_arousal",
    "target_dominance",
    "target_is_emotional",
    "target_emotion_count",
    "target_emotion_anger",
    "target_emotion_anticipation",
    "target_emotion_disgust",
    "target_emotion_fear",
    "target_emotion_joy",
    "target_emotion_negative",
    "target_emotion_positive",
    "target_emotion_sadness",
    "target_emotion_surprise",
    "target_emotion_trust",
]

CONDITION_PREDICTORS = [
    "condition",
    "congruency",
    "congruent",
    "stimulus_type",
    "type",
    "semantic_deviation",
    "syntactic_deviation",
    "semantic_distance",
    "syntactic_distance",
]

DEFAULT_PREDICTORS = (
    CORE_PREDICTORS
    + LANGUAGE_MODEL_PREDICTORS
    + LEXICAL_PREDICTORS
    + PHONOLOGY_PREDICTORS
    + SYNTAX_PREDICTORS
    + EMOTION_PREDICTORS
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_name(name: str) -> str:
    """
    Make column names safe for statsmodels formulas.
    """
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if re.match(r"^\d", name):
        name = "x_" + name

    return name


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)

    if std == 0 or np.isnan(std):
        return values * 0

    return (values - values.mean(skipna=True)) / std


def normalise_channel_name(channel: str) -> str:
    return str(channel).strip()


def load_predictors(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)

    pred.columns = [safe_name(c) for c in pred.columns]

    if "item" not in pred.columns and "sentence_id" in pred.columns:
        pred["item"] = pred["sentence_id"]

    if "trial" not in pred.columns and "sentence_id" in pred.columns:
        pred["trial"] = pred["sentence_id"]

    if "item" in pred.columns:
        pred["item"] = pd.to_numeric(pred["item"], errors="coerce")

    if "trial" in pred.columns:
        pred["trial"] = pd.to_numeric(pred["trial"], errors="coerce")

    return pred


def component_average_chunked(
    erp_long_path: Path,
    component: str,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Compute component mean amplitude without loading the full ERP table.
    """
    spec = COMPONENT_WINDOWS[component]
    y = f"{component}_amplitude"

    required_cols = [
        "subject",
        "condition",
        "trial",
        "item",
        "channel",
        "time",
        "amplitude",
    ]

    partials = []

    log.info("Computing %s average from %s", component, erp_long_path)

    for chunk_number, chunk in enumerate(
        pd.read_csv(erp_long_path, chunksize=chunksize),
        start=1,
    ):
        missing = [c for c in required_cols if c not in chunk.columns]

        if missing:
            raise ValueError(
                f"ERP file is missing required columns: {missing}"
            )

        chunk["channel"] = chunk["channel"].map(normalise_channel_name)
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk["amplitude"] = pd.to_numeric(chunk["amplitude"], errors="coerce")

        sub = chunk[
            (chunk["time"] >= spec["time_min"])
            & (chunk["time"] <= spec["time_max"])
            & (chunk["channel"].isin(spec["roi"]))
        ].copy()

        if sub.empty:
            continue

        grouped = (
            sub.groupby(
                ["subject", "condition", "item", "trial"],
                as_index=False,
            )
            .agg(
                amplitude_sum=("amplitude", "sum"),
                amplitude_n=("amplitude", "count"),
            )
        )

        partials.append(grouped)

        if chunk_number % 10 == 0:
            log.info("Processed %d ERP chunks", chunk_number)

    if not partials:
        raise ValueError(
            f"No ERP rows found for component {component}. "
            f"Check time units and channel labels."
        )

    all_partial = pd.concat(partials, ignore_index=True)

    final = (
        all_partial.groupby(
            ["subject", "condition", "item", "trial"],
            as_index=False,
        )
        .agg(
            amplitude_sum=("amplitude_sum", "sum"),
            amplitude_n=("amplitude_n", "sum"),
        )
    )

    final[y] = final["amplitude_sum"] / final["amplitude_n"]

    final = final.drop(columns=["amplitude_sum", "amplitude_n"])

    log.info("Computed %s averages: %d rows", component, len(final))

    return final


def choose_merge_keys(comp: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
   
    keys = []

    for key in ["item", "trial", "condition"]:
        if key in comp.columns and key in pred.columns:
            keys.append(key)

    if not keys:
        raise ValueError(
            "No shared merge keys found between ERP and predictors. "
            "Expected at least item/trial."
        )

    return keys


def prepare_model_dataframe(
    comp: pd.DataFrame,
    pred: pd.DataFrame,
    component: str,
    output_dir: Path,
) -> pd.DataFrame:
    y = f"{component}_amplitude"

    merge_keys = choose_merge_keys(comp, pred)

    log.info("Merging component data with predictors on: %s", merge_keys)

    df = comp.merge(pred, on=merge_keys, how="left", suffixes=("", "_pred"))

    missing_predictor_rows = df[DEFAULT_PREDICTORS].isna().all(axis=1).sum() if any(
        p in df.columns for p in DEFAULT_PREDICTORS
    ) else len(df)

    log.info("Merged model rows: %d", len(df))
    log.info("Rows with all default predictors missing: %d", missing_predictor_rows)

    model_data_path = output_dir / f"model_data_{component}.csv"
    df.to_csv(model_data_path, index=False)

    log.info("Saved model data: %s", model_data_path)

    if y not in df.columns:
        raise ValueError(f"Outcome column missing: {y}")

    return df


def select_predictors(df: pd.DataFrame, requested: Optional[list[str]] = None) -> list[str]:
    if requested:
        candidates = requested
    else:
        candidates = DEFAULT_PREDICTORS

    selected = []

    for pred in candidates:
        safe_pred = safe_name(pred)

        if safe_pred not in df.columns:
            continue

        # Skip all-missing predictors.
        if df[safe_pred].isna().all():
            continue

        # Skip non-numeric predictors here.
        # Categorical predictors are handled separately.
        if not pd.api.types.is_numeric_dtype(df[safe_pred]):
            converted = pd.to_numeric(df[safe_pred], errors="coerce")

            if converted.notna().sum() == 0:
                continue

            df[safe_pred] = converted

        selected.append(safe_pred)

    return selected


def add_available_categorical_terms(df: pd.DataFrame) -> list[str]:
    terms = []

    for col in CONDITION_PREDICTORS:
        col = safe_name(col)

        if col not in df.columns:
            continue

        if df[col].nunique(dropna=True) < 2:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            terms.append(col)
        else:
            terms.append(f"C({col})")

    return terms


def zscore_predictors(df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    z_predictors = []

    for pred in predictors:
        z_col = f"z_{pred}"

        numeric = pd.to_numeric(df[pred], errors="coerce")

        if numeric.notna().sum() < 3:
            continue

        if numeric.std(skipna=True) == 0:
            continue

        df[z_col] = zscore(numeric)

        z_predictors.append(z_col)

    return df, z_predictors


def build_formula(
    outcome: str,
    predictors: list[str],
    categorical_terms: list[str],
    include_interactions: bool,
) -> str:
    terms = predictors + categorical_terms

    if include_interactions:
        if "z_human_cp" in predictors and "z_target_surprisal_bits" in predictors:
            terms.append("z_human_cp:z_target_surprisal_bits")

        if "z_human_cp" in predictors and "z_context_target_similarity" in predictors:
            terms.append("z_human_cp:z_context_target_similarity")

        if "z_target_surprisal_bits" in predictors and "z_context_target_similarity" in predictors:
            terms.append("z_target_surprisal_bits:z_context_target_similarity")

        if "z_target_valence" in predictors and "z_target_arousal" in predictors:
            terms.append("z_target_valence:z_target_arousal")

    if not terms:
        raise ValueError("No usable predictors available for the model.")

    formula = outcome + " ~ " + " + ".join(terms)

    return formula


def fit_mixed_model(df: pd.DataFrame, formula: str, outcome: str):
    """
    Fit mixed model.

    Random structure:
        groups = subject
        item variance component if item is available
    """
    required = [outcome, "subject"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required model column: {col}")

    model_df = df.copy()

    model_df = model_df.dropna(subset=[outcome, "subject"])

    if len(model_df) < 10:
        raise ValueError("Too few rows for model after dropping missing outcome/subject.")

    vc = {}

    if "item" in model_df.columns:
        vc["item"] = "0 + C(item)"

    log.info("Fitting formula:")
    log.info(formula)

    model = smf.mixedlm(
        formula=formula,
        data=model_df,
        groups=model_df["subject"],
        vc_formula=vc if vc else None,
    )

    try:
        result = model.fit(reml=False, method="lbfgs", maxiter=500)
    except Exception as first_error:
        log.warning("LBFGS failed: %s", first_error)
        log.warning("Trying Powell optimizer.")
        result = model.fit(reml=False, method="powell", maxiter=500)

    return result, model_df


def run_component(
    erp_long_path: Path,
    predictors_path: Path,
    component: str,
    output_dir: Path,
    chunksize: int,
    requested_predictors: Optional[list[str]],
    include_interactions: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictors(predictors_path)

    comp = component_average_chunked(
        erp_long_path=erp_long_path,
        component=component,
        chunksize=chunksize,
    )

    df = prepare_model_dataframe(
        comp=comp,
        pred=pred,
        component=component,
        output_dir=output_dir,
    )

    outcome = f"{component}_amplitude"

    selected = select_predictors(df, requested_predictors)

    df, z_selected = zscore_predictors(df, selected)

    categorical_terms = add_available_categorical_terms(df)

    formula = build_formula(
        outcome=outcome,
        predictors=z_selected,
        categorical_terms=categorical_terms,
        include_interactions=include_interactions,
    )

    result, model_df = fit_mixed_model(
        df=df,
        formula=formula,
        outcome=outcome,
    )

    summary_path = output_dir / f"model_{component}_summary.txt"
    formula_path = output_dir / f"model_{component}_formula.txt"
    coefficients_path = output_dir / f"model_{component}_coefficients.csv"
    used_data_path = output_dir / f"model_{component}_used_rows.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))

    with open(formula_path, "w", encoding="utf-8") as f:
        f.write(formula)

    coef = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
        }
    )

    if hasattr(result, "bse"):
        coef["std_error"] = result.bse.values

    if hasattr(result, "pvalues"):
        coef["p_value"] = result.pvalues.values

    coef.to_csv(coefficients_path, index=False)

    model_df.to_csv(used_data_path, index=False)

    print(result.summary())
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved formula: {formula_path}")
    print(f"Saved coefficients: {coefficients_path}")
    print(f"Saved used model rows: {used_data_path}")


def parse_predictor_list(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None

    predictors = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return predictors if predictors else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit RP/N400/LPC component-level mixed models."
    )

    parser.add_argument(
        "--erp-long",
        required=True,
        help="Path to ERP long CSV, e.g. eeg_outputs/ALL_erp_long.csv",
    )

    parser.add_argument(
        "--predictors",
        required=True,
        help="Path to language predictors CSV, e.g. language_outputs/ALL_language_metrics.csv",
    )

    parser.add_argument(
        "--component",
        choices=["RP", "N400", "LPC", "ALL"],
        required=True,
        help="ERP component to model, or ALL.",
    )

    parser.add_argument(
        "--output-dir",
        default="model_outputs",
        help="Folder where model outputs will be saved.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading ERP long CSV.",
    )

    parser.add_argument(
        "--predictor-list",
        default=None,
        help="Optional comma-separated predictor list. If omitted, uses default predictors.",
    )

    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Include selected theory-relevant interactions.",
    )

    args = parser.parse_args()

    erp_long_path = Path(args.erp_long)
    predictors_path = Path(args.predictors)
    output_dir = Path(args.output_dir)

    if not erp_long_path.exists():
        raise FileNotFoundError(f"ERP long file not found: {erp_long_path}")

    if not predictors_path.exists():
        raise FileNotFoundError(f"Predictors file not found: {predictors_path}")

    requested_predictors = parse_predictor_list(args.predictor_list)

    components = (
        ["RP", "N400", "LPC"]
        if args.component == "ALL"
        else [args.component]
    )

    for component in components:
        print("\n" + "=" * 80)
        print(f"Running component model: {component}")
        print("=" * 80)

        run_component(
            erp_long_path=erp_long_path,
            predictors_path=predictors_path,
            component=component,
            output_dir=output_dir,
            chunksize=args.chunksize,
            requested_predictors=requested_predictors,
            include_interactions=args.interactions,
        )


if __name__ == "__main__":
    main()