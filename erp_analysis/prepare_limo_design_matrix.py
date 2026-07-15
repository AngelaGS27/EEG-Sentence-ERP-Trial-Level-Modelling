"""
Prepare a subject-specific design matrix for LIMO.

The script:

1. Loads the subject events.tsv.
2. Extracts a stable stimulus identifier as stim_key.
3. Optionally keeps only trials listed in a surviving-trials file.
4. Matches each trial to the language predictors using stim_key.
5. Validates duplicates, unmatched trials, and trial order.
6. Saves a subject-specific design matrix.

This script prepares the input table used by the LIMO analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STIMULUS_COLUMNS = [
    "stim_key",
    "stim_file",
    "stimulus",
    "sentence_id",
    "item",
    "trial",
]


def normalise_stim_key(series: pd.Series) -> pd.Series:
    """
    Normalise stimulus identifiers before matching.
    """
    return (
        series
        .astype("string")
        .str.strip()
    )

def normalise_stim_file(
    series: pd.Series,
) -> pd.Series:
    """
    Normalise stimulus filenames before matching.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .str.rsplit(
            "/",
            n=1,
        )
        .str[-1]
    )

def load_stimulus_lookup(
    stimulus_parameters_path: Path,
) -> pd.DataFrame:
    """
    Load the dataset stimulus table and create a validated
    stim_file -> stim_key lookup.

    Expected file:
        N400Stimset_stimuli_parameters.tsv
    """

    stimuli = pd.read_csv(
        stimulus_parameters_path,
        sep="\t",
    )

    required_columns = [
        "stim_file",
        "stim_key",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in stimuli.columns
    ]

    if missing_columns:
        raise ValueError(
            "Stimulus parameters table is missing "
            f"required columns: {missing_columns}"
        )

    stimuli = stimuli.copy()

    stimuli["stim_file"] = normalise_stim_file(
        stimuli["stim_file"]
    )

    stimuli["stim_key"] = normalise_stim_key(
        stimuli["stim_key"]
    )

    if stimuli["stim_file"].isna().any():
        raise ValueError(
            "Stimulus parameters table contains "
            "missing stim_file values."
        )

    if stimuli["stim_key"].isna().any():
        raise ValueError(
            "Stimulus parameters table contains "
            "missing stim_key values."
        )

    conflicting = (
        stimuli
        .groupby(
            "stim_file"
        )["stim_key"]
        .nunique()
    )

    conflicting = conflicting[
        conflicting > 1
    ]

    if not conflicting.empty:
        raise ValueError(
            "Some stim_file values map to more "
            "than one stim_key. Examples: "
            f"{conflicting.index[:10].tolist()}"
        )

    lookup = (
        stimuli[
            [
                "stim_file",
                "stim_key",
            ]
        ]
        .drop_duplicates(
            subset="stim_file",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Loaded {len(lookup)} stimulus mappings "
        f"from {stimulus_parameters_path}"
    )

    return lookup

def load_language_predictors(predictors_path: Path) -> pd.DataFrame:
    """
    Load the z-scored language predictor table.

    Expected input:
        ALL_language_metrics_GLM.tsv
    """
    predictors = pd.read_csv(
        predictors_path,
        sep=None,
        engine="python",
    )

    if "stim_key" not in predictors.columns:
        raise ValueError(
            f"Predictor table does not contain 'stim_key': {predictors_path}"
        )

    predictors["stim_key"] = normalise_stim_key(
        predictors["stim_key"]
    )

    if predictors["stim_key"].isna().any():
        n_missing = predictors["stim_key"].isna().sum()

        raise ValueError(
            f"Predictor table contains {n_missing} missing stim_key values."
        )

    duplicated = predictors["stim_key"].duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            predictors.loc[duplicated, "stim_key"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Predictor table contains duplicate stim_key values. "
            f"Examples: {examples}"
        )

    return predictors

def load_trial_lookup(
    trial_lookup_path: Path,
) -> pd.DataFrame:
    """
    Load the retained-trial lookup created by export_erp_long.py.

    The table contains only trials that survived EEG rejection,
    along with their matched stimulus identifiers.
    """

    lookup = pd.read_csv(
        trial_lookup_path,
        sep="\t",
    )

    required_columns = [
        "subject",
        "condition",
        "condition_label",
        "retained_trial",
        "urevent_index",
        "original_event_row",
        "trial_type",
        "stim_file",
        "stim_key",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in lookup.columns
    ]

    if missing_columns:
        raise ValueError(
            "Trial lookup is missing required "
            f"columns: {missing_columns}"
        )

    lookup = lookup.copy()

    lookup["stim_key"] = normalise_stim_key(
        lookup["stim_key"]
    )

    lookup["stim_file"] = normalise_stim_file(
        lookup["stim_file"]
    )

    if lookup["stim_key"].isna().any():
        raise ValueError(
            "Trial lookup contains missing stim_key values."
        )

    duplicated = lookup.duplicated(
        subset=[
            "condition",
            "retained_trial",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "Trial lookup contains duplicate retained trials."
        )

    lookup = lookup.sort_values(
        [
            "condition",
            "retained_trial",
        ]
    ).reset_index(
        drop=True
    )

    lookup["eeg_trial"] = range(
        1,
        len(lookup) + 1,
    )

    return lookup

def find_stimulus_column(events: pd.DataFrame) -> str:
    stim_col = next(
        (
            col
            for col in STIMULUS_COLUMNS
            if col in events.columns
        ),
        None,
    )

    if stim_col is None:
        raise ValueError(
            "No stimulus identifier was found in events.tsv. "
            f"Checked: {STIMULUS_COLUMNS}"
        )

    return stim_col


def load_subject_events(
    events_path: Path,
    stimulus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load one subject's events.tsv.

    Only NPC and NPI sentence trials are retained.
    stim_key is attached through stim_file.
    """

    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    if events.empty:
        raise ValueError(
            f"Events table is empty: {events_path}"
        )

    required_columns = [
        "trial_type",
        "stim_file",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{events_path} is missing required "
            f"columns: {missing_columns}"
        )

    events = events.copy()

    events["trial_type"] = (
        events["trial_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    events = events[
        events["trial_type"].isin(
            [
                "NPC",
                "NPI",
            ]
        )
    ].copy()

    if events.empty:
        raise ValueError(
            f"No NPC or NPI sentence trials found in {events_path}"
        )

    events["stim_file"] = normalise_stim_file(
        events["stim_file"]
    )

    events = events.merge(
        stimulus_lookup,
        on="stim_file",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    unmatched = (
        events["_merge"]
        != "both"
    )

    if unmatched.any():
        examples = (
            events.loc[
                unmatched,
                "stim_file",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{unmatched.sum()} events could not be "
            "matched to the stimulus parameters table. "
            f"Examples: {examples}"
        )

    events = events.drop(
        columns="_merge"
    )

    events["original_event_row"] = range(
        1,
        len(events) + 1,
    )

    events["subject_trial"] = range(
        1,
        len(events) + 1,
    )

    condition_map = {
        "NPC": 1,
        "NPI": 2,
    }

    events["condition"] = (
        events["trial_type"]
        .map(condition_map)
        .astype(int)
    )

    events["condition_trial"] = (
        events
        .groupby(
            "condition",
            sort=False,
        )
        .cumcount()
        + 1
    )

    print(
        f"Loaded {len(events)} experimental trials: "
        f"{(events['trial_type'] == 'NPC').sum()} NPC and "
        f"{(events['trial_type'] == 'NPI').sum()} NPI"
    )

    return events.reset_index(
        drop=True
    )


def load_surviving_trials(
    surviving_trials_path: Path,
) -> pd.DataFrame:
    """
    Load a table specifying which trials remain in the cleaned EEG data.

    Accepted trial-index columns:
        original_event_row
        subject_trial
        trial
        epoch
        epoch_index
    """
    surviving = pd.read_csv(
        surviving_trials_path,
        sep=None,
        engine="python",
    )

    candidate_columns = [
        "original_event_row",
        "subject_trial",
        "trial",
        "epoch",
        "epoch_index",
    ]

    trial_col = next(
        (
            col
            for col in candidate_columns
            if col in surviving.columns
        ),
        None,
    )

    if trial_col is None:
        raise ValueError(
            "Surviving-trials table does not contain a recognised "
            f"trial-index column. Checked: {candidate_columns}"
        )

    surviving = surviving.copy()

    surviving["original_event_row"] = pd.to_numeric(
        surviving[trial_col],
        errors="raise",
    ).astype(int)

    if surviving["original_event_row"].duplicated().any():
        raise ValueError(
            "Surviving-trials table contains duplicate trial indices."
        )

    return surviving[["original_event_row"]]


def restrict_to_surviving_trials(
    events: pd.DataFrame,
    surviving_trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict events to trials that remain in the cleaned EEG file.

    The order of surviving_trials is preserved.
    """
    surviving_trials = surviving_trials.copy()

    surviving_trials["eeg_trial"] = range(
        1,
        len(surviving_trials) + 1,
    )

    filtered = surviving_trials.merge(
        events,
        on="original_event_row",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if filtered["stim_key"].isna().any():
        missing_rows = filtered.loc[
            filtered["stim_key"].isna(),
            "original_event_row",
        ].tolist()

        raise ValueError(
            "Some surviving trial indices were not found in events.tsv: "
            f"{missing_rows[:10]}"
        )

    filtered = filtered.sort_values(
        "eeg_trial"
    ).reset_index(drop=True)

    return filtered


def build_subject_design_matrix(
    trial_lookup: pd.DataFrame,
    predictors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match each retained EEG trial to language predictors using stim_key.

    The output contains one row per surviving ERP trial.
    """

    design = trial_lookup.merge(
        predictors,
        on="stim_key",
        how="left",
        validate="many_to_one",
        sort=False,
        suffixes=(
            "",
            "_predictor",
        ),
        indicator=True,
    )

    unmatched = (
        design["_merge"] != "both"
    )

    if unmatched.any():
        examples = (
            design.loc[
                unmatched,
                [
                    "stim_key",
                    "stim_file",
                ],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict(
                "records"
            )
        )

        raise ValueError(
            f"{unmatched.sum()} retained EEG trials have "
            "no matched language predictors. "
            f"Examples: {examples}"
        )

    design = design.drop(
        columns="_merge"
    )

    predictor_columns = [
        column
        for column in predictors.columns
        if column != "stim_key"
    ]

    completely_missing = (
        design[
            predictor_columns
        ]
        .isna()
        .all(
            axis=1
        )
    )

    if completely_missing.any():
        raise ValueError(
            f"{completely_missing.sum()} retained trials "
            "contain no predictor values."
        )

    return design


def validate_design_matrix(
    design: pd.DataFrame,
    predictors: pd.DataFrame,
) -> None:
    """
    Run final checks before saving the design matrix.
    """
    if design.empty:
        raise ValueError("The final design matrix is empty.")

    if "stim_key" not in design.columns:
        raise ValueError(
            "The final design matrix does not contain stim_key."
        )

    predictor_columns = [
        col
        for col in predictors.columns
        if col != "stim_key"
    ]

    non_numeric = []

    for col in predictor_columns:
        converted = pd.to_numeric(
            design[col],
            errors="coerce",
        )

        if converted.notna().sum() == 0:
            non_numeric.append(col)

    if non_numeric:
        raise ValueError(
            "These predictor columns are not numeric: "
            + ", ".join(non_numeric)
        )

    print()
    print("Design-matrix validation")
    print("------------------------")
    print(f"Rows: {len(design)}")
    print(f"Unique stimuli: {design['stim_key'].nunique()}")
    print(f"Predictor columns: {len(predictor_columns)}")
    print(
        "Rows with any missing predictor value:",
        design[predictor_columns].isna().any(axis=1).sum(),
    )


def save_design_matrix(
    design: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the subject-specific design matrix.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    design.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved design matrix: {output_path}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a subject-specific design matrix "
            "from the retained ERP trial lookup and "
            "language predictors."
        )
    )

    parser.add_argument(
        "--trial-lookup",
        required=True,
        help=(
            "Path to the subject retained-trial lookup "
            "created by export_erp_long.py."
        ),
    )

    parser.add_argument(
        "--predictors",
        required=True,
        help=(
            "Path to ALL_language_metrics_GLM.tsv."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path for the subject design matrix TSV."
        ),
    )

    args = parser.parse_args()

    trial_lookup_path = Path(
        args.trial_lookup
    )

    predictors_path = Path(
        args.predictors
    )

    output_path = Path(
        args.output
    )

    if not trial_lookup_path.exists():
        raise FileNotFoundError(
            "Trial lookup file not found: "
            f"{trial_lookup_path}"
        )

    if not predictors_path.exists():
        raise FileNotFoundError(
            f"Predictor file not found: {predictors_path}"
        )

    trial_lookup = load_trial_lookup(
        trial_lookup_path
    )

    predictors = load_language_predictors(
        predictors_path
    )

    design = build_subject_design_matrix(
        trial_lookup=trial_lookup,
        predictors=predictors,
    )

    validate_design_matrix(
        design=design,
        predictors=predictors,
    )

    save_design_matrix(
        design=design,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()