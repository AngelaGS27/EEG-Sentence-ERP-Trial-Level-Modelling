"""
Run a Python first-level LIMO-style GLM for one epoched EEGLAB .set file.

This script does:

    EEG.set + subject design matrix TSV
        -> beta estimates at every channel x timepoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import mne
except ImportError as exc:
    raise ImportError("Install MNE first: pip install mne") from exc


DEFAULT_EXCLUDE_COLUMNS = {
    "subject",
    "participant_id",
    "condition",
    "condition_label",
    "trial_type",
    "stim_key",
    "stim_file",
    "stimulus",
    "sentence_id",
    "item",
    "trial",
    "retained_trial",
    "eeg_trial",
    "subject_trial",
    "condition_trial",
    "experimental_trial",
    "original_event_row",
    "urevent_index",
    "urevent_seconds",
    "event_time_difference_seconds",
    "target_onset_seconds",
    "onset",
    "duration",
    "sample",
    "value",
    "event_id",
    "epoch",
    "epoch_index",
    "channel",
    "channel_index",
    "time",
    "time_index",
    "amplitude",
    "x",
    "y",
    "z",
    "sph_theta",
    "sph_phi",
    "sph_radius",
    "theta",
    "radius",
    "channel_status",
    "channel_status_description",
}


def read_epochs(set_path: Path):
    """
    Read an epoched EEGLAB .set file.
    """
    epochs = mne.io.read_epochs_eeglab(str(set_path), verbose="ERROR")
    epochs.load_data()
    return epochs


def load_design_matrix(
    design_path: Path,
) -> pd.DataFrame:
    """
    Load and validate a subject-specific design matrix.

    The design matrix must contain one row per retained EEG epoch,
    in exactly the order of those epochs. prepare_limo_design_matrix.py
    writes an eeg_trial column recording that order.
    """

    design_path = Path(
        design_path
    ).expanduser().resolve()

    if not design_path.exists():
        raise FileNotFoundError(
            f"Design matrix not found: {design_path}"
        )

    design = pd.read_csv(
        design_path,
        sep=None,
        engine="python",
    )

    if design.empty:
        raise ValueError(
            f"Design matrix is empty: {design_path}"
        )

    design = design.copy()

    if "stim_key" not in design.columns:
        raise ValueError(
            "Design matrix does not contain stim_key. "
            "Build it using prepare_limo_design_matrix.py and the "
            "retained-trial lookup created by export_erp_long.py."
        )

    design["stim_key"] = (
        design["stim_key"]
        .astype("string")
        .str.strip()
    )

    missing_stim_keys = (
        design["stim_key"].isna()
        | design["stim_key"].eq("")
    )

    if missing_stim_keys.any():
        raise ValueError(
            "Design matrix contains "
            f"{int(missing_stim_keys.sum())} missing or empty "
            "stim_key values."
        )

    if "eeg_trial" not in design.columns:
        raise ValueError(
            "Design matrix does not contain eeg_trial. "
            "Rebuild it with the updated "
            "prepare_limo_design_matrix.py so epoch order can be "
            "validated."
        )

    design["eeg_trial"] = pd.to_numeric(
        design["eeg_trial"],
        errors="raise",
    ).astype(int)

    if design["eeg_trial"].duplicated().any():
        examples = (
            design.loc[
                design[
                    "eeg_trial"
                ].duplicated(
                    keep=False
                ),
                "eeg_trial",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Design matrix contains duplicate eeg_trial values. "
            f"Examples: {examples}"
        )

    design = design.sort_values(
        "eeg_trial",
        kind="stable",
    ).reset_index(
        drop=True
    )

    expected_order = np.arange(
        1,
        len(design) + 1,
        dtype=int,
    )

    actual_order = design[
        "eeg_trial"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        actual_order,
        expected_order,
    ):
        raise ValueError(
            "eeg_trial must be a complete consecutive sequence "
            f"from 1 to {len(design)}. "
            "The design matrix cannot safely be aligned to the EEG "
            "epochs otherwise."
        )

    if {
        "condition",
        "retained_trial",
    }.issubset(
        design.columns
    ):
        duplicated_trials = design.duplicated(
            subset=[
                "condition",
                "retained_trial",
            ],
            keep=False,
        )

        if duplicated_trials.any():
            examples = (
                design.loc[
                    duplicated_trials,
                    [
                        "condition",
                        "retained_trial",
                    ],
                ]
                .head(10)
                .to_dict(
                    "records"
                )
            )

            raise ValueError(
                "Design matrix contains duplicate retained trials. "
                f"Examples: {examples}"
            )

    return design


def parse_predictor_list(value: str | None) -> list[str] | None:
    if value is None:
        return None

    predictors = [item.strip() for item in value.split(",") if item.strip()]
    return predictors or None


def choose_predictor_columns(
    design: pd.DataFrame,
    requested_predictors: list[str] | None,
) -> list[str]:
    """
    Select numerical scientific predictor columns.

    When --predictor-list is supplied, exactly those predictors are
    considered.

    Otherwise, all columns in DEFAULT_EXCLUDE_COLUMNS are removed
    before numeric validation. This prevents trial indices, event
    rows, urevent timing and other tracking variables from entering
    the GLM as accidental predictors.
    """

    if requested_predictors is not None:
        missing = [
            column
            for column in requested_predictors
            if column not in design.columns
        ]

        if missing:
            raise ValueError(
                "Requested predictors are missing from the design "
                "matrix: "
                + ", ".join(
                    missing
                )
            )

        candidates = list(
            requested_predictors
        )

    else:
        candidates = [
            column
            for column in design.columns
            if column
            not in DEFAULT_EXCLUDE_COLUMNS
        ]

    selected = []
    rejected_non_numeric = []
    rejected_constant = []
    rejected_all_missing = []

    for column in candidates:
        numeric = pd.to_numeric(
            design[column],
            errors="coerce",
        )

        non_missing_count = int(
            numeric.notna().sum()
        )

        if non_missing_count == 0:
            original_non_missing = int(
                design[column]
                .notna()
                .sum()
            )

            if original_non_missing == 0:
                rejected_all_missing.append(
                    column
                )
            else:
                rejected_non_numeric.append(
                    column
                )

            continue

        unique_count = int(
            numeric.nunique(
                dropna=True
            )
        )

        if unique_count < 2:
            rejected_constant.append(
                column
            )
            continue

        selected.append(
            column
        )

    if requested_predictors is not None:
        rejected_requested = (
            rejected_non_numeric
            + rejected_constant
            + rejected_all_missing
        )

        if rejected_requested:
            raise ValueError(
                "Some explicitly requested predictors are unusable. "
                f"Non-numeric: {rejected_non_numeric}; "
                f"constant: {rejected_constant}; "
                f"all missing: {rejected_all_missing}"
            )

    if not selected:
        raise ValueError(
            "No usable numeric scientific predictors were found. "
            "Provide --predictor-list explicitly or check the design "
            "matrix produced from ALL_language_metrics_GLM.tsv."
        )

    print(
        f"Selected {len(selected)} predictor columns."
    )

    if requested_predictors is None:
        if rejected_non_numeric:
            print(
                "Skipped non-numeric candidate columns: "
                + ", ".join(
                    rejected_non_numeric
                )
            )

        if rejected_constant:
            print(
                "Skipped constant candidate columns: "
                + ", ".join(
                    rejected_constant
                )
            )

        if rejected_all_missing:
            print(
                "Skipped all-missing candidate columns: "
                + ", ".join(
                    rejected_all_missing
                )
            )

    return selected


def build_design_array(
    design: pd.DataFrame,
    predictor_columns: list[str],
    add_intercept: bool = True,
):
    """
    Convert the subject design table into numerical matrix X.

    Rows with missing values in any selected predictor are removed.
    The returned mask must be applied to the EEG epochs in exactly
    the same row order.
    """

    missing_columns = [
        column
        for column in predictor_columns
        if column not in design.columns
    ]

    if missing_columns:
        raise ValueError(
            "Predictor columns are missing from the design matrix: "
            + ", ".join(
                missing_columns
            )
        )

    X_df = (
        design[
            predictor_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    valid_trial_mask = (
        X_df
        .notna()
        .all(
            axis=1
        )
        .to_numpy(
            dtype=bool
        )
    )

    n_complete = int(
        valid_trial_mask.sum()
    )

    n_removed = int(
        len(valid_trial_mask)
        - n_complete
    )

    if n_complete == 0:
        raise ValueError(
            "No trials have complete values for all selected "
            "predictors."
        )

    if n_removed > 0:
        print(
            f"Removing {n_removed} EEG trials because at least one "
            "selected predictor is missing."
        )

    X = (
        X_df.loc[
            valid_trial_mask
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        X
    ).all():
        raise ValueError(
            "Design matrix contains infinite predictor values."
        )

    predictor_names = list(
        predictor_columns
    )

    if add_intercept:
        X = np.column_stack(
            [
                np.ones(
                    X.shape[0],
                    dtype=float,
                ),
                X,
            ]
        )

        predictor_names = [
            "intercept",
        ] + predictor_names

    rank = int(
        np.linalg.matrix_rank(
            X
        )
    )

    if rank < X.shape[1]:
        print(
            "Warning: design matrix is rank-deficient. "
            f"Rank: {rank}; columns: {X.shape[1]}. "
            "Some beta estimates will not be uniquely identifiable."
        )

    degrees_of_freedom = (
        X.shape[0]
        - rank
    )

    if degrees_of_freedom <= 0:
        raise ValueError(
            "Not enough complete trials for the selected model. "
            f"Complete trials: {X.shape[0]}; "
            f"design rank: {rank}; "
            f"degrees of freedom: {degrees_of_freedom}."
        )

    return (
        X,
        predictor_names,
        valid_trial_mask,
    )


def fit_mass_univariate_glm(
    data: np.ndarray,
    X: np.ndarray,
):
    """
    Fit ordinary least squares independently at every
    channel-by-timepoint sample.

    data:
        trials x channels x times

    X:
        trials x predictors
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    if data.ndim != 3:
        raise ValueError(
            "EEG data must have shape "
            "trials x channels x times, "
            f"but received {data.shape}."
        )

    if X.ndim != 2:
        raise ValueError(
            "Design matrix X must be two-dimensional, "
            f"but received {X.shape}."
        )

    n_trials, n_channels, n_times = (
        data.shape
    )

    if X.shape[0] != n_trials:
        raise ValueError(
            "Design rows do not match EEG trials after applying "
            f"the complete-case mask: {X.shape[0]} versus {n_trials}."
        )

    if not np.isfinite(
        data
    ).all():
        raise ValueError(
            "EEG data contains NaN or infinite values."
        )

    if not np.isfinite(
        X
    ).all():
        raise ValueError(
            "Design matrix contains NaN or infinite values."
        )

    n_predictors = X.shape[1]

    Y = data.reshape(
        n_trials,
        n_channels * n_times,
    )

    rank = int(
        np.linalg.matrix_rank(
            X
        )
    )

    degrees_of_freedom = int(
        n_trials
        - rank
    )

    if degrees_of_freedom <= 0:
        raise ValueError(
            "The GLM has no residual degrees of freedom. "
            f"Trials: {n_trials}; rank: {rank}."
        )

    pinv_X = np.linalg.pinv(
        X
    )

    beta_2d = (
        pinv_X
        @ Y
    )

    fitted_2d = (
        X
        @ beta_2d
    )

    residuals_2d = (
        Y
        - fitted_2d
    )

    residual_sum_squares = np.sum(
        residuals_2d ** 2,
        axis=0,
    )

    sigma_squared = (
        residual_sum_squares
        / degrees_of_freedom
    )

    xtx_inverse = np.linalg.pinv(
        X.T
        @ X
    )

    beta_variance_factors = np.diag(
        xtx_inverse
    )

    beta_variance_factors = np.maximum(
        beta_variance_factors,
        0.0,
    )

    standard_error_2d = np.sqrt(
        beta_variance_factors[
            :,
            None,
        ]
        * sigma_squared[
            None,
            :,
        ]
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        t_2d = np.divide(
            beta_2d,
            standard_error_2d,
            out=np.full_like(
                beta_2d,
                np.nan,
                dtype=float,
            ),
            where=(
                standard_error_2d > 0
            ),
        )

    beta = beta_2d.reshape(
        n_predictors,
        n_channels,
        n_times,
    )

    t_values = t_2d.reshape(
        n_predictors,
        n_channels,
        n_times,
    )

    residual_variance = sigma_squared.reshape(
        n_channels,
        n_times,
    )

    return {
        "beta": beta,
        "t": t_values,
        "residual_variance": residual_variance,
        "dof": degrees_of_freedom,
        "rank": rank,
        "xtx_inv": xtx_inverse,
    }


def write_string_dataset(h5, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(values, dtype=object), dtype=dtype)


def save_hdf5(
    output_path: Path,
    results: dict,
    X: np.ndarray,
    predictor_names: list[str],
    channel_names: list[str],
    times: np.ndarray,
    valid_trial_mask: np.ndarray,
    metadata: dict,
) -> None:
    """
    Save first-level GLM outputs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("beta", data=results["beta"], compression="gzip")
        h5.create_dataset("t", data=results["t"], compression="gzip")
        h5.create_dataset(
            "residual_variance",
            data=results["residual_variance"],
            compression="gzip",
        )
        h5.create_dataset("design_matrix", data=X, compression="gzip")
        h5.create_dataset("times", data=times)
        h5.create_dataset("valid_trial_mask", data=valid_trial_mask.astype(int))
        h5.create_dataset("xtx_inv", data=results["xtx_inv"])

        h5.attrs["dof"] = results["dof"]
        h5.attrs["rank"] = results["rank"]
        h5.attrs["metadata_json"] = json.dumps(metadata)

        write_string_dataset(h5, "predictor_names", predictor_names)
        write_string_dataset(h5, "channel_names", channel_names)


def save_long_table(
    output_dir: Path,
    beta: np.ndarray,
    t_values: np.ndarray,
    predictor_names: list[str],
    channel_names: list[str],
    times: np.ndarray,
) -> None:
    """
    Save beta and t-values in a readable long-format TSV.
    """
    rows = []

    for pred_idx, predictor in enumerate(predictor_names):
        for ch_idx, channel in enumerate(channel_names):
            rows.append(
                pd.DataFrame(
                    {
                        "predictor": predictor,
                        "channel": channel,
                        "time": times,
                        "beta": beta[pred_idx, ch_idx, :],
                        "t": t_values[pred_idx, ch_idx, :],
                    }
                )
            )

    out = pd.concat(rows, ignore_index=True)

    out_path = output_dir / "first_level_beta_t_long.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    print(f"Saved long beta/t table: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Python LIMO-style first-level GLM for one subject."
    )

    parser.add_argument(
        "--eeg-set",
        required=True,
        help="Path to epoched EEGLAB .set file.",
    )

    parser.add_argument(
        "--design",
        required=True,
        help="Path to subject design matrix TSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--predictor-list",
        default=None,
        help=(
            "Optional comma-separated predictors. "
            "If omitted, all numeric non-metadata columns are used."
        ),
    )

    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Do not add intercept column.",
    )

    args = parser.parse_args()

    eeg_set_path = Path(args.eeg_set)
    design_path = Path(args.design)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not eeg_set_path.exists():
        raise FileNotFoundError(f"EEG .set file not found: {eeg_set_path}")

    if not design_path.exists():
        raise FileNotFoundError(f"Design matrix not found: {design_path}")

    print(f"Reading EEG epochs: {eeg_set_path}")
    epochs = read_epochs(eeg_set_path)

    data = epochs.get_data()
    print(f"EEG data shape: {data.shape} = trials x channels x times")

    design = load_design_matrix(design_path)
    print(f"Design rows: {len(design)}")

    if len(design) != data.shape[0]:
        raise ValueError(
            "Design rows and EEG epochs do not match. "
            f"Design rows: {len(design)}, EEG epochs: {data.shape[0]}. "
            "The design matrix must be in the same trial order as the EEG epochs."
        )

    requested_predictors = parse_predictor_list(args.predictor_list)

    predictor_columns = choose_predictor_columns(
        design=design,
        requested_predictors=requested_predictors,
    )

    print("Predictors used:")
    for col in predictor_columns:
        print(f"  - {col}")

    X, predictor_names, valid_trial_mask = build_design_array(
        design=design,
        predictor_columns=predictor_columns,
        add_intercept=not args.no_intercept,
    )

    data_valid = data[valid_trial_mask, :, :]

    print(f"Complete trials used: {data_valid.shape[0]}")
    print(f"Design matrix shape: {X.shape}")

    results = fit_mass_univariate_glm(
        data=data_valid,
        X=X,
    )

    h5_path = output_dir / "LIMO_first_level.h5"

    metadata = {
        "eeg_set": str(eeg_set_path),
        "design": str(design_path),
        "n_epochs_original": int(data.shape[0]),
        "n_epochs_used": int(data_valid.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_times": int(data.shape[2]),
        "note": (
            "Python LIMO-style first-level mass-univariate GLM; "
            "not official MATLAB LIMO output."
        ),
    }

    save_hdf5(
        output_path=h5_path,
        results=results,
        X=X,
        predictor_names=predictor_names,
        channel_names=list(epochs.ch_names),
        times=epochs.times,
        valid_trial_mask=valid_trial_mask,
        metadata=metadata,
    )

    np.save(output_dir / "beta.npy", results["beta"])
    np.save(output_dir / "t_values.npy", results["t"])
    np.save(output_dir / "residual_variance.npy", results["residual_variance"])

    pd.DataFrame(
        {
            "predictor": predictor_names,
            "column_index": range(len(predictor_names)),
        }
    ).to_csv(output_dir / "predictor_names.tsv", sep="\t", index=False)

    pd.DataFrame(
        {
            "channel": epochs.ch_names,
            "channel_index": range(len(epochs.ch_names)),
        }
    ).to_csv(output_dir / "channel_names.tsv", sep="\t", index=False)

    pd.DataFrame(
        {
            "time": epochs.times,
            "time_index": range(len(epochs.times)),
        }
    ).to_csv(output_dir / "times.tsv", sep="\t", index=False)

    save_long_table(
        output_dir=output_dir,
        beta=results["beta"],
        t_values=results["t"],
        predictor_names=predictor_names,
        channel_names=list(epochs.ch_names),
        times=epochs.times,
    )

    summary = {
        "n_epochs_original": int(data.shape[0]),
        "n_epochs_used": int(data_valid.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_times": int(data.shape[2]),
        "n_predictors_including_intercept": int(X.shape[1]),
        "rank": int(results["rank"]),
        "dof": int(results["dof"]),
        "hdf5_output": str(h5_path),
    }

    with open(output_dir / "first_level_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()