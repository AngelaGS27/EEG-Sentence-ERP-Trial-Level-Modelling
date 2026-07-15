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
    "stim_key",
    "stim_file",
    "stimulus",
    "sentence_id",
    "item",
    "trial",
    "subject_trial",
    "condition_trial",
    "onset",
    "duration",
    "sample",
    "value",
    "event_id",
    "epoch",
}


def read_epochs(set_path: Path):
    """
    Read an epoched EEGLAB .set file.
    """
    epochs = mne.io.read_epochs_eeglab(str(set_path), verbose="ERROR")
    epochs.load_data()
    return epochs


def load_design_matrix(design_path: Path) -> pd.DataFrame:
    """
    Load the subject-specific design matrix.
    """
    design = pd.read_csv(design_path, sep=None, engine="python")

    if design.empty:
        raise ValueError(f"Design matrix is empty: {design_path}")

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
    Select numeric columns to use as predictors.

    If --predictor-list is given, use only those columns.
    Otherwise, use all numeric columns except obvious metadata columns.
    """
    if requested_predictors is not None:
        missing = [col for col in requested_predictors if col not in design.columns]

        if missing:
            raise ValueError(
                "Requested predictors are missing from the design matrix: "
                + ", ".join(missing)
            )

        candidates = requested_predictors
    else:
        candidates = [
            col for col in design.columns
            if col not in DEFAULT_EXCLUDE_COLUMNS
        ]

    selected = []

    for col in candidates:
        numeric = pd.to_numeric(design[col], errors="coerce")

        if numeric.notna().sum() == 0:
            continue

        if numeric.nunique(dropna=True) < 2:
            continue

        selected.append(col)

    if not selected:
        raise ValueError("No usable numeric predictors were found.")

    return selected


def build_design_array(
    design: pd.DataFrame,
    predictor_columns: list[str],
    add_intercept: bool = True,
):
    """
    Convert the design table into a numerical matrix X.

    Rows with missing predictor values are removed.
    The same rows must also be removed from the EEG data.
    """
    X_df = design[predictor_columns].apply(pd.to_numeric, errors="coerce")

    valid_trial_mask = X_df.notna().all(axis=1).to_numpy()

    X = X_df.loc[valid_trial_mask].to_numpy(dtype=float)

    predictor_names = list(predictor_columns)

    if add_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
        predictor_names = ["intercept"] + predictor_names

    rank = np.linalg.matrix_rank(X)

    if rank < X.shape[1]:
        print(
            "Warning: design matrix is rank-deficient. "
            f"Rank {rank}, columns {X.shape[1]}."
        )

    if X.shape[0] <= rank:
        raise ValueError(
            "Not enough complete trials for this model. "
            f"Complete trials: {X.shape[0]}, rank: {rank}"
        )

    return X, predictor_names, valid_trial_mask


def fit_mass_univariate_glm(data: np.ndarray, X: np.ndarray):
    """
    Fit ordinary least squares separately at every channel x timepoint.

    data shape:
        trials x channels x times

    X shape:
        trials x predictors
    """
    n_trials, n_channels, n_times = data.shape
    n_predictors = X.shape[1]

    # Flatten EEG into:
    # trials x all_channel_time_points
    Y = data.reshape(n_trials, n_channels * n_times)

    # OLS beta = pinv(X) @ Y
    pinv_X = np.linalg.pinv(X)
    beta_2d = pinv_X @ Y

    fitted_2d = X @ beta_2d
    residuals_2d = Y - fitted_2d

    rank = np.linalg.matrix_rank(X)
    dof = n_trials - rank

    rss = np.sum(residuals_2d ** 2, axis=0)
    sigma2_2d = rss / dof

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_variance = np.diag(xtx_inv)

    standard_error = np.sqrt(beta_variance[:, None] * sigma2_2d[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        t_2d = beta_2d / standard_error

    beta = beta_2d.reshape(n_predictors, n_channels, n_times)
    t_values = t_2d.reshape(n_predictors, n_channels, n_times)
    residual_variance = sigma2_2d.reshape(n_channels, n_times)

    return {
        "beta": beta,
        "t": t_values,
        "residual_variance": residual_variance,
        "dof": int(dof),
        "rank": int(rank),
        "xtx_inv": xtx_inv,
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