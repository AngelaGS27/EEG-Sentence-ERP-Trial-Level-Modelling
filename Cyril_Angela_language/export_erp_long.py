# Export N400Stimset ERP .mat files into long-format EEG CSV tables.
#
# Input:
#   derivatives/erps/sub-XX/*_erp-CP.mat
#
# Output:
#   eeg_outputs/sub-XX_erp_long.csv
#   eeg_outputs/ALL_erp_long.csv
#
# Output format:
#   subject, condition, trial, item, channel, time, amplitude
#
# Designed for downstream use with:
#   run_component_lmm.py

from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd


def decode_matlab_string(file, ref):
    """Decode MATLAB HDF5 string reference."""
    try:
        obj = file[ref]
        arr = np.array(obj).squeeze()

        if arr.dtype.kind in {"u", "i"}:
            return "".join(chr(int(x)) for x in arr if int(x) != 0)

        return str(arr)

    except Exception:
        return ""


def get_subject_id(path: Path) -> str:
    name = path.name

    if "_task-" in name:
        return name.split("_task-")[0]

    return path.stem


def extract_condition_object(file, erps_dataset, condition_index):
    """Follow ERP object reference."""
    ref = erps_dataset[0, condition_index]
    return file[ref]


def extract_data(condition_group):
    """
    Extract EEG data.

    Expected dataset structure:
        trials x timepoints x channels
    """

    if "data" not in condition_group:
        raise KeyError("No 'data' field found.")

    data = np.array(condition_group["data"])

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got {data.shape}")

    trials, timepoints, channels = data.shape

    return data, trials, timepoints, channels


def extract_times(file):
    """Extract time vector."""

    if "t" not in file:
        raise KeyError("No time vector 't' found.")

    times = np.array(file["t"]).squeeze()

    # Convert ms -> seconds if necessary
    if np.nanmax(np.abs(times)) > 10:
        times = times / 1000.0

    return times


def extract_channel_labels(file, condition_group, n_channels):
    """
    Extract channel labels if possible.
    Otherwise generate generic labels.
    """

    labels = []

    try:
        chanlocs = condition_group["chanlocs"]

        if "labels" in chanlocs:
            label_refs = chanlocs["labels"]

            for i in range(label_refs.shape[0]):
                ref = label_refs[i, 0]
                labels.append(decode_matlab_string(file, ref))

    except Exception:
        labels = []

    if len(labels) != n_channels or any(label == "" for label in labels):
        labels = [f"ch_{i+1:03d}" for i in range(n_channels)]

    return labels


def export_long_for_file(mat_path: Path, output_dir: Path):

    subject = get_subject_id(mat_path)

    print(f"\nProcessing {mat_path.name}")

    with h5py.File(mat_path, "r") as file:

        if "ERPs" not in file:
            raise KeyError("No ERPs dataset found.")

        erps = file["ERPs"]

        times = extract_times(file)

        n_conditions = erps.shape[1]

        all_rows = []

        for condition_idx in range(n_conditions):

            condition_number = condition_idx + 1

            condition_group = extract_condition_object(
                file,
                erps,
                condition_idx
            )

            data, n_trials, n_timepoints, n_channels = extract_data(
                condition_group
            )

            channel_labels = extract_channel_labels(
                file,
                condition_group,
                n_channels
            )

            if len(times) != n_timepoints:
                print(
                    f"  Warning: time vector length {len(times)} "
                    f"!= data timepoints {n_timepoints}"
                )

                times_used = np.arange(n_timepoints)

            else:
                times_used = times

            print(
                f"  Condition {condition_number}: "
                f"{n_trials} trials x "
                f"{n_timepoints} timepoints x "
                f"{n_channels} channels"
            )

            for trial_idx in range(n_trials):

                trial_number = trial_idx + 1

                # Dataset format:
                # trials x timepoints x channels
                trial_data = data[trial_idx, :, :]

                for ch_idx, channel in enumerate(channel_labels):

                    amplitudes = trial_data[:, ch_idx]

                    rows = pd.DataFrame(
                        {
                            "subject": subject,
                            "condition": condition_number,
                            "trial": trial_number,
                            "item": trial_number,
                            "channel": channel,
                            "time": times_used,
                            "amplitude": amplitudes,
                        }
                    )

                    all_rows.append(rows)

        if not all_rows:
            print(f"  No rows extracted for {mat_path.name}")
            return None

        out = pd.concat(all_rows, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{subject}_erp_long.csv"

    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    return out_path


def main():

    parser = argparse.ArgumentParser(
        description="Export ERP .mat files to long-format EEG CSV."
    )

    parser.add_argument(
        "erp_root",
        help="Path to derivatives/erps folder."
    )

    parser.add_argument(
        "--output-dir",
        default="eeg_outputs",
        help="Folder where EEG CSVs will be saved."
    )

    args = parser.parse_args()

    erp_root = Path(args.erp_root)

    output_dir = Path(args.output_dir)

    if not erp_root.exists():
        raise FileNotFoundError(f"ERP root not found: {erp_root}")

    mat_files = sorted(
        erp_root.glob("sub-*/*_erp-CP.mat")
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No *_erp-CP.mat files found under {erp_root}"
        )

    print(f"Found {len(mat_files)} ERP CP files.")

    output_files = []

    for mat_path in mat_files:

        try:
            out_path = export_long_for_file(
                mat_path,
                output_dir
            )

            if out_path is not None:
                output_files.append(out_path)

        except Exception as e:

            print(f"FAILED: {mat_path.name}: {e}")

    if output_files:

        print("\nCombining subject CSV files safely in chunks...")

        combined_path = output_dir / "ALL_erp_long.csv"

        if combined_path.exists():
            combined_path.unlink()

        first_chunk = True

        for path in output_files:
            print(f"  Adding: {path.name}")

            for chunk in pd.read_csv(path, chunksize=500_000):
                chunk.to_csv(
                    combined_path,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )

                first_chunk = False

        print(f"Saved combined EEG long file: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()