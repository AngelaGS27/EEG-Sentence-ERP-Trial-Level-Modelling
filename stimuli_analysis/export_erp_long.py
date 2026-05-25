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

def make_biosemi_128_label(index: int) -> str:
    """
    Convert channel index to BioSemi 128 label.

    index 0  -> A1
    index 31 -> A32
    index 32 -> B1
    index 63 -> B32
    index 64 -> C1
    index 95 -> C32
    index 96 -> D1
    """
    letters = ["A", "B", "C", "D"]

    if index < 0 or index >= 128:
        return f"ch_{index + 1:03d}"

    letter = letters[index // 32]
    number = (index % 32) + 1

    return f"{letter}{number}"

def extract_channel_labels(file, condition_group, n_channels):
    """
    Extract channel labels if possible.

    If labels cannot be extracted from the .mat file, assume BioSemi 128 order:
        A1-A32, B1-B32, C1-C32, D1-D32
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
        labels = [
            make_biosemi_128_label(i)
            for i in range(n_channels)
        ]

    return labels

def clean_electrode_name(value):
    """
    Clean electrode labels.
    """
    value = str(value).strip().strip("'").strip('"')

    if "_" in value:
        value = value.split("_", 1)[0]

    return value.strip()


def split_channel_name(channel_name):
    """
    Split labels like A1_Cz into:
        electrode = A1
        standard_label = Cz
    """
    channel_name = str(channel_name).strip()

    if "_" in channel_name:
        electrode, standard = channel_name.split("_", 1)
        return clean_electrode_name(electrode), standard.strip()

    return clean_electrode_name(channel_name), ""


def load_electrode_coordinates(bids_root: Path) -> pd.DataFrame:
    """
    Load task-N400Stimset_electrodes.tsv.

    Expected columns:
        name, X, Y, Z, sph_theta, sph_phi, sph_radius, theta, radius
    """

    electrodes_path = bids_root / "task-N400Stimset_electrodes.tsv"

    if not electrodes_path.exists():
        raise FileNotFoundError(
            f"Electrodes file not found: {electrodes_path}"
        )

    electrodes = pd.read_csv(electrodes_path, sep="\t")

    electrodes["electrode"] = electrodes["name"].map(clean_electrode_name)

    electrodes = electrodes.rename(
        columns={
            "X": "x",
            "Y": "y",
            "Z": "z",
        }
    )

    keep_cols = [
        "electrode",
        "x",
        "y",
        "z",
        "sph_theta",
        "sph_phi",
        "sph_radius",
        "theta",
        "radius",
    ]

    existing = [col for col in keep_cols if col in electrodes.columns]

    return electrodes[existing]


def load_subject_channel_metadata(bids_root: Path, subject: str) -> pd.DataFrame:
    """
    Load subject channels.tsv.
    """

    possible_paths = [
        bids_root / subject / "eeg" / f"{subject}_task-N400Stimset_channels.tsv",
        bids_root / subject / f"{subject}_task-N400Stimset_channels.tsv",
    ]

    channels_path = None

    for path in possible_paths:
        if path.exists():
            channels_path = path
            break

    if channels_path is None:
        raise FileNotFoundError(
            "Channels file not found. Checked:\n"
            + "\n".join(str(path) for path in possible_paths)
        )

    channels = pd.read_csv(channels_path, sep="\t")

    channels["original_channel"] = channels["name"].astype(str)

    split = channels["name"].map(split_channel_name)

    channels["electrode"] = split.map(lambda x: x[0])
    channels["standard_label"] = split.map(lambda x: x[1])

    channels["channel_clean"] = channels.apply(
        lambda row: (
            row["standard_label"]
            if row["standard_label"] != ""
            else row["electrode"]
        ),
        axis=1,
    )

    return channels[
        [
            "original_channel",
            "electrode",
            "standard_label",
            "channel_clean",
            "type",
            "units",
            "status",
            "status_description",
        ]
    ]


def build_channel_metadata(
    bids_root: Path,
    subject: str,
    channel_labels: list[str],
) -> pd.DataFrame:
    """
    Build channel metadata in the same order as the MAT file channels.

    Handles:
        A1
        A1_Cz
        'A1'
        ch_001 fallback labels

    Combines:
        1. labels from the .mat file
        2. subject channels.tsv
        3. task-N400Stimset_electrodes.tsv coordinates
    """

    electrodes = load_electrode_coordinates(bids_root)
    subject_channels = load_subject_channel_metadata(bids_root, subject)

    mat_channels = pd.DataFrame(
        {
            "channel_index": range(len(channel_labels)),
            "mat_channel": channel_labels,
        }
    )

    mat_channels["mat_channel"] = mat_channels["mat_channel"].astype(str)

    mat_channels["electrode"] = mat_channels.apply(
        lambda row: (
            make_biosemi_128_label(int(row["channel_index"]))
            if row["mat_channel"].lower().startswith("ch_")
            else split_channel_name(row["mat_channel"])[0]
        ),
        axis=1,
    )

    merged = mat_channels.merge(
        subject_channels,
        on="electrode",
        how="left",
    )

    merged = merged.merge(
        electrodes,
        on="electrode",
        how="left",
    )

    merged["channel"] = merged.apply(
        lambda row: (
            row["channel_clean"]
            if pd.notna(row.get("channel_clean"))
            and str(row.get("channel_clean")).strip() != ""
            else row["mat_channel"]
        ),
        axis=1,
    )

    merged["original_channel"] = merged.apply(
        lambda row: (
            row["original_channel"]
            if pd.notna(row.get("original_channel"))
            else row["mat_channel"]
        ),
        axis=1,
    )

    return merged

def export_long_for_file(mat_path: Path, output_dir: Path, bids_root: Path):

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

            channel_metadata = build_channel_metadata(
                bids_root=bids_root,
                subject=subject,
                channel_labels=channel_labels,
            )

            if len(channel_metadata) != n_channels:
                raise ValueError(
                    f"Channel metadata length {len(channel_metadata)} "
                    f"does not match data channels {n_channels}"
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

                trial_data = data[trial_idx, :, :]

                for ch_idx in range(n_channels):

                    meta = channel_metadata.iloc[ch_idx]

                    amplitudes = trial_data[:, ch_idx]

                    rows = pd.DataFrame(
                        {
                            "subject": subject,
                            "condition": condition_number,
                            "trial": trial_number,
                            "item": trial_number,

                            # Clean usable label, e.g. Cz, Pz, Oz, or A2
                            "channel": meta["channel"],

                            # Original BIDS channel label, e.g. A1_Cz
                            "original_channel": meta["original_channel"],

                            # BioSemi electrode, e.g. A1
                            "electrode": meta["electrode"],

                            # Standard scalp label if available, e.g. Cz
                            "standard_label": meta.get("standard_label", ""),

                            # Coordinates
                            "x": meta.get("x", np.nan),
                            "y": meta.get("y", np.nan),
                            "z": meta.get("z", np.nan),
                            "sph_theta": meta.get("sph_theta", np.nan),
                            "sph_phi": meta.get("sph_phi", np.nan),
                            "sph_radius": meta.get("sph_radius", np.nan),
                            "theta": meta.get("theta", np.nan),
                            "radius": meta.get("radius", np.nan),

                            # BIDS quality metadata
                            "channel_status": meta.get("status", np.nan),
                            "channel_status_description": meta.get(
                                "status_description",
                                np.nan,
                            ),

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
        description="Export ERP .mat files to long-format EEG CSV with channel coordinates."
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

    parser.add_argument(
        "--bids-root",
        default=".",
        help=(
            "Root BIDS dataset folder containing "
            "task-N400Stimset_electrodes.tsv and sub-XX/eeg/*channels.tsv."
        ),
    )

    args = parser.parse_args()

    erp_root = Path(args.erp_root)
    output_dir = Path(args.output_dir)
    bids_root = Path(args.bids_root)

    if not erp_root.exists():
        raise FileNotFoundError(f"ERP root not found: {erp_root}")

    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    electrodes_path = bids_root / "task-N400Stimset_electrodes.tsv"

    if not electrodes_path.exists():
        raise FileNotFoundError(
            f"Electrodes file not found: {electrodes_path}"
        )

    mat_files = sorted(
        erp_root.glob("sub-*/*_erp-CP.mat")
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No *_erp-CP.mat files found under {erp_root}"
        )

    print(f"Found {len(mat_files)} ERP CP files.")
    print(f"Using BIDS root: {bids_root}")
    print(f"Using electrodes file: {electrodes_path}")

    output_files = []

    for mat_path in mat_files:

        try:
            out_path = export_long_for_file(
                mat_path=mat_path,
                output_dir=output_dir,
                bids_root=bids_root,
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