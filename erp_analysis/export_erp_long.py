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

def dereference_numeric_cell(
    file: h5py.File,
    dataset,
    row_index: int,
) -> np.ndarray:
    """
    Dereference one MATLAB HDF5 cell containing numeric values.
    """

    reference = dataset[row_index, 0]
    values = np.asarray(
        file[reference]
    ).squeeze()

    return np.atleast_1d(
        values
    ).astype(float)

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

def extract_epoch_urevent_indices(
    file: h5py.File,
    condition_group,
    n_trials: int,
) -> list[int]:
    """
    Extract the original EEGLAB urevent index for each retained epoch.

    The derivative ERP files contain one epoch/eventurevent cell
    for every surviving trial.
    """

    if "epoch" not in condition_group:
        raise KeyError(
            "No epoch structure found in ERP condition."
        )

    epoch_group = condition_group["epoch"]

    if "eventurevent" not in epoch_group:
        raise KeyError(
            "No epoch/eventurevent field found."
        )

    eventurevent = epoch_group[
        "eventurevent"
    ]

    if eventurevent.shape[0] != n_trials:
        raise ValueError(
            "eventurevent row count does not match "
            f"the number of retained trials: "
            f"{eventurevent.shape[0]} versus {n_trials}"
        )

    indices = []

    for trial_index in range(
        n_trials
    ):
        values = dereference_numeric_cell(
            file=file,
            dataset=eventurevent,
            row_index=trial_index,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if len(finite_values) == 0:
            raise ValueError(
                "No valid urevent index found for "
                f"retained trial {trial_index + 1}."
            )

        # Epochs can theoretically contain more than one event.
        # The target epoch in these derivative files contains one
        # relevant original urevent reference.
        urevent_index = int(
            round(
                float(
                    finite_values[0]
                )
            )
        )

        if urevent_index < 1:
            raise ValueError(
                f"Invalid EEGLAB urevent index: {urevent_index}"
            )

        indices.append(
            urevent_index
        )

    return indices

def extract_urevent_latency_seconds(
    file: h5py.File,
    condition_group,
    urevent_index: int,
    sampling_frequency: float,
) -> float:
    """
    Convert an original EEGLAB urevent latency to seconds.
    """

    if "urevent" not in condition_group:
        raise KeyError(
            "No urevent structure found in ERP condition."
        )

    urevent_group = condition_group[
        "urevent"
    ]

    if "latency" not in urevent_group:
        raise KeyError(
            "No urevent/latency field found."
        )

    latency_dataset = urevent_group[
        "latency"
    ]

    zero_based_index = (
        urevent_index - 1
    )

    if (
        zero_based_index < 0
        or zero_based_index
        >= latency_dataset.shape[0]
    ):
        raise IndexError(
            "urevent index is outside the stored "
            f"urevent table: {urevent_index}"
        )

    values = dereference_numeric_cell(
        file=file,
        dataset=latency_dataset,
        row_index=zero_based_index,
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(finite_values) == 0:
        raise ValueError(
            f"No valid latency found for urevent {urevent_index}."
        )

    latency_samples = float(
        finite_values[0]
    )

    latency_seconds = (
        latency_samples - 1.0
    ) / sampling_frequency

    return latency_seconds

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

def normalise_stim_file(series: pd.Series) -> pd.Series:
    """
    Normalise stimulus filenames before matching.
    """
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )


def load_stimulus_lookup(
    bids_root: Path,
) -> pd.DataFrame:
    """
    Load the dataset stimulus table and construct a unique
    stim_file -> stim_key lookup.

    Expected dataset file:
        N400Stimset_stimuli_parameters.tsv
    """
    stimulus_path = (
        bids_root
        / "N400Stimset_stimuli_parameters.tsv"
    )

    if not stimulus_path.exists():
        raise FileNotFoundError(
            "Stimulus parameters file not found: "
            f"{stimulus_path}"
        )

    stimuli = pd.read_csv(
        stimulus_path,
        sep="\t",
    )

    required = [
        "stim_file",
        "stim_key",
    ]

    missing = [
        col
        for col in required
        if col not in stimuli.columns
    ]

    if missing:
        raise ValueError(
            "Stimulus parameters table is missing "
            f"columns: {missing}"
        )

    stimuli = stimuli.copy()

    stimuli["stim_file"] = normalise_stim_file(
        stimuli["stim_file"]
    )

    stimuli["stim_key"] = (
        stimuli["stim_key"]
        .astype("string")
        .str.strip()
    )

    if stimuli["stim_file"].isna().any():
        raise ValueError(
            "Stimulus table contains missing stim_file values."
        )

    if stimuli["stim_key"].isna().any():
        raise ValueError(
            "Stimulus table contains missing stim_key values."
        )

    conflicting = (
        stimuli
        .groupby("stim_file")["stim_key"]
        .nunique()
    )

    conflicting = conflicting[
        conflicting > 1
    ]

    if not conflicting.empty:
        raise ValueError(
            "Some stimulus filenames map to multiple stim_key values. "
            f"Examples: {conflicting.index[:10].tolist()}"
        )

    lookup = (
        stimuli[
            [
                "stim_file",
                "stim_key",
            ]
        ]
        .drop_duplicates(
            subset="stim_file"
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(lookup)} stimulus mappings "
        f"from {stimulus_path}"
    )

    return lookup

def load_subject_events(
    bids_root: Path,
    subject: str,
    stimulus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the subject's experimental sentence events.

    stim_key is attached through:
        stim_file -> N400Stimset_stimuli_parameters.tsv
    """

    events_path = (
        bids_root
        / subject
        / "eeg"
        / f"{subject}_task-N400Stimset_events.tsv"
    )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}"
        )

    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    required_columns = [
        "onset",
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

    # Preserve the row number in the complete BIDS events table.
    events["original_event_row"] = range(
        1,
        len(events) + 1,
    )

    events["trial_type"] = (
        events["trial_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Keep only the actual experimental sentence trials.
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
            f"No NPC or NPI sentence events found in {events_path}"
        )

    # BIDS onset is the final target-word onset.
    events["target_onset_seconds"] = pd.to_numeric(
        events["onset"],
        errors="coerce",
    )

    if events["target_onset_seconds"].isna().any():
        raise ValueError(
            "Some NPC/NPI events have missing or invalid "
            "target-word onset values."
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

    unmatched = events["_merge"] != "both"

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
            f"{unmatched.sum()} event rows could not be matched "
            "to N400Stimset_stimuli_parameters.tsv. "
            f"Examples: {examples}"
        )

    events = events.drop(
        columns="_merge"
    )

    events = events.sort_values(
        "target_onset_seconds"
    ).reset_index(
        drop=True
    )

    events["experimental_trial"] = range(
        1,
        len(events) + 1,
    )

    print(
        f"  Loaded {len(events)} experimental events "
        f"for {subject}: "
        f"{(events['trial_type'] == 'NPC').sum()} NPC and "
        f"{(events['trial_type'] == 'NPI').sum()} NPI"
    )

    return events[
        [
            "original_event_row",
            "experimental_trial",
            "trial_type",
            "target_onset_seconds",
            "stim_file",
            "stim_key",
        ]
    ]

def match_retained_trials_to_events(
    urevent_indices: list[int],
    file: h5py.File,
    condition_group,
    events: pd.DataFrame,
    sampling_frequency: float,
    tolerance_seconds: float,
) -> pd.DataFrame:
    """
    Match retained ERP epochs to subject events using target-word onset.

    For each retained trial:
        eventurevent
            -> original urevent latency
            -> closest BIDS target-word onset
            -> stim_file
            -> stim_key
    """

    matches = []
    used_event_rows = set()

    event_onsets = events[
        "target_onset_seconds"
    ].to_numpy(
        dtype=float
    )

    for retained_trial, urevent_index in enumerate(
        urevent_indices,
        start=1,
    ):
        urevent_seconds = extract_urevent_latency_seconds(
            file=file,
            condition_group=condition_group,
            urevent_index=urevent_index,
            sampling_frequency=sampling_frequency,
        )

        differences = np.abs(
            event_onsets
            - urevent_seconds
        )

        closest_position = int(
            np.argmin(
                differences
            )
        )

        closest_difference = float(
            differences[
                closest_position
            ]
        )

        if closest_difference > tolerance_seconds:
            raise ValueError(
                "Could not safely match retained ERP trial "
                f"{retained_trial}. Urevent {urevent_index} "
                f"occurs at {urevent_seconds:.6f} seconds, "
                "but the closest target-word onset is "
                f"{closest_difference:.6f} seconds away."
            )

        event_row = events.iloc[
            closest_position
        ]

        original_event_row = int(
            event_row[
                "original_event_row"
            ]
        )

        if original_event_row in used_event_rows:
            raise ValueError(
                "Two retained ERP trials matched the same "
                f"events.tsv row: {original_event_row}"
            )

        used_event_rows.add(
            original_event_row
        )

        matches.append(
            {
                "retained_trial": retained_trial,
                "urevent_index": urevent_index,
                "urevent_seconds": urevent_seconds,
                "event_time_difference_seconds": closest_difference,
                "original_event_row": original_event_row,
                "experimental_trial": int(
                    event_row[
                        "experimental_trial"
                    ]
                ),
                "trial_type": event_row[
                    "trial_type"
                ],
                "target_onset_seconds": float(
                    event_row[
                        "target_onset_seconds"
                    ]
                ),
                "stim_file": event_row[
                    "stim_file"
                ],
                "stim_key": event_row[
                    "stim_key"
                ],
            }
        )

    return pd.DataFrame(
        matches
    )

def load_trial_rejection_summary(
    mat_path: Path,
) -> pd.DataFrame:
    """
    Load the trial-rejection summary paired with an ERP MAT file.
    """

    rejection_path = mat_path.with_name(
        mat_path.stem
        + "_trialrej.tsv"
    )

    if not rejection_path.exists():
        raise FileNotFoundError(
            "Trial-rejection summary not found: "
            f"{rejection_path}"
        )

    rejection = pd.read_csv(
        rejection_path,
        sep="\t",
    )

    required_columns = [
        "condition",
        "before_trial_rejection",
        "after_trial_rejection",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rejection.columns
    ]

    if missing_columns:
        raise ValueError(
            "Trial-rejection table is missing "
            f"columns: {missing_columns}"
        )

    rejection = rejection.copy()

    if "#" in rejection.columns:
        rejection["condition_index"] = pd.to_numeric(
            rejection["#"],
            errors="raise",
        ).astype(int)
    else:
        rejection["condition_index"] = range(
            1,
            len(rejection) + 1,
        )

    rejection[
        "after_trial_rejection"
    ] = pd.to_numeric(
        rejection[
            "after_trial_rejection"
        ],
        errors="raise",
    ).astype(int)

    return rejection

def export_long_for_file(
    mat_path: Path,
    output_dir: Path,
    bids_root: Path,
):
    """
    Export one subject's CP ERP derivative to long-format TSV.

    For every retained ERP trial, this function:

    1. Reads the retained trial's EEGLAB eventurevent reference.
    2. Converts the corresponding urevent latency to seconds.
    3. Matches that latency to the final-word onset in events.tsv.
    4. Recovers stim_file and stim_key.
    5. Exports trial-level EEG data with channel metadata.
    6. Saves a retained-trial lookup for design-matrix construction.
    """

    subject = get_subject_id(
        mat_path
    )

    print(
        f"\nProcessing {mat_path.name}"
    )

    stimulus_lookup = load_stimulus_lookup(
        bids_root
    )

    events = load_subject_events(
        bids_root=bids_root,
        subject=subject,
        stimulus_lookup=stimulus_lookup,
    )

    rejection_summary = load_trial_rejection_summary(
        mat_path
    )

    with h5py.File(
        mat_path,
        "r",
    ) as file:

        if "ERPs" not in file:
            raise KeyError(
                "No ERPs dataset found."
            )

        if "fs" not in file:
            raise KeyError(
                "No sampling-frequency dataset 'fs' found."
            )

        erps = file["ERPs"]

        sampling_frequency = float(
            np.asarray(
                file["fs"]
            ).squeeze()
        )

        if (
            not np.isfinite(sampling_frequency)
            or sampling_frequency <= 0
        ):
            raise ValueError(
                "Invalid sampling frequency found in ERP MAT file: "
                f"{sampling_frequency}"
            )

        times = extract_times(
            file
        )

        n_conditions = erps.shape[1]

        if len(rejection_summary) != n_conditions:
            raise ValueError(
                "Number of rows in the trial-rejection table "
                "does not match the number of ERP conditions: "
                f"{len(rejection_summary)} versus {n_conditions}"
            )

        tolerance_seconds = (
            2.0
            / sampling_frequency
        )

        all_rows = []
        all_trial_lookups = []

        for condition_index in range(
            n_conditions
        ):
            condition_number = (
                condition_index + 1
            )

            rejection_row = rejection_summary.iloc[
                condition_index
            ]

            condition_label = str(
                rejection_row[
                    "condition"
                ]
            ).strip()

            expected_retained_trials = int(
                rejection_row[
                    "after_trial_rejection"
                ]
            )

            condition_group = extract_condition_object(
                file,
                erps,
                condition_index,
            )

            (
                data,
                n_trials,
                n_timepoints,
                n_channels,
            ) = extract_data(
                condition_group
            )

            if n_trials != expected_retained_trials:
                raise ValueError(
                    f"Condition {condition_number} "
                    f"({condition_label}) contains "
                    f"{n_trials} trials in the MAT file, "
                    "but the paired trial-rejection table reports "
                    f"{expected_retained_trials} retained trials."
                )

            if len(times) != n_timepoints:
                raise ValueError(
                    "Time vector length does not match the ERP "
                    f"time dimension for condition {condition_number}: "
                    f"{len(times)} versus {n_timepoints}"
                )

            urevent_indices = extract_epoch_urevent_indices(
                file=file,
                condition_group=condition_group,
                n_trials=n_trials,
            )

            trial_lookup = match_retained_trials_to_events(
                urevent_indices=urevent_indices,
                file=file,
                condition_group=condition_group,
                events=events,
                sampling_frequency=sampling_frequency,
                tolerance_seconds=tolerance_seconds,
            )

            if len(trial_lookup) != n_trials:
                raise ValueError(
                    f"Condition {condition_number} produced "
                    f"{len(trial_lookup)} trial matches for "
                    f"{n_trials} retained ERP trials."
                )

            trial_lookup.insert(
                0,
                "subject",
                subject,
            )

            trial_lookup.insert(
                1,
                "condition",
                condition_number,
            )

            trial_lookup.insert(
                2,
                "condition_label",
                condition_label,
            )

            all_trial_lookups.append(
                trial_lookup
            )

            channel_labels = extract_channel_labels(
                file,
                condition_group,
                n_channels,
            )

            channel_metadata = build_channel_metadata(
                bids_root=bids_root,
                subject=subject,
                channel_labels=channel_labels,
            )

            if len(channel_metadata) != n_channels:
                raise ValueError(
                    "Channel metadata count does not match the "
                    f"ERP channel count: {len(channel_metadata)} "
                    f"versus {n_channels}"
                )

            print(
                f"  Condition {condition_number}: "
                f"{condition_label} - "
                f"{n_trials} retained trials x "
                f"{n_timepoints} timepoints x "
                f"{n_channels} channels"
            )

            for trial_index in range(
                n_trials
            ):
                lookup_row = trial_lookup.iloc[
                    trial_index
                ]

                trial_data = data[
                    trial_index,
                    :,
                    :,
                ]

                if trial_data.shape != (
                    n_timepoints,
                    n_channels,
                ):
                    raise ValueError(
                        "Unexpected retained-trial data shape for "
                        f"condition {condition_number}, trial "
                        f"{trial_index + 1}: {trial_data.shape}"
                    )

                for channel_index in range(
                    n_channels
                ):
                    metadata = channel_metadata.iloc[
                        channel_index
                    ]

                    amplitudes = trial_data[
                        :,
                        channel_index,
                    ]

                    rows = pd.DataFrame(
                        {
                            "subject": subject,
                            "condition": condition_number,
                            "condition_label": condition_label,
                            "trial": trial_index + 1,
                            "retained_trial": int(
                                lookup_row[
                                    "retained_trial"
                                ]
                            ),
                            "urevent_index": int(
                                lookup_row[
                                    "urevent_index"
                                ]
                            ),
                            "urevent_seconds": float(
                                lookup_row[
                                    "urevent_seconds"
                                ]
                            ),
                            "event_time_difference_seconds": float(
                                lookup_row[
                                    "event_time_difference_seconds"
                                ]
                            ),
                            "original_event_row": int(
                                lookup_row[
                                    "original_event_row"
                                ]
                            ),
                            "experimental_trial": int(
                                lookup_row[
                                    "experimental_trial"
                                ]
                            ),
                            "trial_type": lookup_row[
                                "trial_type"
                            ],
                            "target_onset_seconds": float(
                                lookup_row[
                                    "target_onset_seconds"
                                ]
                            ),
                            "stim_file": lookup_row[
                                "stim_file"
                            ],
                            "stim_key": lookup_row[
                                "stim_key"
                            ],
                            "item": lookup_row[
                                "stim_key"
                            ],
                            "channel": metadata[
                                "channel"
                            ],
                            "original_channel": metadata[
                                "original_channel"
                            ],
                            "electrode": metadata[
                                "electrode"
                            ],
                            "standard_label": metadata.get(
                                "standard_label",
                                "",
                            ),
                            "x": metadata.get(
                                "x",
                                np.nan,
                            ),
                            "y": metadata.get(
                                "y",
                                np.nan,
                            ),
                            "z": metadata.get(
                                "z",
                                np.nan,
                            ),
                            "sph_theta": metadata.get(
                                "sph_theta",
                                np.nan,
                            ),
                            "sph_phi": metadata.get(
                                "sph_phi",
                                np.nan,
                            ),
                            "sph_radius": metadata.get(
                                "sph_radius",
                                np.nan,
                            ),
                            "theta": metadata.get(
                                "theta",
                                np.nan,
                            ),
                            "radius": metadata.get(
                                "radius",
                                np.nan,
                            ),
                            "channel_status": metadata.get(
                                "status",
                                np.nan,
                            ),
                            "channel_status_description": metadata.get(
                                "status_description",
                                np.nan,
                            ),
                            "time": times,
                            "amplitude": amplitudes,
                        }
                    )

                    all_rows.append(
                        rows
                    )

        if not all_rows:
            raise ValueError(
                f"No ERP rows were extracted for {subject}."
            )

        if not all_trial_lookups:
            raise ValueError(
                f"No retained-trial lookups were created for {subject}."
            )

        trial_lookup_all = pd.concat(
            all_trial_lookups,
            ignore_index=True,
        )

        duplicated_retained_trials = (
            trial_lookup_all
            .duplicated(
                subset=[
                    "condition",
                    "retained_trial",
                ],
                keep=False,
            )
        )

        if duplicated_retained_trials.any():
            examples = (
                trial_lookup_all.loc[
                    duplicated_retained_trials,
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
                "Duplicate condition/retained-trial mappings "
                f"were found. Examples: {examples}"
            )

        duplicated_event_matches = (
            trial_lookup_all
            .duplicated(
                subset=[
                    "original_event_row",
                ],
                keep=False,
            )
        )

        if duplicated_event_matches.any():
            examples = (
                trial_lookup_all.loc[
                    duplicated_event_matches,
                    [
                        "condition",
                        "retained_trial",
                        "original_event_row",
                        "stim_key",
                    ],
                ]
                .head(10)
                .to_dict(
                    "records"
                )
            )

            raise ValueError(
                "The same original events.tsv row was matched "
                "to more than one retained ERP trial. "
                f"Examples: {examples}"
            )

        if trial_lookup_all[
            "stim_key"
        ].isna().any():
            raise ValueError(
                f"{subject} contains retained trials with "
                "missing stim_key values."
            )

        maximum_time_difference = float(
            trial_lookup_all[
                "event_time_difference_seconds"
            ].max()
        )

        print(
            "  Maximum ERP-to-target-onset difference: "
            f"{maximum_time_difference:.6f} seconds"
        )

        out = pd.concat(
            all_rows,
            ignore_index=True,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        output_dir
        / f"{subject}_erp_long.tsv"
    )

    lookup_path = (
        output_dir
        / f"{subject}_trial_lookup.tsv"
    )

    out.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    trial_lookup_all.to_csv(
        lookup_path,
        sep="\t",
        index=False,
    )

    print(
        f"Saved ERP long file: {out_path}"
    )

    print(
        f"Saved retained-trial lookup: {lookup_path}"
    )

    return out_path

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Export ERP .mat files to long-format EEG TSV "
            "with stimulus identifiers and channel coordinates."
        )
    )

    parser.add_argument(
        "erp_root",
        help=(
            "Path to the ERP derivatives folder containing "
            "sub-*/*_erp-CP.mat files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="eeg_outputs",
        help=(
            "Folder where participant and combined EEG TSV "
            "files will be saved."
        ),
    )

    parser.add_argument(
        "--bids-root",
        default=".",
        help=(
            "Root BIDS dataset folder containing "
            "N400Stimset_stimuli_parameters.tsv, "
            "task-N400Stimset_electrodes.tsv, and "
            "sub-XX/eeg/*_channels.tsv and *_events.tsv files."
        ),
    )

    args = parser.parse_args()

    erp_root = Path(
        args.erp_root
    )

    output_dir = Path(
        args.output_dir
    )

    bids_root = Path(
        args.bids_root
    )

    if not erp_root.exists():
        raise FileNotFoundError(
            f"ERP root not found: {erp_root}"
        )

    if not bids_root.exists():
        raise FileNotFoundError(
            f"BIDS root not found: {bids_root}"
        )

    electrodes_path = (
        bids_root
        / "task-N400Stimset_electrodes.tsv"
    )

    if not electrodes_path.exists():
        raise FileNotFoundError(
            f"Electrodes file not found: {electrodes_path}"
        )

    stimulus_parameters_path = (
        bids_root
        / "N400Stimset_stimuli_parameters.tsv"
    )

    if not stimulus_parameters_path.exists():
        raise FileNotFoundError(
            "Stimulus parameters file not found: "
            f"{stimulus_parameters_path}"
        )

    mat_files = sorted(
        erp_root.glob(
            "sub-*/*_erp-CP.mat"
        )
    )

    if not mat_files:
        raise FileNotFoundError(
            "No *_erp-CP.mat files found under "
            f"{erp_root}"
        )

    print(
        f"Found {len(mat_files)} ERP CP files."
    )

    print(
        f"Using BIDS root: {bids_root}"
    )

    print(
        f"Using electrodes file: {electrodes_path}"
    )

    print(
        "Using stimulus parameters file: "
        f"{stimulus_parameters_path}"
    )

    output_files = []

    failed_files = []

    for mat_path in mat_files:

        try:
            out_path = export_long_for_file(
                mat_path=mat_path,
                output_dir=output_dir,
                bids_root=bids_root,
            )

            if out_path is not None:
                output_files.append(
                    out_path
                )

        except Exception as error:
            print(
                f"FAILED: {mat_path.name}: {error}"
            )

            failed_files.append(
                {
                    "file": str(mat_path),
                    "error": str(error),
                }
            )

    if not output_files:
        raise RuntimeError(
            "No participant ERP files were exported successfully."
        )

    print(
        "\nCombining subject TSV files safely in chunks..."
    )

    combined_path = (
        output_dir
        / "ALL_erp_long.tsv"
    )

    if combined_path.exists():
        combined_path.unlink()

    first_chunk = True

    for path in output_files:

        print(
            f"  Adding: {path.name}"
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=500_000,
        ):
            chunk.to_csv(
                combined_path,
                sep="\t",
                mode=(
                    "w"
                    if first_chunk
                    else "a"
                ),
                header=first_chunk,
                index=False,
            )

            first_chunk = False

    print(
        "Saved combined EEG long file: "
        f"{combined_path}"
    )

    if failed_files:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        failed_path = (
            output_dir
            / "failed_erp_exports.tsv"
        )

        pd.DataFrame(
            failed_files
        ).to_csv(
            failed_path,
            sep="\t",
            index=False,
        )

        print(
            f"Saved failed-export report: {failed_path}"
        )

        print(
            f"{len(failed_files)} ERP files failed."
        )

    print(
        f"{len(output_files)} ERP files exported successfully."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()