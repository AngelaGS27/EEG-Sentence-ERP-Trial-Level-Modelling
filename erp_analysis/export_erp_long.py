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
    """
    Extract the subject identifier from an ERP filename.
    """

    name = path.name

    if "_task-" in name:
        return name.split(
            "_task-",
            1,
        )[0]

    return path.stem


def get_analysis_name(path: Path) -> str:
    """
    Extract the ERP analysis name from the filename.

    Examples:
        sub-24_task-N400Stimset_erp-CP.mat
            -> CP

        sub-24_task-N400Stimset_erp-GA.mat
            -> GA

        sub-24_task-N400Stimset_erp-LD.mat
            -> LD

        sub-24_task-N400Stimset_erp-Order.mat
            -> Order

        sub-24_task-N400Stimset_erp-Time.mat
            -> Time
    """

    marker = "_erp-"

    if marker not in path.stem:
        raise ValueError(
            "Cannot identify ERP analysis from filename: "
            f"{path.name}"
        )

    return path.stem.split(
        marker,
        1,
    )[1]


def extract_condition_object(
    file,
    erps_dataset,
    condition_index,
):
    """
    Follow one ERP condition object reference.
    """

    reference = erps_dataset[
        0,
        condition_index,
    ]

    return file[
        reference
    ]


def extract_data(
    condition_group,
):
    """
    Extract ERP data.

    Expected shape:
        trials x timepoints x channels
    """

    if "data" not in condition_group:
        raise KeyError(
            "No 'data' field found in ERP condition."
        )

    data = np.asarray(
        condition_group[
            "data"
        ]
    )

    if data.ndim != 3:
        raise ValueError(
            "Expected three-dimensional ERP data, "
            f"but found shape {data.shape}."
        )

    n_trials = data.shape[0]
    n_timepoints = data.shape[1]
    n_channels = data.shape[2]

    return (
        data,
        n_trials,
        n_timepoints,
        n_channels,
    )


def extract_epoch_urevent_indices(
    file: h5py.File,
    condition_group,
    n_trials: int,
) -> list[int]:
    """
    Extract the original EEGLAB urevent index for every
    retained ERP epoch.

    The resulting indices preserve the retained epoch order
    stored inside the ERP MAT file.
    """

    if "epoch" not in condition_group:
        raise KeyError(
            "No epoch structure found in ERP condition."
        )

    epoch_group = condition_group[
        "epoch"
    ]

    if "eventurevent" not in epoch_group:
        raise KeyError(
            "No epoch/eventurevent field found in ERP condition."
        )

    eventurevent = epoch_group[
        "eventurevent"
    ]

    if eventurevent.shape[0] != n_trials:
        raise ValueError(
            "eventurevent row count does not match retained "
            f"trials: {eventurevent.shape[0]} versus {n_trials}"
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
            np.isfinite(
                values
            )
        ]

        if len(
            finite_values
        ) == 0:
            raise ValueError(
                "No valid urevent index for retained trial "
                f"{trial_index + 1}."
            )

        urevent_index = int(
            round(
                float(
                    finite_values[0]
                )
            )
        )

        if urevent_index < 1:
            raise ValueError(
                "Invalid EEGLAB urevent index: "
                f"{urevent_index}"
            )

        indices.append(
            urevent_index
        )

    return indices


def extract_times(
    file,
):
    """
    Extract the ERP time vector.

    Converts milliseconds to seconds when the stored values
    are clearly in milliseconds.
    """

    if "t" not in file:
        raise KeyError(
            "No time vector 't' found in ERP MAT file."
        )

    times = np.asarray(
        file[
            "t"
        ]
    ).squeeze().astype(
        float
    )

    if times.ndim != 1:
        raise ValueError(
            "ERP time vector is not one-dimensional: "
            f"{times.shape}"
        )

    if len(
        times
    ) == 0:
        raise ValueError(
            "ERP time vector is empty."
        )

    if not np.isfinite(
        times
    ).all():
        raise ValueError(
            "ERP time vector contains missing or infinite values."
        )

    if np.nanmax(
        np.abs(
            times
        )
    ) > 10:
        times = (
            times
            / 1000.0
        )

    return times


def make_biosemi_128_label(
    index: int,
) -> str:
    """
    Convert a zero-based channel index to a BioSemi 128 label.

    Examples:
        0   -> A1
        31  -> A32
        32  -> B1
        63  -> B32
        64  -> C1
        95  -> C32
        96  -> D1
        127 -> D32
    """

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    if (
        index < 0
        or index >= 128
    ):
        return (
            f"ch_{index + 1:03d}"
        )

    letter = letters[
        index // 32
    ]

    number = (
        index % 32
    ) + 1

    return (
        f"{letter}{number}"
    )


def extract_channel_labels(
    file,
    condition_group,
    n_channels,
):
    """
    Extract channel labels from the ERP MAT file.

    If labels cannot be read, use BioSemi 128 acquisition
    labels in channel order.
    """

    labels = []

    try:
        chanlocs = condition_group[
            "chanlocs"
        ]

        if "labels" in chanlocs:
            label_references = chanlocs[
                "labels"
            ]

            for channel_index in range(
                label_references.shape[0]
            ):
                reference = label_references[
                    channel_index,
                    0,
                ]

                labels.append(
                    decode_matlab_string(
                        file,
                        reference,
                    )
                )

    except Exception:
        labels = []

    invalid_labels = (
        len(
            labels
        ) != n_channels
        or any(
            not str(
                label
            ).strip()
            for label in labels
        )
    )

    if invalid_labels:
        labels = [
            make_biosemi_128_label(
                channel_index
            )
            for channel_index in range(
                n_channels
            )
        ]

    return labels


def clean_electrode_name(
    value,
):
    """
    Clean an electrode label.
    """

    value = (
        str(
            value
        )
        .strip()
        .strip("'")
        .strip('"')
    )

    if "_" in value:
        value = value.split(
            "_",
            1,
        )[0]

    return value.strip()


def split_channel_name(
    channel_name,
):
    """
    Split labels such as A1_Cz into:

        electrode = A1
        standard_label = Cz

    Labels without an underscore retain only the acquisition
    electrode name.
    """

    channel_name = str(
        channel_name
    ).strip()

    if "_" in channel_name:
        electrode, standard_label = channel_name.split(
            "_",
            1,
        )

        return (
            clean_electrode_name(
                electrode
            ),
            standard_label.strip(),
        )

    return (
        clean_electrode_name(
            channel_name
        ),
        "",
    )


def build_channel_metadata(
    channel_labels: list[str],
) -> pd.DataFrame:
    """
    Build channel metadata directly from labels stored in the
    ERP MAT file.

    No external channels.tsv, electrodes.tsv, or BIDS directory
    is required.
    """

    rows = []

    for (
        channel_index,
        channel_label,
    ) in enumerate(
        channel_labels
    ):
        channel_label = str(
            channel_label
        ).strip()

        if (
            not channel_label
            or channel_label.lower().startswith(
                "ch_"
            )
        ):
            channel_label = make_biosemi_128_label(
                channel_index
            )

        (
            electrode,
            standard_label,
        ) = split_channel_name(
            channel_label
        )

        channel = (
            standard_label
            if standard_label
            else electrode
        )

        rows.append(
            {
                "channel_index": channel_index,
                "mat_channel": channel_label,
                "channel": channel,
                "original_channel": channel_label,
                "electrode": electrode,
                "standard_label": standard_label,
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "sph_theta": np.nan,
                "sph_phi": np.nan,
                "sph_radius": np.nan,
                "theta": np.nan,
                "radius": np.nan,
                "type": "EEG",
                "units": np.nan,
                "status": np.nan,
                "status_description": np.nan,
            }
        )

    channel_metadata = pd.DataFrame(
        rows
    )

    if len(
        channel_metadata
    ) != len(
        channel_labels
    ):
        raise ValueError(
            "Channel metadata count does not match "
            "the number of channel labels."
        )

    return channel_metadata


def get_trial_rejection_path(
    mat_path: Path,
) -> Path:
    """
    Construct the exact paired trial-rejection TSV path.

    Examples:

        sub-24_task-N400Stimset_erp-CP.mat
        sub-24_task-N400Stimset_erp-CP_trialrej.tsv

        sub-24_task-N400Stimset_erp-GA.mat
        sub-24_task-N400Stimset_erp-GA_trialrej.tsv

        sub-24_task-N400Stimset_erp-LD.mat
        sub-24_task-N400Stimset_erp-LD_trialrej.tsv

        sub-24_task-N400Stimset_erp-Order.mat
        sub-24_task-N400Stimset_erp-Order_trialrej.tsv

        sub-24_task-N400Stimset_erp-Time.mat
        sub-24_task-N400Stimset_erp-Time_trialrej.tsv
    """

    return mat_path.with_name(
        f"{mat_path.stem}_trialrej.tsv"
    )


def load_trial_rejection_summary(
    mat_path: Path,
) -> pd.DataFrame:
    """
    Load the trial-rejection TSV paired with one ERP MAT file.

    Required columns:

        condition
        before_trial_rejection
        after_trial_rejection

    An optional '#' column is used as the condition index when
    present.
    """

    rejection_path = get_trial_rejection_path(
        mat_path
    )

    if not rejection_path.exists():
        raise FileNotFoundError(
            "Trial-rejection TSV not found for ERP MAT file:\n"
            f"MAT: {mat_path}\n"
            f"Expected TSV: {rejection_path}"
        )

    if not rejection_path.is_file():
        raise FileNotFoundError(
            "Trial-rejection path is not a file:\n"
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
            f"{rejection_path} is missing required columns: "
            f"{missing_columns}"
        )

    rejection = rejection.copy()

    rejection["condition"] = (
        rejection[
            "condition"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    missing_condition = (
        rejection[
            "condition"
        ].isna()
        | rejection[
            "condition"
        ].eq(
            ""
        )
    )

    if missing_condition.any():
        raise ValueError(
            f"{rejection_path} contains missing or empty "
            "condition labels."
        )

    if "#" in rejection.columns:
        rejection[
            "condition_index"
        ] = pd.to_numeric(
            rejection[
                "#"
            ],
            errors="raise",
        ).astype(
            int
        )
    else:
        rejection[
            "condition_index"
        ] = range(
            1,
            len(
                rejection
            ) + 1,
        )

    for column in [
        "before_trial_rejection",
        "after_trial_rejection",
    ]:
        rejection[
            column
        ] = pd.to_numeric(
            rejection[
                column
            ],
            errors="raise",
        ).astype(
            int
        )

        if (
            rejection[
                column
            ] < 0
        ).any():
            raise ValueError(
                f"{rejection_path} contains negative values "
                f"in {column}."
            )

    impossible_counts = (
        rejection[
            "after_trial_rejection"
        ]
        > rejection[
            "before_trial_rejection"
        ]
    )

    if impossible_counts.any():
        examples = (
            rejection.loc[
                impossible_counts,
                [
                    "condition",
                    "before_trial_rejection",
                    "after_trial_rejection",
                ],
            ]
            .head(
                10
            )
            .to_dict(
                "records"
            )
        )

        raise ValueError(
            "Some after_trial_rejection counts are larger than "
            "their before_trial_rejection counts. "
            f"Examples: {examples}"
        )

    rejection[
        "analysis"
    ] = get_analysis_name(
        mat_path
    )

    rejection[
        "trial_rejection_file"
    ] = str(
        rejection_path
    )

    return rejection


def export_long_for_file(
    mat_path: Path,
    output_dir: Path,
):
    """
    Export one ERP MAT file using its exact paired
    *_trialrej.tsv summary.

    The trial-rejection TSV contains condition labels and
    before/after trial counts.

    It does not contain stim_file or stim_key. Therefore this
    exporter does not invent stimulus identities or attempt to
    use nonexistent events.tsv files.

    The exporter preserves:

        subject
        ERP analysis
        condition number
        condition label
        retained-trial order
        urevent index
        channel
        time
        amplitude
    """

    subject = get_subject_id(
        mat_path
    )

    analysis = get_analysis_name(
        mat_path
    )

    rejection_summary = load_trial_rejection_summary(
        mat_path
    )

    rejection_path = get_trial_rejection_path(
        mat_path
    )

    print(
        f"\nProcessing {mat_path.name}"
    )

    print(
        f"  Using {rejection_path.name}"
    )

    with h5py.File(
        mat_path,
        "r",
    ) as file:
        if "ERPs" not in file:
            raise KeyError(
                "No ERPs dataset found in MAT file."
            )

        erps = file[
            "ERPs"
        ]

        times = extract_times(
            file
        )

        if erps.ndim != 2:
            raise ValueError(
                "ERPs dataset has an unexpected shape: "
                f"{erps.shape}"
            )

        n_conditions = erps.shape[1]

        if len(
            rejection_summary
        ) != n_conditions:
            raise ValueError(
                "Trial-rejection row count does not match "
                "the number of ERP conditions: "
                f"{len(rejection_summary)} versus "
                f"{n_conditions}"
            )

        all_rows = []
        all_trial_lookups = []

        for condition_index in range(
            n_conditions
        ):
            condition_number = (
                condition_index
                + 1
            )

            rejection_row = rejection_summary.iloc[
                condition_index
            ]

            condition_label = str(
                rejection_row[
                    "condition"
                ]
            ).strip()

            before_trials = int(
                rejection_row[
                    "before_trial_rejection"
                ]
            )

            expected_retained_trials = int(
                rejection_row[
                    "after_trial_rejection"
                ]
            )

            condition_group = extract_condition_object(
                file=file,
                erps_dataset=erps,
                condition_index=condition_index,
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
                    f"{n_trials} retained trials in the MAT file, "
                    f"but {rejection_path.name} reports "
                    f"{expected_retained_trials}."
                )

            if len(
                times
            ) != n_timepoints:
                raise ValueError(
                    "Time-vector length does not match ERP "
                    f"timepoints for condition {condition_number}: "
                    f"{len(times)} versus {n_timepoints}"
                )

            urevent_indices = extract_epoch_urevent_indices(
                file=file,
                condition_group=condition_group,
                n_trials=n_trials,
            )

            if len(
                urevent_indices
            ) != n_trials:
                raise ValueError(
                    "Number of extracted urevent indices does not "
                    f"match retained trials: {len(urevent_indices)} "
                    f"versus {n_trials}"
                )

            trial_lookup = pd.DataFrame(
                {
                    "subject": subject,
                    "analysis": analysis,
                    "condition": condition_number,
                    "condition_label": condition_label,
                    "before_trial_rejection": before_trials,
                    "after_trial_rejection": expected_retained_trials,
                    "retained_trial": np.arange(
                        1,
                        n_trials + 1,
                        dtype=int,
                    ),
                    "urevent_index": urevent_indices,
                }
            )

            trial_lookup[
                "epoch_id"
            ] = (
                trial_lookup[
                    "subject"
                ].astype(
                    str
                )
                + "_"
                + trial_lookup[
                    "analysis"
                ].astype(
                    str
                )
                + "_c"
                + trial_lookup[
                    "condition"
                ].astype(
                    str
                )
                + "_r"
                + trial_lookup[
                    "retained_trial"
                ].astype(
                    str
                )
            )

            duplicated_epochs = trial_lookup.duplicated(
                subset=[
                    "epoch_id",
                ],
                keep=False,
            )

            if duplicated_epochs.any():
                raise ValueError(
                    "Duplicate epoch_id values were created for "
                    f"{mat_path.name}, condition {condition_number}."
                )

            all_trial_lookups.append(
                trial_lookup
            )

            channel_labels = extract_channel_labels(
                file=file,
                condition_group=condition_group,
                n_channels=n_channels,
            )

            channel_metadata = build_channel_metadata(
                channel_labels
            )

            if len(
                channel_metadata
            ) != n_channels:
                raise ValueError(
                    "Channel metadata count does not match ERP "
                    f"channel count: {len(channel_metadata)} "
                    f"versus {n_channels}"
                )

            print(
                f"  Condition {condition_number}: "
                f"{condition_label} - "
                f"{before_trials} before rejection, "
                f"{n_trials} retained, "
                f"{n_timepoints} timepoints, "
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

                expected_shape = (
                    n_timepoints,
                    n_channels,
                )

                if trial_data.shape != expected_shape:
                    raise ValueError(
                        "Unexpected trial shape "
                        f"{trial_data.shape} for condition "
                        f"{condition_number}, retained trial "
                        f"{trial_index + 1}. Expected "
                        f"{expected_shape}."
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

                    if len(
                        amplitudes
                    ) != len(
                        times
                    ):
                        raise ValueError(
                            "Amplitude count does not match the "
                            "time-vector length for condition "
                            f"{condition_number}, retained trial "
                            f"{trial_index + 1}, channel "
                            f"{channel_index + 1}."
                        )

                    rows = pd.DataFrame(
                        {
                            "subject": subject,
                            "analysis": analysis,
                            "condition": condition_number,
                            "condition_label": condition_label,
                            "before_trial_rejection": before_trials,
                            "after_trial_rejection": (
                                expected_retained_trials
                            ),
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
                            "epoch_id": lookup_row[
                                "epoch_id"
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
                            "channel_status_description": (
                                metadata.get(
                                    "status_description",
                                    np.nan,
                                )
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
                "No ERP rows were extracted from "
                f"{mat_path.name}."
            )

        if not all_trial_lookups:
            raise ValueError(
                "No retained-trial lookup rows were created for "
                f"{mat_path.name}."
            )

        output_table = pd.concat(
            all_rows,
            ignore_index=True,
        )

        trial_lookup_all = pd.concat(
            all_trial_lookups,
            ignore_index=True,
        )

        duplicated_lookup_rows = trial_lookup_all.duplicated(
            subset=[
                "subject",
                "analysis",
                "condition",
                "retained_trial",
            ],
            keep=False,
        )

        if duplicated_lookup_rows.any():
            examples = (
                trial_lookup_all.loc[
                    duplicated_lookup_rows,
                    [
                        "subject",
                        "analysis",
                        "condition",
                        "retained_trial",
                    ],
                ]
                .head(
                    10
                )
                .to_dict(
                    "records"
                )
            )

            raise ValueError(
                "Duplicate retained-trial lookup rows were found. "
                f"Examples: {examples}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_analysis = analysis.replace(
        "/",
        "_",
    )

    output_path = (
        output_dir
        / (
            f"{subject}_erp-"
            f"{safe_analysis}_long.tsv"
        )
    )

    lookup_path = (
        output_dir
        / (
            f"{subject}_erp-"
            f"{safe_analysis}_trial_lookup.tsv"
        )
    )

    output_table.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    trial_lookup_all.to_csv(
        lookup_path,
        sep="\t",
        index=False,
    )

    print(
        f"Saved ERP long file: {output_path}"
    )

    print(
        f"Saved retained-trial lookup: {lookup_path}"
    )

    return output_path


def discover_erp_mat_files(
    erp_root: Path,
    analyses: list[str],
) -> list[Path]:
    """
    Find ERP MAT files for the requested analyses under the
    subject subfolders.

    Expected structure:

        erp_root/
            sub-01/
                sub-01_task-N400Stimset_erp-CP.mat
                sub-01_task-N400Stimset_erp-CP_trialrej.tsv
            sub-02/
                ...
    """

    mat_files = []

    for analysis in analyses:
        pattern = (
            f"sub-*/*_erp-{analysis}.mat"
        )

        mat_files.extend(
            erp_root.glob(
                pattern
            )
        )

    return sorted(
        set(
            mat_files
        )
    )


def parse_analysis_list(
    value: str,
) -> list[str]:
    """
    Parse the --analyses argument.

    Valid values:

        CP
        GA
        LD
        Order
        Time
        ALL

    Multiple analyses may be comma-separated.
    """

    valid_analyses = [
        "CP",
        "GA",
        "LD",
        "Order",
        "Time",
    ]

    cleaned_value = str(
        value
    ).strip()

    if cleaned_value.upper() == "ALL":
        return valid_analyses

    requested_analyses = [
        item.strip()
        for item in cleaned_value.split(
            ","
        )
        if item.strip()
    ]

    if not requested_analyses:
        raise ValueError(
            "No ERP analyses were requested."
        )

    normalised_lookup = {
        analysis.lower(): analysis
        for analysis in valid_analyses
    }

    normalised_analyses = []
    invalid_analyses = []

    for requested_analysis in requested_analyses:
        matched_analysis = normalised_lookup.get(
            requested_analysis.lower()
        )

        if matched_analysis is None:
            invalid_analyses.append(
                requested_analysis
            )
        else:
            normalised_analyses.append(
                matched_analysis
            )

    if invalid_analyses:
        raise ValueError(
            f"Unknown ERP analyses: {invalid_analyses}. "
            f"Valid values are {valid_analyses} or ALL."
        )

    return list(
        dict.fromkeys(
            normalised_analyses
        )
    )


def main():
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Export N400Stimset ERP MAT files to long-format "
            "TSV using the exact matching "
            "*_erp-<analysis>_trialrej.tsv file in each "
            "subject folder."
        )
    )

    parser.add_argument(
        "erp_root",
        help=(
            "ERP root folder containing sub-* subject "
            "subfolders."
        ),
    )

    parser.add_argument(
        "--analyses",
        default="CP",
        help=(
            "Comma-separated ERP analyses: "
            "CP,GA,LD,Order,Time, or ALL. "
            "Default: CP."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="eeg_outputs",
        help=(
            "Directory where participant and combined ERP "
            "TSV files will be saved."
        ),
    )

    args = parser.parse_args()

    erp_root = Path(
        args.erp_root
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    analyses = parse_analysis_list(
        args.analyses
    )

    if not erp_root.exists():
        raise FileNotFoundError(
            f"ERP root not found: {erp_root}"
        )

    if not erp_root.is_dir():
        raise NotADirectoryError(
            "ERP root is not a directory: "
            f"{erp_root}"
        )

    mat_files = discover_erp_mat_files(
        erp_root=erp_root,
        analyses=analyses,
    )

    if not mat_files:
        searched_patterns = "\n".join(
            str(
                erp_root
                / "sub-*"
                / f"*_erp-{analysis}.mat"
            )
            for analysis in analyses
        )

        raise FileNotFoundError(
            "No matching ERP MAT files were found. "
            "Searched:\n"
            f"{searched_patterns}"
        )

    missing_rejection_files = [
        get_trial_rejection_path(
            mat_path
        )
        for mat_path in mat_files
        if not get_trial_rejection_path(
            mat_path
        ).exists()
    ]

    if missing_rejection_files:
        examples = "\n".join(
            str(
                path
            )
            for path in missing_rejection_files[
                :20
            ]
        )

        raise FileNotFoundError(
            "Some ERP MAT files do not have their exact "
            "matching *_trialrej.tsv file:\n"
            f"{examples}"
        )

    print(
        f"Found {len(mat_files)} ERP MAT files."
    )

    print(
        f"Using ERP root: {erp_root}"
    )

    print(
        "Analyses: "
        + ", ".join(
            analyses
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = []
    failed_files = []

    for mat_path in mat_files:
        try:
            output_path = export_long_for_file(
                mat_path=mat_path,
                output_dir=output_dir,
            )

            output_files.append(
                output_path
            )

        except Exception as error:
            print(
                f"FAILED: {mat_path.name}: {error}"
            )

            failed_files.append(
                {
                    "file": str(
                        mat_path
                    ),
                    "error": str(
                        error
                    ),
                }
            )

    if failed_files:
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
            "Saved failed-export report: "
            f"{failed_path}"
        )

    if not output_files:
        raise RuntimeError(
            "No ERP MAT files were exported successfully."
        )

    combined_path = (
        output_dir
        / "ALL_erp_long.tsv"
    )

    if combined_path.exists():
        combined_path.unlink()

    print(
        "\nCombining exported TSV files in chunks..."
    )

    first_chunk = True

    for output_path in output_files:
        print(
            f"  Adding: {output_path.name}"
        )

        for chunk in pd.read_csv(
            output_path,
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
        "Saved combined ERP long file: "
        f"{combined_path}"
    )

    print(
        f"{len(output_files)} ERP MAT files "
        "exported successfully."
    )

    if failed_files:
        print(
            f"{len(failed_files)} ERP MAT files failed."
        )

    print(
        "Done."
    )


if __name__ == "__main__":
    main()