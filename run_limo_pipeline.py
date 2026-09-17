from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

try:
    from scipy.io import savemat
except ImportError as exc:
    raise SystemExit(
        "This script requires scipy to create temporary MATLAB input files. "
        "Install it with: pip install scipy"
    ) from exc


# ---------------------------------------------------------------------
# Metadata columns excluded from continuous predictor selection.
# ---------------------------------------------------------------------

DEFAULT_EXCLUDE_COLUMNS = {
    "subject",
    "participant_id",
    "analysis",
    "source_file",
    "source_path",
    "condition",
    "condition_label",
    "before_trial_rejection",
    "after_trial_rejection_reported",
    "after_trial_rejection_mat",
    "trial_type",
    "stim_key",
    "stim_file",
    "stimulus",
    "stimulus_row",
    "sentence_id",
    "item",
    "trial",
    "retained_trial",
    "eeg_trial",
    "epoch_id",
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


DESIGN_FILENAME_RE = re.compile(
    r"^(?P<subject>sub-\d+)_erp-(?P<analysis>.+)_design_matrix\.tsv$"
)


# ---------------------------------------------------------------------
# Predictor set defined in the preprocessing and predictor-diagnostics
# workflow.
#
# target_n_syllables is retained here because it is part of the defined
# diagnostic predictor set. It is skipped automatically when absent from
# a design matrix.
# ---------------------------------------------------------------------

DIAGNOSTIC_PREDICTORS = [
    "human_cp",
    "llm_cp",
    "target_surprisal_bits",
    "context_target_similarity",
    "target_zipf_frequency",
    "target_n_letters",
    "target_n_phonemes",
    "target_n_syllables",
    "syntax_mean_dependency_distance",
    "syntax_max_parse_depth",
    "syntax_n_subordinate_clauses",
    "target_valence",
    "target_arousal",
]


# =====================================================================
# MATLAB helper: first-level LIMO analysis
# =====================================================================

MATLAB_FIRST_LEVEL = r"""
function run_limo_first_level(input_file, output_dir, limo_tools_dir, chanlocs_json, source_file, source_dir, method)

    addpath(genpath(limo_tools_dir));

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    S = load(input_file);

    Y = double(S.Y);
    Cont = double(S.Cont);
    times = double(S.times(:)');
    sampling_rate = double(S.sampling_rate(1));

    % Continuous-regression model without a categorical condition term.
    Cat = 0;

    chanlocs = jsondecode(fileread(chanlocs_json));

    if ~isempty(chanlocs)
        chanlocs = chanlocs(:)';
    end

    predictor_names = cellstr(string(S.predictor_names));

    LIMO = struct();

    LIMO.dir = output_dir;

    LIMO.data = struct();
    LIMO.data.data = source_file;
    LIMO.data.data_dir = source_dir;
    LIMO.data.sampling_rate = sampling_rate;
    LIMO.data.start = times(1);
    LIMO.data.end = times(end);
    LIMO.data.trim1 = 1;
    LIMO.data.trim2 = numel(times);
    LIMO.data.timevect = times;
    LIMO.data.chanlocs = chanlocs;
    LIMO.data.Cat = Cat;
    LIMO.data.Cont = Cont;

    LIMO.Analysis = 'Time';
    LIMO.Type = 'Channels';
    LIMO.Level = 1;

    LIMO.design = struct();

    % Predictors in the design matrices are already standardized.
    LIMO.design.zscore = 0;

    LIMO.design.method = method;
    LIMO.design.type_of_analysis = 'Mass-univariate';
    LIMO.design.fullfactorial = 0;
    LIMO.design.bootstrap = 0;
    LIMO.design.tfce = 0;
    LIMO.design.status = 'to do';

    cd(output_dir);

    [X, nb_conditions, nb_interactions, nb_continuous] = ...
        limo_design_matrix(Y, LIMO, 0);

    LIMO.design.X = X;
    LIMO.design.nb_conditions = nb_conditions;
    LIMO.design.nb_interactions = nb_interactions;
    LIMO.design.nb_continuous = nb_continuous;

    if nb_continuous == 1
        LIMO.design.name = 'GLM Continuous: Simple Regression';
    else
        LIMO.design.name = sprintf( ...
            'GLM Continuous: Multiple Regression with %g continuous variables', ...
            nb_continuous);
    end

    % With Cat = 0, continuous predictors occur first and the constant
    % is appended at the end of the design matrix.
    nlabels = numel(predictor_names) + 1;

    labels = repmat(struct('description', ''), 1, nlabels);

    for k = 1:numel(predictor_names)
        labels(k).description = predictor_names{k};
    end

    labels(end).description = 'constant';

    LIMO.design.labels = labels;

    save(fullfile(output_dir, 'LIMO.mat'), 'LIMO', '-v7.3');

    % Official LIMO first-level mass-univariate analysis.
    limo_eeg(4, fullfile(output_dir, 'LIMO.mat'));

end
"""


# =====================================================================
# MATLAB helper: build expected channel layout for group inference
# =====================================================================

MATLAB_EXPECTED_CHANLOCS = r"""
function build_limo_expected_chanlocs(limo_file, output_file, limo_tools_dir, neighbour_distance)

    addpath(genpath(limo_tools_dir));

    S = load(limo_file);
    LIMO = S.LIMO;

    expected_chanlocs = LIMO.data.chanlocs;

    EEG = struct();

    EEG.chanlocs = expected_chanlocs;
    EEG.nbchan = numel(expected_chanlocs);
    EEG.trials = 1;
    EEG.pnts = 1;
    EEG.srate = LIMO.data.sampling_rate;
    EEG.xmin = 0;
    EEG.xmax = 0;
    EEG.data = zeros(EEG.nbchan, 1, 1);

    [~, channeighbstructmat] = ...
        limo_get_channeighbstructmat(EEG, neighbour_distance);

    if isempty(channeighbstructmat) || ...
            sum(channeighbstructmat(:)) == 0

        error( ...
            ['LIMO neighbouring matrix is empty. ' ...
             'Check neighbour distance and channel coordinates.']);
    end

    save( ...
        output_file, ...
        'expected_chanlocs', ...
        'channeighbstructmat', ...
        '-v7.3');

end
"""


# =====================================================================
# MATLAB helper: second-level group analysis
# =====================================================================

MATLAB_SECOND_LEVEL = r"""
function run_limo_group_level(beta_list_file, expected_chanlocs_file, beta_index, output_dir, limo_tools_dir, nboot, tfce)

    addpath(genpath(limo_tools_dir));

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    lines = readlines(beta_list_file);

    lines = strip(lines);

    lines(lines == "") = [];

    beta_files = cellstr(lines);

    if numel(beta_files) < 2
        error( ...
            ['At least two subject beta files are required ' ...
             'for a one-sample second-level t-test.']);
    end

    old_dir = pwd;

    cleanupObj = onCleanup(@() cd(old_dir));

    cd(output_dir);

    limo_random_select( ...
        'one sample t-test', ...
        expected_chanlocs_file, ...
        'LIMOfiles', beta_files, ...
        'analysis_type', 'Full scalp analysis', ...
        'parameters', {double(beta_index)}, ...
        'type', 'Channels', ...
        'nboot', double(nboot), ...
        'tfce', double(tfce), ...
        'skip design check', 'Yes');

end
"""


# =====================================================================
# General helpers
# =====================================================================

def matlab_quote(value: str | Path) -> str:
    """Return a MATLAB single-quoted string literal."""

    text = str(value).replace("'", "''")

    return f"'{text}'"


def normalise_subject_id(value: str) -> str:

    value = str(value).strip()

    if value.isdigit():
        return f"sub-{int(value):02d}"

    if not value.startswith("sub-"):
        return f"sub-{value}"

    return value


def parse_csv_arg(value: str | None) -> list[str] | None:

    if value is None:
        return None

    output = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return list(dict.fromkeys(output)) or None


# =====================================================================
# Design-matrix handling
# =====================================================================

def load_design_matrix(path: Path) -> pd.DataFrame:

    design = pd.read_csv(path, sep="\t")

    if design.empty:
        raise ValueError(
            f"Design matrix is empty: {path}"
        )

    required = {
        "subject",
        "analysis",
        "source_file",
        "source_path",
        "condition",
        "eeg_trial",
        "urevent_index",
        "stim_key",
    }

    missing = sorted(
        required.difference(design.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    design = design.copy()

    design["eeg_trial"] = pd.to_numeric(
        design["eeg_trial"],
        errors="raise",
    ).astype(int)

    design["urevent_index"] = pd.to_numeric(
        design["urevent_index"],
        errors="raise",
    ).astype(int)

    design = design.sort_values(
        "eeg_trial",
        kind="stable",
    ).reset_index(drop=True)

    expected = np.arange(
        1,
        len(design) + 1,
        dtype=int,
    )

    actual = design[
        "eeg_trial"
    ].to_numpy(dtype=int)

    if not np.array_equal(
        expected,
        actual,
    ):
        raise ValueError(
            f"{path.name}: eeg_trial must be a consecutive "
            f"sequence from 1 to {len(design)}."
        )

    if design[
        "eeg_trial"
    ].duplicated().any():

        raise ValueError(
            f"{path.name}: duplicate eeg_trial values found."
        )

    if {
        "condition",
        "retained_trial",
    }.issubset(design.columns):

        if design.duplicated(
            ["condition", "retained_trial"]
        ).any():

            raise ValueError(
                f"{path.name}: duplicate condition/retained_trial "
                "combinations found."
            )

    subjects = (
        design["subject"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    analyses = (
        design["analysis"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    sources = (
        design["source_file"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if (
        len(subjects) != 1
        or len(analyses) != 1
        or len(sources) != 1
    ):

        raise ValueError(
            f"{path.name}: expected one subject, one analysis, "
            f"and one source_file; got "
            f"subjects={subjects}, "
            f"analyses={analyses}, "
            f"sources={sources}."
        )

    return design


def choose_predictors(
    design: pd.DataFrame,
    requested: list[str] | None,
) -> list[str]:

    if requested is None:

        candidates = [
            column
            for column in DIAGNOSTIC_PREDICTORS
            if column in design.columns
        ]

        missing_predictors = [
            column
            for column in DIAGNOSTIC_PREDICTORS
            if column not in design.columns
        ]

        if missing_predictors:

            print(
                "Diagnostic predictors absent from this design matrix "
                "and skipped: "
                + ", ".join(missing_predictors)
            )

    else:

        missing = [
            column
            for column in requested
            if column not in design.columns
        ]

        if missing:

            raise ValueError(
                f"Requested predictors missing from design matrix: "
                f"{missing}"
            )

        candidates = requested

    selected: list[str] = []

    unusable: list[str] = []

    for column in candidates:

        numeric = pd.to_numeric(
            design[column],
            errors="coerce",
        )

        if (
            numeric.notna().sum() == 0
            or numeric.nunique(
                dropna=True
            ) < 2
        ):

            unusable.append(column)

            continue

        selected.append(column)

    if requested is not None and unusable:

        raise ValueError(
            f"Explicitly requested predictors are unusable: "
            f"{unusable}"
        )

    if unusable and requested is None:

        print(
            "Unusable default predictors skipped: "
            + ", ".join(unusable)
        )

    if not selected:

        raise ValueError(
            "No usable numeric scientific predictors found."
        )

    return selected


# =====================================================================
# ERP file resolution
# =====================================================================

def resolve_erp_path(
    design: pd.DataFrame,
    design_path: Path,
    erp_root: Path | None,
) -> Path:

    source_paths = (
        design["source_path"]
        .astype("string")
        .str.strip()
        .dropna()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )

    if len(source_paths) == 1:

        stored = Path(
            source_paths[0]
        ).expanduser()

        if stored.exists():
            return stored.resolve()

    source_files = (
        design["source_file"]
        .astype("string")
        .str.strip()
        .dropna()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )

    if len(source_files) != 1:

        raise ValueError(
            f"{design_path.name}: source_file is not unique: "
            f"{source_files}"
        )

    if erp_root is None:

        raise FileNotFoundError(
            f"{design_path.name}: stored source_path does not exist. "
            "Provide --erp-root pointing to the directory containing "
            "the ERP .mat files."
        )

    matches = list(
        erp_root.rglob(
            source_files[0]
        )
    )

    if len(matches) == 0:

        raise FileNotFoundError(
            f"Could not find {source_files[0]} "
            f"under {erp_root}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"More than one ERP file named "
            f"{source_files[0]} found under {erp_root}: "
            f"{matches}"
        )

    return matches[0].resolve()


# =====================================================================
# MATLAB HDF5 decoding
# =====================================================================

def decode_matlab_string(
    h5: h5py.File,
    value: Any,
) -> str:

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="ignore",
        ).strip()

    if isinstance(
        value,
        str,
    ):
        return value.strip()

    if isinstance(
        value,
        h5py.Reference,
    ):

        if not value:
            return ""

        return decode_matlab_string(
            h5,
            np.asarray(
                h5[value]
            ).squeeze(),
        )

    array = np.asarray(
        value
    ).squeeze()

    if array.size == 0:
        return ""

    if array.dtype.kind in {
        "u",
        "i",
    }:

        return "".join(
            chr(int(x))
            for x in np.ravel(array)
            if int(x) != 0
        ).strip()

    if array.dtype.kind in {
        "S",
        "U",
    }:

        return "".join(
            str(x)
            for x in np.ravel(array)
        ).strip()

    if array.dtype.kind == "O":

        values = [
            decode_matlab_string(
                h5,
                x,
            )
            for x in np.ravel(array)
        ]

        values = [
            value
            for value in values
            if value
        ]

        return (
            values[0]
            if values
            else ""
        )

    if array.size == 1:
        return str(
            array.item()
        ).strip()

    return str(
        array
    ).strip()


def read_hdf5_cell_string_vector(
    h5: h5py.File,
    dataset: h5py.Dataset,
) -> list[str]:

    return [
        decode_matlab_string(
            h5,
            item,
        )
        for item in np.asarray(
            dataset
        ).ravel()
    ]


def read_hdf5_cell_numeric_vector(
    h5: h5py.File,
    dataset: h5py.Dataset,
) -> np.ndarray:

    output: list[float] = []

    for item in np.asarray(
        dataset
    ).ravel():

        if isinstance(
            item,
            h5py.Reference,
        ):

            array = (
                np.asarray(
                    h5[item]
                ).squeeze()
                if item
                else np.array(
                    np.nan
                )
            )

        else:

            array = np.asarray(
                item
            ).squeeze()

        array = (
            np.asarray(
                array
            )
            .astype(float)
            .ravel()
        )

        output.append(
            np.nan
            if array.size == 0
            else float(array[0])
        )

    return np.asarray(
        output,
        dtype=float,
    )


def get_erp_condition_group(
    h5: h5py.File,
    erps: h5py.Dataset,
    index: int,
):

    return h5[
        erps[0, index]
    ]


def read_condition_times(
    condition_group: h5py.Group,
) -> np.ndarray:

    if "times" not in condition_group:

        raise KeyError(
            "ERP condition does not contain times."
        )

    times = (
        np.asarray(
            condition_group["times"]
        )
        .squeeze()
        .astype(float)
    )

    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(
            times
        ).all()
    ):

        raise ValueError(
            f"Invalid ERP times array with shape "
            f"{times.shape}."
        )

    return times


def _jsonable_scalar(
    value: Any,
) -> Any:

    if isinstance(
        value,
        (np.integer, int),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating, float),
    ):

        value = float(
            value
        )

        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(
        value,
        (str, bytes),
    ):

        return (
            value.decode(
                "utf-8",
                errors="ignore",
            )
            if isinstance(
                value,
                bytes,
            )
            else value
        )

    return value


def _read_chanloc_field(
    h5: h5py.File,
    dataset: h5py.Dataset,
    n_channels: int,
) -> list[Any] | None:

    raw = np.asarray(
        dataset
    )

    # MATLAB cell/reference field.
    if raw.dtype.kind == "O":

        values: list[Any] = []

        for item in raw.ravel():

            if isinstance(
                item,
                h5py.Reference,
            ):

                if not item:

                    values.append(
                        None
                    )

                    continue

                obj = np.asarray(
                    h5[item]
                ).squeeze()

            else:

                obj = np.asarray(
                    item
                ).squeeze()

            array = np.asarray(
                obj
            )

            if (
                array.dtype.kind
                in {"u", "i"}
                and array.size > 1
            ):

                values.append(
                    decode_matlab_string(
                        h5,
                        array,
                    )
                )

            elif (
                array.dtype.kind
                in {"S", "U"}
            ):

                values.append(
                    decode_matlab_string(
                        h5,
                        array,
                    )
                )

            elif array.size == 0:

                values.append(
                    None
                )

            elif array.size == 1:

                values.append(
                    _jsonable_scalar(
                        array.item()
                    )
                )

            else:

                flat = [
                    _jsonable_scalar(x)
                    for x in array.ravel().tolist()
                ]

                values.append(
                    flat
                )

        if len(values) == n_channels:
            return values

        return None

    flat = raw.squeeze()

    if (
        flat.ndim == 0
        and n_channels == 1
    ):

        return [
            _jsonable_scalar(
                flat.item()
            )
        ]

    if flat.size == n_channels:

        return [
            _jsonable_scalar(x)
            for x in flat.ravel().tolist()
        ]

    return None


def read_chanlocs(
    h5: h5py.File,
    condition_group: h5py.Group,
    n_channels: int,
) -> list[dict[str, Any]]:

    if "chanlocs" not in condition_group:

        raise KeyError(
            "ERP condition does not contain chanlocs."
        )

    group = condition_group[
        "chanlocs"
    ]

    if "labels" not in group:

        raise KeyError(
            "ERP condition chanlocs does not contain labels."
        )

    labels = read_hdf5_cell_string_vector(
        h5,
        group["labels"],
    )

    if len(labels) != n_channels:

        raise ValueError(
            f"Channel label count "
            f"{len(labels)} != data channel count "
            f"{n_channels}."
        )

    chanlocs: list[
        dict[str, Any]
    ] = [
        {
            "labels": str(
                label
            ).strip()
        }
        for label in labels
    ]

    # Preserve scalar/per-channel spatial and metadata fields.
    for field, obj in group.items():

        if (
            field == "labels"
            or not isinstance(
                obj,
                h5py.Dataset,
            )
        ):
            continue

        values = _read_chanloc_field(
            h5,
            obj,
            n_channels,
        )

        if values is None:
            continue

        for index, value in enumerate(
            values
        ):

            if value is not None:

                chanlocs[index][
                    field
                ] = value

    return chanlocs


def read_condition_eventurevent(
    h5: h5py.File,
    condition_group: h5py.Group,
    n_trials: int,
) -> np.ndarray:

    if "epoch" not in condition_group:

        raise KeyError(
            "ERP condition does not contain epoch."
        )

    epoch = condition_group[
        "epoch"
    ]

    if "eventurevent" not in epoch:

        raise KeyError(
            "ERP condition epoch does not contain eventurevent."
        )

    values = read_hdf5_cell_numeric_vector(
        h5,
        epoch["eventurevent"],
    )

    if len(values) != n_trials:

        raise ValueError(
            f"eventurevent length "
            f"{len(values)} does not match "
            f"condition trials {n_trials}."
        )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "eventurevent contains NaN or infinite values."
        )

    return values.astype(
        int
    )


# =====================================================================
# ERP loading
# =====================================================================

def load_erp_mat(
    mat_path: Path,
    n_design_rows: int,
) -> dict[str, Any]:

    condition_data: list[
        np.ndarray
    ] = []

    condition_counts: list[
        int
    ] = []

    urevents: list[
        np.ndarray
    ] = []

    times: np.ndarray | None = None

    channel_names: list[
        str
    ] | None = None

    chanlocs: list[
        dict[str, Any]
    ] | None = None

    with h5py.File(
        mat_path,
        "r",
    ) as h5:

        if "ERPs" not in h5:

            raise KeyError(
                f"{mat_path.name}: ERP MAT file "
                "does not contain ERPs."
            )

        erps = h5[
            "ERPs"
        ]

        if erps.ndim != 2:

            raise ValueError(
                f"{mat_path.name}: ERPs must be 2D, "
                f"found {erps.shape}."
            )

        n_conditions = int(
            erps.shape[1]
        )

        if n_conditions < 1:

            raise ValueError(
                f"{mat_path.name}: "
                "ERPs contains no conditions."
            )

        for condition_index in range(
            n_conditions
        ):

            group = get_erp_condition_group(
                h5,
                erps,
                condition_index,
            )

            if "data" not in group:

                raise KeyError(
                    f"Condition "
                    f"{condition_index + 1} "
                    "has no data field."
                )

            raw = np.asarray(
                group["data"],
                dtype=float,
            )

            if raw.ndim != 3:

                raise ValueError(
                    f"Condition "
                    f"{condition_index + 1}: "
                    "expected trials x time x channels, "
                    f"found {raw.shape}."
                )

            (
                n_trials,
                n_timepoints,
                n_channels,
            ) = map(
                int,
                raw.shape,
            )

            current_times = read_condition_times(
                group
            )

            if (
                len(current_times)
                != n_timepoints
            ):

                raise ValueError(
                    f"Condition "
                    f"{condition_index + 1}: "
                    f"times={len(current_times)}, "
                    f"data timepoints={n_timepoints}."
                )

            current_chanlocs = read_chanlocs(
                h5,
                group,
                n_channels,
            )

            current_names = [
                str(
                    channel.get(
                        "labels",
                        "",
                    )
                ).strip()
                for channel in current_chanlocs
            ]

            if times is None:

                times = current_times

            elif not np.array_equal(
                times,
                current_times,
            ):

                raise ValueError(
                    f"{mat_path.name}: "
                    "ERP times differ across conditions."
                )

            if channel_names is None:

                channel_names = current_names
                chanlocs = current_chanlocs

            elif (
                channel_names
                != current_names
            ):

                raise ValueError(
                    f"{mat_path.name}: "
                    "channel labels differ across conditions."
                )

            # Convert:
            #
            # trials x time x channels
            #
            # to:
            #
            # trials x channels x time
            #
            condition_data.append(
                np.transpose(
                    raw,
                    (0, 2, 1),
                )
            )

            condition_counts.append(
                n_trials
            )

            urevents.append(
                read_condition_eventurevent(
                    h5,
                    group,
                    n_trials,
                )
            )

    data = np.concatenate(
        condition_data,
        axis=0,
    )

    eventurevent = np.concatenate(
        urevents,
        axis=0,
    )

    if (
        data.shape[0]
        != n_design_rows
    ):

        raise ValueError(
            f"{mat_path.name}: "
            f"ERP trials={data.shape[0]}, "
            f"design rows={n_design_rows}, "
            f"condition trial counts="
            f"{condition_counts}."
        )

    if (
        len(eventurevent)
        != n_design_rows
    ):

        raise ValueError(
            f"{mat_path.name}: "
            "eventurevent length != design rows."
        )

    assert times is not None
    assert channel_names is not None
    assert chanlocs is not None

    dt = float(
        np.median(
            np.diff(times)
        )
    )

    if (
        not math.isfinite(dt)
        or dt == 0
    ):

        raise ValueError(
            f"{mat_path.name}: "
            "cannot derive sampling rate "
            "from ERP times."
        )

    sampling_rate = (
        abs(
            1000.0 / dt
        )
        if abs(dt) > 0.1
        else abs(
            1.0 / dt
        )
    )

    return {
        "data": data,
        "times": times,
        "sampling_rate": sampling_rate,
        "channel_names": channel_names,
        "chanlocs": chanlocs,
        "eventurevent": eventurevent,
        "condition_trial_counts": condition_counts,
    }


# =====================================================================
# Trial alignment
# =====================================================================

def validate_alignment(
    design: pd.DataFrame,
    erp: dict[str, Any],
    design_path: Path,
) -> None:

    design_urevents = design[
        "urevent_index"
    ].to_numpy(
        dtype=int
    )

    erp_urevents = np.asarray(
        erp["eventurevent"],
        dtype=int,
    )

    if not np.array_equal(
        design_urevents,
        erp_urevents,
    ):

        mismatch = np.flatnonzero(
            design_urevents
            != erp_urevents
        )

        examples = [
            {
                "eeg_trial": int(
                    index + 1
                ),
                "design_urevent": int(
                    design_urevents[
                        index
                    ]
                ),
                "erp_urevent": int(
                    erp_urevents[
                        index
                    ]
                ),
            }
            for index in mismatch[:10]
        ]

        raise ValueError(
            f"{design_path.name}: "
            "design-to-ERP epoch alignment failed. "
            f"Examples: {examples}"
        )


# =====================================================================
# Model preparation
# =====================================================================

def prepare_model_input(
    design: pd.DataFrame,
    erp: dict[str, Any],
    predictors: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:

    predictor_frame = (
        design[
            predictors
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    valid = (
        predictor_frame
        .notna()
        .all(axis=1)
        .to_numpy(
            dtype=bool
        )
    )

    n_removed = int(
        (~valid).sum()
    )

    if not valid.any():

        raise ValueError(
            "No trials have complete values "
            "for all selected predictors."
        )

    continuous = (
        predictor_frame.loc[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        continuous
    ).all():

        raise ValueError(
            "Predictor matrix contains "
            "infinite values."
        )

    # LIMO adds the constant itself.
    # Test rank after including the implied intercept.
    rank_check = np.column_stack(
        [
            continuous,
            np.ones(
                continuous.shape[0]
            ),
        ]
    )

    rank = int(
        np.linalg.matrix_rank(
            rank_check
        )
    )

    if (
        rank
        < rank_check.shape[1]
    ):

        raise ValueError(
            "The selected continuous-regressor matrix "
            "is rank-deficient. "
            f"Rank={rank}, "
            f"columns including constant="
            f"{rank_check.shape[1]}. "
            "Choose a non-collinear predictor set "
            "with --predictors."
        )

    if (
        continuous.shape[0]
        <= rank
    ):

        raise ValueError(
            f"Insufficient complete trials: "
            f"n={continuous.shape[0]}, "
            f"model rank={rank}."
        )

    data = np.asarray(
        erp["data"],
        dtype=float,
    )[valid, :, :]

    # LIMO expects:
    #
    # channels x frames x trials
    #
    y = np.transpose(
        data,
        (1, 2, 0),
    )

    return (
        y,
        continuous,
        valid,
        n_removed,
    )


# =====================================================================
# Design-file discovery
# =====================================================================

def discover_design_files(
    design_dir: Path,
    subjects: list[str] | None,
    analyses: list[str] | None,
) -> list[Path]:

    files: list[
        Path
    ] = []

    for path in sorted(
        design_dir.glob(
            "*_design_matrix.tsv"
        )
    ):

        match = DESIGN_FILENAME_RE.match(
            path.name
        )

        if not match:
            continue

        subject = match.group(
            "subject"
        )

        analysis = match.group(
            "analysis"
        )

        if (
            subjects is not None
            and subject not in subjects
        ):
            continue

        if (
            analyses is not None
            and analysis not in analyses
        ):
            continue

        files.append(
            path
        )

    if not files:

        raise FileNotFoundError(
            "No matching *_design_matrix.tsv "
            "files found."
        )

    return files


# =====================================================================
# MATLAB helper creation
# =====================================================================

def write_matlab_helpers(
    helper_dir: Path,
) -> None:

    helper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        helper_dir
        / "run_limo_first_level.m"
    ).write_text(
        MATLAB_FIRST_LEVEL,
        encoding="utf-8",
    )

    (
        helper_dir
        / "build_limo_expected_chanlocs.m"
    ).write_text(
        MATLAB_EXPECTED_CHANLOCS,
        encoding="utf-8",
    )

    (
        helper_dir
        / "run_limo_group_level.m"
    ).write_text(
        MATLAB_SECOND_LEVEL,
        encoding="utf-8",
    )


# =====================================================================
# MATLAB execution
# =====================================================================

def resolve_matlab_executable(
    command: str,
) -> str:

    found = shutil.which(
        command
    )

    if found:
        return found

    candidate = Path(
        command
    )

    if candidate.exists():
        return str(
            candidate.resolve()
        )

    raise FileNotFoundError(
        f"MATLAB executable not found: "
        f"{command}. "
        "Use --matlab-command with the "
        "MATLAB executable or its full path."
    )


def run_matlab(
    matlab: str,
    command: str,
    cwd: Path | None = None,
) -> None:

    print(
        f"MATLAB: {command}"
    )

    subprocess.run(
        [
            matlab,
            "-batch",
            command,
        ],
        cwd=(
            str(cwd)
            if cwd
            else None
        ),
        check=True,
    )


def safe_name(
    value: str,
) -> str:

    return (
        re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            value,
        ).strip("_")
        or "value"
    )


def locate_betas_file(
    model_dir: Path,
) -> Path:

    matches = sorted(
        model_dir.glob(
            "*Betas.mat"
        )
    )

    if len(matches) != 1:

        raise FileNotFoundError(
            f"Expected exactly one "
            f"*Betas.mat in {model_dir}, "
            f"found {matches}"
        )

    return matches[
        0
    ].resolve()


# =====================================================================
# Output manifests
# =====================================================================

def write_manifest(
    model_dir: Path,
    design_path: Path,
    erp_path: Path,
    predictors: list[str],
    n_design_rows: int,
    n_complete_rows: int,
    n_removed: int,
    channel_names: list[str],
    times: np.ndarray,
    method: str,
) -> None:

    rows = []

    for (
        beta_index,
        predictor,
    ) in enumerate(
        predictors,
        start=1,
    ):

        rows.append(
            {
                "beta_index": beta_index,
                "term": predictor,
                "term_type": "continuous",
            }
        )

    rows.append(
        {
            "beta_index": (
                len(predictors)
                + 1
            ),
            "term": "constant",
            "term_type": "constant",
        }
    )

    pd.DataFrame(
        rows
    ).to_csv(
        model_dir
        / "beta_manifest.tsv",
        sep="\t",
        index=False,
    )

    metadata = {
        "design_matrix": str(
            design_path.resolve()
        ),
        "erp_mat": str(
            erp_path.resolve()
        ),
        "predictors": predictors,
        "categorical_regressors": [],
        "limo_method": method,
        "limo_zscore": 0,
        "n_design_rows": n_design_rows,
        "n_complete_rows": n_complete_rows,
        "n_removed_for_missing_predictors": n_removed,
        "n_channels": len(
            channel_names
        ),
        "channel_names": channel_names,
        "n_timepoints": int(
            len(times)
        ),
        "time_start": float(
            times[0]
        ),
        "time_end": float(
            times[-1]
        ),
    }

    (
        model_dir
        / "model_manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# =====================================================================
# First-level LIMO analysis
# =====================================================================

def first_level(
    args: argparse.Namespace,
    matlab: str,
    helper_dir: Path,
) -> list[dict[str, Any]]:

    subjects = parse_csv_arg(
        args.subjects
    )

    if subjects is not None:

        subjects = [
            normalise_subject_id(
                subject
            )
            for subject in subjects
        ]

    analyses = parse_csv_arg(
        args.analyses
    )

    requested_predictors = (
        parse_csv_arg(
            args.predictors
        )
    )

    design_files = (
        discover_design_files(
            args.design_dir,
            subjects,
            analyses,
        )
    )

    print(
        f"Found {len(design_files)} "
        "design matrices."
    )

    records: list[
        dict[str, Any]
    ] = []

    for index, design_path in enumerate(
        design_files,
        start=1,
    ):

        print(
            f"\n[{index}/"
            f"{len(design_files)}] "
            f"{design_path.name}"
        )

        design = load_design_matrix(
            design_path
        )

        subject = str(
            design[
                "subject"
            ].iloc[0]
        ).strip()

        analysis = str(
            design[
                "analysis"
            ].iloc[0]
        ).strip()

        predictors = choose_predictors(
            design,
            requested_predictors,
        )

        erp_path = resolve_erp_path(
            design,
            design_path,
            args.erp_root,
        )

        erp = load_erp_mat(
            erp_path,
            len(design),
        )

        validate_alignment(
            design,
            erp,
            design_path,
        )

        (
            y,
            continuous,
            valid_mask,
            n_removed,
        ) = prepare_model_input(
            design,
            erp,
            predictors,
        )

        model_dir = (
            args.output_dir
            / subject
            / analysis
        ).resolve()

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        work_dir = (
            model_dir
            / "_limo_input"
        )

        work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        input_mat = (
            work_dir
            / "limo_input.mat"
        )

        chanlocs_json = (
            work_dir
            / "chanlocs.json"
        )

        savemat(
            input_mat,
            {
                "Y": y,
                "Cont": continuous,
                "times": np.asarray(
                    erp["times"],
                    dtype=float,
                ),
                "sampling_rate": np.asarray(
                    [
                        erp[
                            "sampling_rate"
                        ]
                    ],
                    dtype=float,
                ),
                "predictor_names": np.asarray(
                    predictors,
                    dtype=object,
                ),
            },
            do_compression=True,
        )

        chanlocs_json.write_text(
            json.dumps(
                erp["chanlocs"],
                indent=2,
            ),
            encoding="utf-8",
        )

        source_file = str(
            design[
                "source_file"
            ].iloc[0]
        ).strip()

        source_dir = str(
            erp_path.parent
        )

        command = (
            f"addpath("
            f"{matlab_quote(helper_dir)}"
            f"); "
            f"run_limo_first_level("
            f"{matlab_quote(input_mat)},"
            f"{matlab_quote(model_dir)},"
            f"{matlab_quote(args.limo_tools)},"
            f"{matlab_quote(chanlocs_json)},"
            f"{matlab_quote(source_file)},"
            f"{matlab_quote(source_dir)},"
            f"{matlab_quote(args.method)}"
            f");"
        )

        run_matlab(
            matlab,
            command,
        )

        betas_file = (
            locate_betas_file(
                model_dir
            )
        )

        limo_file = (
            model_dir
            / "LIMO.mat"
        )

        if not limo_file.exists():

            raise FileNotFoundError(
                f"LIMO did not create "
                f"{limo_file}"
            )

        write_manifest(
            model_dir=model_dir,
            design_path=design_path,
            erp_path=erp_path,
            predictors=predictors,
            n_design_rows=len(
                design
            ),
            n_complete_rows=int(
                valid_mask.sum()
            ),
            n_removed=n_removed,
            channel_names=erp[
                "channel_names"
            ],
            times=erp[
                "times"
            ],
            method=args.method,
        )

        records.append(
            {
                "subject": subject,
                "analysis": analysis,
                "design_path": design_path,
                "erp_path": erp_path,
                "model_dir": model_dir,
                "limo_file": (
                    limo_file.resolve()
                ),
                "betas_file": (
                    betas_file
                ),
                "predictors": predictors,
                "channel_names": erp[
                    "channel_names"
                ],
                "chanlocs": erp[
                    "chanlocs"
                ],
            }
        )

        if not args.keep_work:

            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

    return records


# =====================================================================
# Load previously completed first-level results
# =====================================================================

def load_existing_first_level(
    output_dir: Path,
) -> list[dict[str, Any]]:

    records: list[
        dict[str, Any]
    ] = []

    for manifest_path in sorted(
        output_dir.glob(
            "sub-*/*/model_manifest.json"
        )
    ):

        model_dir = (
            manifest_path.parent
        )

        metadata = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        beta_manifest = pd.read_csv(
            model_dir
            / "beta_manifest.tsv",
            sep="\t",
        )

        predictors = (
            beta_manifest.loc[
                beta_manifest[
                    "term_type"
                ].eq(
                    "continuous"
                ),
                "term",
            ]
            .astype(str)
            .tolist()
        )

        limo_file = (
            model_dir
            / "LIMO.mat"
        )

        if not limo_file.exists():
            continue

        records.append(
            {
                "subject": (
                    model_dir.parent.name
                ),
                "analysis": (
                    model_dir.name
                ),
                "model_dir": (
                    model_dir
                ),
                "limo_file": (
                    limo_file.resolve()
                ),
                "betas_file": (
                    locate_betas_file(
                        model_dir
                    )
                ),
                "predictors": (
                    predictors
                ),
                "channel_names": metadata.get(
                    "channel_names",
                    [],
                ),
                "chanlocs": [],
            }
        )

    if not records:

        raise FileNotFoundError(
            f"No completed first-level "
            f"model manifests found under "
            f"{output_dir}."
        )

    return records


# =====================================================================
# Second-level group analysis
# =====================================================================

def second_level(
    args: argparse.Namespace,
    matlab: str,
    helper_dir: Path,
    records: list[dict[str, Any]],
) -> None:

    if (
        args.neighbour_distance
        is None
    ):

        raise ValueError(
            "--run-second-level requires "
            "--neighbour-distance. "
            "LIMO uses this montage-specific "
            "threshold to construct the "
            "full-scalp neighbouring-channel matrix."
        )

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for record in records:

        grouped.setdefault(
            record[
                "analysis"
            ],
            [],
        ).append(
            record
        )

    for analysis, group in sorted(
        grouped.items()
    ):

        group = sorted(
            group,
            key=lambda record: record[
                "subject"
            ],
        )

        print(
            f"\nSecond level: "
            f"{analysis} "
            f"({len(group)} subjects)"
        )

        reference_predictors = (
            group[0][
                "predictors"
            ]
        )

        for record in group[1:]:

            if (
                record[
                    "predictors"
                ]
                != reference_predictors
            ):

                raise ValueError(
                    "Predictor order differs "
                    f"within analysis {analysis}: "
                    f"{group[0]['subject']}="
                    f"{reference_predictors}, "
                    f"{record['subject']}="
                    f"{record['predictors']}"
                )

        reference_channels = (
            group[0].get(
                "channel_names",
                [],
            )
        )

        for record in group[1:]:

            if (
                record.get(
                    "channel_names",
                    [],
                )
                != reference_channels
            ):

                raise ValueError(
                    "Channel labels differ "
                    f"across subjects in "
                    f"{analysis}. "
                    "A single expected scalp "
                    "layout cannot be constructed "
                    "safely."
                )

        expected_dir = (
            args.output_dir
            / "group"
            / analysis
        )

        expected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        expected_file = (
            expected_dir
            / "expected_chanlocs.mat"
        )

        if (
            not expected_file.exists()
            or args.overwrite_expected_chanlocs
        ):

            command = (
                f"addpath("
                f"{matlab_quote(helper_dir)}"
                f"); "
                f"build_limo_expected_chanlocs("
                f"{matlab_quote(group[0]['limo_file'])},"
                f"{matlab_quote(expected_file)},"
                f"{matlab_quote(args.limo_tools)},"
                f"{float(args.neighbour_distance):.17g}"
                f");"
            )

            run_matlab(
                matlab,
                command,
            )

        for beta_index, predictor in enumerate(
            reference_predictors,
            start=1,
        ):

            predictor_dir = (
                expected_dir
                / safe_name(
                    predictor
                )
            )

            predictor_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            beta_list = (
                predictor_dir
                / "subject_beta_files.txt"
            )

            beta_list.write_text(
                "\n".join(
                    str(
                        record[
                            "betas_file"
                        ]
                    )
                    for record in group
                )
                + "\n",
                encoding="utf-8",
            )

            command = (
                f"addpath("
                f"{matlab_quote(helper_dir)}"
                f"); "
                f"run_limo_group_level("
                f"{matlab_quote(beta_list)},"
                f"{matlab_quote(expected_file)},"
                f"{beta_index},"
                f"{matlab_quote(predictor_dir)},"
                f"{matlab_quote(args.limo_tools)},"
                f"{int(args.nboot)},"
                f"{1 if args.tfce else 0}"
                f");"
            )

            print(
                f"  predictor "
                f"{beta_index}: "
                f"{predictor}"
            )

            run_matlab(
                matlab,
                command,
            )

            group_metadata = {
                "analysis": analysis,
                "predictor": predictor,
                "beta_index": beta_index,
                "subjects": [
                    record[
                        "subject"
                    ]
                    for record in group
                ],
                "n_subjects": len(
                    group
                ),
                "test": (
                    "LIMO one sample t-test"
                ),
                "nboot": int(
                    args.nboot
                ),
                "tfce": bool(
                    args.tfce
                ),
                "neighbour_distance": float(
                    args.neighbour_distance
                ),
                "expected_chanlocs": str(
                    expected_file.resolve()
                ),
            }

            (
                predictor_dir
                / "group_manifest.json"
            ).write_text(
                json.dumps(
                    group_metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )


# =====================================================================
# Command-line interface
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the official LIMO EEG toolbox on "
            "ERP MATLAB files using existing "
            "subject-specific design matrices, "
            "with optional cross-subject "
            "second-level inference."
        )
    )

    parser.add_argument(
        "--design-dir",
        type=Path,
        default=Path(
            "limo_design_matrices"
        ),
        help=(
            "Directory containing "
            "sub-XX_erp-<analysis>_design_matrix.tsv files."
        ),
    )

    parser.add_argument(
        "--erp-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing ERP .mat files. "
            "Used when source_path stored in the TSV "
            "does not exist on the current machine. "
            "The exact source_file value from each TSV "
            "is searched recursively."
        ),
    )

    parser.add_argument(
        "--limo-tools",
        type=Path,
        default=Path(
            "limo_tools"
        ),
        help=(
            "Path to the official "
            "LIMO EEG toolbox directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "limo_results"
        ),
        help=(
            "Output directory for first- "
            "and second-level LIMO results."
        ),
    )

    parser.add_argument(
        "--matlab-command",
        default="matlab",
        help=(
            "MATLAB executable name "
            "or full path."
        ),
    )

    parser.add_argument(
        "--method",
        choices=[
            "OLS",
            "WLS",
            "IRLS",
        ],
        default="WLS",
        help=(
            "Official LIMO first-level "
            "estimation method. "
            "Default: WLS."
        ),
    )

    parser.add_argument(
        "--predictors",
        default=None,
        help=(
            "Comma-separated predictor columns. "
            "When omitted, the predefined diagnostic "
            "predictor set is used, restricted to "
            "columns present in each design matrix."
        ),
    )

    parser.add_argument(
        "--subjects",
        default=None,
        help=(
            "Optional comma-separated subject filter, "
            "for example sub-01,sub-02 or 1,2."
        ),
    )

    parser.add_argument(
        "--analyses",
        default=None,
        help=(
            "Optional comma-separated ERP-analysis "
            "filter, for example "
            "GA,CP,LD,Order,Time."
        ),
    )

    parser.add_argument(
        "--run-second-level",
        action="store_true",
        help=(
            "After first-level models, "
            "run LIMO full-scalp one-sample "
            "group tests for every predictor."
        ),
    )

    parser.add_argument(
        "--second-level-only",
        action="store_true",
        help=(
            "Use already completed first-level "
            "outputs in --output-dir and run "
            "only second-level group analyses."
        ),
    )

    parser.add_argument(
        "--neighbour-distance",
        type=float,
        default=None,
        help=(
            "Montage-specific LIMO neighbour-distance "
            "threshold for full-scalp group inference. "
            "Required with --run-second-level or "
            "--second-level-only."
        ),
    )

    parser.add_argument(
        "--nboot",
        type=int,
        default=1000,
        help=(
            "Number of LIMO second-level "
            "bootstrap samples. "
            "Default: 1000."
        ),
    )

    parser.add_argument(
        "--tfce",
        action="store_true",
        help=(
            "Request LIMO TFCE "
            "at second level."
        ),
    )

    parser.add_argument(
        "--keep-work",
        action="store_true",
        help=(
            "Keep temporary LIMO input .mat "
            "and channel-location JSON files."
        ),
    )

    parser.add_argument(
        "--overwrite-expected-chanlocs",
        action="store_true",
        help=(
            "Rebuild expected_chanlocs.mat "
            "even if it already exists."
        ),
    )

    return parser


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    args.design_dir = (
        args.design_dir
        .expanduser()
        .resolve()
    )

    args.limo_tools = (
        args.limo_tools
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if args.erp_root is not None:

        args.erp_root = (
            args.erp_root
            .expanduser()
            .resolve()
        )

    if not args.limo_tools.is_dir():

        raise FileNotFoundError(
            "LIMO toolbox directory "
            f"not found: {args.limo_tools}"
        )

    required_limo_files = [
        "limo_design_matrix.m",
        "limo_eeg.m",
        "limo_glm.m",
        "limo_random_select.m",
    ]

    for required in required_limo_files:

        required_path = (
            args.limo_tools
            / required
        )

        if not required_path.exists():

            raise FileNotFoundError(
                "Required LIMO file missing: "
                f"{required_path}"
            )

    if (
        not args.second_level_only
        and not args.design_dir.is_dir()
    ):

        raise FileNotFoundError(
            "Design-matrix directory "
            f"not found: {args.design_dir}"
        )

    if args.nboot < 0:

        raise ValueError(
            "--nboot must be >= 0."
        )

    if (
        args.neighbour_distance
        is not None
        and args.neighbour_distance <= 0
    ):

        raise ValueError(
            "--neighbour-distance "
            "must be > 0."
        )

    matlab = (
        resolve_matlab_executable(
            args.matlab_command
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    helper_dir = (
        args.output_dir
        / "_matlab_helpers"
    ).resolve()

    write_matlab_helpers(
        helper_dir
    )

    if args.second_level_only:

        records = (
            load_existing_first_level(
                args.output_dir
            )
        )

    else:

        records = first_level(
            args,
            matlab,
            helper_dir,
        )

    if (
        args.run_second_level
        or args.second_level_only
    ):

        second_level(
            args,
            matlab,
            helper_dir,
            records,
        )

    print(
        "\nLIMO pipeline complete."
    )

    print(
        f"Results: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()