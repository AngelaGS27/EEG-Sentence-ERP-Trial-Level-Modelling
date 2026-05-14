from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import spacy
from statsmodels.stats.outliers_influence import variance_inflation_factor

from surprisal import SurprisalModel
from sentence_metrics import SentenceMetrics
from word_frequency import get_zipf_frequency, get_word_frequency
from phonology import get_phonology_for_word
from syntax_metrics import compute_syntax

try:
    from emotion import get_emotion_features
except ImportError:
    get_emotion_features = None

try:
    from emotion import get_valence
except ImportError:
    get_valence = None


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
# Column candidates
# ---------------------------------------------------------------------

TEXT_COLUMN_CANDIDATES = [
    "sentence_for_metrics",
    "full_sentence",
    "sentence",
    "sentential_context",
    "sent",
    "sentence_text",
    "stimulus_sentence",
    "stimulus",
    "stimuli",
    "text",
    "trial_text",
    "item_text",
    "context_sentence",
    "word_sentence",
    "sentence_full",
]

TARGET_COLUMN_CANDIDATES = [
    "target_word_for_metrics",
    "key",
    "known_word",
    "critical_word",
    "target_word",
    "target",
    "final_word",
    "cw",
    "critical",
    "anomalous_word",
    "expected_word",
    "completion",
    "noun",
    "word",
]

CONTEXT_COLUMN_CANDIDATES = [
    "context_for_metrics",
    "sentential_context",
    "context",
    "sentence_context",
    "preceding_context",
    "prefix",
    "pre_target_context",
    "sentence_prefix",
    "precritical_context",
]

CONDITION_COLUMN_CANDIDATES = [
    "condition",
    "cond",
    "congruency",
    "congruent",
    "incongruent",
    "type",
    "stimulus_type",
    "category",
]

EXCLUDE_FILE_PATTERNS = [
    "_trialrej",
    "trialrej",
    "_erp",
    "_erp_long",
    "erp_long",
    "_language_metrics",
    "all_language_metrics",
    "_predictor_diagnostics",
    "all_predictor_diagnostics",
    "_correlations",
    "_vif",
    "diagnostics",
    "eeg_outputs",
    "language_outputs",
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def normalise_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def clean_word(word: Any) -> str:
    return (
        str(word)
        .strip()
        .strip(".,;:!?\"'()[]{}<>")
        .lower()
    )


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


def looks_like_sentence(value: Any) -> bool:
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return False

    words = text.split()

    if len(words) < 3:
        return False

    if len(text) < 10:
        return False

    return True


def looks_like_text_column(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)

    if len(non_null) == 0:
        return False

    sample = non_null.head(30)

    sentence_like = sample.apply(looks_like_sentence).mean()

    return sentence_like >= 0.60


def is_probably_real_word(value: Any) -> bool:
    word = clean_word(value)

    if not word:
        return False

    if word in {"nan", "none", "null"}:
        return False

    # Reject phrases or sentence fragments.
    if len(word.split()) != 1:
        return False

    # Reject pure numbers, punctuation, percentages, ranges, etc.
    if re.fullmatch(r"[\d\W_]+", word):
        return False

    if re.fullmatch(r"\d+(\.\d+)?", word):
        return False

    if re.fullmatch(r"\d+\s*-\s*\d+", word):
        return False

    # Reject stimulus/audio/file IDs
    if re.fullmatch(r"sound\d+", word):
        return False

    # Reject one-letter targets unless they are real meaningful English words.
    if len(word) == 1 and word not in {"a", "i"}:
        return False

    # Allow normal alphabetic words and hyphenated words.
    if re.fullmatch(r"[a-zA-Z]+(-[a-zA-Z]+)?", word):
        return True

    return False


def target_column_quality(series: pd.Series) -> float:
    non_null = series.dropna()

    if len(non_null) == 0:
        return 0.0

    sample = non_null.head(100).astype(str)

    # Reject columns that are mostly phrases/sentence fragments.
    phrase_rate = sample.str.strip().str.split().apply(len).gt(1).mean()

    if phrase_rate > 0.20:
        return 0.0

    # Reject columns that look like stimulus IDs.
    sound_id_rate = sample.str.lower().str.strip().str.match(r"^sound\d+$").mean()

    if sound_id_rate > 0.20:
        return 0.0

    # Reject columns where many values are one-letter fragments.
    one_letter_rate = sample.str.strip().str.len().eq(1).mean()

    if one_letter_rate > 0.20:
        return 0.0

    return sample.apply(is_probably_real_word).mean()


def derive_final_word(sentence: Any) -> Optional[str]:
    words = str(sentence).strip().split()

    if not words:
        return None

    return clean_word(words[-1])


def derive_context(sentence: Any, target_word: Optional[str]) -> str:
    sentence = str(sentence).strip()
    words = sentence.split()

    if not words:
        return ""

    if target_word is None or str(target_word).strip() == "":
        return " ".join(words[:-1])

    target_clean = clean_word(target_word)

    for i in range(len(words) - 1, -1, -1):
        if clean_word(words[i]) == target_clean:
            return " ".join(words[:i])

    return " ".join(words[:-1])


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)

    if std == 0 or np.isnan(std):
        return values * 0

    return (values - values.mean(skipna=True)) / std


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

def detect_condition_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, CONDITION_COLUMN_CANDIDATES)


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the real stimulus/sentence column.

    Important:
        Never return condition/congruency columns as text.
    """
    condition_col = detect_condition_column(df)

    direct = find_column(df, TEXT_COLUMN_CANDIDATES)

    if direct is not None:
        if direct != condition_col and looks_like_text_column(df[direct]):
            return direct

    condition_names = {
        normalise_column_name(x)
        for x in CONDITION_COLUMN_CANDIDATES
    }

    possible = []

    for col in df.columns:
        col_norm = normalise_column_name(col)

        if col == condition_col:
            continue

        if col_norm in condition_names:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        if looks_like_text_column(df[col]):
            possible.append(col)

    if possible:
        return possible[0]

    return None


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    The target column should contain single target words, not full sentences,
    contexts, stimulus IDs, metadata, or condition labels.
    """

    text_col = detect_text_column(df)
    context_col = detect_context_column(df)
    condition_col = detect_condition_column(df)

    excluded_cols = {
        col for col in [text_col, context_col, condition_col]
        if col is not None
    }

    direct = find_column(df, TARGET_COLUMN_CANDIDATES)

    if direct is not None and direct not in excluded_cols:
        quality = target_column_quality(df[direct])

        if quality >= 0.70:
            return direct

        log.warning(
            "Rejected target column '%s' because values do not look like single target words. Quality = %.2f",
            direct,
            quality,
        )

    best_col = None
    best_quality = 0.0

    for col in df.columns:
        if col in excluded_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Never allow columns that look like sentence/context columns.
        if looks_like_text_column(df[col]):
            continue

        quality = target_column_quality(df[col])

        if quality > best_quality:
            best_col = col
            best_quality = quality

    if best_col is not None and best_quality >= 0.70:
        return best_col

    # If no reliable target column is found, return None.
    # Then process_table() will derive the target as the final word of the sentence.
    return None


def detect_context_column(df: pd.DataFrame) -> Optional[str]:
    direct = find_column(df, CONTEXT_COLUMN_CANDIDATES)

    if direct is not None and looks_like_text_column(df[direct]):
        return direct

    return None


# ---------------------------------------------------------------------
# Cloze metrics
# ---------------------------------------------------------------------

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


def add_cloze_metrics(out: pd.DataFrame) -> pd.DataFrame:
    human_cp_col = find_column(out, HUMAN_CP_COLUMN_CANDIDATES)
    llm_cp_col = find_column(out, LLM_CP_COLUMN_CANDIDATES)

    if human_cp_col is not None:
        out["human_cp"] = pd.to_numeric(
            out[human_cp_col],
            errors="coerce",
        )
        out["z_human_cp"] = zscore(out["human_cp"])
        out["human_unexpectedness"] = 1 - out["human_cp"]

    if llm_cp_col is not None:
        out["llm_cp"] = pd.to_numeric(
            out[llm_cp_col],
            errors="coerce",
        )

    elif "target_surprisal_bits" in out.columns:
        # Convert GPT-2 target-word surprisal back into a model-derived
        # cloze-probability-like expectancy score.
        #
        # surprisal = -log2(P(target | context))
        # therefore:
        # P(target | context) = 2 ** (-surprisal)
        out["llm_cp"] = 2 ** (
            -pd.to_numeric(out["target_surprisal_bits"], errors="coerce")
        )

    if "llm_cp" in out.columns:
        out["z_llm_cp"] = zscore(out["llm_cp"])
        out["llm_unexpectedness"] = 1 - out["llm_cp"]

    if "human_cp" in out.columns and "llm_cp" in out.columns:
        out["cp_difference_human_minus_llm"] = (
            out["human_cp"] - out["llm_cp"]
        )
        out["abs_cp_disagreement"] = (
            out["cp_difference_human_minus_llm"].abs()
        )

    return out


# ---------------------------------------------------------------------
# Emotion features
# ---------------------------------------------------------------------

def add_emotion_features(out: pd.DataFrame, idx: int, target: str) -> None:

    if get_emotion_features is not None:
        try:
            features = get_emotion_features(target)

            if isinstance(features, dict):
                for key, value in features.items():
                    out.at[idx, key] = value
                return

        except Exception as e:
            log.warning("Emotion features failed for '%s': %s", target, e)

    if get_valence is not None:
        try:
            out.at[idx, "target_valence"] = get_valence(target)
            out.at[idx, "target_is_emotional"] = bool(get_valence(target) != 0.0)
            return

        except Exception as e:
            log.warning("Fallback valence failed for '%s': %s", target, e)

    out.at[idx, "target_valence"] = np.nan
    out.at[idx, "target_is_emotional"] = np.nan


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def save_predictor_diagnostics(df: pd.DataFrame, output_prefix: Path) -> None:
    diagnostic_predictors = [
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

    available = [
        col for col in diagnostic_predictors
        if col in df.columns
    ]

    if len(available) < 2:
        log.warning(
            "Not enough diagnostic predictors available: %s",
            output_prefix,
        )
        return

    numeric = df[available].copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")

    # Remove constant columns because they break correlations/VIF.
    numeric = numeric.loc[:, numeric.std(skipna=True) != 0]

    if numeric.shape[1] < 2:
        log.warning(
            "Not enough non-constant numeric predictors for diagnostics: %s",
            output_prefix,
        )
        return

    numeric = numeric.fillna(numeric.mean())

    corr = numeric.corr()
    corr.to_csv(str(output_prefix) + "_correlations.csv")

    vif_rows = []

    for i, col in enumerate(numeric.columns):
        try:
            vif = variance_inflation_factor(numeric.values, i)
        except Exception:
            vif = np.nan

        vif_rows.append(
            {
                "predictor": col,
                "vif": vif,
            }
        )

    vif_df = pd.DataFrame(vif_rows).sort_values(
        "vif",
        ascending=False,
    )

    vif_df.to_csv(str(output_prefix) + "_vif.csv", index=False)

    log.info("Saved diagnostics: %s", output_prefix)
    log.info("Diagnostic predictors used: %s", list(numeric.columns))


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def should_skip_file(path: Path) -> bool:
    name = path.name.lower()
    full = str(path).lower()

    # Skip cloze-survey metadata JSON.
    # This file describes the task; it is not a trial/stimulus table.
    if name.endswith("cloze-probability-survey.json"):
        return True

    if any(pattern in name for pattern in EXCLUDE_FILE_PATTERNS):
        return True

    if "language_outputs" in full:
        return True

    if "eeg_outputs" in full:
        return True

    if "derivatives" in full and "stim" not in full and "stimuli" not in full:
        return True

    return False


def find_data_files(path: Path) -> list[Path]:
    allowed_names = {
        "n400stimset_cloze-probability-survey_results.tsv",
        "n400stimset_stimuli_parameters.tsv",
    }

    if path.is_file():
        if path.name.lower() in allowed_names and not should_skip_file(path):
            return [path]
        return []

    files = []

    for root, _, names in os.walk(path):
        for name in names:
            p = Path(root) / name

            if p.name.lower() not in allowed_names:
                continue

            if should_skip_file(p):
                continue

            files.append(p)

    return sorted(files)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix == ".json":
        try:
            return pd.read_json(path)

        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)

            if isinstance(data, dict):
                for key in [
                    "data",
                    "items",
                    "records",
                    "stimuli",
                    "sentences",
                ]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])

                return pd.json_normalize(data)

    raise ValueError(f"Unsupported or unreadable file: {path}")


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def validate_detected_table(
    df: pd.DataFrame,
    source_file: Path,
    text_col: Optional[str],
    target_col: Optional[str],
) -> bool:
    if text_col is None:
        log.warning("Skipping file because no real sentence column was found: %s", source_file)
        log.warning("Columns were: %s", list(df.columns))
        return False

    if not looks_like_text_column(df[text_col]):
        log.warning(
            "Skipping file because detected text column does not look like real sentences: %s",
            source_file,
        )
        log.warning("Detected text column was: %s", text_col)
        return False

    if target_col is not None:
        quality = target_column_quality(df[target_col])

        if quality < 0.50:
            log.warning(
                "Skipping file because detected target column looks wrong: %s",
                source_file,
            )
            log.warning("Detected target column was: %s", target_col)
            log.warning("Target quality was: %.2f", quality)
            return False

    return True

def prepare_n400_specific_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cloze survey results:
    # sentential_context = context shown to participant
    # key = expected completion
    if "sentential_context" in df.columns and "key" in df.columns:
        df["context_for_metrics"] = df["sentential_context"].astype(str).str.strip()
        df["target_word_for_metrics"] = df["key"].astype(str).map(clean_word)
        df["sentence_for_metrics"] = (
            df["context_for_metrics"].astype(str).str.rstrip()
            + " "
            + df["target_word_for_metrics"].astype(str).str.strip()
        )

    # Stimuli parameters:
    # columns 1-8 are the actual presented sentence words.
    # The target is the actual final word in the presented sentence.
    numbered_cols = [
        col for col in df.columns
        if str(col).strip().isdigit()
    ]

    if numbered_cols:
        numbered_cols = sorted(numbered_cols, key=lambda x: int(str(x).strip()))

        sentences = []

        for _, row in df.iterrows():
            words = []

            for col in numbered_cols:
                value = row[col]

                if pd.isna(value):
                    continue

                value = str(value).strip()

                if value and value.lower() != "nan":
                    words.append(value)

            sentence = " ".join(words).strip()
            sentences.append(sentence)

        df["sentence_for_metrics"] = sentences
        df["target_word_for_metrics"] = df["sentence_for_metrics"].map(derive_final_word)

        df["context_for_metrics"] = [
            derive_context(sentence, target)
            for sentence, target in zip(
                df["sentence_for_metrics"],
                df["target_word_for_metrics"],
            )
        ]

    return df

def process_table(
    df: pd.DataFrame,
    source_file: Path,
    surprisal_model: SurprisalModel,
    semantic_model: SentenceMetrics,
    syntax_model,
) -> Optional[pd.DataFrame]:

    if df.empty:
        log.warning("Skipping empty file: %s", source_file)
        return None

    df = prepare_n400_specific_columns(df)

    text_col = detect_text_column(df)
    target_col = detect_target_column(df)
    context_col = detect_context_column(df)
    condition_col = detect_condition_column(df)

    if condition_col is not None and text_col == condition_col:
        log.warning(
            "Skipping file because condition was detected as text. This is not a stimulus table: %s",
            source_file,
        )
        return None

    if not validate_detected_table(df, source_file, text_col, target_col):
        return None

    log.info("Processing: %s", source_file)
    log.info("Detected text column: %s", text_col)
    log.info("Detected target column: %s", target_col)
    log.info("Detected context column: %s", context_col)
    log.info("Detected condition column: %s", condition_col)

    out = df.copy()

    out["source_file"] = str(source_file)
    out["detected_text_column"] = text_col

    if target_col is not None:
        out["detected_target_column"] = target_col

    if context_col is not None:
        out["detected_context_column"] = context_col

    if condition_col is not None:
        out["detected_condition_column"] = condition_col

    if "sentence_id" not in out.columns:
        out["sentence_id"] = np.arange(1, len(out) + 1)

    for idx, row in out.iterrows():
        sentence = str(row[text_col]).strip()

        if not looks_like_sentence(sentence):
            continue

        if target_col is not None and pd.notna(row[target_col]):
            target = clean_word(row[target_col])
        else:
            target = derive_final_word(sentence)

        if not is_probably_real_word(target):
            log.warning(
                "Skipping row %s in %s because target does not look like a real word: %s",
                idx,
                source_file,
                target,
            )
            continue

        if context_col is not None and pd.notna(row[context_col]):
            context = str(row[context_col]).strip()
        else:
            context = derive_context(sentence, target)

        out.at[idx, "target_word_used"] = target
        out.at[idx, "context_used"] = context
        out.at[idx, "n_words_sentence"] = len(sentence.split())
        out.at[idx, "target_n_letters"] = len(target)

        out.at[idx, "target_zipf_frequency"] = get_zipf_frequency(target)
        out.at[idx, "target_raw_frequency"] = get_word_frequency(target)

        phon = get_phonology_for_word(target)

        out.at[idx, "target_n_phonemes"] = phon.get("n_phonemes")
        out.at[idx, "target_n_syllables"] = phon.get("n_syllables")
        out.at[idx, "target_onset_phoneme"] = phon.get("onset_phoneme")

        add_emotion_features(out, idx, target)

        try:
            if context:
                out.at[idx, "target_surprisal_bits"] = (
                    surprisal_model.word_surprisal(context, target)
                )
            else:
                out.at[idx, "target_surprisal_bits"] = np.nan

        except Exception as e:
            log.warning(
                "Target surprisal failed at row %s in %s: %s",
                idx,
                source_file,
                e,
            )
            out.at[idx, "target_surprisal_bits"] = np.nan

        try:
            out.at[idx, "sentence_mean_surprisal"] = (
                surprisal_model.sentence_surprisal(sentence)
            )

            out.at[idx, "sentence_perplexity"] = (
                surprisal_model.sentence_perplexity(sentence)
            )

        except Exception as e:
            log.warning(
                "Sentence surprisal failed at row %s in %s: %s",
                idx,
                source_file,
                e,
            )
            out.at[idx, "sentence_mean_surprisal"] = np.nan
            out.at[idx, "sentence_perplexity"] = np.nan

        try:
            if context:
                out.at[idx, "context_target_similarity"] = (
                    semantic_model.context_target_similarity(context, target)
                )
            else:
                out.at[idx, "context_target_similarity"] = np.nan

        except Exception as e:
            log.warning(
                "Semantic similarity failed at row %s in %s: %s",
                idx,
                source_file,
                e,
            )
            out.at[idx, "context_target_similarity"] = np.nan

        try:
            syntax = compute_syntax(sentence, syntax_model)

            for key, value in syntax.items():
                out.at[idx, key] = value

        except Exception as e:
            log.warning(
                "Syntax failed at row %s in %s: %s",
                idx,
                source_file,
                e,
            )

    if "target_surprisal_bits" in out.columns:
        out["z_target_surprisal"] = zscore(out["target_surprisal_bits"])

    if "context_target_similarity" in out.columns:
        out["z_context_target_similarity"] = zscore(
            out["context_target_similarity"]
        )

    if (
        "z_target_surprisal" in out.columns
        and "z_context_target_similarity" in out.columns
    ):
        out["context_shift_index"] = (
            out["z_target_surprisal"]
            - out["z_context_target_similarity"]
        )

    if "context_target_similarity" in out.columns:
        out["prior_context_strength"] = out["context_target_similarity"]

    out = add_cloze_metrics(out)

    return out


def safe_output_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)

    return stem + "_language_metrics.csv"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw stimulus files and compute language predictors."
    )

    parser.add_argument(
        "path",
        help="Path to a raw stimulus CSV/TSV/JSON file or folder.",
    )

    parser.add_argument(
        "--output-dir",
        default="language_outputs",
        help="Folder where output CSV files will be saved.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: cpu, cuda, or leave blank for auto.",
    )

    parser.add_argument(
        "--surprisal-model",
        default="gpt2",
        help="Causal language model for target and sentence surprisal.",
    )

    parser.add_argument(
        "--semantic-model",
        default="bert-base-uncased",
        help="Transformer model for semantic similarity.",
    )

    parser.add_argument(
        "--syntax-model",
        default="en_core_web_sm",
        help="spaCy model for syntax metrics.",
    )

    args = parser.parse_args()

    input_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    files = find_data_files(input_path)

    if not files:
        log.error("No usable raw stimulus CSV/TSV/JSON files found in: %s", input_path)
        return

    log.info("Found %d candidate stimulus files.", len(files))

    log.info("Loading surprisal model: %s", args.surprisal_model)
    surprisal_model = SurprisalModel(
        model_name=args.surprisal_model,
        device=args.device,
    )

    log.info("Loading semantic model: %s", args.semantic_model)
    semantic_model = SentenceMetrics(
        model_name=args.semantic_model,
        device=args.device,
    )

    log.info("Loading syntax model: %s", args.syntax_model)
    syntax_model = spacy.load(args.syntax_model)

    processed_count = 0
    all_results = []

    for file_path in files:
        try:
            df = load_table(file_path)

        except Exception as e:
            log.warning("Could not read %s: %s", file_path, e)
            continue

        result = process_table(
            df=df,
            source_file=file_path,
            surprisal_model=surprisal_model,
            semantic_model=semantic_model,
            syntax_model=syntax_model,
        )

        if result is None:
            continue

        output_path = output_dir / safe_output_name(file_path)
        result.to_csv(output_path, index=False)

        log.info("Saved: %s", output_path)

        diagnostics_prefix = output_dir / safe_output_name(file_path).replace(
            "_language_metrics.csv",
            "_predictor_diagnostics",
        )

        save_predictor_diagnostics(result, diagnostics_prefix)

        all_results.append(result)
        processed_count += 1

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        combined_output = output_dir / "ALL_language_metrics.csv"
        combined.to_csv(combined_output, index=False)

        log.info("Saved combined output: %s", combined_output)

        combined_diagnostics_prefix = output_dir / "ALL_predictor_diagnostics"
        save_predictor_diagnostics(combined, combined_diagnostics_prefix)

    log.info(
        "Finished. Processed %d/%d candidate files.",
        processed_count,
        len(files),
    )


if __name__ == "__main__":
    main()