from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

VALENCE_FILE = BASE_DIR / "valence-NRC-VAD-Lexicon-v2.1.txt"
AROUSAL_FILE = BASE_DIR / "arousal-NRC-VAD-Lexicon-v2.1.txt"
DOMINANCE_FILE = BASE_DIR / "dominance-NRC-VAD-Lexicon-v2.1.txt"
EMOTION_FILE = BASE_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

EMOTION_CATEGORIES = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "negative",
    "positive",
    "sadness",
    "surprise",
    "trust",
]


def clean_word(word: Any) -> str:
    return (
        str(word)
        .strip()
        .lower()
        .strip(".,;:!?\"'()[]{}<>")
    )


def load_vad_file(path: Path, metric_name: str) -> dict[str, float]:
    if not path.exists():
        print(f"WARNING: missing VAD file: {path}")
        return {}

    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lower() for c in df.columns]

    lookup = {}

    for _, row in df.iterrows():
        term = clean_word(row["term"])

        try:
            value = float(row[metric_name])
        except Exception:
            continue

        lookup[term] = value

    return lookup


def load_emotion_file(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        print(f"WARNING: missing emotion file: {path}")
        return {}

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["term", "emotion", "association"],
    )

    lookup = {}

    for _, row in df.iterrows():
        term = clean_word(row["term"])
        emotion = str(row["emotion"]).strip().lower()

        try:
            association = int(row["association"])
        except Exception:
            association = 0

        if emotion not in EMOTION_CATEGORIES:
            continue

        if term not in lookup:
            lookup[term] = {cat: 0 for cat in EMOTION_CATEGORIES}

        lookup[term][emotion] = association

    return lookup


VALENCE = load_vad_file(VALENCE_FILE, "valence")
AROUSAL = load_vad_file(AROUSAL_FILE, "arousal")
DOMINANCE = load_vad_file(DOMINANCE_FILE, "dominance")
EMOTIONS = load_emotion_file(EMOTION_FILE)


SUFFIX_RULES = [
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
]


def lookup_with_suffix(word: str, lookup: dict, default):
    word = clean_word(word)

    if word in lookup:
        return lookup[word]

    for suffix, replacement in SUFFIX_RULES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            candidate = word[: -len(suffix)] + replacement

            if candidate in lookup:
                return lookup[candidate]

    return default


def get_emotion_features(word: Any) -> dict:
    word = clean_word(word)

    valence = lookup_with_suffix(word, VALENCE, 0.5)
    arousal = lookup_with_suffix(word, AROUSAL, 0.5)
    dominance = lookup_with_suffix(word, DOMINANCE, 0.5)

    emotion_flags = lookup_with_suffix(
        word,
        EMOTIONS,
        {cat: 0 for cat in EMOTION_CATEGORIES},
    )

    emotion_count = sum(
        emotion_flags[cat]
        for cat in EMOTION_CATEGORIES
        if cat not in {"positive", "negative"}
    )

    is_emotional = (
        emotion_count > 0
        or emotion_flags.get("positive", 0) == 1
        or emotion_flags.get("negative", 0) == 1
        or abs(float(valence) - 0.5) > 0.15
        or abs(float(arousal) - 0.5) > 0.15
    )

    output = {
        "target_valence": float(valence),
        "target_arousal": float(arousal),
        "target_dominance": float(dominance),
        "target_is_emotional": bool(is_emotional),
        "target_emotion_count": int(emotion_count),
    }

    for cat in EMOTION_CATEGORIES:
        output[f"target_emotion_{cat}"] = int(emotion_flags.get(cat, 0))

    active_emotions = [
        cat
        for cat in EMOTION_CATEGORIES
        if emotion_flags.get(cat, 0) == 1
    ]

    output["target_emotion_labels"] = (
        ";".join(active_emotions)
        if active_emotions
        else "none"
    )

    return output


def get_valence(word: Any) -> float:
    return float(get_emotion_features(word)["target_valence"])