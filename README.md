# Semantically Incongruent or Congruent Eggplants – Revised EEG Language Analysis

A Python-based re-analysis of [Toffolo et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9540983/).

This repository re-analyses trial-level EEG/ERP data by linking ERP component amplitudes to per-trial linguistic predictors, including cloze probability, surprisal, semantic fit, lexical frequency, phonology, syntactic complexity, and affective/emotional properties of the target word.

The main aim is to move beyond condition-level ERP averaging by modelling item/trial-level predictors directly.

---

## Repository layout

    export_erp_long.py          – export ERP .mat files into long-format EEG CSV
    process_stimuli.py          – compute linguistic predictors from stimulus files
    run_component_lmm.py        – fit RP, N400, and LPC mixed-effects models
    predictor_diagnostics.py    – predictor correlations and VIF diagnostics

    word_frequency.py           – lexical frequency using wordfreq
    phonology.py                – phonological features using CMU Pronouncing Dictionary
    emotion.py                  – NRC VAD and NRC Emotion Lexicon predictors
    syntax_metrics.py           – syntactic complexity using spaCy
    surprisal.py                – target and sentence surprisal using GPT-2
    sentence_metrics.py         – context-target semantic similarity using BERT
    cloze_metrics.py            – cloze probability transformations

    valence-NRC-VAD-Lexicon-v2.1.txt
    arousal-NRC-VAD-Lexicon-v2.1.txt
    dominance-NRC-VAD-Lexicon-v2.1.txt
    NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

    requirements.txt            – Python dependencies

---

## Data sources

| Dataset | URL |
|---|---|
| Stimuli | https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffkg |
| EEG / ERP derivatives | https://datadryad.org/dataset/doi:10.5061/dryad.6wwpzgmx4 |

---

## Included affective lexicons

The repository already includes the NRC affective lexicon files required by `emotion.py`.

Included files:

    valence-NRC-VAD-Lexicon-v2.1.txt
    arousal-NRC-VAD-Lexicon-v2.1.txt
    dominance-NRC-VAD-Lexicon-v2.1.txt
    NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

These files are automatically loaded by `emotion.py`.

They are used to compute:

    target_valence
    target_arousal
    target_dominance
    target_is_emotional
    target_emotion_anger
    target_emotion_fear
    target_emotion_joy
    target_emotion_sadness
    target_emotion_disgust
    target_emotion_surprise
    target_emotion_trust
    target_emotion_positive
    target_emotion_negative

---

## Installation

Install dependencies:

    pip install -r requirements.txt

On first run, the scripts automatically download HuggingFace model weights for GPT-2 and BERT.

---

## Recommended run order

1. Compute linguistic predictors with `process_stimuli.py`.

   This also automatically creates predictor correlation and VIF diagnostic files.

2. Export ERP data with `export_erp_long.py`.

3. Fit RP, N400, and LPC mixed-effects models with `run_component_lmm.py`.

---

## Step 1 – Compute linguistic predictors

Run the linguistic predictor pipeline on the stimulus files or stimulus folder.

Example:

    python process_stimuli.py path/to/stimuli_or_folder --output-dir language_outputs

Outputs:

    language_outputs/<original_filename>_language_metrics.csv
    language_outputs/ALL_language_metrics.csv

Predictor diagnostics automatically produced:

    *_predictor_diagnostics_correlations.csv
    *_predictor_diagnostics_vif.csv

The script automatically detects:

- sentence/stimulus columns
- target-word columns
- context columns
- condition columns

The script computes predictors including:

- human cloze probability
- LLM cloze probability
- surprisal
- sentence perplexity
- semantic/context-target similarity
- lexical frequency
- phonology
- syntactic complexity
- affective/emotional predictors

The predictor diagnostics generated during processing can be used to inspect:

- collinearity
- redundant predictors
- highly correlated metrics
- VIF values

before interpreting the mixed-effects models.

---

## Step 2 – Export ERP data into long format

Export the ERP `.mat` derivatives into trial-level long-format EEG CSV files.

Example:

    python export_erp_long.py path/to/derivatives/erps --output-dir eeg_outputs

Outputs:

    eeg_outputs/sub-XX_erp_long.csv
    eeg_outputs/ALL_erp_long.csv

The script combines subject files in chunks for efficient processing of large datasets.

The exported ERP table contains:

- subject
- condition
- item
- trial
- channel
- time
- amplitude

These outputs are later merged with the linguistic predictors during component modelling.

---

## Step 3 – Fit ERP component mixed-effects models

Run all ERP components:

    python run_component_lmm.py --erp-long eeg_outputs/ALL_erp_long.csv --predictors language_outputs/ALL_language_metrics.csv --component ALL --interactions

Run a single component:

    python run_component_lmm.py --erp-long eeg_outputs/ALL_erp_long.csv --predictors language_outputs/ALL_language_metrics.csv --component N400 --interactions

Outputs:

    model_outputs/model_data_RP.csv
    model_outputs/model_RP_summary.txt
    model_outputs/model_RP_formula.txt
    model_outputs/model_RP_coefficients.csv
    model_outputs/model_RP_used_rows.csv

    model_outputs/model_data_N400.csv
    model_outputs/model_N400_summary.txt
    model_outputs/model_N400_formula.txt
    model_outputs/model_N400_coefficients.csv
    model_outputs/model_N400_used_rows.csv

    model_outputs/model_data_LPC.csv
    model_outputs/model_LPC_summary.txt
    model_outputs/model_LPC_formula.txt
    model_outputs/model_LPC_coefficients.csv
    model_outputs/model_LPC_used_rows.csv

The mixed-effects models test whether ERP amplitudes are explained by predictors such as:

- human cloze probability
- LLM-derived cloze probability
- target-word surprisal
- semantic/context-target similarity
- lexical frequency
- phonology
- sentence complexity
- affective/emotional predictors
- condition or deviation predictors, where available

The script automatically:

- computes RP, N400, and LPC component amplitudes
- averages EEG within component time windows and ROIs
- merges ERP amplitudes with linguistic predictors
- z-scores numeric predictors
- processes large ERP CSV files in chunks
- supports interaction terms
- fits subject-level mixed-effects models with item variance components

---

## Main linguistic predictors

| Column | Description |
|---|---|
| `sentence_id` | Sentence/trial identifier |
| `target_word_used` | Target or sentence-final word used for predictors |
| `context_used` | Sentence context before the target word |
| `human_cp` | Human cloze probability, if available |
| `llm_cp` | LLM-derived cloze probability, if available |
| `human_unexpectedness` | 1 − human cloze probability |
| `llm_unexpectedness` | 1 − LLM cloze probability |
| `cp_difference_human_minus_llm` | Human minus LLM cloze probability |
| `abs_cp_disagreement` | Absolute human/LLM CP disagreement |
| `target_surprisal_bits` | GPT-2 surprisal of the target word |
| `sentence_mean_surprisal` | Mean sentence surprisal |
| `sentence_perplexity` | Sentence perplexity |
| `context_target_similarity` | BERT cosine similarity between context and target |
| `context_shift_index` | Combined surprisal/contextual-fit index |
| `prior_context_strength` | Semantic fit between context and target |
| `target_zipf_frequency` | Zipf lexical frequency |
| `target_raw_frequency` | Raw lexical frequency |
| `target_n_letters` | Number of letters in target word |
| `target_n_phonemes` | Number of phonemes |
| `target_n_syllables` | Number of syllables |
| `target_onset_phoneme` | Initial phoneme |
| `target_valence` | NRC VAD valence score |
| `target_arousal` | NRC VAD arousal score |
| `target_dominance` | NRC VAD dominance score |
| `target_is_emotional` | Whether the target has affective/emotional marking |
| `target_emotion_*` | NRC Emotion Lexicon category flags |
| `syntax_n_tokens` | Number of spaCy tokens |
| `syntax_mean_dependency_distance` | Mean dependency distance |
| `syntax_max_parse_depth` | Maximum parse depth |
| `syntax_mean_parse_depth` | Mean parse depth |
| `syntax_n_subordinate_clauses` | Number of subordinate clauses |

---

## ERP components modelled

| Component | Window | Approximate scalp region | Interpretation |
|---|---|---|---|
| RP | 150–250 ms | posterior/parietal-occipital | Recognition potential / early lexical processing |
| N400 | 300–500 ms | centro-parietal | Semantic integration / contextual expectancy |
| LPC | 500–800 ms | centro-parietal/posterior | Later reanalysis, semantic/syntactic difficulty, integration effort |

---

## Analysis rationale

The models test whether ERP amplitudes are explained by:

- human cloze probability
- LLM-derived cloze probability
- target-word surprisal
- context-target semantic similarity
- lexical frequency
- phonology
- sentence complexity
- affective/emotional predictors
- condition or deviation predictors, where available

This allows congruency effects to be decomposed into more specific linguistic mechanisms rather than treated only as broad condition differences.
