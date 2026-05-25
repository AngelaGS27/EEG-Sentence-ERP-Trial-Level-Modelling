% eeg_analysis.m
%
% EEG re-analysis pipeline using EEGLAB and the LIMO EEG plugin.
%
% This script performs a regression-based analysis of EEG data from the
% Semantically Incongruent or Congruent Eggplants study (Pernet et al., 2022).
% It takes the BIDS derivative EEG data and the linguistic metrics produced
% by the Python processing pipeline and fits a general linear model at both
% the single-subject (first-level) and group (second-level) using LIMO EEG.
%
% Prerequisites
% -------------
%   1. EEGLAB (https://sccn.ucsd.edu/eeglab/) on the MATLAB path.
%   2. The LIMO EEG plugin installed inside EEGLAB/plugins/.
%   3. The BIDS derivative folder downloaded from:
%        https://datadryad.org/dataset/doi:10.5061/dryad.6wwpzgmx4
%   4. The stimuli metrics CSV produced by:
%        python/process_stimuli.py stimuli.csv --output stimuli_metrics.csv
%
% Directory layout expected
% -------------------------
%   <bids_root>/
%       derivatives/
%           eeglab/
%               sub-<id>/
%                   eeg/
%                       sub-<id>_task-<task>_eeg.set   (epoched data)
%
% Usage
% -----
%   1. Edit the configuration section below (bids_root, metrics_csv).
%   2. Run this script from the MATLAB command window:
%        >> eeg_analysis
%
% Output
% ------
%   Results are saved in <bids_root>/derivatives/limo/ following the LIMO
%   output conventions (one folder per subject, group-level results at root).

% =========================================================================
% CONFIGURATION  – edit these paths before running
% =========================================================================
bids_root    = '/path/to/bids/dataset';          % BIDS root directory
metrics_csv  = '../python/stimuli_metrics.csv';  % output of process_stimuli.py
task_label   = 'congruency';                     % BIDS task label
output_dir   = fullfile(bids_root, 'derivatives', 'limo');

% Regressors to include in the GLM (must match column names in metrics_csv)
regressors = {
    'cw_zipf_frequency',        % lexical frequency (Zipf scale)
    'cw_n_phonemes',            % number of phonemes
    'cw_n_syllables',           % number of syllables
    'cw_surprisal_bits',        % next-word surprisal (GPT-2)
    'sent_context_target_sim',  % sentence-level semantic congruency
};

% =========================================================================
% SETUP
% =========================================================================
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Ensure EEGLAB is on the path
try
    eeglab nogui;
catch
    error('EEGLAB not found. Please add EEGLAB to the MATLAB path.');
end

% Load stimuli metrics
metrics = readtable(metrics_csv);
fprintf('Loaded metrics for %d stimuli.\n', height(metrics));

% =========================================================================
% FIRST-LEVEL (SINGLE-SUBJECT) ANALYSIS
% =========================================================================

% Collect subject folders
subj_dirs = dir(fullfile(bids_root, 'derivatives', 'eeglab', 'sub-*'));
subj_dirs = subj_dirs([subj_dirs.isdir]);
fprintf('Found %d subjects.\n', numel(subj_dirs));

for s = 1 : numel(subj_dirs)
    subj_id  = subj_dirs(s).name;
    set_file = fullfile(subj_dirs(s).folder, subj_id, 'eeg', ...
        sprintf('%s_task-%s_eeg.set', subj_id, task_label));

    if ~isfile(set_file)
        warning('EEG file not found for %s; skipping.', subj_id);
        continue;
    end

    fprintf('\n[%d/%d] Processing subject: %s\n', s, numel(subj_dirs), subj_id);

    % Load EEG data
    EEG = pop_loadset(set_file);
    fprintf('  Loaded: %d epochs, %d channels, %.1f s epoch\n', ...
        EEG.trials, EEG.nbchan, EEG.xmax - EEG.xmin);

    % Build design matrix from stimulus metrics
    % The EEG epochs must have a 'stimulus_id' field in EEG.event that
    % matches the 'sentence_id' column in the metrics table.
    [X, reg_labels] = build_design_matrix(EEG, metrics, regressors);

    % First-level LIMO GLM
    subj_out = fullfile(output_dir, subj_id);
    if ~exist(subj_out, 'dir'), mkdir(subj_out); end

    limo_data.data      = EEG.data;    % channels x time x trials
    limo_data.data_size = size(EEG.data);
    limo_design.X       = X;
    limo_design.labels  = reg_labels;
    limo_design.name    = sprintf('%s_GLM', subj_id);

    limo_glm(limo_data, limo_design, subj_out);
    fprintf('  First-level GLM saved to: %s\n', subj_out);
end

% =========================================================================
% SECOND-LEVEL (GROUP) ANALYSIS
% =========================================================================
fprintf('\nRunning second-level (group) analysis...\n');
limo_random_robust(output_dir, regressors);
fprintf('Group results saved to: %s\n', output_dir);

fprintf('\nAnalysis complete.\n');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function [X, labels] = build_design_matrix(EEG, metrics, regressors)
% BUILD_DESIGN_MATRIX  Align stimulus metrics to EEG epochs.
%
% For each epoch in EEG, locate the matching row in METRICS using the
% 'stimulus_id' field from EEG.event, then stack the requested REGRESSORS
% into a design matrix X (n_epochs x n_regressors+1) with a constant term.
%
% Parameters
% ----------
% EEG        : EEGLAB EEG struct (must be epoched)
% metrics    : MATLAB table loaded from stimuli_metrics.csv
% regressors : cell array of column names to include
%
% Returns
% -------
% X      : numeric design matrix (n_epochs x n_regressors+1)
% labels : cell array of regressor labels (last element is 'constant')

n_trials = EEG.trials;
n_reg    = numel(regressors);
X        = zeros(n_trials, n_reg + 1);

% Epoch events: each epoch has one time-locking event; grab the first.
for t = 1 : n_trials
    epoch_events = EEG.epoch(t).eventtype;
    if iscell(epoch_events)
        stim_id = epoch_events{1};
    else
        stim_id = epoch_events;
    end

    % Match to metrics table
    if isnumeric(stim_id)
        row_idx = find(metrics.sentence_id == stim_id, 1);
    else
        row_idx = find(strcmp(metrics.sentence_id, stim_id), 1);
    end

    if isempty(row_idx)
        warning('No metrics found for stimulus "%s" (trial %d).', ...
            string(stim_id), t);
        X(t, :) = NaN;
        continue;
    end

    for r = 1 : n_reg
        X(t, r) = metrics.(regressors{r})(row_idx);
    end
end

% Add constant term
X(:, end) = 1;
labels = [regressors, {'constant'}];

% Z-score continuous regressors (exclude constant)
for r = 1 : n_reg
    col    = X(:, r);
    valid  = ~isnan(col);
    mu     = mean(col(valid));
    sigma  = std(col(valid));
    if sigma > 0
        X(valid, r) = (col(valid) - mu) / sigma;
    end
end
end
