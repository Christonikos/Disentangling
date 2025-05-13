function [stimuli_words, stimuli_datasets, training_words] = load_stimuliLocalGlobal(params)
%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the visual stimuli
%-----------------------

% Load the blocks
% ---------------
n_blocks         = numel(params.text_filename);
stimuli_words    = cell(n_blocks,1);
stimuli_datasets = cell(n_blocks,1); 
for b_id = 1:n_blocks % block ID
    warning off;
    curr_filename = ...
        fullfile(params.Visualpath, ['subj_',num2str(params.subject)], ...
        params.text_filename{b_id});
    curr_dataset  = dataset('file',curr_filename);
    stimuli_datasets{b_id} = curr_dataset; 
    % Cell of cells: cell for each sentence, containing cells for each word    
    stimuli_words{b_id} = cellfun(@(x) regexp(x, ' ', 'split'),...
        curr_dataset.sentence, 'UniformOutput',false); 
    for trial = 1:numel(stimuli_words{b_id})
        stimuli_words{b_id}{trial} = ...
        stimuli_words{b_id}{trial}(~cellfun('isempty',stimuli_words{b_id}{trial}));
    end
end

% Load the training
% stimuli.
% -----------------
tr_stimuli       = fullfile(params.path2stim,'training_trials.csv');
training_dataset = readtable(tr_stimuli);
training_words   = cellfun(@(x) regexp(x, ' ', 'split'),...
    training_dataset.sentence, 'UniformOutput',false);
for tt = 1:numel(training_words) % training trial
    training_words{tt} = ...
        training_words{tt}(~cellfun('isempty',training_words{tt}));
end

