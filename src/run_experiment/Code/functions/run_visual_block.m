function run_visual_block(handles, block, stimuli_words, VisualTrialOrder, fid_log, triggers, cumTrial, params, events, stimuli_datasets)
% Initialize table to hold the behavioral responses.
T = cell2table(cell(length(stimuli_words),4));
T.Properties.VariableNames = {'subject_response','RT','Behavioral', 'Behavioral_index'};

% Number of blocks
nblocks = numel(params.text_filename);
if params.photodiode
    %---------------------------------------------------------------------------%
    %--------**PHOTODIODE**-----------------------------------------------------%
    %% Introduce rectangle for the Photodiode
    [screenXpixels, screenYpixels] = Screen('WindowSize', handles.win);
    % Make rectangle to present on top right for the tracker to see
    baseRect = [0 0 30 30]; % size
    xTopRight = screenXpixels * 0.95; % position
    yTopRight = screenYpixels * 0.05; % position
    TopRightRect = CenterRectOnPointd(baseRect, xTopRight, yTopRight);
    %---------------------------------------------------------------------------%
end



%%%%%%% WAIT FOR KEY PRESS
if block == 1
    present_intro_slide(params, handles);
%     KbStrokeWait;
%     KbQueueStart;
    wait_for_key_press()
else
    DrawFormattedText(handles.win, ...
        ['Starting block: ' num2str(block) '/' num2str(nblocks) newline ...
        'Please wait for the block to start.'], ...
        'center', 'center', handles.white);
    Screen('Flip',handles.win);
    wait_for_key_press()
end

%%%%%%% BLOCK START: mark a new block with four 255 triggers separated 200ms from each other
block_start = GetSecs;
for i=1:4
    send_trigger(triggers, handles, params, events, 'event255', 0.2)
end

% %%%%%%% WRITE TO LOG
fprintf(fid_log,['BlockStart\t' ...
    num2str(block) '\t' ...
    num2str(0) '\t' ...
    num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
    '-' '\t' ...
    num2str(block_start) '\t' ...
    '' '\r\n' ...
    ]); % write to log file

% loop through trials
for trial=1:length(stimuli_words)
    %% -------------------------------------------------------------------------------- %%
    % RSVP parameters
    word_cnt       = 0;
    curr_sentence  = stimuli_words{trial};
    curr_violIndex = stimuli_datasets{block}.violIndex(trial);
    
    %% -------------------------------------------------------------------------------- %%
    
    %% DECISION SCREEN SET-UP
    % DESIGN THE DECISION SCREEN AND SWAP RANDONMLY THE ORDER OF APPEARANCE
    ok    = 'OK';
    wrong = 'Wrong';
    space = '     ';
    available_words = {ok, wrong};
    [word_1, idx] = datasample(available_words,1);
    available_words(idx) = [];
    word_2 = available_words;
    decision_screen   = [word_1{1},space,word_2{1}];
    
    % Locate the position of the 'OK' word in the created random panel
    idx = find(ismember(decision_screen,ok));
    if idx(1) > numel(decision_screen)/2
        curr_ok_location     = 'right';
        curr_wrong_location  = 'left';
    else
        curr_ok_location     = 'left';
        curr_wrong_location  = 'right';
    end
    %% -------------------------------------------------------------------------------- %%
    % KEYBOARD INPUT
    [~, ~, keyCode] = KbCheck;
    if keyCode('ESCAPE')
        DisableKeysForKbCheck([]);
        Screen('CloseAll');
        return
    end
    cumTrial = cumTrial+1;
    stimulus = VisualTrialOrder(trial);
    %% -------------------------------------------------------------------------------- %%
    
    %-------------------------------------------------------------------------%
    % LOG-FILE INFORMATION
    curr_condition = stimuli_datasets{block}(trial,:).condition{1};
    base_condition = curr_condition(1:4);
    if contains(curr_condition,'C')==1; emb='objRC'; else emb='PP'; end
    if contains(curr_condition,'sing')==1; num='sing'; else num='plur'; end
    if contains(curr_condition,'semantic')==1; typ='sem'; else typ='synt'; end
    cond_event_name = join([base_condition,'_',emb,'_',num,'_',typ]);
    %-------------------------------------------------------------------------%
    
    fprintf('Block %i, trial %i, Condition %s, Emb %s, Number %s, Type %s\n', ...
        block, trial, base_condition, emb, num, typ)
    
    %% -------------------------- FIXATION ----------------------------------- %%
    %%%%%%%% DRAW FIXATION BEFORE SENTENCE (duration: params.fixation_duration)
    DrawFormattedText(handles.win, '+', 'center', 'center', handles.white);
    if params.photodiode
        %[RECT presentation]
        Screen('FillRect', handles.win, handles.white, TopRightRect); % draw rectangle for tracker to see
    end
    fixation_onset = Screen('Flip', handles.win);
    % %%%%%%% WRITE TO LOG
    fprintf(fid_log,['Fix\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        '+' '\t' ...
        num2str(fixation_onset) '\t' ...
        cond_event_name '\t' ...
        '+' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file
    
    if triggers
        send_trigger(triggers, handles, params, events, 'StartFixation',  0)
    end
    
    WaitSecs('UntilTime',fixation_onset+params.fixation_duration_visual_block); %Wait before trial
    
    %% ------------------------------------------------------------------------- %%
    
    
    
    %% -------------------------- RSVP ----------------------------------- %%
    %%%%%%%%% START RSVP for current sentence

    for word = 1:numel(curr_sentence)
        word_cnt = word_cnt + 1;
        
        %%%%%%%%%%%%% TEXT ON %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        DrawFormattedText(handles.win, curr_sentence{word}, 'center', 'center', handles.white);
        if params.photodiode
            %[RECT presentation]
            Screen('FillRect', handles.win, handles.white, TopRightRect); % draw rectangle for tracker to see
        end
        word_onset = Screen('Flip', handles.win); % Word ON
  
        %%%%%%%%%%%%%%%% ONSETS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if word==1
            onset_id = 'FirstStimVisualOn\t';
            if triggers;send_trigger(triggers, handles, params, events, 'first_word_onset', 0);end
        elseif word==7
            onset_id = 'LastStimVisualOn\t';
            if triggers;send_trigger(triggers, handles, params, events, 'last_word_onset', 0);end
        else
            onset_id = 'StimVisualOn\t';
            if triggers;send_trigger(triggers, handles, params, events, 'StartWord', 0);end
        end
      
        WaitSecs('UntilTime', word_onset + params.stimulus_ontime);
        
        
        % WRITE TO LOG
        fprintf(fid_log,[onset_id ...
            num2str(block) '\t' ...
            num2str(trial) '\t' ...
            num2str(stimulus) '\t' ... % Stimulus serial number in original stimulus text file
            num2str(word_cnt) '\t' ...  %
            num2str(word_onset) '\t' ...
            cond_event_name '\t' ...
            curr_sentence{word} '\t' ...
            base_condition '\t' ...
            emb '\t' ...
            num '\t' ...
            typ '\r\n'  ...
            ]); % write to log file

        
        %%%%%%%%%%%%% TEXT OFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        word_offset = Screen('Flip', handles.win); % Word OFF
        %%%%%%%%%%%%%%%% OFFSETS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if word==1
            offset_id = 'FirstStimVisualOff\t';
            if triggers;send_trigger(triggers, handles, params, events, 'first_word_ofset', 0); end
        elseif word==7
            offset_id = 'LastStimVisualOff\t';
            if triggers;send_trigger(triggers, handles, params, events, 'last_word_ofset', 0); end
        else
            offset_id = 'StimVisualOff\t';
            if triggers;send_trigger(triggers, handles, params, events, 'EndWord', 0);end
        end
        
        fprintf(fid_log,[offset_id ...
            num2str(block) '\t' ...
            num2str(trial) '\t' ...
            num2str(stimulus) '\t' ... % Stimulus serial number in original stimulus text file
            num2str(word_cnt) '\t' ...  %
            num2str(word_offset) '\t' ...
            cond_event_name '\t' ...
            curr_sentence{word} '\t' ...
            base_condition '\t' ...
            emb '\t' ...
            num '\t' ...
            typ '\r\n'  ...
            ]); % write to log file
        
        WaitSecs('UntilTime', word_offset + params.stimulus_offtime);
    end % word
    
    
    %% ---------------------- ISI TO PANEL  ------------------------------ %%
    DrawFormattedText(handles.win, '+', 'center', 'center', [255,255,255]);
    if params.photodiode
        %[RECT presentation]
        Screen('FillRect', handles.win, handles.white, TopRightRect); % draw rectangle for tracker to see
    end
    fix2decision_onset  = Screen('Flip', handles.win);
    if triggers
        send_trigger(triggers, handles, params, events, 'StartFix2Decision', 0)
    end
    
    % %%%%%%% WRITE TO LOG
    fprintf(fid_log,['Fix2DecisionON\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        '+' '\t' ...
        num2str(fix2decision_onset) '\t' ...
        cond_event_name '\t' ...
        '+' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file
    WaitSecs('UntilTime', fix2decision_onset + params.SOA_visual);
    
    fix2decision_offset = Screen('Flip', handles.win);
    if triggers
        send_trigger(triggers, handles, params, events, 'EndFix2Decision', 0)
    end
    
    % %%%%%%% WRITE TO LOG
    fprintf(fid_log,['Fix2DecisionOFF\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        '+' '\t' ...
        num2str(fix2decision_offset) '\t' ...
        cond_event_name '\t' ...
        '+' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file


    
    %% ------------------------- DECISION SCREEN ONLINE ------------------------------- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PANEL ON  %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DrawFormattedText(handles.win, decision_screen, 'center', 'center', handles.white);
    if params.photodiode
        %[RECT presentation]
        Screen('FillRect', handles.win, handles.white, TopRightRect); % draw re
    end
    panel_onset= Screen('Flip', handles.win); % Pannel ON
    if triggers
        send_trigger(triggers, handles, params, events, 'StartPanel', 0)
    end
    
    % %%%%%%% WRITE TO LOG
    fprintf(fid_log,['PanelOn\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file (?)
        'Panel' '\t' ...
        num2str(panel_onset) '\t' ...
        cond_event_name '\t' ...
        'Panel' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file


    
    if strcmp(params.method,'MEG')
        clear Response pressed_decision Key
        Response  = '';
        [~, firstPress] = KbQueueCheck;
        % Define the port 
        port1  = hex2dec('378');
        
        % participant should respond here now !
        Key = 0;
         while (GetSecs <= panel_onset + params.panel_ontime)
             Key  = inp(port1+1);
             if Key~=0
                 firstPress(handles.Key) = GetSecs;
%                  outp(888,0) % Set the trigger channel back to zero, so we only record the key press.
                 if triggers;send_trigger(triggers, handles, params, events, 'PressKey', 0);end
                 pressed_decision =1;
                 break
             end
         end    
        
        
        if Key==32 
            Response = 'Right';
        elseif Key==64
            Response = 'Left';
        else 
            pressed_decision =0;
        end

    elseif strcmp(params.method,'iEEG')
        [pressed_decision, firstPress] = KbQueueCheck; % Collect keyboard events since KbQueueStart was invoked
    end
    
    
    if params.photodiode
        %[RECT presentation]
        Screen('FillRect', handles.win, handles.white, TopRightRect); % draw re
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PANEL OFF %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    panel_offset    = Screen('Flip', handles.win); % Pannel OFF
    outp(888,0) 
    if triggers
        send_trigger(triggers, handles, params, events, 'EndPanel', 0)
    end
    
    fprintf(fid_log,['PanelOff\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        'Panel' '\t' ...
        num2str(panel_offset) '\t' ...
        cond_event_name '\t' ...
        'Panel' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file
    
    
    %%%%%%%%%%%%%%%%%
    % BEHAVIORAL %%%%
    %%%%%%%%%%%%%%%%%
    if pressed_decision
        %%%%%%%%%%%%%%%%%
        % LOG-FILES %%%%
        %%%%%%%%%%%%%%%%%
        
        fprintf(fid_log,['KeyPress\t' ...
            num2str(block) '\t' ...
            num2str(trial) '\t' ...
            num2str(stimulus) '\t' ... % Stimulus serial number in original stimulus text file
            'Left Key\t' ...  %
            num2str(firstPress(handles.Key)) '\t' ...
            cond_event_name '\t' ...
            'Left Key\t' ...  %
            base_condition '\t' ...
            emb '\t' ...
            num '\t' ...
            typ '\r\n'  ...
            ]); % write to log file
        
        
        
        if curr_violIndex == 1
            %% The subjects need to choose "Wrong"
            % True-positive left side
            if strcmp(curr_wrong_location,'left') && strcmp(Response,'Left')
                T.subject_response{trial} = 1;
                T.RT{trial}               = num2str(firstPress(handles.Key)- panel_onset);
                T.Behavioral{trial}       = 'TP';
                T.Behavioral_index{trial} = 1;
                % True-positive right side
            elseif strcmp(curr_wrong_location,'right') && strcmp(Response,'Right')
                T.subject_response{trial} = 1;
                T.RT{trial}               = num2str(firstPress(handles.Key)- panel_onset);
                T.Behavioral{trial}       = 'TP';
                T.Behavioral_index{trial} = 1;
                % False-positive right side
            elseif strcmp(curr_wrong_location,'right') && strcmp(Response,'Left')
                T.subject_response{trial} = 0;
                T.RT{trial}               = num2str(firstPress(handles.Key)- panel_onset);
                T.Behavioral{trial}       = 'FN';
                T.Behavioral_index{trial} = 2;
                % False-positive left side
            elseif strcmp(curr_wrong_location,'left') && strcmp(Response,'Right')
                T.subject_response{trial} = 0;
                T.RT{trial}               = num2str(firstPress(handles.Key)- panel_onset);
                T.Behavioral{trial}       = 'FN';
                T.Behavioral_index{trial} = 2;
            end
        else
            %% The subjects need to choose "OK"
            % True-negative left
            if strcmp(curr_ok_location,'left') && strcmp(Response,'Left')
                T.subject_response{trial} = 0;
                T.RT{trial}               = num2str(firstPress(handles.Key)-panel_onset);
                T.Behavioral{trial}       = 'TN';
                T.Behavioral_index{trial} = 3;
                % True-negative right
            elseif strcmp(curr_ok_location,'right') && strcmp(Response,'Right')
                T.subject_response{trial} = 0;
                T.RT{trial}               = num2str(firstPress(handles.Key)-panel_onset);
                T.Behavioral{trial}       = 'TN';
                T.Behavioral_index{trial} = 3;
                % False-negative left
            elseif strcmp(curr_ok_location,'left') && strcmp(Response,'Right')
                T.subject_response{trial} = 1;
                T.RT{trial}               = num2str(firstPress(handles.Key)-panel_onset);
                T.Behavioral{trial}       = 'FP';
                T.Behavioral_index{trial} = 4;
                % False-negative right
            elseif strcmp(curr_ok_location,'right') && strcmp(Response,'Left')
                T.subject_response{trial} = 1;
                T.RT{trial}               = num2str(firstPress(handles.Key)-panel_onset);
                T.Behavioral{trial}       = 'FP';
                T.Behavioral_index{trial} = 4;
            end
        end
        
        
        %%%%%%%%%%%%%%%%%
        % ESCAPE-KEY %%%%
        %%%%%%%%%%%%%%%%%
        if firstPress(KbName('escape'))
            error('Escape key was pressed')
        end
    else
        % The subject did not press anything
        T.subject_response{trial} = NaN;
        T.RT{trial}               = NaN;
        T.Behavioral{trial}       = 'NR'; % No-Response
        T.Behavioral_index{trial} = 5;
    end
    
    % check RT values
    if str2double(T.RT{trial})<0 || str2double(T.RT{trial}) > panel_offset
        T.RT{trial} = NaN;
    end
    disp(T.RT{trial})
    
    %--------------------------------------%
    %############# FEEDBACK ###############%
    % Use the behavioral index for feedback:
    %--------------------------------------%
    if strcmp(T.Behavioral{trial},'TN') || strcmp(T.Behavioral{trial},'TP')
        % correct - green cross
        % Change the color of the fixation cross depending on the subject's
        % performance.
        color = [0,255,0];
    elseif strcmp(T.Behavioral{trial},'FN') || strcmp(T.Behavioral{trial},'FP')
        % wrong - red cross
        color = [255,0,0];
    else
        % Did not press - blue cross
        color = [0,0,255];
    end
    
    
    %% -------------------------- FEEDBACK FIXATION ----------------------------------- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FEEDBACK FIXATION ON  %%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DrawFormattedText(handles.win, '+', 'center', 'center', color);
    if params.photodiode
        %[RECT presentation]
        Screen('FillRect', handles.win, handles.white, TopRightRect); % draw re
    end
    feed_fixation_onset = Screen('Flip', handles.win);
    if triggers
        send_trigger(triggers, handles, params, events, 'StartFixFeedback', 0)
    end
    %%%%%%%% WRITE TO LOG
    fprintf(fid_log,['FixFeedbackOn\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        '+' '\t' ...
        num2str(feed_fixation_onset) '\t' ...
        cond_event_name '\t' ...
        '+' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file
    
    
    WaitSecs('UntilTime',feed_fixation_onset + params.SOA_visual);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FEEDBACK FIXATION OFF  %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    feed_fixation_offset = Screen('Flip', handles.win);
    if triggers
        send_trigger(triggers, handles, params, events, 'EndFixFeedback', 0)
    end
    %%%%%%%% WRITE TO LOG
    fprintf(fid_log,['FixFeedbackOff\t' ...
        num2str(block) '\t' ...
        num2str(trial) '\t' ...
        num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
        '+' '\t' ...
        num2str(feed_fixation_offset) '\t' ...
        cond_event_name '\t' ...
        '+' '\t' ...
        base_condition '\t' ...
        emb '\t' ...
        num '\t' ...
        typ '\r\n'  ...
        ]); % write to log file
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ISI TO NEXT TRIAL     %%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    WaitSecs('UntilTime',feed_fixation_offset + params.ISI_visual);

end  %trial


%% Concatenate the datasets for the current block:
path2output = fullfile('..','Behavioral',params.defaultmethod, ...
    join(['subj_',num2str(params.subject)]));
if ~exist(path2output , 'dir')
    mkdir(path2output )
end
% append to the dataset
stimuli_datasets{block} = ...
    [dataset2table(stimuli_datasets{block}),T ];
% export to Output
curr_dataset_name = fullfile(path2output, ...
    join([params.defaultmethod, ...
    join(['_subj_',num2str(params.subject)]), ...
    join(['_block_',num2str(block)]),...
    '_session_',num2str(params.session), '.csv']));

writetable(stimuli_datasets{block},curr_dataset_name,'Delimiter','\t')

