function run_training_block(handles, training_words, params)
% This is the training block that includes 5 example sentences to
% get the subjects familiarised with the RSVP.
% ---------------------------------------------------------------%

% Initialize table to hold the behavioral responses.
T = cell2table(cell(length(training_words),4));
T.Properties.VariableNames = {'subject_response','RT','Behavioral', 'Behavioral_index'};

% manual violation indices for the training stimuli
violationIndex = {0,1,1,1,0};

% present the intro slide (same as the one used in the experiment)
present_intro_slide(params, handles);
KbStrokeWait;
% KbQueueStart;


% PRESENT LONG FIXATION ONLY AT THE BEGINING
DrawFormattedText(handles.win, '+', 'center', 'center', handles.white);
Screen('Flip', handles.win);
WaitSecs(1.5); %Wait before experiment start



% loop through the training trials
for trial=1:length(training_words)
    [~, ~, keyCode] = KbCheck;
    if keyCode('ESCAPE')
        DisableKeysForKbCheck([]);
        Screen('CloseAll');
        return
    end
    
    %% -------------------------- FIXATION ----------------------------------- %%
    % %%%%%%% DRAW FIXATION BEFORE SENTENCE (duration: params.fixation_duration)
    DrawFormattedText(handles.win, '+', 'center', 'center', handles.white);
    fixation_onset       = Screen('Flip', handles.win);
    WaitSecs('UntilTime', fixation_onset + params.fixation_duration_visual_block); %Wait before trial
    fixation_offset      = Screen('Flip', handles.win);
    %% ------------------------------------------------------------------------- %%
    
    
    %% DECISION SCREEN SET-UP
    % DESIGN THE DECISION SCREEN AND SWAP RANDONMLY THE ORDER OF APPEARANCE
    ok    = 'OK';
    wrong = 'Wrong';
    space = '     ';
    available_words      = {ok, wrong};
    [word_1, idx]        = datasample(available_words,1);
    available_words(idx) = [];
    word_2               = available_words;
    decision_screen      = [word_1{1},space,word_2{1}];
    % Locate the position of the 'OK' word in the created random panel
    idx = find(ismember(decision_screen,ok));
    if idx(1) > numel(decision_screen)/2
        curr_ok_location     = 'right';
        curr_wrong_location  = 'left';
    else
        curr_ok_location     = 'left';
        curr_wrong_location  = 'right';
    end
    
    %% -------------------------- RSVP ----------------------------------- %%
    %%%%%%%%% START RSVP for current sentence
    word_cnt       = 0;
    curr_sentence  = training_words{trial};
    curr_violIndex = violationIndex{trial};
    
    for word = 1:numel(curr_sentence)
        word_cnt = word_cnt + 1;
        % TEXT ON
        DrawFormattedText(handles.win, curr_sentence{word}, 'centerblock', 'center', handles.white);
        text_onset = Screen('Flip', handles.win); % Word ON
        WaitSecs('UntilTime', text_onset + params.stimulus_ontime);
        % TEXT OFF
        text_offset = Screen('Flip', handles.win); % Word OFF
        WaitSecs('UntilTime', text_offset + params.stimulus_offtime);
    end % word
    
    %% ---------------------- ISI BEFORE DECISION SCREEN ------------------------------ %%
    WaitSecs('UntilTime', text_offset + params.stimulus_offtime + params.ISI_to_response_panel);
    % TO DO: Make the cross red, if the subject presses too early
    [isi_pressed, ~] = KbQueueCheck;
    if isi_pressed
        DrawFormattedText(handles.win, '+', 'center', 'center', [255,0,0]);
        fixation_onset = Screen('Flip', handles.win);
    end
    % -----------------------------------------------------------------------------------%
    
    %% ------------------------- DECISION SCREEN ONLINE ------------------------------- %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PANEL ON  %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DrawFormattedText(handles.win, decision_screen, 'center', 'center', handles.white);

    panel_onset= Screen('Flip', handles.win); % Pannel ON

    
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
                break
            end
        end
        outp(888,0)
        
        if Key==32
            pressed_decision =1;
            Response = 'Right';
            firstPress(handles.Key) = GetSecs;
        elseif Key==64
            pressed_decision =1;
            Response = 'Left';
            firstPress(handles.Key) = GetSecs;
        else
            pressed_decision =0;
        end
        
    elseif strcmp(params.method,'iEEG')
        [pressed_decision, firstPress] = KbQueueCheck; % Collect keyboard events since KbQueueStart was invoked
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PANEL OFF %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    panel_offset    = Screen('Flip', handles.win); % Pannel OFF
    
    
    
    %%%%%%%%%%%%%%%%%
    % BEHAVIORAL %%%%
    %%%%%%%%%%%%%%%%%
    if pressed_decision
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
    feed_fixation_onset = Screen('Flip', handles.win);
    
    WaitSecs('UntilTime',feed_fixation_onset + params.SOA_visual);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FEEDBACK FIXATION OFF  %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    feed_fixation_offset = Screen('Flip', handles.win);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ISI TO NEXT TRIAL     %%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    WaitSecs('UntilTime',feed_fixation_offset + params.ISI_visual);
    
end  %trial

% KbQueueFlush





