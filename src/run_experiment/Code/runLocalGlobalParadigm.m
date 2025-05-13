%% Language local-global paradigm. 
% @UNICOG,NEUROSPIN19
% Christos Zacharopoulos
% ------------------------------------------------
rng('default')       
clear; close all; clc       
debug_mode = 0  ;      
    
if debug_mode
    dbstop if error      
    training = 0;      
else
    training = questdlg('Do yo u want to include a training block?','Training block','Yes','No','Yes');
    if training(1) == 'Y', training = 1; else training = 0; end
end

%% INITIALIZATION
addpath('functions')
KbName('UnifyKeyNames') 
[params, events] = getParamsLocalGlobalParadigm(debug_mode);
rng(str2double(params.subject)*params.session); 

 

% TTL settings 
params.location ='Neurospin';  %options: 'Houston' or 'NeuroSpin', affecting hardware to use for TTL l
params.portA    = 1;
params.portB    = 0;

% Running on PTB-3? Abort otherwise.
% AssertOpenGL;

%% TRIGGERS
%#################################################################
% Send TTLs though the DAQ hardware interface
triggers = questdlg('Send TTLs?','TTLs status','Yes (recording session)','No (just playing)','Yes (recording session)');
if triggers(1) == 'Y', triggers = 1; else triggers = 0; end
if ~triggers, uiwait(msgbox('TTLs  will  *NOT*  be  sent - are you sure you want to continue?','TTLs','modal')); end
%################################################################


handles = initialize_TTL_hardware(triggers, params, events);

%% LOAD LOG, STIMULI, PTB handles.
if triggers 
    for i=1:9 % Mark the beginning of the experiment with NINE consective '255' triggers separated by 0.1 sec
        send_trigger(triggers, handles, params, events, 'event255', 0.1)
    end
end
fid_log = createLogFileLocalGlobalParadigm(params); % OPEN LOG 
%% LOAD STIMULI 
[stimuli_words, stimuli_datasets, training_words] = ...
    load_stimuliLocalGlobal(params);     
% Open screens 
handles = Initialize_PTB_devices(params, handles, debug_mode); 
warning off; 



%% START EXPERIMENT
 try
     if training 
         %%%%%%% LOOP OVER TRAINING STIMULI
         run_training_block(handles, training_words, params); 
     end
     cumTrial=0;
    % PRESENT LONG FIXATION ONLY AT THE BEGINING
    DrawFormattedText(handles.win, '+', 'center', 'center', handles.white);
    Screen('Flip', handles.win);
    WaitSecs(1.5); %Wait before experiment start
    % START LOOP OVER BLOCKS
    nblocks = numel(params.text_filename);
    for block = 1 :nblocks
        % %%%%%% BLOCK TYPE (odd blocks are visual; even auditory)
        if block == 1
            %%%%%%%% WRITE TO LOG
            fprintf(fid_log,['GrandStart\t' ...
                '\t' ...
                '\t' ...
                '\t' ...  %Stimulus serial number in original stimulus text file
                '---' '\t' ...
                num2str(GetSecs) '\t' ...
                '' '\r\n' ...
                ]); % write to log file
        end
        % %%%%%%% RANDOMIZE TRIAL LIST
        VisualTrialOrder=randperm(length(stimuli_words{block}));
        % %%%%%% LOOP OVER STIMULI
         run_visual_block(handles, block, stimuli_words{block}, ...
            VisualTrialOrder, fid_log, triggers, cumTrial, ...
            params, events, stimuli_datasets);   
    end 
catch
    sca
    PsychPortAudio('Close', handles.pahandle);
    psychrethrow(psychlasterror);
    KbQueueRelease;
    fprintf('Error occured\n')
end

%% %%%%%%% CLOSE ALL - END EXPERIMENT
fprintf('Done\n')
KbQueueRelease;
sca