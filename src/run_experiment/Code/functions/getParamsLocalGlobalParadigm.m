function [params, events] = getParamsLocalGlobalParadigm(debug_mode)
%This function makes the struct that holds the parameters for the presentation of the stimuli and such.
%% SUBJECT AND SESSION NUMBERS
% recording method
params.method = 'MEG'; 
% refresh rate
params.r_rate =  60;



if debug_mode
    params.subject = '00';
    params.session = str2double('00');
    % photodiode
    params.photodiode = true;
else
    subject = inputdlg({'Enter subject number'},...
        'Subject Number',1,{''});
    params.subject = subject{1};
    
    session = inputdlg({'Enter session number'},...
        'Subject Number',1,{''});
    params.session=str2double(session{1});
    
    % photodiode
    params.photodiode = false;
end
%%%%%%%%% PATHS
params.path2intro_slide = fullfile('..','Stimuli','instructions_sentences.png');
params.defaultmethod    = params.method;
params.defaultpath      = fullfile('..','runParadigm' ,'Code');
params.Visualpath       = fullfile('..','Stimuli','visual',params.defaultmethod);
params.path2stim        = fullfile('..','Stimuli');

if ismac || isunix %comp == 'h'
    params.sio  = '/dev/tty.usbserial';
elseif ispc % strcmp(comp,'l')
    params.sio  = 'COM1';
end

%%%%%%%%% FILENAMES STIMULI
dr = dir(fullfile(params.path2stim,'visual',params.defaultmethod, ...
    join(['subj_',params.subject])));
dr_n     = {dr.name};
dr_names = dr_n(~[dr.isdir]);

n_blocks = numel(dr_names);

for bID = 1:(n_blocks)
    params.text_filename{bID} = dr_names{bID};
end


%% %%%%%%% TEXT params
params.font_size    = 20; % Fontsize for words presented at the screen center
params.font_name    = 'Courier New';
params.font_color   = 'ffffff';

%% %%%%%%% TIMING params
% VISUAL BLOCK
params.fixation_duration_visual_block   = 0.6; %
params.stimulus_ontime                  = 0.25; % Duration of each word
params.stimulus_offtime                 = 0.25; % Duration of black between stimuli
params.SOA_visual                       = params.stimulus_ontime + params.stimulus_offtime;
params.ISI_to_response_panel            = 0.5;
params.panel_ontime                     = 1.5;  % Duration of panel on the screen
params.max_RT                           = 1.5;    % Maximum allowance for RT.
params.feedback_time                    = 0.5;

if strcmp(params.defaultmethod, 'MEG') || strcmp(params.defaultmethod, 'iEEG')
    params.ISI_visual                      = 1; % from end of last trial to beginning of first trial
else
    params.ISI_visual                      = 7; % for the fMRI, we need at least the same duration as 
                                             % the trial, to get the BOLD response.
end

% covert the default timings to round multiples of the rephresh rate
params = convert_toRR(params);

%% EVENTS NUMBERS (TRIGGERS)


%--------- FIXATION(S) ------------- %
% fixation to first word onset       %
events.StartFixation     = 1;     %
% fixation to Decision screen onset  %
events.StartFix2Decision = 10;       %
events.EndFix2Decision   = 15;       %
% fixation during the feedback period%
events.StartFixFeedback  = 100;     %  
events.EndFixFeedback    = 110;
%------------------------------------%
%------ WORD ONSETS-OFFSET --------- %
%------------------------------------%
% FIRST WORD ------------------------%
% ----------------- First word onset %
events.first_word_onset = 40;
% ----------------- First word offset%
events.first_word_ofset = 50;
% LAST WORD -------------------------%
% ----------------- Last word onset -%
events.last_word_onset = 60;
% ----------------- Last word offset-%
events.last_word_ofset = 70;
%------------------------------------%
% WORDS -----------------------------%
events.StartWord  = 80;
events.EndWord    = 90;
%------------------------------------%


% PANEL
events.StartPanel       = 30;
events.EndPanel         = 35;

% KEY PRESS(ES)
events.PressKey         = 120;


% MISC
events.event255        = 255;
events.eventreset      = 0;
events.ttlwait         = 0.01;
events.audioOnset      = 0;
events.eventResp       = 145;



