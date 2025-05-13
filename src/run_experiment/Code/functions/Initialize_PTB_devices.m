function handles = Initialize_PTB_devices(params, handles, debug_mode)

%% AUDIO
%InitializePsychSound(1);
% handles.pahandle = PsychPortAudio('Open', [], [], 0, params.freq, params.audioChannels, 0);
%handles.pahandle = PsychPortAudio('Open', 1, [], 2, params.freq, params.audioChannels, 0);


%% SCREEN
Screen('Preference', 'SkipSyncTests', 0);
Screen('Preference', 'TextRenderer',  1);
screens = Screen('Screens');


handles.screenNumber = max(screens);
handles.black = BlackIndex(handles.screenNumber);
handles.white = WhiteIndex(handles.screenNumber);


rect = get(0, 'ScreenSize');
if debug_mode
    handles.rect = [0 0 rect(3:4)./2];
    handles.win = Screen('OpenWindow',handles.screenNumber, handles.black, handles.rect);
else
    handles.rect = [0 0 rect(3:4)];
    handles.win = Screen('OpenWindow',handles.screenNumber, handles.black);
end

% if debug_mode
%     PsychDebugWindowConfiguration([0],[0.5])
% end


%% TEXT ON SCREEN
Screen('TextFont',handles.win, 'Arial');
Screen('TextSize',handles.win, 20);   % 160 --> ~25mm text height (from top of `d' to bottom of `g').
Screen('TextStyle', handles.win, 1);   % 0=normal text style. 1=bold. 2=italic.

%% KEYBOARD


handles.escapeKey = KbName('ESCAPE');
handles.LKey = KbName('a');
handles.RKey = KbName('l');
handles.Key  = KbName('g');
keysOfInterest=zeros(1,256);
keysOfInterest(KbName({'a','l','g', 'ESCAPE'}))=1;
KbQueueCreate(-1, keysOfInterest);

























