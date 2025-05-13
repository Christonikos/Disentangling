function fid=createLogFileUCLAParadigm(params)
timestamp=gettimestamp();
subses=['S' num2str(params.subject) '_' num2str(params.session)];
fid=fopen(fullfile('..', 'Logs', ['logUCLAParadigm_' timestamp '_' subses '.csv']) ,'w');
fprintf(fid,['Event\t'...
    'Block\t'...
    'Trial\t'...
    'StimNum\t'...      %=stimCode. Based on stimulus order in text file.
    'WordNum\t'...  % Serial number of word in the sentence (VISUAL blocks only)
    'StimulusName\t'...      %full WAV name '1.wav'
    'Time\r\n']);

% copy code used for running to the log folder
copyfile(fullfile('functions', 'getParamsUCLAParadigm.m'), fullfile('..', 'Logs', sprintf('getParamsUCLAParadigm_%s_%s.m',timestamp,subses)))

end