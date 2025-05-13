function run_auditory_block(handles, block, stimuli_wavs, AudioTrialOrder, fid_log, triggers, cumTrial, params, events)
audioStopTime   = -inf;
% %%%%%% WAIT FOR KEY PRESS
DrawFormattedText(handles.win, 'Press any key...', 'center', 'center', handles.white);
Screen('Flip',handles.win);
wait_for_key_press()    

% %%%%%% BLOCK START: mark a new block with four 255 triggers separated 200ms from each other
block_start = GetSecs;
for i=1:4
    send_trigger(triggers, handles, params, events, 'event255', 0.2)
end
% %%%%%%% WRITE TO LOG
fprintf(fid_log,['BlockStart\t' ...
      num2str(block) '\t' ...
      num2str(0) '\t' ...
      num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
      '' '\t' ...  %
      '-' '\t' ...
      num2str(block_start) '\t' ...
      '' '\r\n' ...
      ]); % write to log file

for trial=1:params.numTrialsAudioBlock
  
  % %%%%%%% DRAW FIXATION BEFORE SENTENCE (duration: params.fixation_duration)
  DrawFormattedText(handles.win, '+', 'center', 'center', handles.white);
  fixation_onset = Screen('Flip', handles.win);
  if triggers
    send_trigger(triggers, handles, params, events, 'StartFixation', 0)
  end
  [pressed, firstPress]=KbQueueCheck; % Collect keyboard events since KbQueueStart was invoked
  cumTrial=cumTrial+1;
  stimulus=AudioTrialOrder(trial);
  clear wavedata;
  wavedata(params.patientChannel,:) = stimuli_wavs{stimulus}(:,params.patientChannel);
  wavedata(params.TTLChannel,:) = stimuli_wavs{stimulus}(:,params.patientChannel);
  % IF SPLITTING THE AUDIO OUTPUT
  %wavedata(params.TTLChannel,:) = zeros(1, length(wavedata(params.patientChannel,:)));
  %wavedata(params.TTLChannel,1:round(params.freq/100)) = 1; % 2ns channel: Make a BEEP during ~10ms at the beginning of the wav
  
  % %%%%%% Echo status
  fprintf('Block %i, trial %i, stimulus %s\n', block, trial, params.WAVnames{stimulus})
  
  % %%%%%%% Present fixation and fill buffer
  PsychPortAudio('FillBuffer', handles.pahandle, wavedata);
  WaitSecs('UntilTime', fixation_onset + params.fixation_duration_audio_block);
  fixation_offset = Screen('Flip', handles.win);      
  
  % %%%%%%% START AUDIO AND SEND TRIGGER AT START AND END
  audioOnset = PsychPortAudio('Start', handles.pahandle, 1, 0, 1); % it takes ~15ms to start the sound
  if triggers
      send_trigger(triggers, handles, params, events, 'StartAudio', 0)
  end
  if pressed
      if triggers
        send_trigger(triggers, handles, params, events, 'PressKey', 0)
      end
  end   
  [~, ~, ~, audioStopTime]=PsychPortAudio('Stop', handles.pahandle,1);
  if triggers
      send_trigger(triggers, handles, params, events, 'EndAudio', 0)
  end

  % %%%%%%% CLEAR-UP (buffer and screen)
  PsychPortAudio('DeleteBuffer',[],1); % clear the buffer
  
  % %%%%%%% WRITE TO LOG
  fprintf(fid_log,['Fix\t' ...
      num2str(block) '\t' ...
      num2str(trial) '\t' ...
      num2str(0) '\t' ... % Stimulus serial number in original stimulus text file
      '' '\t' ...  %
      '+' '\t' ...
      num2str(fixation_onset) '\r\n' ...
      ]); % write to log file
  fprintf(fid_log,['StimAudioOn\t' ...
      num2str(block) '\t' ...
      num2str(trial) '\t' ...
      num2str(stimulus) '\t' ...   
      '' '\t' ...  %
      params.WAVnames{stimulus} '\t' ...
      num2str(audioOnset) '\r\n' ...
      ]); % write to log file
  fprintf(fid_log,['StimAudioOff\t' ...
      num2str(block) '\t' ...
      num2str(trial) '\t' ...
      num2str(stimulus) '\t' ...   
      '\t' ...  %
      '\t' ...
      num2str(audioStopTime) '\r\n' ...
      ]); % write to log file
     if pressed
        if firstPress(KbName('l')) || firstPress(KbName('L'))
            fprintf(fid_log,['KeyPress\t' ...
              num2str(block) '\t' ...
              num2str(trial) '\t' ...
              num2str(stimulus) '\t' ... % Stimulus serial number in original stimulus text file
              '\t' ...  %
              '\t' ...
              num2str(firstPress(KbName('l'))) '\t' ...
              '\r\n' ...
              ]); % write to log file
        end
        if firstPress(KbName('escape'))
            error('Escape key was pressed')
        end
    end
  % %%%%%% WAIT ISI
  WaitSecs('UntilTime', audioStopTime + params.ISI_audio); %Wait before showing next sentencesw

end  %trial
%     
end