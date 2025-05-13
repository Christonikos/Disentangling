function send_trigger(triggers, handles, params, events, event_name, wait_secs)

if triggers && strcmp(params.location,'TLVMC') % Setup for ICHILOV Tel-Aviv
    fwrite(handles.sio,events.(event_name)); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(wait_secs);
elseif triggers && strcmp(params.location,'UCLA') % Setup for UCLA
    DaqDOut(handles.dio,params.portA,events.(event_name)); % send eventX TTL (0-255)
    WaitSecs(events.ttlwait);
    DaqDOut(handles.dio,params.portA,events.eventreset); % reset Daq interface
    WaitSecs(wait_secs);
    %%%%%%%%%%%%%%%%%
    %%% NEUROSPIN %%%
    %%%%%%%%%%%%%%%%%
elseif triggers && strcmp(params.location,'Neurospin') % Setup for NS
    outp(888,events.(event_name))
    WaitSecs(events.ttlwait);
    outp(888,0)
    WaitSecs(wait_secs);
end

