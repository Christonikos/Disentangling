function handles = initialize_TTL_hardware(triggers, params, events)
sio=[]; dio=[]; DaqDOut=[]; hwline=[]; laststim=[];

if triggers && strcmp(params.location,'TLVMC')
    sio = serial(params.sio,'BaudRate',115200,'Terminator', []);
    fopen(sio);     %remember to close manually if you terminal execution early
    fwrite(sio,1); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.5);
    fwrite(sio,2); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.5);
    fwrite(sio,3); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.5);
    fwrite(sio,4); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset);
end

if triggers && strcmp(params.location,'TLVMC') % mark the beginning of the experiment with four 255 triggers separated 100ms from each other
    fwrite(sio,events.event255); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.1);
    fwrite(sio,events.event255); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.1);
    fwrite(sio,events.event255); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset); WaitSecs(0.1);
    fwrite(sio,events.event255); WaitSecs(events.ttlwait); fwrite(sio,events.eventreset);
end

% Initialize DAQ
if triggers && strcmp(params.location,'UCLA')
    dio = initializeDAQ;
end

if triggers && strcmp(params.location,'Neurospin')
    % make sure outp and inp function (for triggers) work
    config_io;
    ioObj = io64;
    status = io64(ioObj);
    outp(888,0); %set all STI channels to 0
end


handles.sio     =   sio;
handles.dio     =   dio;
handles.DaqDOut =   DaqDOut; 
handles.hwline  =   hwline; 
handles.laststim=   laststim;



