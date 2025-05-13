function params = convert_toRR(params)
% Convert the specified timings to round multiples of the refresh rata
% Example timings:
%     fixation_duration_visual_block: 0.6000
%                    stimulus_ontime: 0.2500
%                   stimulus_offtime: 0.2500
%                         SOA_visual: 0.5000
%              ISI_to_response_panel: 0.5000
%                       panel_ontime: 1.5000
%                             max_RT: 1
%                      feedback_time: 0.5000
%                         ISI_visual: 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

target_names = ...
    {'fixation_duration_visual_block','stimulus_ontime', 'stimulus_offtime',...
    'SOA_visual', 'ISI_to_response_panel', 'panel_ontime', 'max_RT', 'feedback_time',...
    'ISI_visual'};
f_names = fieldnames(params);
for fid = 1:length(f_names)    
    if any(strcmp(target_names,f_names(fid)))
        name = target_names{strcmp(target_names,f_names(fid))};
       params.(name) = (round(params.(name)*1e3/params.r_rate)*params.r_rate)*1e-3;
    end
end
