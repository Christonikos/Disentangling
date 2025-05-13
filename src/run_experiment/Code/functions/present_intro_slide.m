function present_intro_slide(params, handles)
% %%%%%% SHOW INTRO SLIDE
intro_img_read = imread(params.path2intro_slide);
intro_img = Screen('MakeTexture', handles.win, intro_img_read, [], [], [], [], 1);
Screen('DrawTexture', handles.win, intro_img, [], [], 0);
Screen('Flip',handles.win);
