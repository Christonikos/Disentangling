function wait_for_key_press()
    t_pressed = false;
    while ~t_pressed
        [~, ~, keyCode] = KbCheck;
        if any(keyCode)
            t_pressed = true;
        elseif keyCode('ESCAPE')
            DisableKeysForKbCheck([]);
            Screen('CloseAll');
            return
        end
    end
end