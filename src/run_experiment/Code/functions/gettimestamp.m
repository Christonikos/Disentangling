function timestamp = gettimestamp()
    formatOut = 'yyyymmmdd_hhMMss';
    timestamp = datestr(now, formatOut);
end