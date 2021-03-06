function [Snd, fs, duration] = import_sound(path)
%import_sound
%filename = 'test1/logs/sound8.log';
delimiterIn = '\t';
headerlinesIn = 4;
S = importdata(path,delimiterIn,headerlinesIn);
fs = str2double(cell2mat(S.textdata(2)));
duration = str2double(cell2mat(S.textdata(4)));
soundsc(S.data,fs)
Snd = S.data;

%save([testname,'/', filename(1:end-3), '.mat'],'Snd','fs','duration')