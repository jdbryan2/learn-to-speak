function [Snd, fs, duration] = import_sound(testname,filename)
%import_sound
%filename = 'test1/logs/recorded8.log';
delimiterIn = '\t';
headerlinesIn = 4;
S = importdata([testname, '/logs/', filename],delimiterIn,headerlinesIn);
fs = str2double(cell2mat(S.textdata(2)));
duration = str2double(cell2mat(S.textdata(4)));
soundsc(S.data,fs)
Snd = S.data;

save([testname,'/', filename(1:end-3), '.mat'],'Snd','fs','duration')