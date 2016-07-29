%import_sound
filename = 'test1/logs/recorded8.log';
delimiterIn = '\t';
headerlinesIn = 4;
S = importdata(filename,delimiterIn,headerlinesIn);
fs = str2double(cell2mat(S.textdata(2)));
duration = str2double(cell2mat(S.textdata(4)));
soundsc(S.data,fs)

save('test1/recorded8.mat','S','fs','duration')