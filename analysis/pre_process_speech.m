% Pre-process data into spectral features
% Generate Signal to test spectrogram on
clear
load('test1/recorded8.mat')
win_time = 20/1000;
nfft = fs*win_time; % To get x ms long window
%noverlap = nfft-round(nfft/3);
noverlap = 0;
win = hamming(nfft,'symmetric'); % Might need to be 'periodic' not sure

% Phrase 1
fignum = 1;
[mag_spect, f, t] = my_spectrogram(S.data,win,noverlap,nfft,fs,fignum);

%% Save recording
save('test1/recorded1_spect.mat')