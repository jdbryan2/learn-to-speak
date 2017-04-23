function [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
         DFA_preprocess_speech(testname,testdir,win_time,max_length,f,p,save_figs)
% Import and Preprocess Speech for subsequent DFA analysis
load([testname,'/logs/sample1_30sec.mat']) % Hardcoded filename
nfft = fs*win_time; % To get x ms long window
%noverlap = nfft-round(nfft/2);
noverlap = 0;
win = hamming(nfft,'symmetric'); % Might need to be 'periodic' not sure

% Phrase 1
fignum = 1;
if length(y)>max_length*fs
    [mag_spect, freq, t] = my_spectrogram(y(1:max_length*fs),win,noverlap,nfft,fs,fignum);
else
    [mag_spect, freq, t] = my_spectrogram(y(1:length(y)),win,noverlap,nfft,fs,fignum);
end
if save_figs == true
    saveas(fignum,[testdir,'spectrogram'],'epsc');
    saveas(fignum,[testdir,'spectrogram'],'fig');
end
dt = t(2)-t(1);
L = f+p;
[num_f,num_samp] = size(mag_spect);
%Remove any zeros and replace with small value to not mess up svd
zs = mag_spect==0;
mag_spect(zs) = 1e-10;
logmag = log10(mag_spect.^2);
f_ind = 1:5:num_f;
freqs = freq(f_ind);
num_freqs = length(freqs);
length_fact = length(t)-L;
XPF = zeros(length(f_ind),L*length_fact);
Xp_ = zeros(length(f_ind)*p,length_fact);
Xf_ = zeros(length(f_ind)*f,length_fact);
for n=1:length_fact
    xp = logmag(f_ind,n:n+p-1);
    xf = logmag(f_ind,n+p:n+p+f-1);
    XPF(:,1+(n-1)*L:n*L) = [xp,xf];
    Xp_(:,n) = xp(:);
    Xf_(:,n) = xf(:);
end
%     XPM = mean(XP,2);
%     XFM = mean(XF,2);
%     Xp = Xp_ - repmat(XPM,p,length_fact);
%     Xf = Xf_ - repmat(XFM,f,length_fact);

Xpm = mean(Xp_')';
Xfm = mean(Xf_')';
Xp = Xp_-Xpm*ones(1,length_fact);
Xf = Xf_-Xfm*ones(1,length_fact);

% Translation from old spectrogram variable names to D names
num_vars = num_freqs;
num_logs = length_fact;
samp_freq = 1/dt;
% Round frequencies for labels
D_lab = num2cell(floor(freqs));
dmean = [Xpm;Xfm];
stdevs = std(XPF,0,2);
% Was Not scaling Spectrogram features!
%Xp = Xp./repmat(stdevs,[p,num_logs]);
%Xf = Xf./repmat(stdevs,[f,num_logs]);

fs_ = fs;
save([testdir,'speech_preprocess.mat'],'noverlap','nfft','fs_');
end