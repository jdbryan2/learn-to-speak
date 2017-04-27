function [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
    DFA_preprocess_stubart(testname,smooth,max_num_files,scaling,f,p)
% VT Definitions
num_tubes = 89;
num_art = 29;

L = f+p;
logs = dir([testname, '/logs/datalog*.log']);
snd_logs = dir([testname, '/logs/sound*.log']);
num_files = length(logs);
assert(num_files == length(snd_logs));
if num_files > max_num_files
    num_files = max_num_files;
end
dvec = [];
dmat = [];
nan_inds = [];
for i=1:num_files
    %Import Datalog
    [VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog([testname,'/logs/',logs(i).name]);
    % Flip matrix to make more similar to how the spectrogram was processed
    % in earlier code.
    vt = VT_log(:,1:end-1)'; %remove sound
    if sum(sum(isnan(vt)))
        nan_inds = [nan_inds, i];
        continue
    end
    if i==1
%              zinds = find(mean(vt,2)~=0);
%             num_vars = length(VT_lab);
%             if(length(intersect(zinds,1:num_vars))>length(zinds))
%                 error('Zero average for variable other than tubes')
%             end
%             num_tubes = num_tubes - length(zinds);
%             zinds = intersect(1:num_vars-1,zinds);
        zinds = 1:length(VT_lab)-1; %From removing sound
        VT_labs = VT_lab(zinds);
        num_vt_vars = length(VT_labs);
    end
    % Remove first sample to line up data with STFT
    vt = vt(zinds,2:end);
    %samp_time = (samp_len-1)/samp_freq;

    %Import Sound
    [snd, fs, duration] = import_sound([testname,'/logs/',snd_logs(i).name],false);
    %Perform Spectrogram
    %win_time = (samp_time)/L;
    win_time = 1/(floor(0.02*samp_freq)*samp_freq);
    nfft = floor(fs*win_time); % To get x ms long window
    assert(mod(nfft,1)==0)
    %noverlap = nfft-round(nfft/2);
    noverlap = 0;
    win = hamming(nfft,'symmetric'); % Might need to be 'periodic' not sure
    fignum = 1;
    [mag_spect, freq, t] = my_spectrogram(snd,win,noverlap,nfft,fs,fignum);
    dt = t(2)-t(1);
    [num_f,num_samp] = size(mag_spect);
    %Remove any zeros and replace with small value to not mess up svd
    zs = mag_spect==0;
    mag_spect(zs) = 1e-10;
    logmag = log10(mag_spect.^2);
    % Pull out every 5th frequency for analysis
    f_ind = 1:5:num_f;
    freqs = freq(f_ind);
    num_freqs = length(freqs);
    spect = logmag(f_ind,:);

    num_vars = num_freqs+num_vt_vars;
    if ~smooth
        % Cut off sample to length L
        length_fact = 1;
    else
        length_fact = samp_len-L;
    end
    Xpf = zeros(num_vars,L*length_fact);
    xpf = zeros(num_vars*L,length_fact);
    for n=1:length_fact
        xp = [vt(:,n:n+p-1);spect(:,n:n+p-1)];
        xf = [vt(:,n+p:n+p+f-1);spect(:,n+p:n+p+f-1)];
        Xpf(:,1+(n-1)*L:n*L) = [xp,xf];
        xpf(:,n) = [xp(:);xf(:)];
    end

    dmat = [dmat,Xpf];
    dvec = [dvec,xpf];
end
num_logs = length_fact*num_files;
%Round labels
D_lab = [VT_labs,num2cell(floor(freqs))];
% Remove mean of each feature at each timestep from data
dmean = mean(dvec,2);
% Assuming all file lengths are the same currently
Dmat = (dmat - repmat(reshape(dmean,[num_vars,L]),[1,num_files*length_fact]));
% Scaling
stdevs = std(Dmat,0,2);

% Scaling tubes and arts by the respective mean stddevs
if strcmp(scaling,'original')
    rng1 = 1:num_tubes;
    rng2 = num_tubes+1:num_vars;
    stdevs(rng1) = mean(stdevs(rng1));
    stdevs(rng2) = mean(stdevs(rng2));
% Scaling non-lung tubes, lung tubes ish?, and arts differently
elseif strcmp(scaling,'short_lung_scale')
    rng1 = [1:6,19:num_tubes];
    rng2 = 7:18;
    rng3 = num_tubes+1:num_vars;
    stdevs(rng1) = mean(stdevs(rng1));
    stdevs(rng2) = mean(stdevs(rng2));
    stdevs(rng3) = mean(stdevs(rng3));
elseif strcmp(scaling,'individual')
    % Scaling by individual feature variance
end

% First remove zero std devs from tube sections that don't move
%tub_stds = stdevs(1:num_tubes);
%ztubs = tub_stds~=0;
%z_std = mean(tub_stds(ztubs));
%tub_stds(tub_stds==0) = z_std;
%stdevs(1:num_tubes) = tub_stds;

%Dmat = Dmat./repmat(stdevs,[1,L*num_logs]);
D = reshape(Dmat,[L*num_vars,num_logs]);
Xp = D(1:p*num_vars,:);
Xf = D(p*num_vars+1:end,:);