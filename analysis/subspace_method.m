% Subspace Method
%% Load log files and combine data into one array
clear

% TEST CONFIGURATION
% Select 4 parameters
% - k = number of synergies/primitives
% - data_type = type of data being analyzed can be:
%   'speech','tubart','stubart'
% - testname = name of test that generated the data and name of the folder
%   containing the data
% - config = the name of the configuraiton that is being used to generate
%   the synergies/primitives. Must line up with if statements in data
%   import

% Model Parameters
k = 8;

% Save figures or not
save_figs = false;

% Human Speech
% data_type = 'speech';
% testname = 'TestMySpeech1';
% %config = 'simple'; % noisy ish ball with some rotation
% %config = 'texture';
% %config = 'original'; % Super smooth and lots of rotation
% config = 'short';
% %config = 'smooth2';

% VT-articulatory
data_type = 'tubart';
%testname = 'testThesis4';
%testname = 'testFeatureTrack';
%testname = 'testFeatureTrack2';
%testname = 'testFeatureTrack4'; noisy oblong spheriod broken up wrt time
%testname = 'testRevised1'; %original sound maker
%testname = 'testRevised3';
%testname = 'testRevised4';
%testname = 'testBatch3';
testname = 'testRandArt1';
%config = 'original';
%config = 'original_no_zeros';
%config = 'long';
%config = 'short';
%config = 'short_lung_scale';
%config = 'short_no_zeros';
config = 'scale_fix_perm';
%config = 'scale_fix_long';

% VT-acoustic-articulatory
% data_type = 'stubart';
% %testname = 'testThesis4';
% testname = 'testFeatureTrack'; %smooth ish ball with a lot of rotation (.1,.25) 10 sec 5 logs
% %testname = 'testFeatureTrack2'; %smooth ish ball with some rotation (.25,.25) 10 sec 5 logs
% %testname = 'testFeatureTrack3'; % smooth ish Ball, no rotation (.25,.25) 10 sec 10 logs
% %testname = 'testFeatureTrack4'; %smooth ish ball with some rotation (.1,.25) 10 sec 5 logs
% config = 'long'; % Long random stim trials
% %config = 'longer'; % longer f and p

% VT Definitions
num_tubes = 89;
num_art = 29;

% Subplot figure helper variables
mk = ceil(sqrt(k));
nk = ceil((k-mk)/mk)+1;

% Create directory for this specific test
testdir = [testname,'/',config,num2str(k),'/'];
mkdir(testdir);

% Import Data and Preprocess accordingly
if strcmp(data_type, 'speech')
    load([testname,'/logs/sample1_30sec.mat']) % Hardcoded filename
    if strcmp(config,'simple') || strcmp(config,'original') || strcmp(config,'smooth2') ||strcmp(config,'short')
        win_time = 20/1000;
    elseif strcmp(config,'texture')
        win_time = 5/1000; %textured one
    end
    nfft = fs*win_time; % To get x ms long window
    %noverlap = nfft-round(nfft/2);
    noverlap = 0;
    win = hamming(nfft,'symmetric'); % Might need to be 'periodic' not sure

    % Phrase 1
    fignum = 1;
    if strcmp(config,'simple') || strcmp(config,'original') || strcmp(config,'smooth2')||strcmp(config,'short')
        [mag_spect, freq, t] = my_spectrogram(y(1:length(y)),win,noverlap,nfft,fs,fignum);
    elseif strcmp(config,'texture')
        [mag_spect, freq, t] = my_spectrogram(y(1:length(y)/5),win,noverlap,nfft,fs,fignum);
    end
    if save_figs == true
        saveas(fignum,[testdir,'spectrogram'],'epsc');
        saveas(fignum,[testdir,'spectrogram'],'fig');
    end
    dt = t(2)-t(1);
    
    if strcmp(config,'simple')
        tlen = 0.15;
        f = round(tlen/dt);
        p = round(tlen/dt);
    elseif strcmp(config,'original')
        tlen = 0.3;
        f = round(tlen/dt);
        p = round(tlen/dt);
    elseif strcmp(config,'smooth2')
        tlen = 0.3;
        f = round(tlen/dt);
        p = 2;
    elseif strcmp(config,'texture')
        f =1*4; %textured one
        p = 29*4; %textured one
    elseif strcmp(config,'short')
        f = 2;
        p = 2;
    end
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
    Xp = Xp./repmat(stdevs,[p,num_logs]);
    Xf = Xf./repmat(stdevs,[f,num_logs]);
elseif strcmp(data_type, 'tubart')
    if strcmp(config,'long') || strcmp(config,'scale_fix_long') || strcmp(config,'scale_fix_perm')
        f = 13;
        p = 13;
    elseif strcmp(config,'original') || strcmp(config,'original_no_zeros')
        f = 13;
        p = 13;
    elseif strcmp(config,'longer')
        f = 15;
        p = 15;
    elseif strcmp(config,'short') || strcmp(config,'short_lung_scale') || strcmp(config,'short_no_zeros') || strcmp(config,'scale_fix')
        f = 2;
        p = 2;
    end
    L = f+p;
    logs = dir([testname, '/logs/datalog*.log']);
    num_files = length(logs);
    Xp_ = [];
    Xf_ = [];
    XPF = [];
    nan_inds = [];
    for i=1:num_files
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
            zinds = 1:length(VT_lab)-1;
            D_lab = VT_lab(zinds);
            num_vars = length(D_lab);
        end
        vt = vt(zinds,:);
        length_fact = samp_len+1-L;
        xpf = zeros(num_vars,L*length_fact);
        XP_ = zeros(num_vars*p,length_fact);
        XF_ = zeros(num_vars*f,length_fact);
        for n=1:length_fact
            xp = vt(:,n:n+p-1);
            xf = vt(:,n+p:n+p+f-1);
            xpf(:,1+(n-1)*L:n*L) = [xp,xf];
            XP_(:,n) = xp(:);
            XF_(:,n) = xf(:);
        end
        Xp_ = [Xp_,XP_];
        Xf_ = [Xf_,XF_];
        XPF = [XPF,xpf];
    end
    [~,num_logs] = size(Xp_);
    dt = 1/samp_freq;
    
    %Remove mean of time-varying features. i.e. mean computed over time and
    %feature space
    Xpm = mean(Xp_,2);
    Xfm = mean(Xf_,2);
    dmean = [Xpm;Xfm];
    Xp = Xp_-Xpm*ones(1,num_logs);
    Xf = Xf_-Xfm*ones(1,num_logs);
    
    % Scale features
    stdevs = std(XPF,0,2);
    % Scaling tubes and arts by the respective mean stddevs
    if strcmp(config,'original') || strcmp(config,'short')
        rng1 = 1:num_tubes;
        rng2 = num_tubes+1:num_vars;
        stdevs(rng1) = mean(stdevs(rng1));
        stdevs(rng2) = mean(stdevs(rng2));
    elseif strcmp(config,'short_lung_scale')
        %Scaling non-lung tubes, lung tubes ish?, and arts differently
        rng1 = [1:6,22:num_tubes];
        rng2 = 7:21; % technically lungs are 1:23/22?, but first 6 aren't used
        rng3 = num_tubes+1:num_vars;
        stdevs(rng1) = mean(stdevs(rng1));
        stdevs(rng2) = mean(stdevs(rng2));
        stdevs(rng3) = mean(stdevs(rng3));
    %else % Scaling by individual feature variance
    elseif strcmp(config,'scale_fix') ||strcmp(config,'scale_fix_long')
        % Currenlty just change it so that non-zero features are weighted
        % very low for the DFA.
        % TODO:Change code to only output non_zero features, but that
        % requires editing the primitive math in the simulator.
%         non_zero_feats = stdevs>=1e-6;
%         stdevs(~non_zero_feats) = max(stdevs)*10;
%         non_zero_p = repmat(non_zero_feats,[p,1]);
%         non_zero_f = repmat(non_zero_feats,[f,1]);
%         %Xp = Xp(non_zero_p,:)./repmat(stdevs(non_zero_feats),[p,num_logs]);
%         %Xf = Xf(non_zero_f,:)./repmat(stdevs(non_zero_feats),[f,num_logs]);
    end
        
    non_zero_feats = stdevs>=1e-6;
    stdevs(~non_zero_feats) = max(stdevs)*10;
    non_zero_p = repmat(non_zero_feats,[p,1]);
    non_zero_f = repmat(non_zero_feats,[f,1]);
    Xp = Xp(non_zero_p,:)./repmat(stdevs(non_zero_feats),[p,num_logs]);
    Xf = Xf(non_zero_f,:)./repmat(stdevs(non_zero_feats),[f,num_logs]);
    % Make tube sections with 0 std dev = the mean tube std dev
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!BAD IDEA POTENTIALLY bc still could be
    % fairly small. Maybe make it really big so we don't care about it or
    % not bc xp for those values is zero anyway
    % May need to set a tolerance here instead of just 0
%     tub_stds = stdevs(1:num_tubes);
%     ztubs = tub_stds~=0;
%     z_std = mean(tub_stds(ztubs));
%     tub_stds(tub_stds==0) = z_std;
%     stdevs(1:num_tubes) = tub_stds;
    
%     Xp = Xp./repmat(stdevs,[p,num_logs]);
%     Xf = Xf./repmat(stdevs,[f,num_logs]);
elseif strcmp(data_type, 'stubart') %FIX THIS!!!!!!!!!!!!!!!!
    % Keep f+p<=samp_len-1
    p = 13;
    f = 12;
    L = f+p;
    logs = dir([testname, '/logs/datalog*.log']);
    snd_logs = dir([testname, '/logs/sound*.log']);
    num_files = length(logs);
    assert(num_files == length(snd_logs));
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
            VT_lab = VT_lab(zinds);
            num_vt_vars = length(VT_lab);
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
        length_fact = samp_len-L;
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
    D_lab = [VT_lab,num2cell(floor(freqs))];
    % Remove mean of each feature at each timestep from data
    dmean = mean(dvec,2);
    % Assuming all file lengths are the same currently
    Dmat = (dmat - repmat(reshape(dmean,[num_vars,L]),[1,num_files*length_fact]));
    % Scaling
    stdevs = std(Dmat,0,2);
    
    % Scaling tubes and arts by the respective mean stddevs
    if strcmp(config,'')
        rng1 = 1:num_tubes;
        rng2 = num_tubes+1:num_vars;
        stdevs(rng1) = mean(stdevs(rng1));
        stdevs(rng2) = mean(stdevs(rng2));
    % Scaling non-lung tubes, lung tubes ish?, and arts differently
    elseif strcmp(config,'')
        rng1 = [1:6,19:num_tubes];
        rng2 = 7:18;
        rng3 = num_tubes+1:num_vars;
        stdevs(rng1) = mean(stdevs(rng1));
        stdevs(rng2) = mean(stdevs(rng2));
        stdevs(rng3) = mean(stdevs(rng3));
    %else % Scaling by individual feature variance
    end

    % First remove zero std devs from tube sections that don't move
    tub_stds = stdevs(1:num_tubes);
    ztubs = tub_stds~=0;
    z_std = mean(tub_stds(ztubs));
    tub_stds(tub_stds==0) = z_std;
    stdevs(1:num_tubes) = tub_stds;
    
    Dmat = Dmat./repmat(stdevs,[1,L*num_logs]);
    D = reshape(Dmat,[L*num_vars,num_logs]);
    Xp = D(1:p*num_vars,:);
    Xf = D(p*num_vars+1:end,:);
else
    error('Not a supported Data Type');
end

%Remove any zeros and replace with small value to not mess up svd
zs = Xp ==0;
Xp(zs) = 1e-10;
zs = Xf == 0;
Xf(zs) = 1e-10;
%Xp = log10(Xp.^2);
%Xf = log10(Xf.^2);
% Re-order the Columns of Xf and Xp
perm = randperm(num_logs);
Xp = Xp(:,perm);
Xf = Xf(:,perm);
%% Perform Least Squares Regression
skip = 0;
prms = skip+1:k+skip;
%F = Xf*(Xp'*(Xp*Xp')^-1);
F = Xf*pinv(Xp);
Qp_ = cov(Xp')';
Qf_ = cov(Xf')';
Qp_ = eye(size(Qp_));
Qf_ = eye(size(Qf_));
% Take real part of scale factor
F_sc = real(Qf_^(-.5))*F*real(Qp_^(.5));
[U,S,V] = svd(F_sc);
Sk = S(prms,prms);
Vk = V(:,prms);
Uk = U(:,prms);
K = Sk^(1/2)*Vk'*real(Qp_^(-.5));
O = real(Qf_^(.5))*Uk*Sk^(1/2);
Factors = K*Xp;
%%
Kz = zeros(k,num_vars*p);
Oz = zeros(num_vars*f,k);
Xpz = zeros(num_vars*p,num_logs);
Xfz = zeros(num_vars*f,num_logs);
Kz(:,non_zero_p) = K;
Oz(non_zero_f,:) = O;
Xpz(non_zero_p,:) = Xp;
Xfz(non_zero_f,:) = Xf;
%% View Pimitives
view_prim_image(K,O,p,f,k,sum(non_zero_feats),D_lab(non_zero_feats),dt,mk,nk,save_figs,[2,3],testdir,1:sum(non_zero_feats),stdevs,dmean);
%view_prim_image(K,O,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[2,3],testdir,1:num_vars,stdevs,dmean);
if strcmp(data_type,'tubart')
    view_prim_image(Kz,Oz,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[32,33],[testdir,'tube_'],1:num_tubes,stdevs,dmean);
    view_prim_image(Kz,Oz,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[34,35],[testdir,'art_'],num_tubes+1:num_tubes+num_art,stdevs,dmean);
elseif strcmp(data_type,'stubart')
    view_prim_image(Kz,O,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[32,33],[testdir,'tube_'],1:num_tubes,stdevs,dmean);
    view_prim_image(Kz,O,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[34,35],[testdir,'art_'],num_tubes+1:num_tubes+num_art,stdevs,dmean);
    view_prim_image(Kz,O,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[37,38],[testdir,'spectrogram_'],num_tubes+num_art+1:num_vars,stdevs,dmean);
end
%% View Factors
f4 = figure(4); % Graph common factors vs sample/time
plot(Factors')
leg = [repmat('Factor ',[k,1]),num2str((1:k)'),repmat('  Factors',[k,1])];
if k>=10
    leg = [leg; [repmat('Primitive ',[k,1]),num2str((1:k)'),repmat(' Factors',[k,1])]];
end
legend(leg)
title('Latent Variables values over Time/Logs')

% Try a different plot of first 3 factors in 3D
% Has color vary overtime up to time tt
f16 = figure(16);clf;
tt = floor(num_logs);
st = 1;%ceil(num_logs*1);
col = st:st-1+tt;
fac1 = 1; fac2 = 2; fac3 = 3;
xx = Factors(fac1,col); yy=Factors(fac2,col);zz = Factors(fac3,col);
surface([xx;xx],[yy;yy],[zz;zz],[col;col],'facecol','no','edgecol','interp','linew',2);
grid on
title('3 Latent Variables as 3D Trajectory')
xlabel(['Factor ',num2str(fac1)]);
ylabel(['Factor ',num2str(fac2)]);
zlabel(['Factor ',num2str(fac3)]);

% And plot them as 3D scatter plot
f17 = figure(17);clf
plot3(xx,yy,zz,'.')
grid on
title('3 Latent Variables as 3D Scatter Plot')
xlabel(['Factor ',num2str(fac1)]);
ylabel(['Factor ',num2str(fac2)]);
zlabel(['Factor ',num2str(fac3)]);

if save_figs
    set(f4,'PaperPosition',[.25,1.5,8,5])
    print('-f4',[testdir,'factor_v_time'],'-depsc','-r150');
    saveas(f4,[testdir,'factor_v_time'],'fig');
    set(f16,'PaperPosition',[.25,1.5,8,5])
    print('-f16',[testdir,'factor_3d_traj'],'-depsc','-r150');
    saveas(f16,[testdir,'factor_3d_traj'],'fig');
    set(f17,'PaperPosition',[.25,1.5,8,5])
    print('-f17',[testdir,'factor_3d_scatter'],'-depsc','-r150');
    saveas(f17,[testdir,'factor_3d_scatter'],'fig');
end
%%
KKs = Kz.*repmat(stdevs',[k,p]);
OOs = Oz.*repmat(stdevs,[f,k]);
FFs = OOs*KKs;
mfps = dmean(num_vars*p+1:end)-FFs*dmean(1:num_vars*p);
%mfs = (dmean(num_vars*p+1:end)*ones(1,k)).*repmat(stdevs,[f,k]);
%mps = O*K*dmean(1:num_vars*p).*repmat(stdevs,[f,1]);
ffs(FFs>1) = 1;
ffs(FFs<-1) = -1;
imagesc(ffs)
figure(39);plot(mfps)
figure(38);imagesc(abs(log(FFs)))
ffs = FFs;
ffs(abs(FFs)<1) = mean(ffs(abs(FFs)<1));
ffs(FFs<-1) = mean(ffs(FFs<-1));
ffs(FFs>1) = mean(ffs(FFs>1));
%figure(38);imagesc(ffs)
%% Compute Errors
% Scale Yf back to correct units
Xf_pred = Oz*Kz*Xpz;
Xf_mean = dmean(p*(num_vars)+1:end)*ones(1,num_logs);
Xf_unscaled_pred = Xf_pred.*repmat(stdevs,[f,num_logs])+Xf_mean;
Xf_unscaled = Xfz.*repmat(stdevs,[f,num_logs])+Xf_mean;
%Xf_unscaled(zs) = 0;
% Limit predictions to max and min articulator activations as the sim does
% Should technically pull out just art values and test them not look at
% tubes as well. 
%  The two lines below need fixed if we are actually going to limit them
% Xf_unscaled_pred(Xf_unscaled_pred>1) = 1;
% Xf_unscaled_pred(Xf_unscaled_pred<0) = 0;

if strcmp(data_type,'tubart') || strcmp(data_type,'stubart')
    %nzero_f_inds = repmat([ztubs;logical(ones(num_vars-num_tubes,1))],[f,1]);
    %nzero_f_inds = repmat(non_zero_f,[1,num_logs]);
elseif strcmp(data_type,'speech')
    nzero_f_inds = logical(ones(num_vars*f,1));
    ztubs = 1;
end

errors_scaled = (Xf_pred-Xfz);
%errors_scaled = errors_scaled(nzero_f_inds,:);
errors = (Xf_unscaled-Xf_unscaled_pred);
errors = errors(non_zero_f,:);
errors = reshape(errors,[(num_vars-sum(~non_zero_feats))*f,num_logs]);
figure(20); imagesc(errors);
title('Combined Unscaled Xf Prediction Error')
colorbar
figure(21);imagesc(errors_scaled)
title('Combined Xf Prediction Error')
colorbar

% Plot an example error from each component
if strcmp(data_type,'tubart') || strcmp(data_type, 'stubart')
    tube_error = errors_scaled(1:num_tubes,:);
    art_error = errors_scaled(num_tubes+1:num_tubes+num_art,:);
    figure(22); imagesc(tube_error)
    title('Tube Area Xf Prediction Error Example')
    colorbar
    % Scale art_error by std dev
    %art_error = ./repmat(arts_std,[1,num_logs*f]);
    figure(23); imagesc(art_error)
    title('Articulation Xf Prediction Error Example')
    colorbar
    % pull out part of O matrix that corresponds to generating Area predictions
    % in Xf
    Oarea = zeros(f*num_tubes,k);
    for i=0:f-1
            ind = i*num_vars;
            Oarea(i*num_tubes+1:(i+1)*num_tubes,:) = Oz(ind+1:ind+num_tubes,:);
    end
    Oarea_inv = pinv(Oarea);
end
if strcmp(data_type, 'stubart')
    spect_error = errors_scaled(num_tubes+num_art+1:num_vars,:);
    figure(24); imagesc(spect_error)
    title('Spectrogram Xf Prediction Error Example')
    colorbar
end

%% Save K, O, Oarea_inv, VT_mean, tub_std, art_std, f, p, and samp_freq to output files
% compatible with GSL matrix files (vectorization of the matrix transpose)
% precision comes from default : digits
% Using 32 digits of precision to get the best accuracy I can without 
% using binary or hex values in the log files
% TODO: Use hex or binary log files
if strcmp(data_type, 'tubart')
kt = Kz';
fid=fopen([testdir,'K_mat.prim'],'wt');
fprintf(fid,'%.32e\n',kt);
fclose(fid);

ot = Oz';
fid=fopen([testdir,'O_mat.prim'],'wt');
fprintf(fid,'%.32e\n',ot);
fclose(fid);

oait = Oarea_inv';
fid=fopen([testdir,'Oa_inv_mat.prim'],'wt');
fprintf(fid,'%.32e\n',oait);
fclose(fid);

fid=fopen([testdir,'mean_mat.prim'],'wt');
fprintf(fid,'%.32e\n',dmean);
fclose(fid);

fid=fopen([testdir,'stddev.prim'],'wt');
fprintf(fid,'%.32e\n',stdevs);
fclose(fid);

fp = [f,p];
fid=fopen([testdir,'f_p_mat.prim'],'wt');
fprintf(fid,'%d\n',fp);
fclose(fid);

fid=fopen([testdir,'samp_freq.prim'],'wt');
fprintf(fid,'%.32e\n',samp_freq);
fclose(fid);

fid=fopen([testdir,'num_prim.prim'],'wt');
fprintf(fid,'%d\n',k);
fclose(fid);

% Trick to print out artword code from mean. Could be useful Later
%xmm = reshape(Xf_mean,[num_vars,f*num_logs]);
%codes = [repmat('artw.setTarget(i, utterance_length,',[29,1]),num2str(xmm(90:end,1)),repmat(');',[29,1])];
end

if strcmp(data_type,'speech')
    fs_ = fs;
    save([testdir,'prims.mat'],'K','O','dmean','stdevs','f','p','samp_freq','k',...
        'noverlap','nfft','fs_');
else
    save([testdir,'prims.mat'],'K','O','dmean','stdevs','f','p','samp_freq','k');
end

%% Load Area function Reference and Export
% save([testname,'/',config,'/prims.mat'],'K','O','Oarea_inv','dmean','stdevs','f','p','samp_freq','k');
% [VT_log, VT_lab, samp_freq, samp_len] = ...
%         import_datalog([testname,'/artword_logs/apa1.log']);
% VT_log = VT_log(:,1:end-1)'; %remove sound
% VT_log = VT_log(:);
% 
% tub_ind = [];
% art_ind = [];
% for ind = 0:samp_len-1
%     z = ind*(num_tubes+num_art);
%     tub_ind = [tub_ind, z+1:z+num_tubes];
%     art_ind = [art_ind, z+num_tubes+1:z+num_tubes+num_art];
% end
% 
% Aref = VT_log(tub_ind);
% 
% fid=fopen([testname,'/Aref.alog'],'wt');
% fprintf(fid,'%.32e\n',Aref);
% fclose(fid);
% save([testname,'/Aref.mat'],'Aref');