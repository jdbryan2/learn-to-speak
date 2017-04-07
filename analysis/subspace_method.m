% Subspace Method
%% Load log files and combine data into one array
clear

% Function Selection
%save_figs = true;
save_figs = false;
% Human Speech
% data_type = 'spectrum';
% testname = 'testSpeech1';
% filename = 'sample1_30sec';
% %config = 'simple';
% %config = 'texture';
% config = 'smooth';

% VT-articulatory
data_type = 'tubart';
testname = 'testThesis4';
config = 'test';

% VT-acoustic-articulatory
% data_type = 'stubart';
% testname = 'testThesis4';
% config = 'test';

%data_type = 'stubart';

naming = {testname,data_type,config};

% Model Parameters
% History lengths
% f = 1;
% p = 17;
%mk = ceil(sqrt(k));
%nk = mk;
k = 5;
mk = floor(sqrt(k));
nk = ceil(sqrt(k));

% Import Data and Preprocess accordingly
if strcmp(data_type, 'spectrum')
    load([testname,'/',filename,'.mat'])
    if strcmp(config,'simple') || strcmp(config,'smooth')
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
    if strcmp(config,'simple') || strcmp(config,'smooth')
        [mag_spect, freq, t] = my_spectrogram(y(1:length(y)),win,noverlap,nfft,fs,fignum);
    elseif strcmp(config,'texture')
        [mag_spect, freq, t] = my_spectrogram(y(1:length(y)/5),win,noverlap,nfft,fs,fignum);
    end
    if save_figs == true
        saveas(fignum,[testname,'/',data_type,'_spectrogram','_',config],'epsc');
        saveas(fignum,[testname,'/',data_type,'_spectrogram','_',config],'fig');
    end
    dt = t(2)-t(1);
    
    if strcmp(config,'simple')
        tlen = 0.15;
        f = round(tlen/dt);
        p = round(tlen/dt);
    elseif strcmp(config,'smooth')
        tlen = 0.3;
        f = round(tlen/dt);
        p = round(tlen/dt);
    elseif strcmp(config,'texture')
        f =1*4; %textured one
        p = 29*4; %textured one
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
    XP = zeros(length(f_ind),p*length_fact);
    XF = zeros(length(f_ind),f*length_fact);
    XPF = zeros(length(f_ind),L*length_fact);
    Xp_ = zeros(length(f_ind)*p,length_fact);
    Xf_ = zeros(length(f_ind)*f,length_fact);
    for n=1:length_fact
        xp = logmag(f_ind,n:n+p-1);
        xf = logmag(f_ind,n+p:n+p+f-1);
        XP(:,1+(n-1)*p:n*p) = xp;
        XF(:,1+(n-1)*f:n*f) = xf;
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
    % Round frequencies for labels
    D_lab = num2cell(floor(freqs));
    dmean = [Xpm;Xfm];
    stdevs = std(XPF,0,2);
elseif strcmp(data_type, 'tubart')
    f = 1;
    p = 17;
    logs = dir([testname, '/logs/datalog*.log']);
    num_logs = length(logs);
    VT = [];
    VT1 = [];
    nan_inds = [];
    for i=1:num_logs
        [VT_log, VT_lab, samp_freq, samp_len] = ...
            import_datalog([testname,'/logs/',logs(i).name]);
        % Flip matrix to make more similar to how the spectrogram was processed
        % in earlier code.
        vt = VT_log(:,1:end-1)'; %remove sound
        if sum(sum(isnan(vt)))
            nan_inds = [nan_inds, i];
            continue
        end
        % Clip data to exact length of future and past vectors
        vt = vt(:,1:f+p);
        VT1 = [VT1,vt];
        VT = [VT,vt(:)];
    end
    samp_len = f+p;
    num_logs = num_logs - length(nan_inds);
    num_vars = length(VT_lab)-1;
    dt = 1/samp_freq;
    num_tubes = 89;
    num_art = 29;

    % Remove mean of each feature at each timestep from data
    VT_mean = mean(VT,2);
    VTs1 = (VT1 - repmat(reshape(VT_mean,[num_vars,samp_len]),[1,num_logs]));
    % Scale by std dev of features over all timesteps
    % Remove mean first because stddev is over all features not time varying features
    stdevs = std(VTs1,0,2);
    % Make tube sections with 0 std dev = the mean tube std dev
    % May need to set a tolerance here instead of just 0

    % Trying to match up with last version of code
    % tub_ind = [];
    % art_ind = [];
    % for ind = 0:samp_len-1
    %     z = ind*(num_tubes+num_art);
    %     tub_ind = [tub_ind, z+1:z+num_tubes];
    %     art_ind = [art_ind, z+num_tubes+1:z+num_tubes+num_art];
    % end
    % VTs_ = VT-VT_mean*ones(1,num_logs);
    % tubs = VTs_(tub_ind,:);
    % arts = VTs_(art_ind,:);
    % stdevs(1:num_tubes) = std(tubs(:));
    % tub_std = stdevs(1);
    % stdevs(num_tubes+1:end) = std(arts(:));
    % %art_std = stdevs(num_tubes+1);

    % Scaling non-lung tubes, lung tubes ish?, and arts differently
    % rng1 = [1:6,19:num_tubes];
    % rng2 = 7:18;
    % rng3 = num_tubes+1:num_vars;
    % stdevs(rng1) = mean(stdevs(rng1));
    % stdevs(rng2) = mean(stdevs(rng2));
    % stdevs(rng3) = mean(stdevs(rng3));

    % Scaling tubes and arts by the respective mean stddevs
    % rng1 = 1:num_tubes;
    % rng2 = num_tubes+1:num_vars;
    % stdevs(rng1) = mean(stdevs(rng1));
    % stdevs(rng2) = mean(stdevs(rng2));

    % Scaling each variable individually
    % Do nothing

    tub_stds = stdevs(1:num_tubes);
    z_std = mean(tub_stds(tub_stds~=0));
    tub_stds(tub_stds==0) = z_std;
    stdevs(1:89) = tub_stds;
    VTs1 = VTs1./repmat(stdevs,[1,samp_len*num_logs]);
    VTs = reshape(VTs1,[samp_len*num_vars,num_logs]);
    Xp = VTs(1:p*num_vars,:);
    Xf = VTs(p*num_vars+1:end,:);
    
    %Translation of variable names
    D_lab = VT_lab;
    dmean = VT_mean;
elseif strcmp(data_type, 'stubart')
    % Keep f+p<=samp_len-1
    p = 13;
    f = 12;
    L = f+p;
    logs = dir([testname, '/logs/datalog*.log']);
    snd_logs = dir([testname, '/logs/sound*.log']);
    num_logs = length(logs);
    assert(num_logs == length(snd_logs));
    dvec = [];
    dmat = [];
    nan_inds = [];
    for i=1:num_logs
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
        % Clip data to exact length of future and past vectors
        % Also remove first sample to line up data with STFT
        vt = vt(:,2:end);
        samp_time = (samp_len-1)/samp_freq;
        
        %Import Sound
        [snd, fs, duration] = import_sound([testname,'/logs/',snd_logs(i).name],false);
        %Perform Spectrogram
        win_time = (samp_time)/L;
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
        
        combo = [vt;spect];
        dmat = [dmat,combo];
        dvec = [dvec,combo(:)];
    end
    %Round labels
    D_lab = [VT_lab(1:end-1),num2cell(floor(freqs))];
    num_vars = length(D_lab);
    num_tubes = 89;
    num_art = 29;
    % Remove mean of each feature at each timestep from data
    dmean = mean(dvec,2);
    Dmat = (dmat - repmat(reshape(dmean,[num_vars,L]),[1,num_logs]));
    stdevs = std(Dmat,0,2);

    % Scaling each variable individually
    % Scale by std dev of features over all timesteps

    % First remove zero std devs from tube sections that don't move
    tub_stds = stdevs(1:num_tubes);
    z_std = mean(tub_stds(tub_stds~=0));
    tub_stds(tub_stds==0) = z_std;
    stdevs(1:num_tubes) = tub_stds;
    
    Dmat = Dmat./repmat(stdevs,[1,L*num_logs]);
    D = reshape(Dmat,[L*num_vars,num_logs]);
    Xp = D(1:p*num_vars,:);
    Xf = D(p*num_vars+1:end,:);

    %samp_len =- L
    %VT =- dvec
    %VT1 =- dmat
    %VT_lab =- combo_lab
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

%% View Pimitives
view_prim_image(K,O,p,f,k,num_vars,D_lab,dt,num_logs,mk,nk,save_figs,naming);
%% View Factors
figure(4) % Graph common factors vs sample/time
plot(Factors')
leg = [repmat('Primitive ',[k,1]),num2str((1:k)'),repmat('  Factors',[k,1])];
if k>=10
    leg = [leg; [repmat('Primitive ',[k,1]),num2str((1:k)'),repmat(' Factors',[k,1])]];
end
legend(leg)

% Try a different plot of first 3 factors in 3D
% Has color vary overtime up to time tt
figure(16);clf;
tt = floor(num_logs*1);
st = 1;%ceil(num_logs*1);
col = st:st-1+tt;
xx = Factors(1,col); yy=Factors(2,col);zz = Factors(3,col);
surface([xx;xx],[yy;yy],[zz;zz],[col;col],'facecol','no','edgecol','interp','linew',2);
grid on

% And plot them as 3D scatter plot
figure(17);clf
plot3(xx,yy,zz,'.')
grid on
%% Compute Errors
% Scale Yf back to correct units
Xf_pred = O*K*Xp;
Xf_mean = dmean(p*(num_vars)+1:end)*ones(1,num_logs);
Xf_unscaled_pred = Xf_pred.*repmat(stdevs,[f,num_logs])+Xf_mean;
Xf_unscaled = Xf.*repmat(stdevs,[f,num_logs])+Xf_mean;
%Xf_unscaled(zs) = 0;
% Limit predictions to max and min articulator activations as the sim does
% Should technically pull out just art values and test them not look at
% tubes as well. 
%  The two lines below need fixed if we are actually going to limit them
% Xf_unscaled_pred(Xf_unscaled_pred>1) = 1;
% Xf_unscaled_pred(Xf_unscaled_pred<0) = 0;

errors_scaled = Xf_pred-Xf;
errors = (Xf_unscaled-Xf_unscaled_pred);
errors = reshape(errors,[num_vars*f,num_logs]);
figure(20); imagesc(errors);
title('Combined Xf Prediction Error')
colorbar
figure(21);imagesc(errors_scaled)
title('Combined Normalized Xf Prediction Error')
colorbar

% Plot an example error from each component
if strcmp(data_type,'tubart') || strcmp(data_type, 'stubart')
    tube_error = errors(1:num_tubes,:);
    art_error = errors(num_tubes+1:num_tubes+num_art,:);
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
            Oarea(i*num_tubes+1:(i+1)*num_tubes,:) = O(ind+1:ind+num_tubes,:);
    end
    Oarea_inv = pinv(Oarea);
end
if strcmp(data_type, 'stubart')
    spect_error = errors(num_tubes+num_art+1:num_vars,:);
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
kt = K';
fid=fopen([testname,'/K_mat.prim'],'wt');
fprintf(fid,'%.32e\n',kt);
fclose(fid);

ot = O';
fid=fopen([testname,'/O_mat.prim'],'wt');
fprintf(fid,'%.32e\n',ot);
fclose(fid);

oait = Oarea_inv';
fid=fopen([testname,'/Oa_inv_mat.prim'],'wt');
fprintf(fid,'%.32e\n',oait);
fclose(fid);

fid=fopen([testname,'/mean_mat.prim'],'wt');
fprintf(fid,'%.32e\n',dmean);
fclose(fid);

fid=fopen([testname,'/stddev.prim'],'wt');
fprintf(fid,'%.32e\n',stdevs);
fclose(fid);

fp = [f,p];
fid=fopen([testname,'/f_p_mat.prim'],'wt');
fprintf(fid,'%d\n',fp);
fclose(fid);

fid=fopen([testname,'/samp_freq.prim'],'wt');
fprintf(fid,'%.32e\n',samp_freq);
fclose(fid);

fid=fopen([testname,'/num_prim.prim'],'wt');
fprintf(fid,'%d\n',k);
fclose(fid);

%save([testname,'/prims.mat'],'K','O','Oarea_inv','VT_mean','stdevs','tub_std','art_std','f','p','samp_freq','k');
save([testname,'/prims.mat'],'K','O','Oarea_inv','VT_mean','stdevs','f','p','samp_freq','k');
% figure(1)
% surf(t,freq,logmag,'EdgeColor','none');
% axis xy; axis tight; colormap(hot); view(0,90);
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Log Magnitude Squared Spectrogram')
% set(gca,'FontSize',12)
% colorbar
end
%% Load Area function Reference and Export

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