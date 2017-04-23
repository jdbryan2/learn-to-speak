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
% data_type = 'tubart';
% smooth = false;
% %testname = 'testThesis4';
% %testname = 'testFeatureTrack';
% %testname = 'testFeatureTrack2';
% %testname = 'testFeatureTrack4'; noisy oblong spheriod broken up wrt time
% %testname = 'testRevised1'; %original sound maker
% %testname = 'testRevised3';
% %testname = 'testRevised4';
% %testname = 'testBatch3';
% %testname = 'testBatch2';
% testname = 'testBatch1000';
% %config = 'original';
% %config = 'original_no_zeros';
% %config = 'long';
% %config = 'short';
% %config = 'short_lung_scale';
% %config = 'short_no_zeros';
% %config = 'scale_fix_perm_no_smooth';
% config = 'scale_fix';
% %config = 'scale_fix_long';

% VT-acoustic-articulatory
data_type = 'stubart';
%testname = 'testThesis4';
%testname = 'testFeatureTrack'; %smooth ish ball with a lot of rotation (.1,.25) 10 sec 5 logs
%testname = 'testFeatureTrack2'; %smooth ish ball with some rotation (.25,.25) 10 sec 5 logs
%testname = 'testFeatureTrack3'; % smooth ish Ball, no rotation (.25,.25) 10 sec 10 logs
%testname = 'testFeatureTrack4'; %smooth ish ball with some rotation (.1,.25) 10 sec 5 logs
testname = 'testBatch1000';
config = 'long'; % Long random stim trials
%config = 'longer'; % longer f and p
%config = 'default';

% DFA Configuration Settings
if strcmp(config,'default')
    % General Model Parameters
    k = 8; % Number of hidden state/primitives
    f = 3; % Number of timesteps in the future
    p = 3; % Number of timesteps in the past
    
    % Specific to Speech data type
    max_length = 30; % Maximum length of sample in seconds
    win_time = 20/1000; % Width of FFT window in seconds
    
    % Specific to tubart datatype
    skip_first_samp = true; % Boolean determining discarding of first log sample
    
    % Specific to tubart or stubart
    smooth = false; % Boolean determining use of multiple samples from same log for training
    scaling = 'individual'; % String specifying feature scaling method
    max_num_files = 200; % Max number of log files to analyze
elseif strcmp(config,'original_594')
    tlen = 0.3;
    % General Model Parameters
    k = 8; % Number of hidden state/primitives
    f = round(tlen/win_time); % Number of timesteps in the future
    p = round(tlen/win_time); % Number of timesteps in the past
    
    % Specific to Speech data type
    max_length = 30; % Maximum length of sample in seconds
    win_time = 20/1000; % Width of FFT window in seconds
elseif strcmp(config,'textured_input')
    % General Model Parameters
    k = 8; % Number of hidden state/primitives
    f = 4; % Number of timesteps in the future
    p = 116; % Number of timesteps in the past
    
    % Specific to Speech data type
    max_length = 6; % Maximum length of sample in seconds
    win_time = 5/1000; % Width of FFT window in seconds
elseif strcmp(config,'broad_phonetic_cat')
    % General Model Parameters
    k = 8; % Number of hidden state/primitives
    f = 2; % Number of timesteps in the future
    p = 2; % Number of timesteps in the past
    
    % Specific to Speech data type
    max_length = 30; % Maximum length of sample in seconds
    win_time = 20/1000; % Width of FFT window in seconds
elseif strcmp(config,'long')
    % General Model Parameters
    k = 8; % Number of hidden state/primitives
    f = 12; % Number of timesteps in the future
    p = 13; % Number of timesteps in the past
    
    % Specific to Speech data type
    max_length = 30; % Maximum length of sample in seconds
    win_time = 20/1000; % Width of FFT window in seconds
    
    % Specific to tubart datatype
    skip_first_samp = false; % Boolean determining discarding of first log sample
    
    % Specific to tubart or stubart
    smooth = false; % Boolean determining use of multiple samples from same log for training
    scaling = 'individual'; % String specifying feature scaling method
    max_num_files = 200; % Max number of log files to analyze
end

% if strcmp(data_type,'speech');
%     if strcmp(config,'simple')
%         tlen = 0.15;
%         win_time = 20/1000;
%         max_length = 30;
%         f = round(tlen/win_time);
%         p = round(tlen/win_time);
%     elseif strcmp(config,'original')
%         tlen = 0.3;
%         win_time = 20/1000;
%         max_length = 30;
%         f = round(tlen/win_time);
%         p = round(tlen/win_time);
%     elseif strcmp(config,'smooth2')
%         win_time = 20/1000;
%         max_length = 30;
%         tlen = 0.3;
%         f = round(tlen/win_time);
%         p = 2;
%     elseif strcmp(config,'texture')
%         win_time = 5/1000; %textured one
%         max_length = 6;
%         f =1*4; %textured one
%         p = 29*4; %textured one
%     elseif strcmp(config,'short')
%         win_time = 20/1000;
%         max_length = 30;
%         f = 2;
%         p = 2;
%     end
% elseif strcmp(data_type,'tubart')
%     if strcmp(config,'long') || strcmp(config,'scale_fix_long') || strcmp(config,'scale_fix_perm')% || strcmp(config,'scale_fix_perm_no_smooth')
%         f = 13;
%         p = 13;
%         smooth = true;
%         max_num_files = 1000;
%         scaling = 'individual';
%         skip_first_samp = false;
%     elseif strcmp(config,'original') || strcmp(config,'original_no_zeros')
%         f = 13;
%         p = 13;
%         smooth = true;
%         max_num_files = 50;
%         scaling = 'original';
%         skip_first_samp = false;
%     elseif strcmp(config,'longer')
%         f = 15;
%         p = 15;
%         smooth = true;
%         max_num_files = 1000;
%         scaling = 'individual';
%         skip_first_samp = false;
%     elseif strcmp(config,'short')
%         f = 3;
%         p = 3;
%         smooth = true;
%         max_num_files = 200;
%         scaling = 'original';
%         skip_first_samp = false;
%     elseif strcmp(config,'short_lung_scale') 
%         f = 3;
%         p = 3;
%         smooth = true;
%         max_num_files = 1000;
%         scaling = 'short_lung_scale';
%         skip_first_samp = false;
%     elseif strcmp(config,'short_no_zeros')
%         f = 3;
%         p = 3;
%         smooth = true;
%         max_num_files = 1000;
%         scaling = 'individual';
%         skip_first_samp = false;
%     elseif strcmp(config,'scale_fix')
%         f = 3;
%         p = 3;
%         smooth = false;
%         max_num_files = 200;
%         scaling = 'individual';
%         skip_first_samp = true;
%     elseif strcmp(config,'scale_fix_perm_no_smooth')
%         f=13;
%         p=13;
%         smooth = false;
%         max_num_files = 200;
%         scaling = 'individual';
%         skip_first_samp = false;
%     end
% elseif strcmp(data_type,'stubart')
%     if strcmp(config, 'long')
%         % Keep f+p<=samp_len-1
%         p = 13;
%         f = 12;
%         max_num_files = 200;
%         scaling = 'individual';
%         smooth = false;
%     elseif strcmp(config,'longer')
%         % Keep f+p<=samp_len-1
%         p = 15;
%         f = 15;
%         max_num_files = 200;
%         scaling = 'individual';
%         smooth = false;
%     end
% else
%     error('Unsupported Data Type')
% end
% VT Definitions
num_tubes = 89;
num_art = 29;

% Subplot figure helper variables
mk = ceil(sqrt(k));
nk = ceil((k-mk)/mk)+1;

% Create directory for this specific test
testdir = [testname,'/',data_type,'-',config,num2str(k),'/'];
mkdir(testdir);

% Import Data and Preprocess accordingly
if strcmp(data_type, 'speech')
    [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
        DFA_preprocess_speech(testname,testdir,win_time,max_length,f,p,save_figs);
elseif strcmp(data_type, 'tubart')
    [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
        DFA_preprocess_tubart(testname,smooth,max_num_files,scaling,f,p,skip_first_samp);
    samp_freq = 1/dt;
elseif strcmp(data_type, 'stubart')
    [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
        DFA_preprocess_stubart(testname,smooth,max_num_files,scaling,f,p);
    samp_freq = 1/dt;
else
    error('Not a supported Data Type');
end

non_zero_feats = stdevs>=1e-6;
%stdevs(~non_zero_feats) = max(stdevs)*10;
stdevs(~non_zero_feats) = 0;%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!---
non_zero_p = repmat(non_zero_feats,[p,1]);
non_zero_f = repmat(non_zero_feats,[f,1]);
Xp = Xp(non_zero_p,:)./repmat(stdevs(non_zero_feats),[p,num_logs]);
Xf = Xf(non_zero_f,:)./repmat(stdevs(non_zero_feats),[f,num_logs]);
% Make tube sections with 0 std dev = the mean tube std dev
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!BAD IDEA POTENTIALLY bc still could be
% fairly small. Maybe make it really big so we don't care about it or
% not bc xp for those values is zero anyway

%Remove any zeros and replace with small value to not mess up svd
zs = Xp ==0;
Xp(zs) = 1e-10;
zs = Xf == 0;
Xf(zs) = 1e-10;
%Xp = log10(Xp.^2);
%Xf = log10(Xf.^2);
% Re-order the Columns of Xf and Xp
%perm = randperm(num_logs); %!!!!!!!!!!!!!!!!!!!!!!!!!-------
%Xp = Xp(:,perm);
%Xf = Xf(:,perm);
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
    view_prim_image(Kz,Oz,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[32,33],[testdir,'tube_'],1:num_tubes,stdevs,dmean);
    view_prim_image(Kz,Oz,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[34,35],[testdir,'art_'],num_tubes+1:num_tubes+num_art,stdevs,dmean);
    view_prim_image(Kz,Oz,p,f,k,num_vars,D_lab,dt,mk,nk,save_figs,[36,37],[testdir,'spectrogram_'],num_tubes+num_art+1:num_vars,stdevs,dmean);
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
% figure(42);imagesc(abs(log(reshape(OOs(:,1),[num_vars,f]))))
% figure(41);imagesc(abs(log(reshape(KKs(1,:),[num_vars,p]))))
%mfs = (dmean(num_vars*p+1:end)*ones(1,k)).*repmat(stdevs,[f,k]);
%mps = O*K*dmean(1:num_vars*p).*repmat(stdevs,[f,1]);
ffs(FFs>1) = 1;
ffs(FFs<-1) = -1;
%imagesc(ffs)
figure(38);plot(mfps)
figure(39);imagesc(abs(log(FFs)))
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
    save([testdir,'prims.mat'],'K','O','dmean','stdevs','f','p','k');
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