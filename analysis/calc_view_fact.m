% Calculate and View Factors of log
clear; clf(figure(4));clf(figure(16));clf(figure(17));
% VT Definitions
num_tubes = 89;
num_art = 29;

% Log definitions
% {Speech IPA, Artword Logs or Primlogs, IPA Artword Logs}
log_type_id = [1,2,3]; % Need to fix this
log_types = {'ipa*.mat'; '*.log';'ipa*.log'};
log_expr = {'ipa\d\d\d_ex';'^((?!sound).)*.log$';'^ipa*((?!sound).)*.log$'};

save_figs = false;
fac1 = 1; fac2 = 2; fac3 = 3;
%fac1 = 4; fac2 = 5; fac3 = 6;
%fac1 = 6; fac2 = 7; fac3 = 8;

% Human Speech
% data_type = 'speech';
% testname = 'TestMySpeech1';
% log_fldr = '/logs/';
% log_type = 1;
% %config = 'original';
% %config = 'broad_phonetic_cat';
% config = 'medium';
% %config = 'long';
% %config = 'default';
% %config = 'textured_input';
% %config = 'original_594';
% k = 8;
% testdir = [testname,'/',data_type,'-',config,num2str(k),'/'];
% % Load Speech specific preprocessing parameters
% load([testdir,'speech_preprocess.mat']);

% VT Tubes Artwords
%data_type = 'tub';
%data_type = 'tubart';
data_type = 'tubart';

%testname = 'testBatch1000';
testname = 'testStim3Batch300';
%config = 'original_noisemaker';

%testname = 'testRevised1';
%config = 'original_noisemaker';
log_type = 2;
%testname = 'testBatch300';
%config = 'original_50noisemaker';
%testname = 'testBatch1000';
%testname = 'testStim1Batch50';
%config = 'default';
%config = 'medium_original_scale';
%config = 'short_original_scale';
%config = 'long_original_scale';
config = 'default';
%config = 'medium';
%config = 'long';
k = 8;
testdir = [testname,'/',data_type,'-',config,num2str(k),'/'];
if log_type == 3
    log_fldr = '/artword_logs/';
elseif log_type == 2
    log_fldr = ['/',data_type,'-',config,num2str(k),'/prim_logs/'];
end


% Load Primitives
load([testdir,'prims.mat']);
non_zero_feats = find(stdevs~=0);

% Load Logs
logs = dir([testname,log_fldr,log_types{log_type}]);
num_files = length(logs);
num_logs = 0;
for i=1:num_files
    if isempty(regexp([logs(i).name],log_expr{log_type}))
        continue;
    end
    num_logs = num_logs+1;
end
nl = 0;
log_id = cell(num_logs,1);
log_len = zeros(num_logs,1);
Y = cell(num_logs,1);
for i=1:num_files
    fname = logs(i).name;
    ind = regexp(fname,log_expr{log_type});
    if isempty(ind)
        continue
    end
    nl = nl+1;
    if strcmp(data_type,'speech')
        log_id{nl} = fname(4:6);
        load([testname, '/logs/',fname]);
        if(fs~=fs_)
            error('Speech Log was sampled at different rate than Synergy')
        end
        win = hamming(nfft,'symmetric');
        [mag_spect, freq, t] = my_spectrogram(y,win,noverlap,nfft,fs_,0);
        [num_f,num_samp] = size(mag_spect);
        log_len(nl) = num_samp;
        %Remove any zeros and replace with small value to not mess up svd
        %zs = mag_spect==0;
        %mag_spect(zs) = 1e-10;
        logmag = log10(mag_spect.^2);
        f_ind = 1:5:num_f;
        freqs = freq(f_ind);
        num_freqs = length(freqs);
        Y{nl} = logmag(f_ind,:);
    elseif strcmp(data_type,'tub') || strcmp(data_type,'tubart') || strcmp(data_type,'stubart')
        if log_type == 3
            ind = regexp(fname,'()\d*\.log')-1-3; % Fix for non ipa
            log_id{nl} = fname(1+3:ind);
        elseif log_type == 2
            ind = regexp(fname,'()\d*\.log'); % Fix for non primlog
            log_id{nl} = fname(1:ind);
        end %clugey way of fixing this.
        [VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog([testname,log_fldr,logs(i).name]);
        % Flip matrix to make more similar to how the spectrogram was processed
        % in earlier code.
        vt = VT_log(:,1:end-1)'; %remove sound
        if strcmp(data_type,'tub')
            VT = vt(1:num_tubes,:); % extract tubes
        elseif strcmp(data_type,'tubart')
            % leave it alone
            VT = vt;
        elseif strcmp(data_type,'stubart')
            if nl ==1
                snd_logs = dir([testname,log_fldr,'*sound*.log']);
                snd_logs = {snd_logs(:).name};
            end
            for j=1:length(snd_logs)
                same = regexp(snd_logs{j},fname(1:6));
                if ~isempty(same)
                    n_ind = j;
                    break;
                end
            end
            %Import Sound
            [snd, fs, duration] = import_sound([testname,log_fldr,snd_logs{n_ind}],false);
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
            VT = [vt(:,2:end);spect];
            samp_len = samp_len-1;
        end
        VT = VT(non_zero_feats,:); % remove zero_feats
        if sum(sum(isnan(VT)))
            nl = nl -1;
            num_logs = num_logs-1;
            log_id = log_id(1:end-1);
            log_len = log_len(1:end-1);
            Y = Y(1:end-1);
            continue
        end
        Y{nl} = VT;
        log_len(nl) = samp_len;
    end
end

num_nonz_vars = length(non_zero_feats);
num_vars = length(stdevs);
[log_class_ids,~,log_class_nums] = unique(log_id); %cell2mat?
% cats_label = {'Consonants';'Vowels'};
num_unique_logs = length(log_class_ids);
cats = zeros(num_unique_logs,1);
if log_type == 1 || log_type == 3
    log_class_ids = str2num(cell2mat(log_class_ids));
    c_inds = log_class_ids>=100 & log_class_ids<=200; v_inds = log_class_ids>=300 &log_class_ids<=399;
    cats(c_inds) = 1; cats(v_inds) = 2;
elseif log_type == 2
    cats = 1:num_unique_logs;
else
    error('Unsupported Data Type')
end
markers = ['v','*','+','o','.','x','s','d','^'];%;'v';'>';'<';'p';'h'];
clrs = ['r';'b';'g';'c';'m';'y';'k';'w'];
XX = cell(num_unique_logs,1); YY = cell(num_unique_logs,1); ZZ = cell(num_unique_logs,1);
xpast_mean = zeros(k,num_logs);
%% Cycle through all of the logs computing the factors and plot
for l=1:num_logs
% First set Yp_unscaled to have a p long constant history of the inital
% feature vector.
Yp_unscaled = zeros(num_nonz_vars*p,1);
samp_len = log_len(l);
y = Y{l};
yvec = y(:);
Yp_unscaled(num_nonz_vars+1:end) = yvec(1:num_nonz_vars*(p-1));

% Loop through each sample
num_pts = samp_len-p+1;
X_past = zeros(k,num_pts);
YfU = zeros(num_nonz_vars*f,num_pts);
Yp_mean = zeros(num_nonz_vars*p,1);
for s=1:p
    Yp_mean((s-1)*num_nonz_vars+1:s*num_nonz_vars) = dmean(non_zero_feats+(s-1)*num_vars);
end
for i=1:num_pts
    % Shift feature sample backward by one in Yp_unscaled
    Yp_unscaled(1:end-num_nonz_vars) = Yp_unscaled(num_nonz_vars+1:end);
    Yp_unscaled(end-num_nonz_vars+1:end) = y(:,p+i-1);
    % Remove mean from Yp_unscaled
    Yp = Yp_unscaled - Yp_mean;
    % Scale features by their std devs
    Yp = Yp./repmat(stdevs(non_zero_feats),[p,1]);
    % Use primitives to find next art
    x_past = K*Yp;
    X_past(:,i) = x_past;
    Yf = O*x_past;
    
    % Scale Yf back to correct units
    %Yf_unscaled = Yf.*repmat(stdevs(non_zero_feats),[f,1]);
    
    % Add back mean to Yf
    %Yf_unscaled = Yf_unscaled + dmean(num_vars*p+1:end);
    %YfU(:,i) = Yf_unscaled;
end

% Try a different plot of first 3 factors in 3D
% Has color vary overtime up to time tt
f16 = figure(16); hold on;
tt = floor(num_pts);
st = 1;%ceil(num_logs*1);
col = st:st-1+tt;
xx = X_past(fac1,col); yy=X_past(fac2,col);zz = X_past(fac3,col);
xpast_mean(:,l) = mean(X_past,2);

XX{log_class_nums(l)}(end+1:end+num_pts) = xx';
YY{log_class_nums(l)}(end+1:end+num_pts) = yy';
ZZ{log_class_nums(l)}(end+1:end+num_pts) = zz';
surface([xx;xx],[yy;yy],[zz;zz],[col;col],'facecol','no','edgecol','interp',...
    'linew',2,'marker',markers(cats(log_class_nums(l))));
grid on
title('3 Latent Variables as 3D Trajectory')
xlabel(['Factor ',num2str(fac1)]);
ylabel(['Factor ',num2str(fac2)]);
zlabel(['Factor ',num2str(fac3)]);
set(gca,'FontSize',12);
hold off;

% View Factors vs time in 2d plot
if ~mod(l-1,1)
f4 = figure(4); % Graph common factors vs sample/time
hold on;
if log_type == 1
    tt = t(p:end);
elseif log_type == 2 || log_type==3
    tt = ((1:num_pts)+p)*1/samp_freq;
end
clr_len = length(clrs);
if cats(log_class_nums(l))>clr_len
    clr_ind = mod(cats(log_class_nums(l)),clr_len);
    if clr_ind == 0
        clr_ind = clr_len;
    end
    clr = clrs(clr_ind);
else
    clr = clrs(cats(log_class_nums(l)));
end
pl = plot3(tt,X_past(fac1,:),X_past(fac2,:),'-','Color',clr);
title('Latent Variables values over Time/Logs')
ylabel(['Factor ',num2str(fac1)]);
zlabel(['Factor ',num2str(fac2)]);
xlabel('Time');
grid on
set(gca,'FontSize',12);
hold off;
%for j = 1:k
    %pl = plot(X_past(j,:),['-',markers(cats(ic(l)))],'Color',line_clrs(ic(l)));%,...
                     %'MarkerFaceColor',marker_clrs(j));%,...
                     %markers(cats(j)));
    %pl = plot(X_past(j,:),'-','Color',line_clrs(cats(ic(l))));
%end
end

if save_figs
%     set(f4,'PaperPosition',[.25,1.5,8,5])
%     print('-f4',[testdir,'factor_v_time'],'-depsc','-r150');
%     saveas(f4,[testdir,'factor_v_time'],'fig');
%     set(f16,'PaperPosition',[.25,1.5,8,5])
%     print('-f16',[testdir,'factor_3d_traj'],'-depsc','-r150');
%     saveas(f16,[testdir,'factor_3d_traj'],'fig');
    set(f17,'PaperPosition',[.25,1.5,8,5])
    print('-f17',[testdir,'factor_3d_scatter'],'-depsc','-r150');
    saveas(f17,[testdir,'factor_3d_scatter'],'fig');
end
end
facts = 1:k;%[1,2,3];
Dist_mat = zeros(num_logs);
for i=1:num_logs
    for j=1:num_logs
        Dist_mat(i,j) = norm(xpast_mean(facts,i)-xpast_mean(facts,j));
    end
end
figure(51);imagesc(Dist_mat)
n_class_logs = num_logs/2;
c_norm = mean(mean(Dist_mat(1:n_class_logs,1:n_class_logs)))
v_norm = mean(mean(Dist_mat(n_class_logs+1:end,n_class_logs+1:end)))
vc_norm = mean(mean(Dist_mat(1:n_class_logs,n_class_logs+1:end)))

for i=1:num_unique_logs
    % And plot them as 3D scatter plot
    f17 = figure(17); hold on;
    clr_len = length(clrs);
    if i>clr_len
        clr_ind = mod(i,clr_len);
        if clr_ind == 0
            clr_ind = clr_len;
        end
        clr = clrs(clr_ind);
    else
        clr = clrs(i);
    end
    pl=plot3(XX{i},YY{i},ZZ{i},[markers(cats(i)),clr]);
    if strcmp(clr,'w')
        set(pl,'Color',[.7 .7 .7]);
    end
    grid on
    title('Latent Variables Scatter Plot')
    xlabel(['Factor ',num2str(fac1)]);
    ylabel(['Factor ',num2str(fac2)]);
    zlabel(['Factor ',num2str(fac3)]);
    set(gca,'FontSize',12);
    hold off;
end
%Put in legends
%figure(4); legend(leg);
figure(17); 
if log_type == 1 || log_type == 3
    legend([repmat('IPA # ',[num_unique_logs,1]),num2str(log_class_ids)])
elseif log_type == 2
    legend(log_class_ids)
else
    error('Log_type Not Supported')
end
%figure(16); legend(num2str(ids(ic)))