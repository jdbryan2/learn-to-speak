% Calculate and View Factors of log
clear; clf(figure(4));clf(figure(16));clf(figure(17));

save_figs = false;

k = 8;
% Human Speech
data_type = 'speech';
testname = 'TestMySpeech1';
%config = 'original';
config = 'short';
log_type = 'ipa*.mat';

% log_type = 'datalog*.log'
% VT Definitions
num_tubes = 89;
num_art = 29;

% Load Primitives
testdir = [testname,'/',config,num2str(k),'/'];
load([testdir,'prims.mat']);
load([testdir,'speech_preprocess.mat']);

% Load Logs
logs = dir([testname, '/logs/',log_type]);
num_files = length(logs);
num_logs = length(regexp([logs(:).name],'ipa\d\d\d_ex'));
nl = 0;
log_id = zeros(num_logs,1);
log_len = zeros(num_logs,1);
Y = cell(num_logs,1);
for i=1:num_files
    fname = logs(i).name;
    if strcmp(log_type,'ipa*.mat')
        ind = regexp(fname,'ipa\d\d\d_ex');
        if isempty(ind) || ind ~= 1
            continue
        end
        nl = nl+1;
        log_id(nl) = str2num(fname(4:6));
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
    else
        continue;
    end
end

num_vars = length(stdevs);
[ids,ia,ic] = unique(log_id);
cats_label = {'Consonants';'Vowels'};
num_unique_logs = length(ids);
cats = zeros(num_unique_logs,1);
c_inds = ids>=100 & ids<=200; v_inds = ids>=300 &ids<=399;
cats(c_inds) = 1; cats(v_inds) = 2;
markers_bpa = ['v','*']; %['+';'o';'*';'.';'x';'s';'d';'^'];%;'v';'>';'<';'p';'h'];
markers_snd = ['+';'o';'.';'x';'s';'d';'^';'>'];
clrs = ['r';'b';'g';'c';'m';'y';'k';'w'];
line_styles = {'-','--',':','-.'};
XX = cell(num_unique_logs,1); YY = cell(num_unique_logs,1); ZZ = cell(num_unique_logs,1);
%% Cycle through all of the logs computing the factors and plot
for l=1:num_logs
% First set Yp_unscaled to have a p long constant history of the inital
% feature vector.
Yp_unscaled = zeros(num_vars*p,1);
samp_len = log_len(l);
y = Y{l};
yvec = y(:);
Yp_unscaled(num_vars+1:end) = yvec(1:num_vars*(p-1));

% Loop through each sample
num_pts = samp_len-p+1;
X_past = zeros(k,num_pts);
YfU = zeros(num_vars*f,num_pts);
for i=1:num_pts
    % Shift feature sample backward by one in Yp_unscaled
    Yp_unscaled(1:end-num_vars) = Yp_unscaled(num_vars+1:end);
    Yp_unscaled(end-num_vars+1:end) = y(:,p+i-1);
    % Remove mean from Yp_unscaled
    Yp = Yp_unscaled - dmean(1:num_vars*p);
    % Scale features by their std devs
    Yp = Yp./repmat(stdevs,[p,1]);
    % Use primitives to find next art
    x_past = K*Yp;
    X_past(:,i) = x_past;
    Yf = O*x_past;
    
    % Scale Yf back to correct units
    Yf_unscaled = Yf.*repmat(stdevs,[f,1]);
    
    % Add back mean to Yf
    Yf_unscaled = Yf_unscaled + dmean(num_vars*p+1:end);
    YfU(:,i) = Yf_unscaled;
end

% Try a different plot of first 3 factors in 3D
% Has color vary overtime up to time tt
f16 = figure(16); hold on;
tt = floor(num_pts);
st = 1;%ceil(num_logs*1);
col = st:st-1+tt;
fac1 = 1; fac2 = 2; fac3 = 4;
xx = X_past(fac1,col); yy=X_past(fac2,col);zz = X_past(fac3,col);

XX{ic(l)}(end+1:end+log_len(l)+1-f) = xx';
YY{ic(l)}(end+1:end+log_len(l)+1-f) = yy';
ZZ{ic(l)}(end+1:end+log_len(l)+1-f) = zz';
surface([xx;xx],[yy;yy],[zz;zz],[col;col],'facecol','no','edgecol','interp',...
    'linew',2,'marker',markers_bpa(cats(ic(l))));
grid on
title('3 Latent Variables as 3D Trajectory')
xlabel(['Factor ',num2str(fac1)]);
ylabel(['Factor ',num2str(fac2)]);
zlabel(['Factor ',num2str(fac3)]);
hold off;

% View Factors vs time in 2d plot
if ~mod(l-1,1)
f4 = figure(4); % Graph common factors vs sample/time
hold on;
pl = plot3(t(p:end),X_past(1,:),X_past(3,:),'-','Color',clrs(cats(ic(l))));
title('Latent Variables values over Time/Logs')
grid on
hold off;
%for j = 1:k
    %pl = plot(X_past(j,:),['-',markers(cats(ic(l)))],'Color',line_clrs(ic(l)));%,...
                     %'MarkerFaceColor',marker_clrs(j));%,...
                     %markers(cats(j)));
    %pl = plot(X_past(j,:),'-','Color',line_clrs(cats(ic(l))));
%end
end

% if save_figs
%     set(f4,'PaperPosition',[.25,1.5,8,5])
%     print('-f4',[testdir,'factor_v_time'],'-depsc','-r150');
%     saveas(f4,[testdir,'factor_v_time'],'fig');
%     set(f16,'PaperPosition',[.25,1.5,8,5])
%     print('-f16',[testdir,'factor_3d_traj'],'-depsc','-r150');
%     saveas(f16,[testdir,'factor_3d_traj'],'fig');
%     set(f17,'PaperPosition',[.25,1.5,8,5])
%     print('-f17',[testdir,'factor_3d_scatter'],'-depsc','-r150');
%     saveas(f17,[testdir,'factor_3d_scatter'],'fig');
% end
end
for i=1:num_unique_logs
    % And plot them as 3D scatter plot
    f17 = figure(17); hold on;
    pl=plot3(XX{i},YY{i},ZZ{i},[markers_bpa(cats(i)),clrs(i)]);
    if strcmp(clrs(i),'w')
        set(pl,'Color',[.7 .7 .7]);
    end
    grid on
    title('3 Latent Variables as 3D Scatter Plot')
    xlabel(['Factor ',num2str(fac1)]);
    ylabel(['Factor ',num2str(fac2)]);
    zlabel(['Factor ',num2str(fac3)]);
    hold off;
end
%Put in legends
%figure(4); legend(leg);
figure(17); legend([repmat('IPA # ',[num_unique_logs,1]),num2str(ids)])
%figure(16); legend(num2str(ids(ic)))