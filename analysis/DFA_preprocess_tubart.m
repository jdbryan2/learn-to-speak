function [Xp,Xf,stdevs,dmean,D_lab,num_vars,num_logs,dt] = ...
    DFA_preprocess_tubart(testname,smooth,max_num_files,scaling,f,p,skip_first_samp)
% Import and Preprocess VT Tube and Art data for subsequent DFA analysis
% VT Definitions
num_tubes = 89;
num_art = 29;

L = f+p;
logs = dir([testname, '/logs/datalog*.log']);
num_files = length(logs);
if num_files > max_num_files
    num_files = max_num_files;
end
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
    if ~smooth
        % Cut off sample to length L
        length_fact = 1;
    else
        length_fact = samp_len+1-L;
    end
    if skip_first_samp
        vt = vt(:,2:end);
    else
        vt = vt(:,1:end);
    end
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
if strcmp(scaling,'original')
    rng1 = 1:num_tubes;
    rng2 = num_tubes+1:num_vars;
    stdevs(rng1) = mean(stdevs(rng1));
    stdevs(rng2) = mean(stdevs(rng2));
elseif strcmp(scaling,'short_lung_scale')
    %Scaling non-lung tubes, lung tubes ish?, and arts differently
    rng1 = [1:6,22:num_tubes];
    rng2 = 7:21; % technically lungs are 1:23/22?, but first 6 aren't used
    rng3 = num_tubes+1:num_vars;
    stdevs(rng1) = mean(stdevs(rng1));
    stdevs(rng2) = mean(stdevs(rng2));
    stdevs(rng3) = mean(stdevs(rng3));
%else % Scaling by individual feature variance
elseif strcmp(scaling,'individual')
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
    % May need to set a tolerance here instead of just 0
%     tub_stds = stdevs(1:num_tubes);
%     ztubs = tub_stds~=0;
%     z_std = mean(tub_stds(ztubs));
%     tub_stds(tub_stds==0) = z_std;
%     stdevs(1:num_tubes) = tub_stds;
    
% Xp = Xp./repmat(stdevs,[p,num_logs]);
% Xf = Xf./repmat(stdevs,[f,num_logs]);
end