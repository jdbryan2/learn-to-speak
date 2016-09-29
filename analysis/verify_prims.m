% Attempting to verify primitive controller by making sure that doing the
% math in MATLAB as opposed to c++ gives me the same commanded articulator
% positions.
% Subspace Method
%% Load log file and primitives
clear
load prims.mat
logname = 'test3Area/prim_logs/primlog1.log';
[VT_log, VT_lab, samp_freq2, samp_len] = ...
    import_datalog(logname);
vt = VT_log(:,1:end-1)'; %remove sound
VT = vt(:);
num_vars = length(VT_lab)-1;
if samp_freq~=samp_freq2
    error('Primitive sample frequency is different from log sample frequency')
end

num_tubes = 89;
num_art = 29;

%% Now attempt to replicate the primitive controller code
% First set Yp_unscaled to have a p long constant history of the inital
% feature vector.
Yp_unscaled = zeros(num_vars*p,1);
for i=0:p-1
    Yp_unscaled(i*num_vars+1:(i+1)*num_vars) = vt(:,1);
end

% Loop through each sample and compute error
error = zeros(samp_len-1,1);
error2 = error;
error3 = error;
for i=1:samp_len-1
    % Shift feature sample backward by one in Yp_unscaled
    Yp_unscaled(1:end-num_vars) = Yp_unscaled(num_vars+1:end);
    Yp_unscaled(end-num_vars+1:end) = vt(:,i);
    % Remove mean from Yp_unscaled
    Yp = Yp_unscaled - VT_mean(1:num_vars*p);
    % Scale features by their std devs
    for j=0:p-1
        ind = j*num_vars;
        Yp(ind+1:ind+num_tubes) = Yp(ind+1:ind+num_tubes)/tub_std;
        ind = j*num_vars+num_tubes;
        Yp(ind+1:ind+num_art) = Yp(ind+1:ind+num_art)/art_std;
    end
    
    % Use primitives to find next art
    x = K*Yp;
%     primno = 1;
%     xno = x(primno);
%     x = zeros(k,1);
%     x(primno) = xno;
    %x(2:end) = zeros(k-1,1);
    %x(1) = 1;
    % Remove later. Just for testing activation of individual primitives
%     x(1) = 100;
%     zer_ind = [2,4,5,6,7,8];
%     x(zer_ind) = 0;
    Yf = O*x;
    
    % Scale Yf back to correct units
    Yf_unscaled = zeros(f*num_vars,1);
    for j=0:f-1
        ind = j*num_vars;
        Yf_unscaled(ind+1:ind+num_tubes) = Yf(ind+1:ind+num_tubes)*tub_std;
        ind = j*num_vars+num_tubes;
        Yf_unscaled(ind+1:ind+num_art) = Yf(ind+1:ind+num_art)*art_std;
    end
    
    % Add back mean to Yf
    Yf_unscaled = Yf_unscaled + VT_mean(num_vars*p+1:end);
    
    art = Yf_unscaled(num_tubes+1:num_vars);
    art(art>1) = 1;
    art(art<0) = 0;
    artlog = vt(num_tubes+1:num_vars,i+1);
    artmean_f = VT_mean(num_vars*p+num_tubes+1:num_vars*(p+1));
    
    %i
    %art-artlog
    error(i) = sum(abs(art-artlog));
    error2(i) = sum(abs(art-(O(num_tubes+1:num_vars,1)*art_std+artmean_f)));
    error3(i) = sum(abs(artlog-artmean_f));
end
sum(error)