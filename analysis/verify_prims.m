% Attempting to verify primitive controller by making sure that doing the
% math in MATLAB as opposed to c++ gives me the same commanded articulator
% positions.
% Subspace Method
%% Load log file and primitives
clear
testname = 'test3Area';
load([testname,'/prims.mat']);
load([testname,'/Aref.mat']);
doAref = true;
%doAref = false;

if doAref
    refctrllog = '/prim_logs/Areflog1.log';
    refctrlsnd = '/prim_logs/Arefsound1.log';
    reflog = '/artword_logs/apa.log';
    refsnd = '/artword_logs/apasound1.log';

    [VT_log, VT_lab, samp_freq2, samp_len_aref] = ...
    import_datalog([testname,reflog]);
    vt_ref = VT_log(:,1:end-1)'; %remove sound
else
    refctrllog = '/prim_logs/primlog1.log';
    refctrlsnd = '/prim_logs/sound1.log';
end
[VT_log, VT_lab, samp_freq2, samp_len] = ...
    import_datalog([testname,refctrllog]);
vt = VT_log(:,1:end-1)'; %remove sound
num_vars = length(VT_lab)-1;
if samp_freq~=samp_freq2
    errors('Primitive sample frequency is different from log sample frequency')
end

num_tubes = 89;
num_art = 29;
num_feat = num_tubes+num_art;

%% Now attempt to replicate the primitive controller code
% First set Yp_unscaled to have a p long constant history of the inital
% feature vector.
Yp_unscaled = zeros(num_vars*p,1);
for i=0:p-1
    Yp_unscaled(i*num_vars+1:(i+1)*num_vars) = vt(:,1);
end

% Loop through each sample and compute error
errors = zeros(samp_len-1,1);
errors2 = errors;
errors3 = errors;
X_past = zeros(k,samp_len-1);
EK = X_past;
X = X_past;
% Initialize PID history vectors
Ek1 = zeros(k,1);
I1 = Ek1;
for i=1:samp_len-1
    % Shift feature sample backward by one in Yp_unscaled
    Yp_unscaled(1:end-num_vars) = Yp_unscaled(num_vars+1:end);
    Yp_unscaled(end-num_vars+1:end) = vt(:,i);
    % Remove mean from Yp_unscaled
    Yp = Yp_unscaled - VT_mean(1:num_vars*p);
    % Setup the f future Area function that we are tracking and remove mean
    % and scale
    Af_mean = zeros((num_tubes)*f,1);
    for j=1:f
        ind = (j-1)*(num_tubes+num_art)+p*(num_tubes+num_art);
        Af_mean((j-1)*num_tubes+1:j*num_tubes) = VT_mean(ind+1:ind+num_tubes);
    end
    
    if(doAref)
    Afref = zeros(size(Af_mean));
    % Start at sample 2 because first sample is not generated through feedback
    ind_start = i;
    ind_end = ind_start+f;
    if ind_end>samp_len_aref
        if ind_start+1>samp_len_aref
            Afref = (repmat(Aref(end-num_tubes+1:end),[f,1]) - Af_mean)/tub_std;
        else
            ovlap = f-(ind_end-samp_len_aref);
            ind_end = samp_len_aref;
            Afref(1:ovlap*num_tubes) = Aref(ind_start*num_tubes+1:ind_end*num_tubes);
            Afref(ovlap*num_tubes+1:end) = repmat(Aref(end-num_tubes+1:end),[f-ovlap,1]);
            Afref = (Afref - Af_mean)/tub_std;
        end
    else
        Afref = (Aref(ind_start*num_tubes+1:ind_end*num_tubes) - Af_mean)/tub_std;
    end
    end
    % Scale features by their std devs
    for j=0:p-1
        ind = j*num_vars;
        Yp(ind+1:ind+num_tubes) = Yp(ind+1:ind+num_tubes)/tub_std;
        ind = j*num_vars+num_tubes;
        Yp(ind+1:ind+num_art) = Yp(ind+1:ind+num_art)/art_std;
    end
    
    % Use primitives to find next art
    x_past = K*Yp;
    X_past(:,i) = x_past;
    if(doAref)
        % PID Gains
        Kp = 0.2;
        Ki = 0.5;
        Kd = 0.02;
        I_limit = Ki*5;
        Ts = 1/samp_freq;
        %Kp = 0.0;
        %Ki = 0.0;
        %Kd = 0.1;
        % Coefficients of discrete controller
        Ek = Oarea_inv*Afref - x_past;
        EK(:,i) = Ek;
        P = Kp*Ek;
        I = I1 + Ki*Ts/2*(Ek+Ek1);
        % Integral anti-windup
        I(I>I_limit) = I_limit;
        I(I<-I_limit) = -I_limit;
        D = Kd/Ts*(Ek-Ek1);
        x = P+I+D;
        I1 = I;
        Ek1 = Ek;
        X(:,i) = x;
    else
        x = x_past;
    end
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
    errors(i) = sum(abs(art-artlog));
    errors2(i) = sum(abs(art-(O(num_tubes+1:num_vars,1)*art_std+artmean_f)));
    errors3(i) = sum(abs(artlog-artmean_f));
end
% pull out part of O matrix that corresponds to generating Area predictions
% in Xf
O_area = zeros(f*num_tubes,k);
Yf_area = zeros(f*num_tubes,1);
for i=0:f-1
        ind = i*num_vars;
        O_area(i*num_tubes+1:(i+1)*num_tubes,:) = O(ind+1:ind+num_tubes,:);
        Yf_area(i*num_tubes+1:(i+1)*num_tubes,1) = Yf(ind+1:ind+num_tubes,1);
end
FB = O*pinv(O_area);
art_fb = zeros(f*num_art,f*num_tubes);
area_fb = zeros(f*num_tubes);
for i=0:f-1
        ind = i*num_vars;
        art_fb(i*num_art+1:(i+1)*num_art,:) = FB(ind+num_tubes+1:ind+num_vars,:);
        area_fb(i*num_tubes+1:(i+1)*num_tubes,:) = FB(ind+1:ind+num_tubes,:);
end

sum(errors)

%% Plot things
figure(5)
clrordr = get(gca,'colororder');
clf
hold on
leg = [];
dt = 1/samp_freq;
ii = 0;
for i=1:k
    ii = ii+1;
    if i>length(clrordr)
        ii = 1;
    end
    plot((1:samp_len-1)*dt,X_past(i,:),'.-','Color',clrordr(ii,:))
    leg = [leg ; ['Past Primitive ', num2str(i),'Factors']];
    if(doAref)
        plot((1:samp_len-1)*dt,EK(i,:),'Color',clrordr(ii,:))
        leg = [leg ; ['Aref Error     ', num2str(i),'Factors']];
        
        plot((1:samp_len-1)*dt,X(i,:),'--','Color',clrordr(ii,:))
        leg = [leg ; ['Control Signal ', num2str(i),'Factors']];
    end
    legend(leg)
end
hold off

figure(6);imagesc(vt(90:end,:))
figure(7);imagesc(log(vt(1:89,:)))
[Snd,fs,duration] = import_sound([testname,refctrlsnd]);
figure(11);plot(linspace(0,duration,duration*fs),Snd)

if doAref
    pause(duration+1)
    hold on
    [Snd_ref,fs,duration] = import_sound([testname,refsnd]);
    figure(11);plot(linspace(0,duration,duration*fs),Snd_ref)
    hold off
end

figure(42); art_cmd = reshape(art_fb*Yf_area,[num_art,f]); art_cmd(art_cmd<0) = 0;imagesc(art_cmd);
visual_tract