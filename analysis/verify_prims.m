% Attempting to verify primitive controller by making sure that doing the
% math in MATLAB as opposed to c++ gives me the same commanded articulator
% positions.
% Subspace Method
%% Load log file and primitives
clear
testname = 'testRevised8';
load([testname,'/prims.mat']);
%load([testname,'/Aref.mat']);
%doAref = true;
doAref = false;

if doAref
    refctrllog = '/prim_logs/Areflog1.log';
    refctrlsnd = '/prim_logs/Arefsound1.log';
    reflog = '/artword_logs/apa1.log';
    refsnd = '/artword_logs/apa_sound1.log';

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
X_past = zeros(k,samp_len-1);
EK = X_past;
X = X_past;
% Initialize PID history vectors
Ek1 = zeros(k,1);
I1 = Ek1;
Art = []; Artlog = [];
for i=1:samp_len-1
    % Shift feature sample backward by one in Yp_unscaled
    Yp_unscaled(1:end-num_vars) = Yp_unscaled(num_vars+1:end);
    Yp_unscaled(end-num_vars+1:end) = vt(:,i);
    % Remove mean from Yp_unscaled
    Yp = Yp_unscaled - dmean(1:num_vars*p);
    % Setup the f future Area function that we are tracking and remove mean
    % and scale
    Af_mean = zeros((num_tubes)*f,1);
    for j=1:f
        ind = (j-1)*(num_tubes+num_art)+p*(num_tubes+num_art);
        Af_mean((j-1)*num_tubes+1:j*num_tubes) = dmean(ind+1:ind+num_tubes);
    end
    
    if(doAref)
    Afref = zeros(size(Af_mean));
    % Start at sample 2 because first sample is not generated through feedback
    ind_start = i;
    ind_end = ind_start+f;
    if ind_end>samp_len_aref
        if ind_start+1>samp_len_aref
            Afref = (repmat(Aref(end-num_tubes+1:end),[f,1]) - Af_mean)./repmat(stdevs(1:num_tubes),[f,1]);
        else
            ovlap = f-(ind_end-samp_len_aref);
            ind_end = samp_len_aref;
            Afref(1:ovlap*num_tubes) = Aref(ind_start*num_tubes+1:ind_end*num_tubes);
            Afref(ovlap*num_tubes+1:end) = repmat(Aref(end-num_tubes+1:end),[f-ovlap,1]);
            Afref = (Afref - Af_mean)./repmat(stdevs(1:num_tubes),[f,1]);
        end
    else
        Afref = (Aref(ind_start*num_tubes+1:ind_end*num_tubes) - Af_mean)./repmat(stdevs(1:num_tubes),[f,1]);
    end
    end
    % Scale features by their std devs
    Yp = Yp./repmat(stdevs,[p,1]);
    
    % Use primitives to find next art
    x_past = K*Yp;
    X_past(:,i) = x_past;
    if(doAref)
        % PID Gains
        skip = 0;
%         Kp =          [0.2/2,   0.8/3,       0.4/3,     2.5/3,     10.0/3]';
%         Ki =          [20.0,    20.0/2,     10.0/3,   75.0/4,    10.0]';
%         Kd =          [0.005/3, 0.02/3,     0.001,    0.041/3,   0.15/2]';
%         I_limit =     [Ki(1)*100, Ki(2)*100, Ki(3)*100, Ki(4)*100, 1]';
        
        scale = 0.0031;
        nn = 40;
        Kp = [38.1375, 33.3272, 29.4786, 27.0174, 21.1427, 19.4681, 17.2987, 12.9520, 12.6968, 11.4316, 9.4109, 8.8003, 7.1776, 6.9008, 5.8787, 5.6424, 5.3387, 4.5657, 4.3664, 3.7702, 3.5929, 3.4918, 3.2031, 3.0276, 2.9124, 2.3938, 2.3026, 2.1132, 1.9882, 1.9070, 1.7922, 1.7125, 1.5319, 1.4327, 1.3016, 1.2827, 1.1766, 1.1448, 1.0682, 1.0356]';
        Kp = Kp*scale;
        Ki = zeros(nn,1);
        Kd = zeros(nn,1);
        I_limit = zeros(nn,1);
        Kp = Kp(skip+1:k+skip);
        Ki = Ki(skip+1:k+skip);
        Kd = Kd(skip+1:k+skip);
        I_limit = I_limit(skip+1:k+skip);
        Ts = 1/samp_freq;
        % Coefficients of discrete controller
        Ek = Oarea_inv*Afref - x_past;
        EK(:,i) = Ek;
        P = Kp.*Ek;
        I = I1 + Ts/2*Ki.*(Ek+Ek1);
        % Integral anti-windup
        max_ind = I>I_limit;
        I(max_ind) = I_limit(max_ind);
        min_ind = I<-I_limit;
        I(min_ind) = -I_limit(min_ind);
        D = Kd.*(Ek-Ek1)/Ts;
        x = P+I+D;
        I1 = I;
        Ek1 = Ek;
        X(:,i) = x;
    else
        x = x_past;
        X(:,i) = x;
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
    Yf_unscaled = Yf.*repmat(stdevs,[f,1]);
    
    % Add back mean to Yf
    Yf_unscaled = Yf_unscaled + dmean(num_vars*p+1:end);
    
    art = Yf_unscaled(num_tubes+1:num_vars);
    art(art>1) = 1;
    art(art<0) = 0;
    Art = [Art,art];
    artlog = vt(num_tubes+1:num_vars,i+1);
    Artlog = [Artlog,artlog];
    artmean_f = dmean(num_vars*p+num_tubes+1:num_vars*(p+1));
    
    errors(i) = sum(abs(art-artlog));
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
    if i<10
        leg = [leg ; ['Past Primitive ', num2str(i),'  Factors']];
    else
        leg = [leg ; ['Past Primitive ', num2str(i),' Factors']];
    end
    if(doAref)
        plot((1:samp_len-1)*dt,EK(i,:),'Color',clrordr(ii,:))
        if i<10
            leg = [leg ; ['Aref Error     ', num2str(i),'  Factors']];
        else
            leg = [leg ; ['Aref Error     ', num2str(i),' Factors']];
        end
        
        plot((1:samp_len-1)*dt,X(i,:),'--','Color',clrordr(ii,:))
        if i<10
            leg = [leg ; ['Control Signal ', num2str(i),'  Factors']];
        else
            leg = [leg ; ['Control Signal ', num2str(i),' Factors']];
        end
    end
    legend(leg)
end
hold off

% Plot Factors in 3D
figure(18);clf;
col = (1:samp_len-1)*dt;
xx = X_past(1,:); yy=X_past(2,:);zz = X_past(3,:);
surface([xx;xx],[yy;yy],[zz;zz],[col;col],'facecol','no','edgecol','interp','linew',2);
grid on

figure(6);imagesc(vt(90:end,:))
figure(7);imagesc(log(vt(1:89,:)))
[Snd,fs,duration] = import_sound([testname,refctrlsnd],true);
figure(11);plot(linspace(0,duration,duration*fs),Snd)

if doAref
    pause(duration+1)
    hold on
    [Snd_ref,fs,duration] = import_sound([testname,refsnd],true);
    figure(11);plot(linspace(0,duration,duration*fs),Snd_ref)
    hold off
end

figure(42); art_cmd = reshape(art_fb*Yf_area,[num_art,f]); art_cmd(art_cmd<0) = 0;imagesc(art_cmd);
%visual_tract