% Subspace Method
%% Load log files and combine data into one array
clear
testname = 'test3Area';
logs = dir([testname, '/logs/datalog*.log']);
num_logs = length(logs);
VT = [];
VT1 = [];
for i=1:num_logs
    [VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog([testname,'/logs/',logs(i).name]);
    % Flip matrix to make more similar to how the spectrogram was processed
    % in earlier code.
    vt = VT_log(:,1:end-1)'; %remove sound
    VT1 = [VT1,vt];
    VT = [VT,vt(:)];
end
num_vars = length(VT_lab)-1;
dt = 1/samp_freq;
f = round(samp_len/2);
p = samp_len-f;
%L = f+p;
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
% art_std = stdevs(num_tubes+1);
rng1 = [1:6,19:num_tubes];
rng2 = 7:18;
rng3 = num_tubes+1:num_vars;
stdevs(rng1) = mean(stdevs(rng1));
stdevs(rng2) = mean(stdevs(rng2));
stdevs(rng3) = mean(stdevs(rng3));
stdevs(num_tubes+1:end) = mean(stdevs(num_tubes+1:end));
tub_stds = stdevs(1:num_tubes);
z_std = mean(tub_stds(tub_stds~=0));
tub_stds(tub_stds==0) = z_std;
stdevs(1:89) = tub_stds;
VTs1 = VTs1./repmat(stdevs,[1,samp_len*num_logs]);
VTs = reshape(VTs1,[samp_len*num_vars,num_logs]);
Xp = VTs(1:p*num_vars,:);
Xf = VTs(p*num_vars+1:end,:);

%Remove any zeros and replace with small value to not mess up svd
zs = Xp ==0;
Xp(zs) = 1e-10;
zs = Xf == 0;
Xf(zs) = 1e-10;
%Xp = log10(Xp.^2);
%Xf = log10(Xf.^2);
%% Perform Least Squares Regression
k = 3;
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

% pull out part of O matrix that corresponds to generating Area predictions
% in Xf
Oarea = zeros(f*num_tubes,k);
for i=0:f-1
        ind = i*num_vars;
        Oarea(i*num_tubes+1:(i+1)*num_tubes,:) = O(ind+1:ind+num_tubes,:);
end
Oarea_inv = pinv(Oarea);

mn = ceil(sqrt(k));
leg = [];
figure(4); clf;
for i=1:k
    Ps{i} = reshape(K(i,:),[num_vars,p]);
    Fs{i} = reshape(O(:,i),[num_vars,f]);
    figure(2)
    subplot(mn,mn,i)
    % Hacky way to make surf not delete 1 row and 1 col at end of data
    % could use imagesc instead, but I think label editing is harder
    % http://www.mathworks.com/examples/matlab/community/6386-offsets-and-discarded-data-via-pcolor-and-surf
    
    surf(((0:p))*dt,(1:num_vars+1)-.5,[[Ps{i}; zeros(1,p)],zeros(num_vars+1,1)],'EdgeColor','none');
    % Or use interpolation which doesn't get rid of values
    %surf((1:p)*dt,1:num_vars,Ps{i},'EdgeColor','none');
    %shading interp
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('VT variables')
    title(['Primitive ', num2str(i), ' Input Mapping'])
    set(gca,'FontSize',12)
    set(gca,'YTick',1:num_vars)
    set(gca,'YTickLabel',VT_lab(1:end-1))
    colorbar
    
    figure(3)
    subplot(mn,mn,i)
    % Hacky way to make surf not delete 1 row and 1 col at end of data
    % could use imagesc instead, but I think label editing is harder
    surf(((p:p+f))*dt,(1:num_vars+1)-.5,[[Fs{i}; zeros(1,f)],zeros(num_vars+1,1)],'EdgeColor','none');
    % Or use interpolation which doesn't get rid of values
    %surf((p+1:p+f)*dt,1:num_vars,Fs{i},'EdgeColor','none');
    %shading interp
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('VT variables')
    title(['Primitive ', num2str(i), ' Output Mapping'])
    set(gca,'FontSize',12)
    set(gca,'YTick',1:num_vars)
    set(gca,'YTickLabel',VT_lab(1:end-1))
    colorbar
    
    figure(4)
    hold on
    plot(1:num_logs,Factors(i,:))
    if i<10
        leg = [leg ; ['Primitive ', num2str(i),'  Factors']];
    else
        leg = [leg ; ['Primitive ', num2str(i),' Factors']];
    end
    legend(leg)
    hold off
end

%% Scale Yf back to correct units
Xf_unscaled = zeros(f*num_vars,num_logs);
Xf_unscaled_pred = Xf_unscaled;
Xf_pred = O*K*Xp;

Xf_mean = VT_mean(p*(num_art+num_tubes)+1:end)*ones(1,num_logs);
Xf_unscaled = Xf.*repmat(stdevs,[f,num_logs])+Xf_mean;
%Xf_unscaled(zs) = 0;
% Limit predictions to max and min articulator activations as the sim does
% Should technically pull out just art values and test them not look at
% tubes as well
Xf_unscaled_pred(Xf_unscaled_pred>1) = 1;
Xf_unscaled_pred(Xf_unscaled_pred<0) = 0;

errors = (Xf_unscaled-Xf_unscaled_pred);
errors = reshape(errors,[num_vars*f,num_logs]);
tube_error = errors(1:num_tubes,:);
figure(10); imagesc(tube_error)
art_error = errors(num_tubes+1:num_vars,:);
% Scale art_error by std dev
%art_error = ./repmat(arts_std,[1,num_logs*f]);
figure(11); imagesc(art_error)


%% Save K, O, Oarea_inv, VT_mean, tub_std, art_std, f, p, and samp_freq to output files
% compatible with GSL matrix files (vectorization of the matrix transpose)
% precision comes from default : digits
% Using 32 digits of precision to get the best accuracy I can without 
% using binary or hex values in the log files
% TODO: Use hex or binary log files
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
fprintf(fid,'%.32e\n',VT_mean);
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
%% Load Area function Reference and Export
[VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog([testname,'/artword_logs/apa.log']);
VT_log = VT_log(:,1:end-1)'; %remove sound
VT_log = VT_log(:);

tub_ind = [];
art_ind = [];
for ind = 0:samp_len-1
    z = ind*(num_tubes+num_art);
    tub_ind = [tub_ind, z+1:z+num_tubes];
    art_ind = [art_ind, z+num_tubes+1:z+num_tubes+num_art];
end

Aref = VT_log(tub_ind);

fid=fopen([testname,'/Aref.alog'],'wt');
fprintf(fid,'%.32e\n',Aref);
fclose(fid);
save([testname,'/Aref.mat'],'Aref');