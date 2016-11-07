% Subspace Method
%% Load log files and combine data into one array
clear
testname = 'test3Area';
logs = dir([testname, '/logs/datalog*.log']);
num_logs = length(logs);
VT = [];
for i=1:num_logs
    [VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog([testname,'/logs/',logs(i).name]);
    % Flip matrix to make more similar to how the spectrogram was processed
    % in earlier code.
    vt = VT_log(:,1:end-1)'; %remove sound
    VT = [VT,vt(:)];
end
num_vars = length(VT_lab)-1;
dt = 1/samp_freq;
f = round(samp_len/2);
p = samp_len-f;
%L = f+p;

% Scale areas to be simialar to arts in units in variance
% num_tubes = 89;
% num_art = 29;
% tub_ind = [];
% art_ind = [];
% for ind = 0:samp_len-1
%     z = ind*(num_tubes+num_art);
%     tub_ind = [tub_ind, z+1:z+num_tubes];
%     art_ind = [art_ind, z+num_tubes+1:z+num_tubes+num_art];
% end
% tubes = VT(tub_ind,:);
% arts = VT(art_ind,:);
% % tub_std = std(tubes(:));
% % art_std = std(arts(:));
% % VT(tub_ind,:) = VT(tub_ind,:)/tub_std;
% % VT(art_ind,:) = VT(art_ind,:)/art_std;
% % Instead of variance use range
% tub_rng = range(tubes(:));
% art_rng = range(arts(:));
% VT(tub_ind,:) = VT(tub_ind,:)/tub_rng;
% VT(art_ind,:) = VT(art_ind,:)/art_rng;
% 
% % Remove mean from data
% Xpm = mean(VT(1:p*num_vars,:)')';
% Xfm = mean(VT(p*num_vars+1:end,:)')';
% Xp = VT(1:p*num_vars,:)-Xpm*ones(1,num_logs);
% Xf = VT(p*num_vars+1:end,:)-Xfm*ones(1,num_logs);

% Remove mean from data
VT_mean = mean(VT,2);
VTs = VT-VT_mean*ones(1,num_logs);
% Scale areas to be simialar to arts in units in variance
num_tubes = 89;
num_art = 29;
tub_ind = [];
art_ind = [];
for ind = 0:samp_len-1
    z = ind*(num_tubes+num_art);
    tub_ind = [tub_ind, z+1:z+num_tubes];
    art_ind = [art_ind, z+num_tubes+1:z+num_tubes+num_art];
end
tubes = VTs(tub_ind,:);
arts = VTs(art_ind,:);
tub_std = std(tubes(:));
art_std = std(arts(:));
VTs(tub_ind,:) = VTs(tub_ind,:)/tub_std;
VTs(art_ind,:) = VTs(art_ind,:)/art_std;
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
k = 8;
%F = Xf*(Xp'*(Xp*Xp')^-1);
F = Xf*pinv(Xp);
Qp_ = cov(Xp')';
Qf_ = cov(Xf')';
Qp_ = eye(size(Qp_));
Qf_ = eye(size(Qf_));
% Take real part of scale factor
F_sc = real(Qf_^(-.5))*F*real(Qp_^(.5));
[U,S,V] = svd(F_sc);
Sk = S(1:k,1:k);
Vk = V(:,1:k);
Uk = U(:,1:k);
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
    leg = [leg ; ['Primitive ', num2str(i),'Factors']];
    legend(leg)
    hold off
end

%% Scale Yf back to correct units
tube_inds = [];
art_inds = [];
for j=0:f-1
    ind = j*num_vars;
    tube_inds = [tube_inds,ind+1:ind+num_tubes];
    ind = j*num_vars+num_tubes;
    art_inds = [art_inds,ind+1:ind+num_art];
end
Xf_unscaled = zeros(f*num_vars,num_logs);
Xf_unscaled_pred = Xf_unscaled;
Xf_pred = O*K*Xp;

Xf_mean = VT_mean(p*(num_art+num_tubes)+1:end)*ones(1,num_logs);
Xf_unscaled(tube_inds,:) = Xf(tube_inds,:)*tub_std;
Xf_unscaled(art_inds,:) = Xf(art_inds,:)*art_std;
Xf_unscaled = Xf_unscaled+Xf_mean;
Xf_unscaled2 = VT(p*(num_art+num_tubes)+1:end,:);
Xf_unscaled_pred(tube_inds,:) = Xf_pred(tube_inds,:)*tub_std;
Xf_unscaled_pred(art_inds,:) = Xf_pred(art_inds,:)*art_std;
Xf_unscaled_pred = Xf_unscaled_pred+Xf_mean;
%Xf_unscaled(zs) = 0;
% Limit predictions to max and min articulator activations as the sim does
% Should technically pull out just art values and test them not look at
% tubes as well
Xf_unscaled_pred(Xf_unscaled_pred>1) = 1;
Xf_unscaled_pred(Xf_unscaled_pred<0) = 0;

errors = (Xf_unscaled-Xf_unscaled_pred);
tubes_std = std(reshape(tubes,[num_tubes,(f+p)*num_logs]),0,2);
tube_error = errors(tube_inds,:);
tube_error = reshape(tube_error,[num_tubes,f*num_logs])./repmat(tubes_std,[1,num_logs*f]);
%tube_error = errors(tube_inds,:)/tub_std;
figure(10); imagesc(tube_error)
arts_std = std(reshape(arts,[num_art,(f+p)*num_logs]),0,2);
art_error = errors(art_inds,:);
art_error = reshape(art_error,[num_art,f*num_logs])./repmat(arts_std,[1,num_logs*f]);
%art_error = errors(art_inds,:)/art_std;
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

fid=fopen([testname,'/area_std.prim'],'wt');
fprintf(fid,'%.32e\n',tub_std);
fclose(fid);

fid=fopen([testname,'/art_std.prim'],'wt');
fprintf(fid,'%.32e\n',art_std);
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

save([testname,'/prims.mat'],'K','O','Oarea_inv','VT_mean','tub_std','art_std','f','p','samp_freq','k');

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
Aref = VT_log(tub_ind);

fid=fopen([testname,'/Aref.alog'],'wt');
fprintf(fid,'%.32e\n',Aref);
fclose(fid);
save([testname,'/Aref.mat'],'Aref');