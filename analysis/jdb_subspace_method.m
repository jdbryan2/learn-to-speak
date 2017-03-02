% Subspace Method
%% Load log files and combine data into one array
clear
testname = 'rand_gesture3';
file_string = ['data/logs/', testname, '/datalog%i.log'];
logs = dir(['data/logs/', testname, '/datalog*.log']);
num_logs = length(logs);

% setup ranges we want
num_tubes = 89;
num_pressure = 89;
num_art = 29;
num_sound = 1; % gets removed 
num_vars = num_tubes+num_art;

rng1 = 7:18; % lungs
rng2 = 19:num_tubes; % trachea to lips
rng3 = (num_tubes+1):num_vars; % articulators

% setup prediction variables
f = 1; %round(samp_len/2);
p = 60;%samp_len-f;
L = f+p;



VTs = [];
%VT1 = [];
for i=1:num_logs
    %logs(i).name
    [VT_log, VT_lab, samp_freq, samp_len] = ...
        import_datalog(sprintf(file_string, i));
    
    % Throw out unstable datas
    if sum(sum(isnan(VT_log))) == 0
        % Flip matrix to make more similar to how the spectrogram was processed
        % in earlier code.
        VT = VT_log(:,1:end-1)'; %remove sound
        VT((num_tubes+1):(num_tubes+num_pressure), :) = []; % remove pressure for now

        dt = 1/samp_freq;

        % Remove mean of each feature at each timestep from data
        VT_mean = mean(VT,2);% ???
        VTs1 = (VT - repmat(mean(VT, 2),[1,size(VT, 2)]));

        % Scale by std dev of features over all timesteps
        % Remove mean first because stddev is over all features not time varying features
        stdevs = std(VTs1,0,2);
        % Make tube sections with 0 std dev = the mean tube std dev
        % May need to set a tolerance here instead of just 0


        stdevs(rng1) = mean(stdevs(rng1)); % lungs
        stdevs(rng2) = mean(stdevs(rng2)); % trachea to lips
        stdevs(rng3) = mean(stdevs(rng3)); % articulators
       
        stdevs(stdevs==0) = mean(stdevs(stdevs~=0)); % make all zero values equal to average
        
        %redundant with rng3
        %stdevs(num_tubes+1:end) = mean(stdevs(num_tubes+1:end));
        
        %tub_stds = stdevs(1:num_tubes);
        %z_std = mean(tub_stds(tub_stds~=0));
        %tub_stds(tub_stds==0) = z_std;
        %stdevs(1:num_tubes) = tub_stds;
        
        % normalize by standard deviation
        VTs1 = VTs1./repmat(stdevs,[1,size(VTs1, 2)]); 
        
        % reshape into sliding window format
        VTs1 = reshape(VTs1,[],1); % turn into column vector
        %VTs2 = buffer(VTs2, num_vars*L, num_vars*p); % buffer turns it into a sliding window format
        VTs1 = buffer(VTs1, num_vars*L, 0); % buffer turns it into a sliding window format
        VTs1(:, end) = []; % last row ends with zeros (usually) so we throw it out
        %VTs2(:, 1) = [];
        VTs = [VTs,VTs1];
    end
end

num_logs = size(VTs, 2); % num_logs being used pretty differently here

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
[K, O, ~, ~, ~] = SubspaceDFA(Xf, Xp, k);
% skip = 0;
% prms = skip+1:k+skip;
% %F = Xf*(Xp'*(Xp*Xp')^-1);
% F = Xf*pinv(Xp);
% Qp_ = cov(Xp')';
% Qf_ = cov(Xf')';
% Qp_ = eye(size(Qp_));
% Qf_ = eye(size(Qf_));
% % Take real part of scale factor
% F_sc = real(Qf_^(-.5))*F*real(Qp_^(.5));
% [U,S,V] = svd(F_sc);
% Sk = S(prms,prms);
% Vk = V(:,prms);
% Uk = U(:,prms);
% K = Sk^(1/2)*Vk'*real(Qp_^(-.5));
% O = real(Qf_^(.5))*Uk*Sk^(1/2);
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

Xf_mean = repmat(mean(Xf, 2), [1, num_logs]);%VT_mean((p*(num_art+num_tubes)+1):end)*ones(1,num_logs);
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
dirname = ['data/primitives/', testname];
mkdir(dirname);
fid=fopen([dirname,'/K_mat.prim'],'wt');
fprintf(fid,'%.32e\n',kt);
fclose(fid);

ot = O';
fid=fopen([dirname,'/O_mat.prim'],'wt');
fprintf(fid,'%.32e\n',ot);
fclose(fid);

oait = Oarea_inv';
fid=fopen([dirname,'/Oa_inv_mat.prim'],'wt');
fprintf(fid,'%.32e\n',oait);
fclose(fid);

fid=fopen([dirname,'/mean_mat.prim'],'wt');
fprintf(fid,'%.32e\n',VT_mean);
fclose(fid);

fid=fopen([dirname,'/stddev.prim'],'wt');
fprintf(fid,'%.32e\n',stdevs);
fclose(fid);

fp = [f,p];
fid=fopen([dirname,'/f_p_mat.prim'],'wt');
fprintf(fid,'%d\n',fp);
fclose(fid);

fid=fopen([dirname,'/samp_freq.prim'],'wt');
fprintf(fid,'%.32e\n',samp_freq);
fclose(fid);

fid=fopen([dirname,'/num_prim.prim'],'wt');
fprintf(fid,'%d\n',k);
fclose(fid);

%save([testname,'/prims.mat'],'K','O','Oarea_inv','VT_mean','stdevs','tub_std','art_std','f','p','samp_freq','k');
save([dirname,'/prims.mat'],'K','O','Oarea_inv','VT_mean','stdevs','f','p','samp_freq','k');
% figure(1)
% surf(t,freq,logmag,'EdgeColor','none');
% axis xy; axis tight; colormap(hot); view(0,90);
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Log Magnitude Squared Spectrogram')
% set(gca,'FontSize',12)
% colorbar

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