% Subspace Method
%% Load log files and combine data into one array
clear
testname = 'test1';
logs = dir([testname, '/logs/datalog*.log']);
num_logs = length(logs);
for i=1:num_logs
    [VT_log, VT_lab, samp_freq, samp_len, des_samp_freq] = ...
        import_datalog(testname,logs.name(i));
    % Flip matrix to make more similar to how the spectrogram was processed
    % in earlier code.
    vt = VT_log';
    VT = vt(:);
end
num_vars = length(VT_lab);
f = round(samp_len/2);
p = samp_len-f;
%L = f+p;
%Remove any zeros and replace with small value to not mess up svd
zs = VT==0;
VT(zs) = 1e-10;
% Remove mean from data
Xpm = mean(VT(1:p,:)')';
Xfm = mean(VT(p+1:end,:)')';
Xp = Xp_-Xpm*ones(1,samp_len);
Xf = Xf_-Xfm*ones(1,samp_len);
%% Perform Least Squares Regression
k = 8;
%F = Xf*(Xp'*(Xp*Xp')^-1);
F = Xf*pinv(Xp);
Qp_ = cov(Xp')';
Qf_ = cov(Xf')';
%Qp_ = eye(size(Qp_));
%Qf_ = eye(size(Qf_));
% Take real part of scale factor
F_sc = real(Qf_^(-.5))*F*real(Qp_^(.5));
[U,S,V] = svd(F_sc);
Sk = S(1:k,1:k);
Vk = V(:,1:k);
Uk = U(:,1:k);
K = Sk^(1/2)*Vk'*real(Qp_^(-.5));
O = real(Qf_^(.5))*Uk*Sk^(1/2);
Factors = K*Xp;
mn = ceil(sqrt(k));
leg = [];
figure(4); clf;
for i=1:k
    Ps{i} = reshape(K(i,:),[num_vars,p]);
    Fs{i} = reshape(O(:,i),[num_vars,f]);    
    figure(2)
    subplot(mn,mn,i)
    surf((1:p)*dt,1:num_vars,Ps{i},'EdgeColor','none');
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('VT variables')
    title(['Primitive ', num2str(i), ' Input Mapping'])
    set(gca,'FontSize',12)
    set(gca,'YTick',1:num_vars)
    set(gca,'YTickLabel',VT_lab)
    colorbar
    
    figure(3)
    subplot(mn,mn,i)
    surf((p+1:p+f)*dt,1:num_vars,Fs{i},'EdgeColor','none');
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('VT variables')
    title(['Primitive ', num2str(i), ' Output Mapping'])
    set(gca,'FontSize',12)
    set(gca,'YTick',1:num_vars)
    set(gca,'YTickLabel',VT_lab)
    colorbar
    
    figure(4)
    hold on
    plot(t(1:length_fact),Factors(i,:))
    leg = [leg ; ['Primitive ', num2str(i),'Factors']];
    legend(leg)
    hold off
end

% figure(1)
% surf(t,freq,logmag,'EdgeColor','none');
% axis xy; axis tight; colormap(hot); view(0,90);
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Log Magnitude Squared Spectrogram')
% set(gca,'FontSize',12)
% colorbar