% Subspace Method
%% Separate data into past and future
clear
load('recorded1_spect.mat')
freq = f;
dt = t(2)-t(1);

% Randomly select N samples of length L from data
%N = 10000;
f = round(.3/dt);
p = round(.3/dt);
L = f+p;
[num_f,num_samp] = size(mag_spect);
%Remove any zeros and replace with small value to not mess up svd
zs = mag_spect==0;
mag_spect(zs) = 1e-10;
logmag = log10(mag_spect.^2);
f_ind = 1:5:num_f;
freqs = freq(f_ind);
n_freqs = length(freqs);
length_fact = length(t)-L;
Xp_ = zeros(length(f_ind)*p,length_fact);
Xf_ = zeros(length(f_ind)*f,length_fact);
for n=1:length_fact
    xp = logmag(f_ind,n:n+p-1);
    xf = logmag(f_ind,n+p:n+p+f-1);
    Xp_(:,n) = xp(:);
    Xf_(:,n) = xf(:);
end
Xpm = mean(Xp_')';
Xfm = mean(Xf_')';
Xp = Xp_-Xpm*ones(1,length_fact);
Xf = Xf_-Xfm*ones(1,length_fact);
% Xp = zeros(length(freqs)*p,N);
% Xf = zeros(length(freqs)*f,N);
% for n=1:N
%     start_p = ceil(rand(1)*(num_samp-L));
%     xp = logmag(freqs,start_p:start_p+p-1);
%     xf = logmag(freqs,start_p+p:start_p+p+f-1);
%     Xp(:,n) = xp(:);
%     Xf(:,n) = xf(:);
% end

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
    Ps{i} = reshape(K(i,:),[n_freqs,p]);
    Fs{i} = reshape(O(:,i),[n_freqs,f]);    
    figure(2)
    subplot(mn,mn,i)
    surf((1:p)*dt,freqs,Ps{i},'EdgeColor','none');
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Primitive ', num2str(i), ' Input Mapping'])
    set(gca,'FontSize',12)
    colorbar
    
    figure(3)
    subplot(mn,mn,i)
    surf((p+1:p+f)*dt,freqs,Fs{i},'EdgeColor','none');
    axis xy; axis tight; colormap(hot); view(0,90);
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Primitive ', num2str(i), ' Output Mapping'])
    set(gca,'FontSize',12)
    colorbar
    
    figure(4)
    hold on
    plot(t(1:length_fact),Factors(i,:))
    leg = [leg ; ['Primitive ', num2str(i),'Factors']];
    legend(leg)
    hold off
end

figure(1)
surf(t,freq,logmag,'EdgeColor','none');
axis xy; axis tight; colormap(hot); view(0,90);
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Log Magnitude Squared Spectrogram')
set(gca,'FontSize',12)
colorbar