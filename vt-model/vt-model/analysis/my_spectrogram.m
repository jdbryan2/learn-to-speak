function [mag_spect, f, t] = my_spectrogram(x,win,noverlap,nfft,fs,fignum)

win_corr = 1/mean(win);
z_pad = 1000;

xlen = length(x);
wlen = length(win);
ncol = floor(1+(xlen-wlen)/(wlen-noverlap));
colidx = 1+ (0:ncol-1)*(wlen-noverlap);
colidxs = repmat(colidx,wlen,1);
rowidx = repmat(0:(wlen-1),1,ncol);
xin = zeros(wlen,ncol);
xin(:) = x(rowidx+colidxs(:)');
xin = win(:,ones(1,ncol)).*xin;
nfftz = nfft+z_pad;
Y = fft(xin,nfftz);
s = win_corr*abs(Y/nfftz);
mag_spect = s(1:nfftz/2+1,:);
mag_spect(2:end-1,:) = 2*mag_spect(2:end-1,:);
f = fs*(0:(nfftz/2))/nfftz;
t = (colidx+wlen/2-1)/fs;

figure(fignum)
% Need to correct for windowing on energy spectrum
surf(t,f,log10(mag_spect.^2),'EdgeColor','none');
axis xy; axis tight; colormap(hot); view(0,90);
xlabel('Time (s)')
ylabel('Frequency (Hz)')
title('Log Magnitude Squared Spectrogram')
set(gca,'FontSize',18)
colorbar