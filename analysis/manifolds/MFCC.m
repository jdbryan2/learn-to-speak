function [ mfccs ] = MFCC( signal, fs, p, nfft, window, overlap)
%MFCC compute p mel frequency cepstral coefficients of signal
%   signal: input signal from which mfccs are computed
%   p: number of coefficients to compute
%   nfft: number of points in the fft, defaults to 1024
%   fs: sampling frequency of signal
%   mfccs : output coefficients

if nargin < 3
    p = 25;
end
if nargin < 4 
    nfft = 1024;
end 
if nargin < 5
    window = hamming(round(20/1000*fs)); % default to 20ms window
end
if nargin < 6
    overlap = floor(0.75*length(window));
end

spectrum = abs(spectrogram(signal, window, overlap, nfft)).^2;
mel_filters = MelFilters(p, nfft, fs);

%mfccs = spectrum;
mfccs = dct(log(abs(mel_filters*spectrum)));

%mfccs(0, :) = []; % remove the zeroth row because it tells us nothing


end

