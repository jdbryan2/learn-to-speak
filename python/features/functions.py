import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct

# Original Matlab code for mel filter bank generator
################################################################################
#function [ filter_bank ] = MelFilters( p, nfft, fs )
#%UNTITLED5 Summary of this function goes here
#%   Detailed explanation goes here
#
#%f0 = fs/700;
#f_max = fs/2;
#length = floor(nfft/2);
#
#mel_spacing = log(1+f_max/700)/(p+1);
#
#% convert fft bin numbers to mel
#mel_bins = nfft*(700/fs*(exp([0 1 p p+1]*mel_spacing)-1));
#
#% clever matlab-fu to compute the filter bank efficiently
#b1 = floor(mel_bins(1)) + 1;
#b2 = ceil(mel_bins(2));
#b3 = floor(mel_bins(3));
#b4 = min(length, ceil(mel_bins(4))) - 1;
#
#pf = log(1 + fs*(b1:b4)/nfft/700) / mel_spacing;
#fp = floor(pf);
#pm = pf - fp;
#
#r = [fp(b2:b4) 1+fp(1:b3)];
#c = [b2:b4 1:b3] + 1;
#v = 2 * [1-pm(b2:b4) pm(1:b3)];
#
#filter_bank = sparse(r, c, v, p, 1+length);
#
#end

def matlab_MelFilters( p, nfft, fs ):
    """
    Generate numpy array of mel filter bank.
    """

    #f0 = fs/700;
    f_max = fs/2.;
    length = floor(nfft/2.);

    mel_spacing = log(1.+f_max/700.)/(p+1.);

    # convert fft bin numbers to mel
    mel_bins = nfft*(700/fs*(np.exp(np.array([0., 1., p, p+1])*mel_spacing)-1));

    # clever numpy-fu to compute the filter bank efficiently
    b1 = np.floor(mel_bins[1]) + 1;
    b2 = np.ceil(mel_bins[2]);
    b3 = np.floor(mel_bins[3]);
    b4 = np.min(length, ceil(mel_bins[4])) - 1;

    pf = np.log(1 + fs*np.arange(b1,b4)/nfft/700.) / mel_spacing;
    fp = np.floor(pf);
    pm = pf - fp;

    # define matrix sparsely
    # r, c, v -> row, column, value
    r = np.array([fp[b2:b4], 1+fp[1:b3]]);
    c = np.append(np.arange(b2,b4), np.arange(1,b3)) + 1;
    v = 2 * np.append(1-pm[b2:b4], pm[1:b3]);

    # need to find the numpy equivalent of this
    # matrix dimensions are (p x 1+length)
    filter_bank = sparse(r, c, v, p, 1+length);

    return filter_bank

def Freq2Mel(freq):
    return 2595. * np.log10(1.+freq/700.)

def Mel2Freq(mel):
    return 700. * (10.**(mel/2595.0)-1.)

def PreemphasisFilter(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def MelFilters(nbins, nfft, sample_freq, low_freq=0, high_freq=None):
    # largely borrowed from python_speech_features by James Lyons

    # set upper frequency
    high_freq = high_freq or sample_freq/2. # clever use of 'or' 
    assert high_freq <= sample_freq/2., "high_freq is greater than sample_freq/2"

    # compute mel bins (evenly spaced on mel scale)
    low_mel = Freq2Mel(low_freq)
    high_mel = Freq2Mel(high_freq)
    mel_bins = np.linspace(low_mel, high_mel, nbins+2)
    print mel_bins

    fft_bins = np.floor((nfft+1)*Mel2Freq(mel_bins)/sample_freq)
    print fft_bins

    # note: the '//' operator is divides and truncates to integer value
    filter_bank = np.zeros([nbins, nfft//2+1])
    for j in range(0, nbins):

        for i in range(int(fft_bins[j]), int(fft_bins[j+1])):
            filter_bank[j, i] = (i-fft_bins[j])/(fft_bins[j+1]-fft_bins[j])

        for i in range(int(fft_bins[j+1]), int(fft_bins[j+2])):
            filter_bank[j, i] = (fft_bins[j+2]-i)/(fft_bins[j+2]-fft_bins[j+1])

    return filter_bank

def MFCC(data, ncoeffs, nfilters, nfft=512, sample_freq=16000, low_freq=0,
         high_freq=None, preemph = 0.95, window='hamming', nperseg=512,
         noverlap=0):


    data = PreemphasisFilter(data, preemph)

    filter_bank = MelFilters(nfilters, nfft, sample_freq, low_freq, high_freq)

    f, t, spectrum = signal.stft(data, 
                                 fs=sample_freq, 
                                 window=window, 
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft)

    print spectrum.shape

    #spectrum = spectrum[:, :, :-1] # trim off last element
    #spectrum = spectrum.reshape(spectrum.shape[1], spectrum.shape[2]) # remove first dim
    spectrum = np.abs(spectrum)**2. # convert to power spectrum

    energy = np.sum(spectrum,1)
    energy = np.where(energy == 0,np.finfo(float).eps,energy)

    features = np.dot(filter_bank, spectrum)
    features = np.where(features == 0,np.finfo(float).eps,features)

    features = features.T

    features = np.log(features)
    print features.shape

    # I don't really know why the ortho paramter is important
    # it applies a scaling factor but no explanation is given in scipy docs
    features = dct(features, type=2, axis=1, norm='ortho')[:, :ncoeffs]

    return features, energy

 
    # Dynamic programming function
    # Copy and paste from Matlab 
    #
    #function [ distance, lattice, backpointers ] = dynamicLPC( x, y, window, step, p)
    #%dynamicLPC Dynamic time warping applied to the LPC distance
    #%   x,y: input signals (should be 2D like a spectrogram)
    #%   window: window function defines dimensions and shape of window
    #%   step: size of a single increment (default 1)
    #%   a,b,c: weighting parameters used in local SSIM (defaulted to 1)


    #    
    #    %% Format x and y
    #    %%%%%%%%%%%%%%%%%
    #    
    #    [xrows, xcols] = size(x);
    #    [yrows, ycols] = size(y);
    #    [wrows, wcols] = size(window);
    #    wcols = 1;
    #    
    #    
    #    %% Looping parameters
    #    
    #    x_max_steps = floor((xrows-wrows)/step);
    #    y_max_steps = floor((yrows-wrows)/step);
    #    
    #    lattice = 100*ones(x_max_steps, y_max_steps);
    #    backpointers = zeros(x_max_steps, y_max_steps);
    #    
    #    %% walk through the lattice
    #    
    #    for x_index = 1:x_max_steps
    #        
    #        % compute the global constraints
    #        y_min = ceil(max([0.5+0.5*x_index, y_max_steps-2*(x_max_steps-x_index)]));
    #        y_max = floor(min([2*x_index-1, y_max_steps-0.5*(x_max_steps-x_index)]));
    #        
    #        
    #        for y_index = y_min:y_max            
    #            x_local = x((1+(x_index-1)*step):((x_index-1)*step+wrows));
    #            x_local = x_local.*window;
    #            x_lpc = lpc(x_local, p);
    #            Rx = xcorr(x_local);
    #            Rx = toeplitz(Rx((0:p)+wrows));
    #            
    #            y_local = y((1+(y_index-1)*step):((y_index-1)*step+wrows));
    #            y_local = y_local.*window;     
    #            y_lpc = lpc(y_local, p);
    #            
    #            % find best previous step within local constraints
    #            k = 2;
    #            if y_index < 3
    #                k = y_index-1;
    #            end
    #            
    #            %bleh = y_lpc*Rx*y_lpc'
    #            %blah = x_lpc*Rx*x_lpc'
    #            
    #            if x_index > 1
    #                [min_val, min_index] = min(lattice(x_index-1, (y_index-k):y_index));
    #                lattice(x_index, y_index) = log((y_lpc*Rx*y_lpc')/(x_lpc*Rx*x_lpc'))+min_val;
    #                backpointers(x_index, y_index) = y_index-k+(min_index-1);
    #            else
    #                lattice(x_index, y_index) = log((y_lpc*Rx*y_lpc')/(x_lpc*Rx*x_lpc'));
    #                backpointers(x_index, y_index) = 0;
    #            end
    #            

    #        end

    #    end
    #    
    #    if ~isempty(lattice(x_index, y_index))
    #        distance = lattice(x_index, y_index);
    #    else 
    #        distance = nan;
    #    end
    #    
    #    
    #end


if __name__ == '__main__':
    import pylab as plt
    from scipy.io.wavfile import read as wav_read

    # Spoken Language Processing: only first 13 MFCC needed for speech recog
    x = MelFilters(13, 512, 16000, 300., 8000.)
    #for filt in x:
        #plt.plot(filt)

    rate, data = wav_read('/home/jacob/Projects/learn-to-speak/analysis/manifolds/timit/sa1.wav')
    data = 1.0*data/(2**15) # convert from 16 bit integer encoding to [-1, 1]
    print rate, max(data), min(data)
    plt.figure()
    plt.plot(data)
    plt.show()

    #y, e = MFCC(np.random.random(2000), 10)
    y, e = MFCC(data, ncoeffs=13, nfilters=26)
    y2, e = MFCC(data, ncoeffs=13, nfilters=26, nfft=512, nperseg=512,
                 noverlap=384)

    print y.shape
    plt.imshow(np.abs(y[:, 1:].T))
    plt.figure()
    plt.imshow(np.abs(y2[:, 1:].T))
    plt.show()

