import numpy as np
import scipy.signal as signal

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

def PreemphasisFilter(signal, coeff=0.95)
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

def MFCC(signal, nbins, nfft=512, sample_freq=16000, low_freq=0,
         high_freq=None, preemph = 0.95, window='hamming', nperseg=512,
         noverlap=0):


    signal = PreemphasisFilter(signal, preemph)
    f, power_spectrum = signal.periodogram(signal, sample_freq, window, nfft,
                                           scaling='spectrum')

    filter_bank = MelFilters(nbins, nfft, sample_freq, low_freq, high_freq)

    f, t, spectrum = signal.stft(signal, 
                                 fs=sample_freq, 
                                 window=window, 
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft)


    spectrum = spectrum[:, :, :-1] # trim off last element
    spectrum = spectrum.reshape(spectrum.shape[1], spectrum.shape[2]) # remove first dim
    spectrum = np.abs(spectrum)**2. # convert to power spectrum

    energy = numpy.sum(spectrum,1)
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy)

    features = np.dot(spectrum, filter_bank)
    features = numpy.where(features == 0,numpy.finfo(float).eps,features)

    return features, energy

 


if __name__ == '__main__':
    import pylab as plt
    x = MelFilters(10, 512, 16000, 300., 8000.)
    for filt in x:
        plt.plot(filt)

    plt.show()
