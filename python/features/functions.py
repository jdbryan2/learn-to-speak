import numpy as np

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

def MelFilters( p, nfft, fs )
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

if __name__ == '__main__':
    x = MelFilters(10, 256, 8000)
