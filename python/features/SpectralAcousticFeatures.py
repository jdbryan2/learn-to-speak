import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import pylab as plt

from features.BaseFeatures import BaseFeatures
from features.BaseFeatures import moving_average

from features.functions import MFCC

class SpectralAcousticFeatures(BaseFeatures):

    def __init__(self, **kwargs):
        super(SpectralAcousticFeatures, self).__init__(**kwargs)

    def DefaultParams(self):
        super(SpectralAcousticFeatures, self).DefaultParams()
        self.fs = 8000
        self.window = 'hamming'
        self.nfft = 512
        self.ncoeffs = 13 # number of MFCC features to return
        self.nfilters = 26 # number of mel spaced filters
        self.periodsperseg = 5 # measured in sample periods

    def InitializeParams(self, **kwargs):
        super(SpectralAcousticFeatures, self).InitializeParams(**kwargs)

        self.fs = kwargs.get('fs', self.fs)
        self.window = kwargs.get('window', self.window)
        self.nfft = kwargs.get('nfft', self.nfft)
        self.periodsperseg = kwargs.get('periodsperseg', self.periodsperseg)


    def Extract(self, data, sample_period=1):

        # sample period is measured in ms
        sample_period = sample_period*8

        nperseg = self.periodsperseg*sample_period 

        if self.nfft < nperseg:
            print "nfft too small (%i vs %i), " % (self.nfft, nperseg),
            self.nfft = int(2**np.ceil(np.log2(nperseg)))
            self.nfft = int(nperseg)
            print "rounded up to %i" % self.nfft

        sound_wave = data['sound_wave'].flatten()

        #f, t, spectrum = signal.stft(sound_wave, 
        #                                    fs=self.fs,
        #                                    window=self.window, 
        #                                    nperseg=self.periodsperseg*sample_period,
        #                                    noverlap=(self.periodsperseg-1)*sample_period,
        #                                    nfft=self.nfft)
        #
        #spectrum = spectrum[:, :, :-1] # trim off last element
        #spectrum = spectrum.reshape(spectrum.shape[1], spectrum.shape[2]) # remove first dim
        #print spectrum.shape

        print "nperseg: " , self.periodsperseg*sample_period

        spectrum, energy = MFCC(sound_wave, 
                                     ncoeffs=self.ncoeffs,
                                     nfilters=self.nfilters,
                                     sample_freq=self.fs,
                                     window=self.window, 
                                     nperseg=nperseg,
                                     noverlap=(self.periodsperseg-1)*sample_period,
                                     nfft=self.nfft)
        # transpose so that spectrum.shape[1]==time dimension
        #               and spectrum.shape[0]==feature dimension
        spectrum=spectrum.T

        # start with articulator inputs and down sample
        _data = data[self.control_action]
        _data = moving_average(_data, n=sample_period)
        _data = _data[:, ::sample_period]

        start = 0
        self.pointer[self.control_action] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        # only the first 
        _data = np.append(_data, spectrum[:, :_data.shape[1]], axis=0) 
        self.pointer['spectrum'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        return _data
    

