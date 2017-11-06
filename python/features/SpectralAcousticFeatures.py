import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import pylab as plt

from features.BaseFeatures import BaseFeatures

def moving_average(a, n=3):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = (ret[:, n:] - ret[:, :-n])
    return ret[:, n - 1:]/n

class SpectralAcousticFeatures(BaseFeatures):

    def __init__(self, **kwargs):
        super(SpectralAcousticFeatures, self).__init__(**kwargs)

    def DefaultParams(self):
        super(SpectralAcousticFeatures, self).DefaultParams()
        self.fs = 8000
        self.window = 'hamming'
        self.nfft = None
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

        sound_wave = data['sound_wave']
        f, t, spectrum = signal.stft(sound_wave, 
                                            fs=self.fs,
                                            window=self.window, 
                                            nperseg=self.periodsperseg*sample_period,
                                            noverlap=(self.periodsperseg-1)*sample_period,
                                            nfft=self.nfft)

        
        spectrum = spectrum[:, :, :-1] # trim off last element
        spectrum = spectrum.reshape(spectrum.shape[1], spectrum.shape[2]) # remove first dim
        


        # start with articulator inputs and down sample
        # TODO: figure out how to generalize this as generic control inputs
        _data = data['art_hist']
        _data = moving_average(_data, n=sample_period)
        _data = _data[:, ::sample_period]

        start = 0
        self.pointer['art_hist'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, np.abs(spectrum), axis=0)
        self.pointer['spectrum'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        return _data
    

