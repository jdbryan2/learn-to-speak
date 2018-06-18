import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import pylab as plt

from features.BaseFeatures import BaseFeatures
from features.BaseFeatures import moving_average

from features.functions import MFCC

class SpectralAcousticFeatures(BaseFeatures):

    def __init__(self, **kwargs):
        # does this actually call the functions at this class level or the
        # parent class level? 
        # Yes! I just tested by printing a variable defined at child level
        super(SpectralAcousticFeatures, self).__init__(**kwargs)

    def DefaultParams(self):
        super(SpectralAcousticFeatures, self).DefaultParams()
        self.fs = 8000
        self.window = 'hamming'
        self.nfft = 512
        self.ncoeffs = 13 # number of MFCC features to return
        self.nfilters = 26 # number of mel spaced filters
        self.periodsperseg = 5 # measured in sample periods
        self.min_data_length = self.sample_period*self.periodsperseg
        self.control_sample_period = 8

    def InitializeParams(self, **kwargs):
        super(SpectralAcousticFeatures, self).InitializeParams(**kwargs)

        self.fs = kwargs.get('fs', self.fs)
        self.window = kwargs.get('window', self.window)
        self.nfft = kwargs.get('nfft', self.nfft)
        self.periodsperseg = kwargs.get('periodsperseg', self.periodsperseg)
        self.min_sound_length = self.sample_period*(self.periodsperseg-1)
        self.control_sample_period = kwargs.get('control_sample_period', self.control_sample_period)

    # TODO: Add function for checking that all necessary items are in data dict

    def Extract(self, data, **kwargs):

        # sample period is measured in ms
        self.InitializeParams(**kwargs)

        nperseg = self.periodsperseg*self.sample_period 

        if self.nfft < nperseg:
            print "nfft too small (%i vs %i), " % (self.nfft, nperseg),
            #self.nfft = int(2**np.ceil(np.log2(nperseg)))
            self.nfft = int(nperseg)
            print "rounded up to %i" % self.nfft

        sound_wave = data['sound_wave'].flatten()

        #print "nperseg: " , self.periodsperseg*sample_period

        spectrum, energy = MFCC(sound_wave, 
                                ncoeffs=self.ncoeffs,
                                nfilters=self.nfilters,
                                sample_freq=self.fs,
                                window=self.window, 
                                nperseg=nperseg,
                                noverlap=(self.periodsperseg-1)*self.sample_period,
                                nfft=self.nfft)
        # transpose so that spectrum.shape[1]==time dimension
        #               and spectrum.shape[0]==feature dimension
        spectrum=spectrum.T

        # start with articulator inputs and down sample
        _data = data[self.control_action]
        if self.sample_period%self.control_sample_period:
            print "Sample period must be an integer multiple of controller's sample period"
        _data = moving_average(_data, n=int(self.sample_period/self.control_sample_period))
        _data = _data[:, ::int(self.sample_period/self.control_sample_period)]

        start = 0
        self.pointer[self.control_action] = np.arange(start, _data.shape[0])
        start=_data.shape[0]
        print _data.shape, spectrum.shape, sound_wave.shape, nperseg, self.sample_period

        # only the first 
        _data = np.append(_data, spectrum[:, :_data.shape[1]], axis=0) 
        #_data = np.append(_data, spectrum, axis=0) 
        self.pointer['spectrum'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        return _data
    
    def ExtractLast(self, data, **kwargs):

        self.InitializeParams(**kwargs)

        nperseg = self.periodsperseg*self.sample_period 

        # default start of sound wave to zero if not enough data provided
        # otherwise grab last data points off the array
        sound_length = len(data['sound_wave'])
        if sound_length < nperseg:
            sound_wave = data['sound_wave'].flatten() # creates new instance
            sound_wave = np.append(np.zeros(nperseg-sound_length), sound_wave)

            data['sound_wave'] = sound_wave
            # note: dictionary only saves a pointer to the array, this does not
            # actually overwrite any data, it just changes the pointer to the
            # augmented sound_wave variable (which is a new instance due to flatten)

        elif sound_length > nperseg:
            data['sound_wave'] = data['sound_wave'][-nperseg:]

        # default start of control action to copy of first element if not
        # enough data provided
        action_length = data[self.control_action].shape[1]
        if action_length < self.sample_period:
            action = np.copy(data[self.control_action]) # create new instance
            default = np.zeros((action.shape[0],
                               self.sample_period-action_length))
            default = (default.T+action[:, 0].flatten()).T # fast copy over cols
            action = np.append(default, action, axis=1)

            data[self.control_action] = action

        elif action_length > self.sample_period:
            data[self.control_action] = data[self.control_action][:, -self.sample_period:]

        # get extracted data and flatten that shit
        return self.Extract(data).flatten()
    

