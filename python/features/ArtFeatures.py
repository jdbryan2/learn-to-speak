import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import pylab as plt

from features.BaseFeatures import BaseFeatures

def moving_average(a, n=3):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = (ret[:, n:] - ret[:, :-n])
    return ret[:, n - 1:]/n

class ArtFeatures(BaseFeatures):

    def __init__(self, **kwargs):
        super(ArtFeatures, self).__init__(**kwargs)

    def DefaultParams(self):
        super(ArtFeatures, self).DefaultParams()

    def InitializeParams(self, **kwargs):
        super(ArtFeatures, self).InitializeParams(**kwargs)

    def Extract(self, data, sample_period=1):

        area_function = data['area_function'][self.tubes['glottis_to_velum'], :]
        lung_pressure = np.mean(data['pressure_function'][self.tubes['lungs'], :], axis=0)
        nose_pressure = np.mean(data['pressure_function'][self.tubes['nose'], :], axis=0)

        start = 0
        _data = data['art_hist']
        self.pointer['art_hist'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, area_function, axis=0)
        self.pointer['area_function'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, lung_pressure.reshape((1, -1)), axis=0)
        self.pointer['lung_pressure'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]
        
        self.pointer['all'] = np.arange(0, _data.shape[0])
        self.pointer['all_out'] = np.arange(self.pointer['art_hist'][-1], _data.shape[0])

        _data = moving_average(_data, n=8*sample_period)
        _data = _data[:, ::8*sample_period]

        ## decimate to 1ms sampling period
        #_data = signal.decimate(_data, 8, axis=1, zero_phase=True) 

        ## decimate to 5ms sampling period
        #if sample_period > 1:
        #    _data = signal.decimate(_data, sample_period, axis=1, zero_phase=True)

        return _data
    

