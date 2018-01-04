import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import pylab as plt

from features.BaseFeatures import BaseFeatures
from features.BaseFeatures import moving_average


class ArtFeatures(BaseFeatures):

    def __init__(self, **kwargs):
        super(ArtFeatures, self).__init__(**kwargs)

    def DefaultParams(self):
        super(ArtFeatures, self).DefaultParams()

    def InitializeParams(self, **kwargs):
        super(ArtFeatures, self).InitializeParams(**kwargs)

    def Extract(self, data, **kwargs):
        # default sample period is 8samples == 1ms
        self.InitializeParams(**kwargs)

        # self.control_action is defaulted to "art_hist"
        # can be changed directly or by passing a new string into InitializeParams

        area_function = data['area_function'][self.tubes['glottis_to_velum'], :]
        lung_pressure = np.mean(data['pressure_function'][self.tubes['lungs'], :], axis=0)
        nose_pressure = np.mean(data['pressure_function'][self.tubes['nose'], :], axis=0)

        start = 0
        _data = data[self.control_action]
        self.pointer[self.control_action] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, area_function, axis=0)
        self.pointer['area_function'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, lung_pressure.reshape((1, -1)), axis=0)
        self.pointer['lung_pressure'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]
        
        self.pointer['all'] = np.arange(0, _data.shape[0])
        self.pointer['all_out'] = np.arange(self.pointer[self.control_action][-1], _data.shape[0])

        # compute average value for each sample to match what controller does 
        # (averages samples over control sample period)
        _data = moving_average(_data, n=self.sample_period)
        _data = _data[:, ::self.sample_period]

        return _data

    def DirectExtract(self, articulation, full_area_function, pressure_function):
        # I don't like the code repetition here

        area_function = full_area_function[self.tubes['glottis_to_velum']]
        lung_pressure = np.mean(pressure_function[self.tubes['lungs']])
        nose_pressure = np.mean(pressure_function[self.tubes['nose']])

        start = 0
        _data = articulation
        self.pointer[self.control_action] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, area_function)
        self.pointer['area_function'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, lung_pressure.reshape((1, -1)))
        self.pointer['lung_pressure'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]
        
        self.pointer['all'] = np.arange(0, _data.shape[0])
        self.pointer['all_out'] = np.arange(self.pointer[self.control_action][-1], _data.shape[0])

        # compute average value for each sample to match what controller does 
        # (averages samples over control sample period)
        #_data = np.mean(_data, axis=1)

        return _data
    

