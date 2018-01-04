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

    def ExtractLast(self, data, **kwargs):

        self.InitializeParams(**kwargs)

        # default start of control action to copy of first element if not
        # enough data provided
        for key in data:
            if key != 'sound_wave': # ignore sound_wave

                data_length = data[key].shape[1]
                if data_length < self.sample_period:
                    key_data = np.copy(data[key]) # create new instance
                    default = np.zeros((key_data.shape[0],
                                       self.sample_period-data_length))
                    default = (default.T+key_data[:, 0].flatten()).T # fast copy over cols
                    key_data = np.append(default, key_data, axis=1)

                    data[key] = key_data

                elif data_length > self.sample_period:
                    data[key] = data[key][:, -self.sample_period:]

        # get extracted data and flatten that shit
        return self.Extract(data).flatten()
    

