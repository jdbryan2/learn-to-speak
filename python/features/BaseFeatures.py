import numpy as np

def moving_average(a, n=3):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = (ret[:, n:] - ret[:, :-n])
    return ret[:, n - 1:]/n

class BaseFeatures(object):

    def __init__(self, **kwargs):
        self.DefaultParams()
        self.InitializeParams(**kwargs)

    def DefaultParams(self):
        self.tubes = {}
        self.pointer = {}
        self.control_action = 'art_hist'
        self.sample_period = 8

    def InitializeParams(self, **kwargs):
        # Pass tube segments dictionary through 
        self.tubes = kwargs.get("tubes", self.tubes)
        self.control_action = kwargs.get("control_action", self.control_action)
        self.pointer = kwargs.get("pointer", self.pointer)
        self.sample_period = kwargs.get("sample_period", self.sample_period)

    def Extract(self, data, **kwargs):
        print "BaseFeatures.Extract cannot be called directly."
        return 0

    






