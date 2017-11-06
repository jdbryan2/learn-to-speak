import numpy as np

class BaseFeatures(object):

    def __init__(self, **kwargs):
        self.DefaultParams()
        self.InitializeParams(**kwargs)

    def DefaultParams(self):
        self.tubes = {}
        self.pointer = {}

    def InitializeParams(self, **kwargs):
        # Pass tube segments dictionary through 
        self.tubes = kwargs.get("tubes", self.tubes)

    def Extract(self, data):
        print "BaseFeatures.Extract cannot be called directly."
        return 0

    






