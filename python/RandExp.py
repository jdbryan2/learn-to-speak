import os
import time
# import numpy as np
# from scipy.io.wavfile import write
# import PyRAAT as vt
# import Artword as aw

# import pylab as plt


class RandExp:

    home_dir = 'data'

    def __init__(self, **kwargs):
        method = kwargs.get("method", "gesture")
        self.gender = kwargs.get("gender", "Female")
        self.sample_freq = kwargs.get("sample_freq", 8000)
        self.oversamp = kwargs.get("oversamp", 70)
        self.glottal_masses = kwargs.get("glottal_masses", 2)
        self.utterance_length = 1.0  # seconds
        self.loops = 10

        if method == "gesture":
            print "Gesture exploration method initializing."
            self.method = method
            self.max_increment = kwargs.get("max_increment", 0.1)  # sec
            self.min_increment = kwargs.get("min_increment", 0.01)  # sec
            self.max_delta_target = kwargs.get("max_delta_target", 1.0)  # sec
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.min_increment + 1

        elif method == "brownian":
            print "Brownian exploration method initializing."
            self.method = method

        else:
            print "Unknown method type: %s" % method
            print "Gesture exploration method initializing."
            self.method = "gesture"

        # setup directory for saving files
        self.directory = self.method + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')

        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)

        self.directory = self.home_dir + '/' + self.directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/'

if __name__ == "__main__":
    rando = RandExp(method="brownian")
