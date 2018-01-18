#from config import *
import config
import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
import RandomArtword as raw
from Utterance import Utterance

import pylab as plt

# TODO: Add this to constants in library
#MAX_NUMBER_OF_TUBES = 89


class RandExcite(Utterance):

    def __init__(self, **kwargs):
        self.DefaultParams()
        #self.InitializeParams(**kwargs)
        self.InitializeAll(**kwargs)

    def DefaultParams(self):
        # load parent defaults first
        Utterance.DefaultParams(self)

        # change default dirname
        # DTS is automatically tacked on to file name
        # change by passing addDTS=False on initialization
        self.directory="data/random" # default directory name data/random_<current dts>

        # gesture default params
        self.method = "gesture"

        self.initial_art = np.zeros(aw.kArt_muscle.MAX,
                                    dtype=np.dtype('double'))

        self.manual_targets = {}
        self.manual_times = {}

        # flag for seeing if simulation has already been initialized
        self._sim_init = False

    def InitializeParams(self, **kwargs):
    
        # call parent method first
        Utterance.InitializeParams(self, **kwargs)

        # initialize random excitation params
        if self.method == "gesture":
            self._art = raw.RandomArtword(**kwargs)
            self._art_init = True


        else:
            print "Unknown method type: %s" % self.method
            print "Gesture exploration method initializing."
            self._art = raw.RandomArtword(**kwargs)
            self._art_init = True
            self.method = "gesture"

        # Articulator parameters
        #self.initial_art = kwargs.get("initial_art", self.initial_art)

        self.ResetOutputVars()

    def SetManualArticulation(self, muscle, targets, times):
        if muscle > aw.kArt_muscle.MAX or muscle < 0:
            print "Invalid muscle ID: " + str(muscle)
            return 0
        if targets.shape[0] != times.shape[0]:
            print "Invalid targets/times for muscle ID: " + str(muscle)

        for k in range(times.shape[0]):
            self._art.SetManualTarget(muscle, targets[k], times[k])


    def SaveParams(self):
        # TODO: 
        #   find better way to just add more parameters to what is already saved in parent class


        # TODO: 
        #   Add remaining parameters for RandomArtword
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 method=self.method,
                 loops=self.loops,
                 max_delta_target=self._art.max_delta_target)

    #def Run(self, **kwargs):
    #    self.InitializeAll(**kwargs)

    #    for k in range(self.loops):
    #        print "Loop: " + str(k)
    #        self.Simulate()
    #        self.Save()


if __name__ == "__main__":
    
    loops = 5 
    utterance_length = 1.0
    full_utterance = loops*utterance_length

    rando = RandExcite(dirname="full_random_5", 
                    method="gesture",
                    loops=loops,
                    utterance_length=utterance_length,
                    initial_art=np.random.random((aw.kArt_muscle.MAX, )))

    # manually pump the lungs
    #rando.SetManualSequence(aw.kArt_muscle.LUNGS,
    #                        np.array([0.2, 0.0]),  # targets
    #                        np.array([0.0, 0.5]))  # times


    # manually open the jaw
    ###jaw_period = 0.5
    ###jaw_period_var = 0.2
    ###jaw_times = np.cumsum(np.random.rand(int(full_utterance/(jaw_period-jaw_period_var))))
    ###jaw_times = jaw_times[jaw_times<full_utterance]
    ###jaw_targets = np.random.rand(jaw_times.size).reshape((-1, 2))*0.5
    ###jaw_targets[:, 1] += 0.5
    ###jaw_targets = jaw_targets.flatten()

    ###plt.plot(jaw_times)
    ###plt.show()

    ###exit()

    ###rando.SetManualSequence(aw.kArt_muscle.LUNGS,
    ###                        np.array([0.2, 0.0]),  # targets
    ###                        np.array([0.0, 0.5]))  # times


    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.5)
    # rando.Run(max_increment=0.5, min_increment=0.05, max_delta_target=0.5,
    #           initial_art=np.zeros(aw.kArt_muscle.MAX))
    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.3)
    rando.Run(max_increment=0.3, min_increment=0.01, max_delta_target=0.2,
              initial_art=np.random.random((aw.kArt_muscle.MAX, )))

    for k in range(rando.art_hist.shape[0]):
        plt.plot(rando.art_hist[k])

    plt.show()
