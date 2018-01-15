import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
from Utterance import Utterance

import pylab as plt

# TODO: Add this to constants in library
MAX_NUMBER_OF_TUBES = 89


class RandExcite(Utterance):

    home_dir = 'data'  # changes to this changes all instances of the class

    def __init__(self, **kwargs):
        self.DefaultParams()
        self.InitializeParams(**kwargs)

    def DefaultParams(self):
        # load parent defaults first
        Utterance.DefaultParams(self)

        # change default dirname
        self.dirname="random" # default directory name ../data/random_<current dts>

        # gesture default params
        self.method = "gesture"
        self.max_increment = 0.1
        self.min_increment = 0.01
        self.max_delta_target = 1.0
        self.total_increments = self.loops * \
            self.utterance_length / \
            self.min_increment + 1

        self.initial_art = np.zeros(aw.kArt_muscle.MAX,
                                    dtype=np.dtype('double'))

        self.manual_targets = {}
        self.manual_times = {}

    def InitializeParams(self, **kwargs):
    
        # call parent method first
        Utterance.InitializeParams(self, **kwargs)

        # initialize random excitation params
        if self.method == "gesture":
            print "Gesture exploration method initializing."
            self.max_increment = kwargs.get("max_increment", 0.1)  # sec
            self.min_increment = kwargs.get("min_increment", 0.01)  # sec
            self.max_delta_target = kwargs.get("max_delta_target", 0.5)  
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.min_increment + 1

        else:
            print "Unknown method type: %s" % self.method
            print "Gesture exploration method initializing."
            self.method = "gesture"
            self.max_increment = kwargs.get("max_increment", 0.1)  # sec
            self.min_increment = kwargs.get("min_increment", 0.01)  # sec
            self.max_delta_target = kwargs.get("max_delta_target", 0.5)  # sec
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.min_increment + 1

        # Articulator parameters
        self.initial_art = kwargs.get("initial_art", self.initial_art)

        self.ResetOutputVars()

    def SetManualArticulation(self, muscle, targets, times):
        if muscle > aw.kArt_muscle.MAX or muscle < 0:
            print "Invalid muscle ID: " + str(muscle)
            return 0
        if targets.shape[0] != times.shape[0]:
            print "Invalid targets/times for muscle ID: " + str(muscle)

        if muscle in self.manual_targets:
            self.manual_targets[muscle] = np.append(self.manual_targets[muscle],
                                                    targets)
        else:
            self.manual_targets[muscle] = targets

        if muscle in self.manual_times:
            self.manual_times[muscle] = np.append(self.manual_times[muscle],
                                                  times)
        else:
            self.manual_times[muscle] = times

    def GenerateGestureSequence(self, **kwargs):
        # if len(kwargs.keys()):
        #     self.InitializeParams(**kwargs)
        ########################################################################
        # Generate random target sequences
        ########################################################################
        # Should simply create an artword class in python

        total_length = self.utterance_length*self.loops

        self.randart = aw.Artword(total_length)

        for k in range(aw.kArt_muscle.MAX):
            # check if manually defined
            if k in self.manual_targets:
                for t in range(len(self.manual_targets[k])):
                    self.randart.setTarget(k,
                                           self.manual_times[k][t],
                                           self.manual_targets[k][t])
                time_hist = self.manual_times[k][:]
                target_hist = self.manual_targets[k][:]

            # random gestures if not
            else:
                # generate random sequence!
                time = 0.0
                target = self.initial_art[k]  # np.random.random()
                self.randart.setTarget(k, time, target)
                
                time_hist = np.array([time])
                target_hist = np.array([target])
                while True:
                    delta = (np.random.random()-0.5)*self.max_delta_target
                    increment = np.random.random() * \
                        (self.max_increment-self.min_increment) + \
                        self.min_increment

                    # if we're at the boundary, force delta to drive inward
                    if target == 1.0:
                        delta = -1.0 * abs(delta)
                    elif target == 0.0:
                        delta = abs(delta)

                    # if delta pushed past boundaries, interpolate to the
                    # boundary and place target there
                    if target + delta > 1.0:
                        increment = (1.0-target) * increment / delta
                        delta = 1.0-target
                    elif target + delta < 0.0:
                        increment = (0.0-target) * increment / delta
                        delta = 0.0-target

                    # set target if it's still in the utterance duration
                    if time+increment < total_length:
                        time = time+increment
                        target = target+delta
                        self.randart.setTarget(k, time, target)
                        time_hist = np.append(time_hist, time)
                        target_hist = np.append(target_hist, target)

                    # interp between current and next target at end of utterance
                    else:
                        self.randart.setTarget(
                                k,
                                total_length,
                                np.interp(total_length,
                                          [time, time+increment],
                                          [target, target+delta]))

                        time_hist = np.append(time_hist, total_length)
                        target_hist = np.append(target_hist,
                                                np.interp(
                                                    total_length,
                                                    [time, time+increment],
                                                    [target, target+delta]))

                        break  # we've already hit the end of the utterance
            plt.plot(time_hist, target_hist)
            #plt.show()
            #plt.hold(True)
        plt.show()

    def SaveParams(self):
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 method=self.method,
                 loops=self.loops,
                 max_delta_target=self.max_delta_target)

    def Run(self, **kwargs):
        # initialize parameters if anything new is passed in
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        self.InitializeDir(self.dirname)  # appends DTS to folder name
        self.SaveParams()  # save parameters before anything else
        self.InitializeSpeaker()
        self.InitializeSim()
        self.InitializeArticulation()
        if self.method == "gesture":
            self.GenerateGestureSequence()
        else:
            print "Excitation method is undefined: " + self.method
            return False

        for k in range(self.loops):
            print "Loop: " + str(k)
            self.Simulate()
            self.Save()


if __name__ == "__main__":
    
    loops = 15
    utterance_length = 1.0
    full_utterance = loops*utterance_length

    rando = RandExcite(dirname="full_random_15", 
                    method="gesture",
                    loops=loops,
                    utterance_length=utterance_length,
                    initial_art=np.random.random((aw.kArt_muscle.MAX, )))

    # manually pump the lungs
    #rando.SetManualSequence(aw.kArt_muscle.LUNGS,
    #                        np.array([0.2, 0.0]),  # targets
    #                        np.array([0.0, 0.5]))  # times


    # manually open the jaw
    jaw_period = 0.5
    jaw_period_var = 0.2
    jaw_times = np.cumsum(np.random.rand(int(full_utterance/(jaw_period-jaw_period_var))))
    jaw_times = jaw_times[jaw_times<full_utterance]
    jaw_targets = np.random.rand(jaw_times.size).reshape((-1, 2))*0.5
    jaw_targets[:, 1] += 0.5
    jaw_targets = jaw_targets.flatten()

    plt.plot(jaw_times)
    plt.show()

    exit()

    rando.SetManualSequence(aw.kArt_muscle.LUNGS,
                            np.array([0.2, 0.0]),  # targets
                            np.array([0.0, 0.5]))  # times


    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.5)
    # rando.Run(max_increment=0.5, min_increment=0.05, max_delta_target=0.5,
    #           initial_art=np.zeros(aw.kArt_muscle.MAX))
    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.3)
    rando.Run(max_increment=0.3, min_increment=0.01, max_delta_target=0.2,
              initial_art=np.random.random((aw.kArt_muscle.MAX, )))
