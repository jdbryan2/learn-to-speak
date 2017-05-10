import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw

import pylab as plt

# TODO: Add this to constants in library
MAX_NUMBER_OF_TUBES = 89


class RandExp:

    home_dir = 'data'  # changes to this changes all instances of the class

    def __init__(self, **kwargs):
        self.method = "gesture"
        self.gender = "Female"
        self.sample_freq = 8000
        self.oversamp = 70
        self.glottal_masses = 2
        self.utterance_length = 1.0  # seconds
        self.loops = 10

        # gesture default params
        self.max_increment = 0.1
        self.min_increment = 0.01
        self.max_delta_target = 1.0
        self.total_increments = self.loops * \
            self.utterance_length / \
            self.min_increment + 1

        # brownian default params
        self.increment = 0.01
        self.sigma = 0.1
        self.total_increments = self.loops * \
            self.utterance_length / \
            self.increment + 1

        self.initial_art = np.zeros(aw.kArt_muscle.MAX,
                                    dtype=np.dtype('double'))

        self.manual_targets = {}
        self.manual_times = {}

        self.InitializeParams(**kwargs)

    def InitializeParams(self, **kwargs):
        self.method = kwargs.get("method", self.method)
        self.gender = kwargs.get("gender", self.gender)
        self.sample_freq = kwargs.get("sample_freq", self.sample_freq)
        self.oversamp = kwargs.get("oversamp", self.oversamp)
        self.glottal_masses = kwargs.get("glottal_masses", self.glottal_masses)
        self.utterance_length = kwargs.get("utterance_length",
                                           self.utterance_length)
        self.loops = kwargs.get("loops", self.loops)

        if self.method == "gesture":
            print "Gesture exploration method initializing."
            self.max_increment = kwargs.get("max_increment", 0.1)  # sec
            self.min_increment = kwargs.get("min_increment", 0.01)  # sec
            self.max_delta_target = kwargs.get("max_delta_target", 1.0)  # sec
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.min_increment + 1

        elif self.method == "brownian":
            print "Brownian exploration method initializing."
            self.increment = kwargs.get("increment", self.increment)
            self.sigma = kwargs.get("sigma", self.sigma)
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.increment + 1

        else:
            print "Unknown method type: %s" % self.method
            print "Gesture exploration method initializing."
            self.method = "gesture"
            self.max_increment = kwargs.get("max_increment", 0.2)  # sec
            self.min_increment = kwargs.get("min_increment", 0.05)  # sec
            self.max_delta_target = kwargs.get("max_delta_target", 0.5)  # sec
            self.total_increments = self.loops * \
                self.utterance_length / \
                self.min_increment + 1

        # Articulator parameters
        self.initial_art = kwargs.get("initial_art", self.initial_art)

        self.ResetOutputVars()

    def ResetOutputVars(self):
        self.sound_wave = np.zeros(np.ceil(self.sample_freq *
                                           self.utterance_length))

        self.area_function = np.zeros((MAX_NUMBER_OF_TUBES,
                                       np.ceil(self.sample_freq *
                                               self.utterance_length)))

        self.art_hist = np.zeros((aw.kArt_muscle.MAX,
                                  np.ceil(self.sample_freq *
                                          self.utterance_length)))

    def InitializeDir(self, dirname):
        # setup directory for saving files
        self.directory = dirname + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')

        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)

        self.directory = self.home_dir + '/' + self.directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/'

    def InitializeSpeaker(self, **kwargs):
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        self.speaker = vt.Speaker(self.gender,
                                  self.glottal_masses,
                                  self.sample_freq,
                                  self.oversamp)
        self.iteration = 0  # reset counter

    def InitializeSim(self, **kwargs):
        # note: changing speaker params requires calling InitializeSpeaker
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        self.speaker.InitSim(self.utterance_length, self.initial_art)

    def SetManualSequence(self, muscle, targets, times):
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
                # generate manual sequence!
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
            plt.hold(True)
        plt.show()

    def Simulate(self):

        self.ResetOutputVars()

        articulation = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))

        while self.speaker.NotDone():

            # pass the current articulation in
            self.randart.intoArt(articulation, self.speaker.NowSecondsLooped())
            self.speaker.SetArticulation(articulation)

            self.speaker.IterateSim()

            # Save sound data point
            self.sound_wave[self.speaker.Now()-1] = self.speaker.GetLastSample()

            self.speaker.GetAreaFcn(self.area_function[:, self.speaker.Now()-1])

            self.art_hist[:, self.speaker.Now()-1] = articulation

        self.speaker.LoopBack()
        self.iteration += 1

    def Save(self):

        scaled = np.int16(self.sound_wave/np.max(np.abs(self.sound_wave))*32767)
        write(self.directory + 'audio' + str(self.iteration) + '.wav',
              self.sample_freq,
              scaled)

        np.savez(self.directory + 'data' + str(self.iteration),
                 sound_wave=self.sound_wave,
                 area_function=self.area_function,
                 art_hist=self.art_hist)

    def SaveGestureParams(self):
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 method=self.method,
                 loops=self.loops,
                 initial_art=self.initial_art,
                 max_increment=self.max_increment,
                 min_increment=self.min_increment,
                 max_delta_target=self.max_delta_target)

    def SaveBrownianParams(self):
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 method=self.method,
                 loops=self.loops,
                 initial_art=self.initial_art,
                 increment=self.increment,
                 sigma=self.sigma)

    def Run(self, **kwargs):
        # initialize parameters if anything new is passed in
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        self.InitializeDir(self.method)  # appends DTS to folder name
        self.SaveGestureParams()  # save parameters before anything else
        self.InitializeSpeaker()
        self.InitializeSim()
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
    rando = RandExp(method="gesture",
                    loops=10,
                    initial_art=np.random.random((aw.kArt_muscle.MAX, )))

    # manually pump the lungs
    rando.SetManualSequence(aw.kArt_muscle.LUNGS,
                            np.array([0.4, 0.0]),  # targets
                            np.array([0.0, 0.5]))  # times

    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.5)
    # rando.Run(max_increment=0.5, min_increment=0.05, max_delta_target=0.5,
    #           initial_art=np.zeros(aw.kArt_muscle.MAX))
    # rando.Run(max_increment=0.5, min_increment=0.1, max_delta_target=0.3)
    rando.Run(max_increment=0.3, min_increment=0.05, max_delta_target=0.3)
