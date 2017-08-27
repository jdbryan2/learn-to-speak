import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw

import pylab as plt

# TODO: Add this to constants in library
MAX_NUMBER_OF_TUBES = 89


class Utterance:

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

        ## brownian default params
        #self.increment = 0.01
        #self.sigma = 0.1
        #self.total_increments = self.loops * \
        #    self.utterance_length / \
        #    self.increment + 1

        self.initial_art = np.zeros(aw.kArt_muscle.MAX,
                                    dtype=np.dtype('double'))


        self.dir_name = "utterance"

        self._art_init = False  # flag for whether self.InitializeArticulation has been called

        self.InitializeParams(**kwargs)

    def InitializeParams(self, **kwargs):
        self.dir_name = kwargs.get("dir_name", self.dir_name)

        self.gender = kwargs.get("gender", self.gender)
        self.sample_freq = kwargs.get("sample_freq", self.sample_freq)
        self.oversamp = kwargs.get("oversamp", self.oversamp)
        self.glottal_masses = kwargs.get("glottal_masses", self.glottal_masses)
        self.utterance_length = kwargs.get("utterance_length",
                                           self.utterance_length)
        self.loops = kwargs.get("loops", self.loops)


        self.ResetOutputVars()

    def ResetOutputVars(self):
        self.sound_wave = np.zeros(int(np.ceil(self.sample_freq *
                                           self.utterance_length)))

        self.area_function = np.zeros((MAX_NUMBER_OF_TUBES,
                                       int(np.ceil(self.sample_freq *
                                               self.utterance_length))))

        self.art_hist = np.zeros((aw.kArt_muscle.MAX,
                                  int(np.ceil(self.sample_freq *
                                          self.utterance_length))))

    def InitializeDir(self, dirname, addDTS=True):
        # setup directory for saving files
        if addDTS:
            self.directory = dirname + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.directory = dirname

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

        initial_art = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
        if self._art_init:
            self.articulation.intoArt(initial_art, self.speaker.NowSecondsLooped())

        self.speaker.InitSim(self.utterance_length, initial_art)

    def InitializeArticulation(self):
        # note: changing speaker params requires calling InitializeSpeaker
        if self._art_init == False:

            # Initialize artword for driving a manual sequence
            total_length = self.utterance_length*self.loops
            self.articulation = aw.Artword(total_length)

            self._art_init = True  # flag for whether self.InitializeArticulation has been called

    def SetManualSequence(self, muscle, times, targets):

        self.InitializeArticulation()

        if muscle > aw.kArt_muscle.MAX or muscle < 0:
            print "Invalid muscle ID: " + str(muscle)
            return 0
        if len(targets) != len(times):
            print "Invalid targets/times for muscle ID: " + str(muscle)


        for t in range(len(targets)):
            self.articulation.setTarget(muscle, times[t], targets[t])
    

    def Simulate(self):

        self.ResetOutputVars()

        articulation = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))

        while self.speaker.NotDone():

            # pass the current articulation in
            self.articulation.intoArt(articulation, self.speaker.NowSecondsLooped())
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

    def SaveParams(self):
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 loops=self.loops)

    def Run(self, **kwargs):
        # initialize parameters if anything new is passed in
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        #self.InitializeDir(self.method)  # appends DTS to folder name
        self.InitializeDir(self.dir_name)  # appends DTS to folder name
        self.SaveParams()  # save parameters before anything else
        self.InitializeSpeaker()
        self.InitializeSim()
        if self._art_init == False:
            print "No articulations to simulate."
            return False

        for k in range(self.loops):
            print "Loop: " + str(k)
            self.Simulate()
            self.Save()


if __name__ == "__main__":
    
    # Default initial_art is all zeros
    apa = Utterance(dir_name="apa",
                    loops=1,
                    utterance_length=0.5)

    apa.SetManualSequence(aw.kArt_muscle.INTERARYTENOID, [0,   0.5], # time 
                                                         [0.5, 0.5]) # target

    apa.SetManualSequence(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                           [1.0, 1.0])

    apa.SetManualSequence(aw.kArt_muscle.LUNGS, [0.0, 0.2], 
                                                [0.1, 0.0])

    apa.SetManualSequence(aw.kArt_muscle.MASSETER, [0.25], [0.7])

    apa.SetManualSequence(aw.kArt_muscle.ORBICULARIS_ORIS, [0.25], [0.2])

    apa.Run()

    ############################################################################
    sigh = Utterance(dir_name="sigh",
                    loops=1,
                    utterance_length=0.5)

    sigh.SetManualSequence(aw.kArt_muscle.LUNGS, [0.0, 0.1], 
                                                [0.1, 0.0])


    sigh.SetManualSequence(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                           [1.0, 1.0])


    sigh.Run()
    ############################################################################
    ejective = Utterance(dir_name="ejective",
                    loops=1,
                    utterance_length=0.5)

    ejective.SetManualSequence(aw.kArt_muscle.LUNGS, [0.0, 0.1], 
                                                     [0.1, 0.0])


    ejective.SetManualSequence(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                                [1.0, 1.0])

    ejective.SetManualSequence(aw.kArt_muscle.INTERARYTENOID, [0.0, 0.17, 0.2, 0.35, 0.38, 0.5], 
                                                              [0.5, 0.5 , 1.0, 1.0 , 1.0 , 0.5])

    ejective.SetManualSequence(aw.kArt_muscle.MASSETER, [0.0, 0.5], 
                                                        [-.3, -.3])

    ejective.SetManualSequence(aw.kArt_muscle.HYOGLOSSUS, [0.0, 0.5], 
                                                          [0.5, 0.5])

    ejective.SetManualSequence(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.1, 0.15, 0.29, 0.32], 
                                                            [0.0, 0.0, 1.0 , 1.0 , 0.0 ])

    ejective.SetManualSequence(aw.kArt_muscle.STYLOHYOID, [0.0, 0.22, 0.27, 0.35, 0.38, 0.5], 
                                                          [0.0,  0.0, 1.0 , 1.0 , 0.0 , 0.0])

    ejective.Run()
    ############################################################################

    click = Utterance(dir_name="click",
                    loops=1,
                    utterance_length=0.5)

    click.SetManualSequence(aw.kArt_muscle.MASSETER, [0.0 , 0.2 ,  0.3 ,  0.5 ], 
                                                     [0.25, 0.25, -0.25, -0.25])

    click.SetManualSequence(aw.kArt_muscle.ORBICULARIS_ORIS, [0.0 , 0.2 , 0.3, 0.5], 
                                                             [0.75, 0.75, 0.0, 0.0])

    click.SetManualSequence(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.5], 
                                                         [0.9, 0.9])
    click.Run()
