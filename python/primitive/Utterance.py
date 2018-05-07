import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
import RandomArtword as rand_aw

import pylab as plt

# TODO: Add this to constants in library
MAX_NUMBER_OF_TUBES = 89


class Utterance(object):


    def __init__(self, **kwargs):
        self.DefaultParams()
        self.UpdateParams(**kwargs)

    def DefaultParams(self):
        self.directory="data/utterance" # default directory name data/utterance_<current dts>
        self._dir_DTS=True

        self.gender = "Female"
        self.sample_freq = 8000
        self.oversamp = 70
        self.glottal_masses = 2
        self.utterance_length = 1.0  # seconds
        self.loops = 1

        self.initial_art = np.zeros(aw.kArt_muscle.MAX,
                                    dtype=np.dtype('double'))

        self._addDTS = True
        self._art_init = False  # flag for whether self.InitializeArticulation has been called
        
        self._sim_init = False  # flag for seeing if simulation has already been initialized

    def UpdateParams(self, **kwargs):
        self.directory = kwargs.get("dirname", self.directory) # keeps backward compatible
        self.directory = kwargs.get("directory", self.directory)
        self._dir_DTS=kwargs.get("addDTS", self._dir_DTS)

        self.gender = kwargs.get("gender", self.gender)
        self.sample_freq = kwargs.get("sample_freq", self.sample_freq)
        self.oversamp = kwargs.get("oversamp", self.oversamp)
        self.glottal_masses = kwargs.get("glottal_masses", self.glottal_masses)
        self.utterance_length = kwargs.get("utterance_length",
                                           self.utterance_length)
        self.loops = kwargs.get("loops", self.loops)
        self.initial_art = kwargs.get("initial_art", self.initial_art)

        self._addDTS=kwargs.get("addDTS", self._addDTS)
        #print self.initial_art


        self.ResetOutputVars()

    def ResetOutputVars(self):
        # outputs are directly saved as dictionary
        # makes saving much quicker and allows expansion of outputs without adding attributes to class

        self.data = {}
        self.data['sound_wave'] = np.zeros(int(np.ceil(self.sample_freq * self.utterance_length)))

        self.data['area_function'] = np.zeros((MAX_NUMBER_OF_TUBES, 
                                       int(np.ceil(self.sample_freq * self.utterance_length))))

        self.data['pressure_function'] = np.zeros((MAX_NUMBER_OF_TUBES,
                                       int(np.ceil(self.sample_freq * self.utterance_length))))


        self.data['art_hist'] = np.zeros((aw.kArt_muscle.MAX,
                                  int(np.ceil(self.sample_freq *
                                          self.utterance_length))))
        self.data['art_hist'][:, 0] = self.initial_art

    def GetOutputVars(self, time):
        _data = {}
        #print time
        #print self.data['art_hist'][:, 0]
        _data['art_hist'] = self.data['art_hist'][:, :time] 
        _data['area_function'] = self.data['area_function'][:, :time]
        _data['pressure_function'] = self.data['pressure_function'][:, :time]
        _data['sound_wave'] = self.data['sound_wave'][:time]

        return _data



    def InitializeDir(self, directory, addDTS=True):
        # setup directory for saving files
        if self._addDTS:
            self.directory = directory + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.directory = directory

        #if (not os.path.exists(self.home_dir)) and (not self.home_dir == None): 
        #    os.makedirs(self.home_dir)

        #self.directory = self.home_dir + '/' + self.directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/'

    def InitializeSim(self, **kwargs):
        # note: changing speaker params requires calling InitializeSpeaker
        if len(kwargs.keys()):
            self.UpdateParams(**kwargs)

        if not self._art_init:
            print "####################\n"*3
            print "No articulator has been initialized."
            print "Simulator cannot run without articulator."
            print "####################\n"*3

        # reset counter
        self.iteration = 0  

        # initialize speaker 
        self.speaker = vt.Speaker(self.gender,
                                  self.glottal_masses,
                                  self.sample_freq,
                                  self.oversamp)
        

        initial_art = self._art.GetArt() # defaults to getting articulation at t=0
        self.speaker.InitSim(self.utterance_length, initial_art)
        self.initial_art = np.copy(initial_art) # store for later use
        #self.data['art_hist'][0] = np.copy(initial_art) # store for later use

        # element zero of output arrays will be filled with initial value of
        # simulator variables. 
        #self.SaveOutputs()
        self.data['art_hist'][:,0] = np.copy(initial_art)
        #print self.data['art_hist'][:, 0]
        #plt.figure()
        #plt.title('initial_art')
        #plt.show()
        #print self.art_hist[:, 0]

    def InitializeArticulation(self, **kwargs):
        # note: changing speaker params requires calling InitializeSpeaker
        if self._art_init == False:

            # Initialize artword for driving a manual sequence
            self._art = rand_aw.Artword(**kwargs)

            self._art_init = True  # flag for whether self.InitializeArticulation has been called

    def IsInitialized(self):
        return self._sim_init

    def InitializeAll(self, **kwargs):
        # initialize parameters if anything new is passed in
        self.UpdateParams(**kwargs)

        # Quick message to let us know whether we are initializing for a second time
        if not self._sim_init: 
            self._sim_init = True
            print "Setting up simulation..."
            self.InitializeArticulation(**kwargs)
            self.InitializeDir(self.directory)  # appends DTS to folder name
            #self.SaveParams()  # save parameters before anything else
            self.InitializeSim() # speaker now initialized in sim

        else: 
            print "Resetting simulation..."
            initial_art = kwargs.get("initial_art", None)
            self.Reset(initial_art)


        #self.InitializeSpeaker()
        #self.InitializeSim() # speaker now initialized in sim

    def Reset(self, initial_art=None):

        print "Resetting simulation..."
        self._art.Reset(initial_art)
        self.initial_art = initial_art
        self.InitializeSim()
        self.ResetOutputVars()

    ## set control targets in old Artword style
    def SetManualArticulation(self, muscle, times, targets):

        #self.InitializeArticulation()
        if not self._art_init: 
            print "Articulation has not been initialized. Cannot define manual controls."
            return 0

        if muscle > aw.kArt_muscle.MAX or muscle < 0:
            print "Invalid muscle ID: " + str(muscle)
            return 0
        if len(targets) != len(times):
            print "Invalid targets/times for muscle ID: " + str(muscle)


        for t in range(len(targets)):
            #self._art.SetManualTarget(muscle, times[t], targets[t])
            self._art.SetManualTarget(muscle, targets[t], times[t])

    def Simulate(self):

        self.ResetOutputVars()
        #self.SaveOutputs()

        #articulation = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))

        while self.speaker.NotDone():

            # pass the current articulation in
            #self._art.intoArt(articulation, self.speaker.NowSecondsLooped())
            articulation = self._art.GetArt(self.speaker.NowSecondsLooped())
            self.speaker.SetArticulation(articulation)
            self.speaker.IterateSim()

            # save output data
            # save new articulation
            #self.data['art_hist'][:, self.speaker.Now()-1] = np.copy(articulation) # save
            self.UpdateActionHistory(articulation, self.Now()-1)
            self.UpdateOutputs(self.Now()-1)

            # Save sound data point
            #self.SaveOutputs()
            #self.sound_wave[self.speaker.Now()-1] = self.speaker.GetLastSample()
            #self.speaker.GetAreaFcn(self.area_function[:, self.speaker.Now()-1])
            #self.speaker.GetPressureFcn(self.pressure_function[:, self.speaker.Now()-1])

            #self.art_hist[:, self.speaker.Now()] = np.copy(articulation)

        self.speaker.LoopBack()
        self.iteration += 1

    # not totally sure that this use of kwargs will work properly
    # previously named Save
    def SaveOutputs(self, fname = None, wav_file=True, **kwargs):

        self.SaveParams()  # save parameters before anything else

        if fname == None: 
            fname = self.iteration

        if wav_file:
            scaled = np.int16(self.data['sound_wave']/np.nanmax(np.abs(self.data['sound_wave']))*32767)
            write(self.directory + 'audio' + str(fname) + '.wav',
                  self.sample_freq,
                  scaled)

        #kwargs['sound_wave'] = self.sound_wave
        #kwargs['area_function'] = self.area_function
        #kwargs['pressure_function'] = self.pressure_function
        #kwargs['art_hist'] = self.art_hist

        np.savez( self.directory + 'data' + str(fname), **(self.data) )

    def SaveParams(self, **kwargs):
        
        kwargs['gender'] = self.gender
        kwargs['sample_freq'] = self.sample_freq
        kwargs['oversamp'] = self.oversamp
        kwargs['glottal_masses'] = self.glottal_masses
        kwargs['loops'] = self.loops

        np.savez(self.directory + 'params', **kwargs)

    def Run(self, **kwargs):
        # initialize parameters if anything new is passed in
        
        #self.InitializeAll(**kwargs)

        for k in range(self.loops):
            print "Loop: " + str(k)
            self.Simulate()
            self.SaveOutputs()


# wrapper functions for driving the simulator
    # this needs a new name, something other than save
    def UpdateActionHistory(self, action, index):
        self.data['art_hist'][:, index] = np.copy(action)

    def UpdateOutputs(self, index=0):
        # Save sound data point
        self.data['sound_wave'][index] = self.speaker.GetLastSample()

        self.speaker.GetAreaFcn(self.data['area_function'][:, index])

        self.speaker.GetPressureFcn(self.data['pressure_function'][:, index])

    def SetControl(self, action):
        self.speaker.SetArticulation(action)

    def GetLastControl(self):
        return self.data['art_hist'][:, self.speaker.Now()-1]
        
    def IterateSim(self):
        self.speaker.IterateSim()

    def Now(self):
        return self.speaker.Now()

    def NowSecondsLooped(self):
        return self.speaker.NowSecondsLooped()

    def NotDone(self):
        return self.speaker.NotDone()

    def Level(self):
        return 0

    def GetInitialControl(self):
        return self.data['art_hist'][:,0]


if __name__ == "__main__":
    
    # Default initial_art is all zeros
    apa = Utterance(directory="../data/apa",
                    loops=1,
                    utterance_length=0.5)

    apa.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0,   0.5], # time 
                                                         [0.5, 0.5]) # target

    apa.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                           [1.0, 1.0])

    apa.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.2], 
                                                [0.1, 0.0])

    apa.SetManualArticulation(aw.kArt_muscle.MASSETER, [0., 0.25, 0.5], [0., 0.7, 0.])

    apa.SetManualArticulation(aw.kArt_muscle.ORBICULARIS_ORIS, [0., 0.25, 0.5], [0., 0.2, 0.])

    apa.Run()

    ############################################################################
    sigh = Utterance(directory="../data/sigh",
                    loops=1,
                    utterance_length=0.5)

    sigh.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1], 
                                                [0.1, 0.0])


    sigh.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                           [1.0, 1.0])


    sigh.Run()
    ############################################################################
    ejective = Utterance(directory="../data/ejective",
                    loops=1,
                    utterance_length=0.5)

    ejective.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1], 
                                                     [0.1, 0.0])


    ejective.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5], 
                                                                [1.0, 1.0])

    ejective.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0.0, 0.17, 0.2, 0.35, 0.38, 0.5], 
                                                              [0.5, 0.5 , 1.0, 1.0 , 1.0 , 0.5])

    ejective.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.0, 0.5], 
                                                        [-.3, -.3])

    ejective.SetManualArticulation(aw.kArt_muscle.HYOGLOSSUS, [0.0, 0.5], 
                                                          [0.5, 0.5])

    ejective.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.1, 0.15, 0.29, 0.32], 
                                                            [0.0, 0.0, 1.0 , 1.0 , 0.0 ])

    ejective.SetManualArticulation(aw.kArt_muscle.STYLOHYOID, [0.0, 0.22, 0.27, 0.35, 0.38, 0.5], 
                                                          [0.0,  0.0, 1.0 , 1.0 , 0.0 , 0.0])

    ejective.Run()
    ############################################################################

    click = Utterance(dirname="../data/click",
                    loops=1,
                    utterance_length=0.5)

    click.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.0 , 0.2 ,  0.3 ,  0.5 ], 
                                                     [0.25, 0.25, -0.25, -0.25])

    click.SetManualArticulation(aw.kArt_muscle.ORBICULARIS_ORIS, [0.0 , 0.2 , 0.3, 0.5], 
                                                             [0.75, 0.75, 0.0, 0.0])

    click.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.5], 
                                                         [0.9, 0.9])
    click.Run()
