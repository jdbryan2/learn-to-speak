import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

# TODO: Add functionality to automatically load feature extractor based on primitive meta data

class PrimitiveUtterance(Utterance):

    def __init__(self, **kwargs):

        #super(PrimLearn, self).__init__(**kwargs)
        DataHandler.__init__(self, **kwargs)

        # initialize the variables
        self.Xf = np.array([])
        self.Xp = np.array([])

        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0

        # data vars
        self.art_hist = np.array([])
        self.area_function = np.array([])
        self.sound_wave = np.array([])
        self._data = np.array([]) # internal data var
        self.features = {}


        # downsample data such that it's some number of ms
        # period at which controller can make observations
        self._observation_period = 1 # in ms

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")

    def LoadPrimitives(self, fname=None):

        if fname==None:
            fname = 'primitives.npz'

        # save for later
        self.fname = fname

        primitives = np.load(os.path.join(self.home_dir, fname))

        self.K = primitives['K']
        self.O = primitives['O']
        self._ave = primitives['ave']
        self._std = primitives['std']
        self._past = primitives['past'].item()
        self._future = primitives['future'].item()

        self.dim = self.K.shape[0]
        print "Primitive dimension: ", self.dim

        feature_class= primitives['features'].item()
        feature_pointer = primitives['feature_pointer'].item() # should pull out the dict
        feature_tubes = primitives['feature_tubes'].item() # should pull out the dict
        control_action = primitives['control_action'].item() 

        if feature_class == 'ArtFeatures':
            from features.ArtFeatures import ArtFeatures
            self.Features= ArtFeatures(pointer=feature_pointer, 
                                       tubes=feature_tubes,
                                       control_action=control_action)
            
            
        else: 
            print 'Invalid features.'


    def EstimateStateHistory(self, data):
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()

    def EstimateState(self, data, normalize=True):
        data_length = data.shape[1]

        if normalize:
            # convert data by mean and standard deviation
            _data = data
            _data = ((_data.T-self._ave)/self._std).T
        
        if data_length < self._past:
            _data = np.append(np.zeros((data.shape[0], self._past-data_length)), _data, axis=1)
            data_length = self._past
            
        # only grabs the last values of the data 
        _Xp = np.reshape(_data[:, data_length-self._past:data_length].T, (-1, 1))
        current_state = np.dot(self.K, _Xp).flatten()
        return current_state

    def GetControl(self, current_state):

        _Xf = np.dot(self.O, current_state)
        _Xf = _Xf.reshape((self._data.shape[0], self._future))
        predicted = _Xf[:, 0]
        predicted = ((predicted.T*self._std)+self._ave).T
        return self.ClipArticulation(predicted[self.Features.pointer['art_hist']])

    def GetFeatures(self):
        self.speaker.GetAreaFcn(self.area_function[:, 0]) # grab initial area_function
        self.speaker.GetPressureFcn(self.pressure_function[:,0]) # grab initial area_function
        articulation = self.art_hist[:, self.speaker.Now()-1] 

        return self.Features.DirectExtract(articulation,
                                           self.area_function, 
                                           self.pressure_function)


    def ClipArticulation(self, articulation):
        articulation[articulation<0] = 0.
        articulation[articulation>1] = 1.
        return articulation

    def SaveParams(self):
        np.savez(self.directory + 'params',
                 gender=self.gender,
                 sample_freq=self.sample_freq,
                 oversamp=self.oversamp,
                 glottal_masses=self.glottal_masses,
                 loops=self.loops, 
                 primitives=self.fname)

    def ResetOutputVars(self):
        # reset the normal utterance output vars
        super(PrimitiveUtterance, self).ResetOutputVars()
        
        # internal primitive state "h"
        self.state_hist = np.zeros((self.K.shape[0],
                                  int(np.ceil(self.sample_freq *
                                          self.utterance_length))))

        # action_hist = high level control actions
        self.action_hist = np.zeros((self.K.shape[0],
                                  int(np.ceil(self.sample_freq *
                                          self.utterance_length))))

    def InitializeControl(self, **kwargs):
        # initialize parameters if anything new is passed in
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)

        #self.InitializeDir(self.method)  # appends DTS to folder name
        self.InitializeDir(self.dirname, addDTS=kwargs.get('addDTS', False))  # appends DTS to folder name
        self.SaveParams()  # save parameters before anything else
        self.InitializeSpeaker()

        # intialize simulator
        self.InitializeSim()

        # setup feedback variables
        #self.control_target = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double')) 
        features = self.GetFeatures()

        # past_data stores history of features used to compute current state
        self.past_data = np.zeros((features.shape[0], self._past))
        self.past_data = (past_data.T+features).T

        # estimate current state and do appropriate action
        self.state_hist[:, 0] = ss.EstimateState(self.past_data, normalize=True)
        self.control_target = ss.GetControl(np.zeros(dim))
        self.speaker.SetArticulation(self.control_target)


    # need a better way to pass sample period from one class to the other
    def SimulatePeriod(self, control_action, control_period):

        self.past_data = np.roll(self.past_data, -1, axis=1) # roll to the left
        self.past_data[:, -1] = self.GetFeatures()

        # get current state and choose target articulation
        current_state = self.EstimateState(past_data, normalize=True)
        target = ss.GetControl(current_state+control_action)

        # save control action and state
        self.state_hist[:, self.speaker.Now()/control_period-1] = current_state
        self.action_hist[:, self.speaker.Now()/control_period-1] = control_action
        
        # initialize features and articulation
        features = np.zeros(self.past_data.shape[0])
        articulation = np.zeros(target.shape[0])
        last_art = self.art_hist[:, control.speaker.Now()-1] # used for interpolation

        # loop over control period and implement interpolated articulation
        for t in range(control_period):
            if control.speaker.NotDone():

                # interpolate to target in order to make smooth motions
                for k in range(target.size):
                    articulation[k] = np.interp(t, [0, control_period-1], [last_art[k], target[k]])

                self.speaker.SetArticulation(articulation)
                self.speaker.IterateSim()

                self.SaveOutputs()
                # Save sound data point

                #area_function += control.area_function[:, control.speaker.Now()-1]/down_sample/8.
                #last_art += articulation/down_sample/8.

                self.art_hist[:, control.speaker.Now()-1] = articulation

            features += self.GetFeatures()/control_period.


if __name__ == "__main__":


