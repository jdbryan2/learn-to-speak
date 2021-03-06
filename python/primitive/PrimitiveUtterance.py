import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from Utterance import Utterance

# TODO: Add functionality to automatically load feature extractor based on primitive meta data

# this should actually take "utterance" as an attribute, not an inheritance
#class PrimitiveUtterance(Utterance):
class PrimitiveUtterance(object):

    def __init__(self, **kwargs):

        #super(PrimLearn, self).__init__(**kwargs)

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
        #self.art_hist = np.array([])
        #self.area_function = np.array([])
        #self.sound_wave = np.array([])
        self._data = np.array([]) # internal data var
        self.features = {}


        # downsample data such that it's some number of ms
        # period at which controller can make observations
        self.control_period = 1 # in ms

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")

        prim_fname = kwargs.get('prim_fname', None)
        if not prim_fname == None:
            self.LoadPrimitives(prim_fname)

        #super(PrimitiveUtterance, self).__init__(**kwargs)
        #self.utterance = Utterance(**kwargs) # default utterance object
        #self.utterance = kwargs.get('utterance', self.utterance)

    def SetUtterance(self, utterance):
        """
        Pass an utterance object to the controller.
        Object can be an instance of this PrimitiveUtterance class.

        """
        # function is somewhat trivial 
        self.utterance = utterance

    def LoadPrimitives(self, fname=None):

        if fname==None:
            print "No primitive file given"
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

        # pull control_period from primitive
        if 'control_period' in primitives:
            self.control_period = primitives['control_period'].item()

        if feature_class == 'ArtFeatures':
            from features.ArtFeatures import ArtFeatures
            self.Features= ArtFeatures(pointer=feature_pointer, 
                                       tubes=feature_tubes,
                                       control_action=control_action,
                                       sample_period=self.control_period)

        if feature_class == 'SpectralAcousticFeatures':
            from features.SpectralAcousticFeatures import SpectralAcousticFeatures

            # need to set this up to pass all relevant feature parameters
            # through. This will only work with all defaults set
            self.Features= SpectralAcousticFeatures(pointer=feature_pointer, 
                                                    tubes=feature_tubes,
                                                    control_action=control_action,
                                                    sample_period=self.control_period)
            
            
        else: 
            print 'Invalid features.'

    def EstimateStateHistory(self, data):
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()

    def EstimateState(self, data, normalize=True, extend_edge=False):
        data_length = data.shape[1]
        _data = data

        if normalize:
            # convert data by mean and standard deviation
            _data = ((_data.T-self._ave)/self._std).T
        
        # note: appending zeros actually works here because the feature data has been
        # shifted to be zero mean. Features of 0 are the average value.
        if data_length < self._past:
            # Either do 1 or 2

            # 1.)Take last element of history and assume that was what the data was up til self._past
            if extend_edge:
                data_init = np.ones((_data.shape[0], self._past-data_length))*np.atleast_2d(_data[:,0]).T
                _data = np.concatenate([data_init,_data],axis=1)
            
            # 2.)Set rest of history to zero
            else:
                _data = np.append(np.zeros((data.shape[0], self._past-data_length)), _data, axis=1)

            data_length = self._past
            
        # only grabs the last values of the data 
        _Xp = np.reshape(_data[:, data_length-self._past:data_length].T, (-1, 1))
        current_state = np.dot(self.K, _Xp).flatten()
        return current_state

    def GetControl(self, current_state):

        _Xf = np.dot(self.O, current_state)
        _Xf = _Xf.reshape((self.past_data.shape[0], self._future))
        predicted = _Xf[:, 0]
        predicted = ((predicted.T*self._std)+self._ave).T
        return self.ClipArticulation(predicted[self.Features.pointer['art_hist']])

    def GetFeatures(self):

        # TODO: figure out a way of generalizing the 'art_hist' variable to a
        # more general control_action variable as done in the feature extractor
        _data = {}
        _data['art_hist'] = self.utterance.art_hist[:, :self.speaker.Now()+1] 
        _data['area_function'] = self.utterance.area_function[:, :self.speaker.Now()+1]
        _data['pressure_function'] = self.utterance.pressure_function[:, :self.speaker.Now()+1]
        _data['sound_wave'] = self.utterance.sound_wave[:self.speaker.Now()+1]

<<<<<<< HEAD
        return self.Features.ExtractLast(_data)
=======
        # action_hist = high level control actions
        self.action_hist = np.zeros((self.K.shape[0],
                                  int(np.ceil(self.sample_freq *
                                              self.utterance_length /
                                              self.control_period))))
    
    # Note: I'm overiding the base class because I want to use the actual
    #       initial_art that is passed in not the default artword at 0.0 seconds.
    # TODO: Find a better way to do this. Possibly clean dependence on artwords
    #       for Utterance classes.
    def InitializeSim(self, **kwargs):
        # note: changing speaker params requires calling InitializeSpeaker
        if len(kwargs.keys()):
            self.InitializeParams(**kwargs)
        
        # Set the initial_art that the sim will be initialized with
        self.speaker.InitSim(self.utterance_length, self.initial_art)

    # Note: I'm overiding the base class because I want to use the actual
    #       initial_art that is passed in not the default artword at 0.0 seconds.
    #       This is requred in addition to Initialize Sim so that I can pass in a
    #       new intitial_art each time I want to start over the sim.
    def InitializeParams(self, **kwargs):
        if len(kwargs.keys()):
            self.initial_art = kwargs.get("initial_art", self.initial_art)
            super(PrimitiveUtterance, self).InitializeParams(**kwargs)
>>>>>>> master

    def ClipControl(self, action, lower=0., upper=1.):
        action[action<lower] = lower 
        action[action>upper] = upper 
        return action

    # TODO: InitControl needs updating to the new cascaded class format
    def InitializeControl(self, **kwargs):
        # initialize parameters if anything new is passed in
        if len(kwargs.keys()):
            self.UpdateParams(**kwargs)

        print "Controller period:", self.control_period

        #self.InitializeDir(self.method)  # appends DTS to folder name
        self.InitializeDir(self.dirname)  # appends DTS to folder name
        self.SaveParams()  # save parameters before anything else

        # intialize simulator
        self.InitializeSpeaker() # Should only have to call this once unless chaning speaker parameters
        #self.InitializeArticulation()
        self.InitializeSim()
        self.ResetOutputVars()

        # setup feedback variables
        #self.control_target = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double')) 
        features = self.GetFeatures()

        # past_data stores history of features used to compute current state
        
        # Currenlty not using this method. It relies on EstimateState to fill in the rest of the past_data vector
        # self.past_data = features
        # self.past_data = self.past_data.reshape(self.past_data.shape[0],1)
        
        # Either choose 1 or 2 for initializing past_data
        
        # 1.)Set all of past but last state to 0 then estimate state without normalization to avoid subtraction of mean
        self.past_data = np.zeros((features.shape[0], self._past))
        self.past_data[:,-1] = features
        self.current_state = self.EstimateState(self.past_data, normalize=False)

        # 2.)Set whole past to initial features then estimate state with normalization
        #self.past_data = np.zeros((features.shape[0], self._past))
        #self.past_data = (self.past_data.T+features).T
        #self.current_state = self.EstimateState(self.past_data, normalize=True)

        # Append to state_history
        self.state_hist[:, 0] = self.current_state

        #articulation = self.GetControl(self.current_state)
        #self.speaker.SetArticulation(articulation)


    # need a better way to pass sample period from one class to the other
    # maybe it's worth setting it up to pass a pointer to a control policy as input?
    def SimulatePeriod(self, control_action=0, control_period=0):
        if control_period > 0:
            self.control_period=control_period

        # get target articulation
        #target = self.GetControl(self.current_state+control_action)
        target = self.GetControl(control_action)
        # Save control action history
        self.action_hist[:, self.speaker.Now()/self.control_period-1] = control_action

        # initialize features and articulation
        features = np.zeros(self.past_data.shape[0])
        action = np.zeros(target.shape[0])
        prev_target = self.utterance.GetLastControl()

        # loop over control period and implement interpolated articulation
        for t in range(self.control_period):
            if self.speaker.NotDone():

                # interpolate to target in order to make smooth motions
                # TODO: For some reason, this interpolation modifies the variable target.
                for k in range(target.size):
                    action[k] = np.interp(t, [0, self.control_period-1], [prev_target[k], target[k]])
                    # it may be worth wrapping this function too
                    # effectively, it would do something like getting the control from the lower level controller
                    # in order to reach the desired target. 

                #self.speaker.SetArticulation(articulation)
                #self.speaker.IterateSim()
                self.utterance.SetControl(articulation)
                self.utterance.IterateSim()

                self.UpdateOutputs()
                # Save sound data point

                #area_function += control.area_function[:, control.speaker.Now()-1]/down_sample/8.
                #last_art += articulation/down_sample/8.

                self.art_hist[:, self.utterance.Now()-1] = action

            #features += self.GetFeatures()/self.control_period
        features = self.GetFeatures()

        # add features to past data
        # First if statement won't get used currently how we are initializing past_data, but here for future proofing
        if self.past_data.shape[1]< self._past:
            features = features.reshape(features.shape[0],1)
            self.past_data = np.concatenate([self.past_data,features],axis=1)
        else:
            self.past_data = np.roll(self.past_data, -1, axis=1) # roll to the left
            self.past_data[:, -1] = features
        

        # get current state and choose target articulation
        self.current_state = self.EstimateState(self.past_data, normalize=True)

        # save control action and state
        self.state_hist[:, self.utterance.Now()/self.control_period-1] = self.current_state
        self.action_hist[:, self.utterance.Now()/self.control_period-1] = control_action

        return self.current_state


    # wrapper functions for driving the simulator
    # PrimitiveUtterance can be used as the utterance attribute
    def Level(self):
        return self.utterance.Level()+1

    def SaveParams(self, **kwargs):

        # add pointer to primitive operators
        kwargs['primitives_'+str(self.Level())] = self.fname

        # pass down the chain
        self.utterance.SaveParams(**kwargs)
        #np.savez(self.directory + 'params', **kwargs)

    def Save(self, fname=None, wav_file=True, **kwargs):

        # state and state-level control action
        kwargs['state_hist_'+str(self.Level())] = self.state_hist
        kwargs['action_hist_'+str(self.Level())] = self.action_hist

        # pass down the chain
        self.utterance.Save(fname=fname, wav_file=wav_file, **kwargs)

    def ResetOutputVars(self):
        # reset the normal utterance output vars
        #super(PrimitiveUtterance, self).ResetOutputVars()
        self.utterance.ResetOutputVars()
        
        # internal primitive state "h"
        self.state_hist = np.zeros((self.K.shape[0],
                                  int(np.ceil(self.sample_freq *
                                              self.utterance_length / 
                                              self.control_period))))

        # action_hist = high level control actions
        self.action_hist = np.zeros((self.K.shape[0],
                                  int(np.ceil(self.sample_freq *
                                              self.utterance_length /
                                              self.control_period))))

    def UpdateOutputs(self, index=0):
        self.utterance.SaveOutputs(self.Now()-1)

        ## Save sound data point
        #self.sound_wave[index] = self.speaker.GetLastSample()
        #self.speaker.GetAreaFcn(self.area_function[:, index])
        #self.speaker.GetPressureFcn(self.pressure_function[:, index])

    def InitializeDir(self, directory, addDTS=True):
        self.utterance.IntializeDir(directory, addDTS)

    def SetControl(self, action):
        self.utterance.SetControl(action)

    def GetLastControl(self):
        return self.utterance.GetLastControl()
        
    def IterateSim(self):
        self.utterance.IterateSim()

    def Now(self):
        return self.utterance.Now()



if __name__ == "__main__":

    print "Some test script here."

