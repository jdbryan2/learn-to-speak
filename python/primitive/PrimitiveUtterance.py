import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from ActionSequence import ActionSequence
from matplotlib2tikz import save as tikz_save

from PrimitiveHandler import PrimitiveHandler
from Utterance import Utterance

# Note:  
    # PrimitiveUtterance must have methods that match with the Utterance class
    # these matching methods must in some way call the corresponding method from 
    # the attribute PrimitiveUtterance.utterance
    #
    # The attribute utterance can either be an instance of PrimitiveUtterance or Utterance
    # This gives the abilility to call a function that cascades down the chain until the bottom - which must be an
    # instance of Utterance
class PrimitiveUtterance(PrimitiveHandler):

    def __init__(self, **kwargs):

        super(PrimitiveUtterance, self).__init__(**kwargs)

        self.InitVars()
        self.DefaultParams()
        self.UpdateParams(**kwargs)

        self._data = np.array([]) # internal data var

        prim_fname = kwargs.get('prim_fname', None)
        if not prim_fname == None:
            self.LoadPrimitives(prim_fname)

        self._act = ActionSequence() # leave as default, can be overridden by user

        #super(PrimitiveUtterance, self).__init__(**kwargs)
        #self.utterance = Utterance(**kwargs) # default utterance object
        #self.utterance = kwargs.get('utterance', self.utterance)

    def SetUtterance(self, utterance):
        """
        Pass an utterance object to the controller.
        Object can be an instance of this PrimitiveUtterance class.

        """
        # assign utterance class at the bottom of the cascade
        try:
            self.utterance.SetUtterance(utterance)
        except AttributeError:
            self.utterance = utterance

    def EstimateState(self, data, normalize=True, extend_edge=False):
        data_length = data.shape[1]
        _data = data

        if normalize:
            # convert data by mean and standard deviation
            _data = ((_data.T-self._mean)/self._std).T
        
        # note: appending zeros actually works here because the feature data has been
        # shifted to be zero mean. Features of 0 are the average value.
        if data_length < self._past:
            print "Not enough data provided"
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

    def GetControlMean(self, level=-1):
        if self.Level() == level or level<=0:
            return self._mean[self.Features.pointer[self.Features.control_action]]
        else: 
            return self.utterance.GetControlMean(level)

    #TODO: change input variable to something like "action"
    def GetControl(self, current_state):

        _Xf = np.dot(self.O, current_state)
        _Xf = _Xf.reshape((-1, self._future))

        predicted = _Xf[:, 0]
        predicted = ((predicted.T*self._std)+self._mean).T
        return self.ClipControl(predicted[self.Features.pointer[self.Features.control_action]])

    def GetFeatures(self):
        #print self.utterance.GetOutputVars(self.Now()+1)
        #print self.Now()+1
        #plt.figure()
        #plt.show()
        print self.NowPeriods()
        return self.Features.ExtractLast(self.utterance.GetOutputVars(self.Now()))


    def ClipControl(self, action, lower=0., upper=1.):
        action[action<lower] = lower 
        action[action>upper] = upper 
        return action

    # TODO: InitControl needs updating to the new cascaded class format
    def InitializeControl(self, **kwargs):

        print "Controller period:", self._sample_period

        # pass kwargs through to utterance initializer
        #self.utterance.InitializeAll(**kwargs)
        self.utterance.InitializeControl(**kwargs)
        self.SaveParams()  # save parameters before anything else

        self.ResetOutputVars()

        # setup feedback variables
        #self.control_target = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double')) 

        # run the simulator for one loop to get the initial output
        for k in range(1):
            self.IterateSim()
            self.UpdateOutputs()
        features = self.GetFeatures()

        # Either choose 1 or 2 for initializing past_data
        # past_data stores history of features used to compute current state
        edge_copy = kwargs.get("edge_copy", True)
        
        if not edge_copy:
            # 1.)Set all of past but last state to 0 then estimate state without normalization to avoid subtraction of mean
            self.past_data = np.zeros((features.shape[0], self._past))
            self.past_data[:,-1] = features
            self.current_state = self.EstimateState(self.past_data, normalize=False)

        else:
            # 2.)Set whole past to initial features then estimate state with normalization
            self.past_data = np.zeros((features.shape[0], self._past))
            self.past_data = (self.past_data.T+features).T
            self.current_state = self.EstimateState(self.past_data, normalize=True)

        # Append to state_history
        self.state_hist[:, 0] = self.current_state


    # need a better way to pass sample period from one class to the other
    # maybe its worth setting it up to pass a pointer to a control policy as input?
    def SimulatePeriod(self, control_action=None, sample_period=0, hold=False):

        # I don't think this is actually good though
        if np.any(control_action) == None: 
            control_action = np.zeros(self.O.shape[1])

        if sample_period > 0:
            self._sample_period=sample_period

        # get target articulation
        #target = self.GetControl(self.current_state+control_action)
        target = self.GetControl(control_action) # only use high level input, no state feedback

        # Save control action history
        self.action_hist[:, self.Now2Periods(self.Now())-1] = control_action

        # initialize features and articulation
        features = np.zeros(self.past_data.shape[0])
        action = np.copy(self.utterance.GetInitialControl()) #np.zeros(target.shape[0])
        prev_target = self.utterance.GetLastControl()

        # loop over control period and implement interpolated articulation
        for t in range(self._sample_period):
            if self.NotDone():

                # interpolate to target in order to make smooth motions
                # TODO: For some reason, this interpolation modifies the variable target.
                if hold:
                    print "Holding"
                else:
                    for k in range(target.size):
                        action[k] = np.interp(t, [0, self._sample_period-1], [prev_target[k], target[k]])
                        # it may be worth wrapping this function too
                        # effectively, it would do something like getting the control from the lower level controller
                        # in order to reach the desired target. 

                    self.utterance.SetControl(action)

                self.utterance.IterateSim()

                self.UpdateOutputs()
                # Save sound data point

                self.utterance.UpdateActionHistory(action, self.Now2Periods(self.Now())-1)

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
        self.state_hist[:, self.utterance.Now()/self._sample_period-1] = self.current_state
        self.action_hist[:, self.utterance.Now()/self._sample_period-1] = control_action

        return self.current_state

    def Simulate(self):
            
        while self.NotDone():
            self.SimulatePeriod(control_action=self._act.GetAction(self.NowSecondsLooped()))

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

    def GetParams(self, **kwargs):

        # add pointer to primitive operators
        kwargs['primitives_'+str(self.Level())] = self.fname

        # pass down the chain
        return self.utterance.GetParams(**kwargs)
        #np.savez(self.directory + 'params', **kwargs)

    def GetOutputVars(self, time):
        # TODO: verify that I'm getting the time indexing right here
        _data = self.utterance.GetOutputVars(time) 
        _data['state_hist_'+str(self.Level())] = self.state_hist[:self.Now2Periods(time)]
        _data['action_hist_'+str(self.Level())] = self.action_hist[:self.Now2Periods(time)]
        return _data

    def SaveOutputs(self, fname=None, wav_file=True, **kwargs):

        # state and state-level control action
        kwargs['state_hist_'+str(self.Level())] = self.state_hist
        kwargs['action_hist_'+str(self.Level())] = self.action_hist

        # pass down the chain
        self.utterance.SaveOutputs(fname=fname, wav_file=wav_file, **kwargs)

    def GetOutputs(self, **kwargs):
        # state and state-level control action
        kwargs['state_hist_'+str(self.Level())] = self.state_hist
        kwargs['action_hist_'+str(self.Level())] = self.action_hist

        return self.utterance.GetOutputs(**kwargs)

    def ResetOutputVars(self):
        # reset the normal utterance output vars
        #super(PrimitiveUtterance, self).ResetOutputVars()
        self.utterance.ResetOutputVars()
        
        if self.Level() == 1:
            self.state_hist = np.zeros((self.K.shape[0],
                                      int(np.ceil(self.utterance.sample_freq *
                                                  self.utterance.utterance_length / 
                                                  self._sample_period))))

            # action_hist = high level control actions
            self.action_hist = np.zeros((self.K.shape[0],
                                      int(np.ceil(self.utterance.sample_freq *
                                                  self.utterance.utterance_length /
                                                  self._sample_period))))
        else:
            self.state_hist = np.zeros((self.K.shape[0],
                                      int(np.ceil(self.utterance.state_hist.shape[1] / 
                                                  self._sample_period))))

            # action_hist = high level control actions
            self.action_hist = np.zeros((self.K.shape[0],
                                      int(np.ceil(self.utterance.action_hist.shape[1] /
                                                  self._sample_period))))

    def UpdateActionHistory(self, action, index):
        self.action_hist[:, index] = np.copy(action)

    # only updates simulator outputs
    def UpdateOutputs(self, index=0):
        self.utterance.UpdateOutputs(self.Now()-1)

        ## Save sound data point
        #self.sound_wave[index] = self.speaker.GetLastSample()
        #self.speaker.GetAreaFcn(self.area_function[:, index])
        #self.speaker.GetPressureFcn(self.pressure_function[:, index])

    def InitializeDir(self, directory, addDTS=True):
        self.utterance.IntializeDir(directory, addDTS)

    def SetControl(self, action):
        self.utterance.SetControl(action)

    def GetControlHistory(self, level=-1):
        if level == self.Level() or level==-1:
            return self.action_hist

        else: # if it's at a lower level, pass it down the line
            return self.utterance.GetControlHistory(level)

    def GetStateHistory(self, level=-1):
        if level == self.Level() or level==-1:
            return self.state_hist

        else: # if it's at a lower level, pass it down the line
            return self.utterance.GetStateHistory(level)

    def GetAreaHistory(self):
        return self.utterance.GetAreaHistory()

    def GetPressureHistory(self):
        return self.utterance.GetAreaHistory()

    def GetSoundWave(self):
        return self.utterance.GetSoundWave()

    def GetLastControl(self):
        return self.utterance.GetLastControl()
        
    def IterateSim(self):
        self.utterance.IterateSim()

    def Now(self):
        return self.utterance.Now()

    def Now2Periods(self, time):
        if self.Level() == 1:
            return time/self._sample_period
        elif self.Level() > 1: 
            return int(self.utterance.Now2Periods(time)/self._sample_period)

    def NowPeriods(self):
        return self.utterance.NowPeriods()/self._sample_period

    def NowSecondsLooped(self):
        return self.utterance.NowSecondsLooped()

    def NotDone(self):
        return self.utterance.NotDone()

    def LoopBack(self):
        return self.utterance.LoopBack()


    #def LoadPrimitives(self, fname=None):

    #    if fname==None:
    #        print "No primitive file given"
    #        fname = 'primitives.npz'

    #    # append file extension if not given
    #    if fname[-4:] != ".npz":
    #        fname+=".npz"
    #    # save for later
    #    self.fname = fname

    #    #primitives = np.load(os.path.join(self.directory, fname))
    #    primitives = np.load(fname)

    #    self.K = primitives['K']
    #    self.O = primitives['O']
    #    self._mean = primitives['mean']
    #    self._std = primitives['std']
    #    self._past = primitives['past'].item()
    #    self._future = primitives['future'].item()

    #    self._dim = self.K.shape[0]
    #    print "Primitive dimension: ", self._dim


    #    self.Features = primitives['features'].item() # should load an instance of the Feature extractor object used
    #    #feature_class= primitives['features'].item()
    #    #feature_pointer = primitives['feature_pointer'].item() # should pull out the dict
    #    #feature_tubes = primitives['feature_tubes'].item() # should pull out the dict
    #    #control_action = primitives['control_action'].item() 
    #    #control_pointer = self.Features.pointer[control_action]

    #    # pull sample_period from primitive
    #    if 'sample_period' in primitives:
    #        self._sample_period = primitives['sample_period'].item()

    #    if 'downpointer_fname' in primitives:
    #        self._downpointer_fname = primitives['downpointer_fname'].item()
    #    else: 
    #        self._downpointer_fname = None

    #    if 'downpointer_directory' in primitives:
    #        print "downpointer directory", primitives['downpointer_directory'].item()
    #        self._downpointer_directory = primitives['downpointer_directory'].item()
    #    else: 
    #        self._downpointer_directory  = None

    #    #if feature_class == 'ArtFeatures':
    #    #    from features.ArtFeatures import ArtFeatures
    #    #    self.Features= ArtFeatures(pointer=feature_pointer, 
    #    #                               tubes=feature_tubes,
    #    #                               control_action=control_action,
    #    #                               sample_period=self._sample_period)

    #    #if feature_class == 'SpectralAcousticFeatures':
    #    #    from features.SpectralAcousticFeatures import SpectralAcousticFeatures

    #    #    # need to set this up to pass all relevant feature parameters
    #    #    # through. This will only work with all defaults set
    #    #    self.Features= SpectralAcousticFeatures(pointer=feature_pointer, 
    #    #                                            tubes=feature_tubes,
    #    #                                            control_action=control_action,
    #    #                                            sample_period=self._sample_period)
    #    #    
    #        
    #    #else: 
    #        #print 'Invalid features.'

# defined in PrimitiveHandler
###########################################
#    def EstimateStateHistory(self, data):
#        self.h = np.zeros((self.K.shape[0], data.shape[1]))
#
#        #for t in range(0, self._past):
#        #    self.h[:, t] = self.EstimateState(data[:, 0:t])
#
#        for t in range(self._past, data.shape[1]):
#            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
#            self.h[:, t] = np.dot(self.K, _Xp).flatten()


if __name__ == "__main__":

    print "Some test script here."

