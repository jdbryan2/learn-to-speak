import numpy as np
import Artword as aw
import os

class ActionSequence(object):
#class Artword:

#TODO: Add save and load functions to this class
    def __init__(self, **kwargs):
        self.DefaultParams()
        self.UpdateParams(**kwargs)

    def DefaultParams(self):
        self.directory = '.'

        self._dim = aw.kArt_muscle.MAX 
        self._max_action = 1.
        self._min_action = -1.


        self.max_increment = 0.1  # sec
        self.min_increment = 0.01  # sec
        self.max_delta_target = 0.5 

        self.delayed_start = 0.

        self.current_target = np.zeros((self._dim, 2))
        #self.current_target[:, 0] += self.delayed_start
        self.current_target[:, 0] = np.ones(self._dim)*self.delayed_start
        self.previous_target = np.zeros((self._dim, 2))
        

        self.time = 0.
        self.sample_freq = 8000. # in seconds
        self.sample_period =  1./self.sample_freq # in seconds
        self.manual_targets = {}
        self._random = False # flag for generating random targets (or not)

    def UpdateParams(self, **kwargs):

        self.max_increment = kwargs.get("max_increment", self.max_increment)  # sec
        self.min_increment = kwargs.get("min_increment", self.min_increment)  # sec
        self.max_delta_target = kwargs.get("max_delta_target", self.max_delta_target)  
        
        self._min_action = kwargs.get("min_action", self._min_action) 
        self._max_action = kwargs.get("max_action", self._max_action) 

        self.delayed_start = kwargs.get("delayed_start", self.delayed_start)

        self._dim = kwargs.get("dimension", self._dim)
        self._dim = kwargs.get("dim", self._dim) 

        # reset target arrays if the dimension has been changed
        if not self.current_target.shape[0] == self._dim:
            self.current_target = np.zeros((self._dim, 2))
            self.previous_target = np.zeros((self._dim, 2))
            

        initial_action = kwargs.get("initial_action", np.copy(self.current_target[:, 1]))
        if not initial_action.size == self._dim:
            print "initial_action does not match specified dimension. Changing dimension to match initial_action."
            self._dim = initial_action.size
            self.current_target = np.zeros((self._dim, 2))
            self.previous_target = np.zeros((self._dim, 2))


        self.current_target[:, 1] = initial_action
        self.current_target[:, 0] = np.ones(self._dim)*self.delayed_start

        self.previous_target = np.copy(self.current_target)
        

        self.time = 0.
        # note: only sample_period is actually used
        self.sample_freq = kwargs.get("sample_freq", self.sample_freq) # in seconds
        self.sample_period = kwargs.get("sample_period", 1./self.sample_freq) # in seconds

        self._random = kwargs.get("random", self._random) # flag for generating random targets (or not)

    def SaveSequence(self, fname=None, directory=None):
        # Note: Only saves manual targets
        # randomly generated sequence is only saved through simulation because it's generated on the fly
        if directory != None: 
            self.directory = directory

        if fname==None:
            fname = 'sequence'

        data = {'manual_targets':self.manual_targets}
        
        # create save directory if needed
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        np.savez(os.path.join(self.directory, fname), **data)


    def LoadSequence(self, fname=None, directory=None):
        if directory != None: 
            self.directory = directory

        if fname==None:
            fname = 'sequence'

        if fname[-4:] != ".npz":
            fname += ".npz"

        
        if not os.path.exists(os.path.join(self.directory, fname)):
            print "Failed to load data sequence, file not found."
            self.manual_targets = {}
        else:
            data = np.load(os.path.join(self.directory, fname))
            self.manual_targets = data['manual_targets'].item()

        self.UpdateManualInit()

        
    def UpdateManualInit(self):
        # take care of initial_action
        for muscle in self.manual_targets:
            where_init = np.where(self.manual_targets[muscle][:, 0] == 0.)
            if len(where_init)>0:
                self.current_target[muscle, 0] = 0.;
                self.current_target[muscle, 1] = self.manual_targets[muscle][where_init, 1];


    def Reset(self, initial_art=None):
        if initial_art == None:
            if self._random:
                initial_art=np.random.random((self._dim, ))
            else:
                initial_art=np.zeros((self._dim, ))

        #print intial_art
        self.time=0.
        self.current_target = np.zeros((self._dim, 2))
        self.current_target[:, 1] = np.copy(initial_art)
        self.previous_target = np.copy(self.current_target)

    def Randomize(self, flag=True):
        self._random = flag

    def RandomTimeIncrement(self):
        return np.random.random() * \
                (self.max_increment-self.min_increment) + \
                self.min_increment

    def RandomDeltaTarget(self):
        return (np.random.random()-0.5)*self.max_delta_target

    def RandomTarget(self, target):
        #self._max_action = 1.
        #self._min_action = -1.
        #print self._max_action, self._min_action
        increment = self.RandomTimeIncrement()
        delta = self.RandomDeltaTarget()

        # if we're at the boundary, force delta to drive inward
        #print target
        if target == self._max_action:
            delta = -self._max_action * abs(delta)
        elif target == self._min_action:
            delta = -self._min_action * abs(delta)

        # if delta pushed past boundaries, interpolate to the
        # boundary and place target there
        if target + delta > self._max_action:
            increment = (self._max_action-target) * increment / delta
            delta = self._max_action-target
        elif target + delta < self._min_action:
            increment = (self._min_action-target) * increment / delta
            delta = self._min_action-target

        return increment, target+delta

    def UpdateTargets(self):
        # check if any targets are no longer in the future
        out_of_date = np.nonzero(self.current_target[:, 0] < self.time)[0]
        #print out_of_date
        for muscle in out_of_date:
            #print "Art: ", art

            # save current target as previous target
            self.previous_target[muscle, :] = np.copy(self.current_target[muscle, :])

            if muscle in self.manual_targets:
                times = self.manual_targets[muscle][:, 0]
                targets = self.manual_targets[muscle][:, 1]
                mask = times > self.time
                if np.sum(mask):
                    ind = np.argmin(times[mask])
                    self.current_target[muscle, 0] = (times[mask])[ind]
                    self.current_target[muscle, 1] = (targets[mask])[ind]
                else:
                    # if we don't have any more targets, leave constant and set target time to infinite
                    # this will prevent it from getting marked as "out_of_date"
                    self.current_target[muscle, 0] = np.infty

            else:
                if self._random:
                # generate new set point
                    time_inc, target = self.RandomTarget(self.previous_target[muscle, 1])
                    self.current_target[muscle, 0] = self.previous_target[muscle, 0]+time_inc
                    self.current_target[muscle, 1] = target
                else:
                    self.current_target[muscle, 0] = np.infty # never change, never go out of date



        # update targets with new random set point
        return 0

    def SetManualTarget(self, muscle, target, time):
        if self.time == 0. and time == 0.:
            self.current_target[muscle, 0] = 0.;
            self.current_target[muscle, 1] = target;

        # add manual target to the list
        if muscle in self.manual_targets:
            # over write if a target already exists at a given point in time
            where_equal = self.manual_targets[muscle][:, 0] == time
            #if np.any(self.manual_targets[muscle][:, 0] == time):
            if np.any(where_equal):
                self.manual_targets[muscle][np.where(where_equal), 1] = target
            else:
                self.manual_targets[muscle] = np.vstack((self.manual_targets[muscle], [time, target]))
        else:
            self.manual_targets[muscle] = np.array([[time, target]])

        sort_ind = np.argsort(self.manual_targets[muscle][:, 0])
        self.manual_targets[muscle][:, 0] = self.manual_targets[muscle][sort_ind, 0]
        self.manual_targets[muscle][:, 1] = self.manual_targets[muscle][sort_ind, 1]

    def PerturbManualTargets(self, epsilon=0.1):
        for muscle in self.manual_targets:
            self.manual_targets[muscle][:, 1] += (np.random.random(self.manual_targets[muscle].shape[0])-0.5)*epsilon

        self.UpdateManualInit()
    
    def UpdateTime(self, now):
        # set current time to now
        self.time = now

    def Now(self):
        return self.time

    def GetAction(self, time = None):
        if not time == None: 
            self.UpdateTime(time)
            self.UpdateTargets()

        # return articulation array 
        action = np.zeros(self._dim)
        for k in range(self._dim):
            action[k] = np.interp(self.Now(), 
                                        [self.previous_target[k, 0], self.current_target[k, 0]],
                                        [self.previous_target[k, 1], self.current_target[k, 1]])

        return action

    # function wrapper to match original Artword class (backward compatibility)
    def intoArt(self, art, time):
        _art = self.GetAction(time)
        for k in range(len(art)):
           art[k] = _art[k]

#class Artword(ActionSequence):

if __name__ == '__main__':
    d = 9
    rand = ActionSequence(dim=d, initial_action=np.zeros(d), sample_period=1./8000, delayed_start = 0.0, random=True)
                        #initial_art=np.random.random(self._dim))
    print rand._dim

    rand.SetManualTarget(0, 0.0, 0.5)
    rand.SetManualTarget(0, 0.5, 1.)
    rand.PerturbManualTargets()
    N= 10000
    x = np.zeros((N, rand._dim))

    for n in range(N): 
        time = 1.*n/8000

        #articulation = np.zeros(d)
        #rand.intoArt(articulation, time)
        action = rand.GetAction(time)
        x[n,:] = np.copy(action)

    rand.SaveSequence(fname='sequence')
    print rand.manual_targets

    rand.LoadSequence()
    print rand.manual_targets
    
    import pylab as plt
#    for k in range(x.shape[1]):
#        plt.plot(x[:, k])
    plt.plot(x[:, 0])
    plt.show()


