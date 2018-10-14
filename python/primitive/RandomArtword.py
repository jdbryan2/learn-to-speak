import numpy as np
import Artword as aw

class Artword:

    def __init__(self, **kwargs):
        self.DefaultParams()
        self.UpdateParams(**kwargs)

    def UpdateParams(self, **kwargs):

        self.max_increment = kwargs.get("max_increment", self.max_increment)  # sec
        self.min_increment = kwargs.get("min_increment", self.min_increment)  # sec
        self.max_delta_target = kwargs.get("max_delta_target", self.max_delta_target)  

        self.delayed_start = kwargs.get("delayed_start", self.delayed_start)
        #initial_art = kwargs.get("initial_art", np.copy(self.current_target[:, 1])) 
        #if np.any(initial_art == None):
        #    if self._random:
        #        initial_art=np.random.random((aw.kArt_muscle.MAX, ))
        #    else:
        #        initial_art=np.zeros((aw.kArt_muscle.MAX, ))
        self.current_target[:, 1] = kwargs.get("initial_art", np.copy(self.current_target[:, 1]))
        self.current_target[:, 0] = np.ones(aw.kArt_muscle.MAX)*self.delayed_start
        #print self.current_target
        #print self.current_target[:,1]

        self.previous_target = np.copy(self.current_target)
        

        self.time = 0.
        # note: only sample_period is actually used
        self.sample_freq = kwargs.get("sample_freq", self.sample_freq) # in seconds
        self.sample_period = kwargs.get("sample_period", 1./self.sample_freq) # in seconds

        self._random = kwargs.get("random", self._random) # flag for generating random targets (or not)


    def DefaultParams(self):
        self.max_increment = 0.1  # sec
        self.min_increment = 0.01  # sec
        self.max_delta_target = 0.5 

        self.delayed_start = 0.

        self.current_target = np.zeros((aw.kArt_muscle.MAX, 2))
        self.current_target[:, 0] += self.delayed_start
        self.previous_target = np.zeros((aw.kArt_muscle.MAX, 2))
        

        self.time = 0.
        self.sample_freq = 8000. # in seconds
        self.sample_period =  1./self.sample_freq # in seconds
        self.manual_targets = {}
        self._random = False # flag for generating random targets (or not)

    def Reset(self, initial_art=None):
        if any(initial_art) == None:
            if self._random:
                initial_art=np.random.random((aw.kArt_muscle.MAX, ))
            else:
                initial_art=np.zeros((aw.kArt_muscle.MAX, ))

        #print intial_art
        self.time=0.
        self.current_target = np.zeros((aw.kArt_muscle.MAX, 2))
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
        increment = self.RandomTimeIncrement()
        delta = self.RandomDeltaTarget()

        # if we're at the boundary, force delta to drive inward
        #print target
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
            self.manual_targets[muscle] = np.vstack((self.manual_targets[muscle], [time, target]))
        else:
            self.manual_targets[muscle] = np.array([[time, target]])

        sort_ind = np.argsort(self.manual_targets[muscle][:, 0])
        self.manual_targets[muscle][:, 0] = self.manual_targets[muscle][sort_ind, 0]
        self.manual_targets[muscle][:, 1] = self.manual_targets[muscle][sort_ind, 1]
    

    def UpdateTime(self, now):
        # set current time to now
        self.time = now

    def Now(self):
        return self.time

    def GetArt(self, time = None):
        if not time == None: 
            self.UpdateTime(time)
            self.UpdateTargets()

        # return articulation array 
        articulation = np.zeros(aw.kArt_muscle.MAX)
        for k in range(aw.kArt_muscle.MAX):
            articulation[k] = np.interp(self.Now(), 
                                        [self.previous_target[k, 0], self.current_target[k, 0]],
                                        [self.previous_target[k, 1], self.current_target[k, 1]])

        return articulation

    # function wrapper to match original Artword class
    def intoArt(self, art, time):
        _art = self.GetArt(time)
        for k in range(len(art)):
           art[k] = _art[k]

if __name__ == '__main__':
    rand = Artword(sample_period=1./8000, delayed_start = 0.3, random=True)
                        #initial_art=np.random.random(aw.kArt_muscle.MAX))

    rand.SetManualTarget(0, 0.0, 0.5)
    rand.SetManualTarget(0, 0.5, 1.)
    N= 10000
    x = np.zeros((N, aw.kArt_muscle.MAX))

    for n in range(N): 
        time = 1.*n/8000

        articulation = np.zeros(aw.kArt_muscle.MAX)
        rand.intoArt(articulation, time)
        #articulation = rand.GetArt(time)
        x[n,:] = np.copy(articulation)

    
    import pylab as plt
    for k in range(x.shape[1]):
        plt.plot(x[:, k])
    plt.show()


