import numpy as np
import Artword as aw

class RandomArtword:

    def __init__(self, **kwargs):
        self.max_increment = kwargs.get("max_increment", 0.1)  # sec
        self.min_increment = kwargs.get("min_increment", 0.01)  # sec
        self.max_delta_target = kwargs.get("max_delta_target", 0.5)  

        self.initial_art = kwargs.get("initial_art", np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double')))

        self._art = np.copy(self.initial_art)
        
        # target[articulator, 0] = time 
        # target[articulator, 1] = position
        self.current_target = np.zeros((aw.kArt_muscle.MAX, 2))
        self.previous_target = np.zeros((aw.kArt_muscle.MAX, 2))

        self.time = 0.
        self.sample_period = kwargs.get("sample_period", 0.) # in seconds

    def RandomTimeIncrement(self):
        return np.random.random() * \
                (self.max_increment-self.min_increment) + \
                self.min_increment

    def RandomDeltaTarget(self):
        return (np.random.random()-0.5)*self.max_delta_target

    def RandomTarget(self, target):
        increment = self.RandomIncrement()
        delta = self.RandomDeltaTarget()

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

        return increment, target+delta


    def UpdateTargets(self):
        # check if any targets are no longer in the future
        out_of_date = np.nonzero(self.current_target[:, 0] < self.time)
        print out_of_date
        for art in out_of_date:
            # save current target as previous target
            self.previous_target[art, :] = np.copy(self.current_target[art, :])

            # generate new set point
            time_inc, target = self.RandomTarget(self.previous_target[art, 1])
            self.current_target[art, 0] = self.previous_target[art, 0]+time_inc
            self.current_target[art, 1] = target


        # update targets with new random set point
        return 0

    def SetManualTarget(self, articulator, time, target):
        print "Manual Targets are not yet supported"
        return 0

    def UpdateTime(self, now):
        # set current time to now
        self.time = now

    def Now(self):
        return self.time

    def GetArt(self, time = None):
        if not time == None: 
            self.UpdateTime(time)

        # return articulation array 
        articulation = np.zeros(aw.kArt_muscle.MAX)
        for k in range(aw.kArt_muscle.MAX):
            articulation[k] = np.interp(self.Now(), 
                                        [self.previous_target[k, 0], self.current_target[k, 0]],
                                        [self.previous_target[k, 1], self.current_target[k, 1]])

        return articulation



    # function wrapper to match original Artword class
    def intoArt(self, art, time):
        art = self.GetArt(time)

if __name__ == '__main__':
    rand = RandomArtword()

