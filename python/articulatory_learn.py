# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
from primitive.SubspacePrim import PrimLearn
## pretty sure these aren't used ##
#import scipy.signal as signal
#import numpy.linalg as ln
#import os

down_sample = 10
ss = PrimLearn()
ss.LoadDataDir('full_random_100')
ss.PreprocessData(50, 10, sample_period=down_sample)
ss.SubspaceDFA(dim)

ss.EstimateStateHistory(ss._data)
plt.plot(ss.h.T)
plt.show()

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.K[k, :].reshape(ss._past, 88))
#    plt.title('Input: '+str(k))

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.O[:, k].reshape(ss._future, 88), aspect=2)
#    plt.title('Output: '+str(k))

for k in range(dim):
    plt.figure();
    K = ss.K[k,:].reshape(ss._past, 88)
    for p in range(ss._past):
        plt.plot(K[p, :], 'b-', alpha=1.*(p+1)/(ss._past+1))
    plt.title('Input: '+str(k))

for k in range(dim):
    plt.figure();
    O = ss.O[:, k].reshape(ss._future, 88)
    for f in range(ss._future):
        dat = O[f, :]
        dat = ((dat.T*ss._std)+ss._ave).T

        plt.plot(dat, 'b-', alpha=1.*(ss._future-f+1)/(ss._future+1))
    plt.title('Output: '+str(k))

plt.show()

####
# This stuff should get moved to primitive control class
from primitive.Utterance import Utterance


control = Utterance(dir_name="prim_out",
        loops=1,
        utterance_length=4)

control.InitializeDir(dirname="prim_out")

control.InitializeManualControl()


target = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
articulation = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
area_function = np.zeros(control.area_function.shape[0]) 
control.speaker.GetAreaFcn(area_function) # grab initial area_function


last_art = np.zeros(control.art_hist.shape[0])
past_data = np.zeros((ss._data.shape[0], ss._past))
# area function should be back filled
past_data = (past_data.T+np.append(articulation, area_function[ss.tubes['all']])).T 

_h = np.zeros((1, dim))

while control.speaker.NotDone():

    past_data = np.roll(past_data, -1, axis=1) # roll to the left
    past_data[:, -1] = np.append(articulation, area_function[ss.tubes['all']]) # put data in last column

    h = ss.EstimateState(past_data, normalize=True)
    target = ss.GetControl(h)
    print h
    #plt.plot(target)
    #plt.show()
    _h = np.append(_h, h.reshape((1, dim)), axis=0)

    # reset variables so they will get loaded again
    last_art = np.copy(articulation)


    for t in range(down_sample*8):
        if control.speaker.NotDone():

            # interpolate to target in order to make smooth motions
            for k in range(target.size):
                articulation[k] = np.interp(t, [0, down_sample*8-1], [last_art[k], target[k]])
                if articulation[k] < 0.:
                    articulation[k] = 0.
                elif articulation[k] > 1.:
                    articulation[k] = 1.

            # pass the current articulation in
            control.speaker.SetArticulation(articulation)

            control.speaker.IterateSim()

            control.SaveOutputs()
            # Save sound data point

            #area_function += control.area_function[:, control.speaker.Now()-1]/down_sample/8.
            #last_art += articulation/down_sample/8.

    area_function = control.area_function[:, control.speaker.Now()-1]
control.Save()
plt.plot(_h)
plt.show()

