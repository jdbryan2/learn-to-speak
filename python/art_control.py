# load primitives from file
# control vocal tract 
# generate figures with constant control inputs at various values
# save vocal tract data of utterance to file, and save output audio to file


import os
import numpy as np
import pylab as plt
from primitive.SubspacePrim import PrimLearn
import Artword as aw
from matplotlib2tikz import save as tikz_save

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    if data.shape[1] < max_length:
        max_length= data.shape[1]

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/1000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/1000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/1000.))

#import test_params
from test_params import *
#dim = 8
#sample_period = 10
#dirname = 'full_random_500'
primdir = dirname+'_primv'
#savedir = 'data/' + dirname + '/figures/in_out/'
#load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals

#max_delta = 0.2/0.01/1000*sample_period

if not os.path.exists(savedir):
    os.makedirs(savedir)

ss = PrimLearn()

ss.ConvertDataDir(dirname, sample_period=sample_period)

ss.LoadPrimitives(load_fname)

#ss.EstimateStateHistory(ss._data)
#plt.plot(ss.h.T)
#plt.show()



####
# This stuff should get moved to primitive control class
from primitive.Utterance import Utterance


control = Utterance(dir_name=primdir,
        loops=1,
        utterance_length=5)

#control.InitializeDir(dirname=primdir, addDTS=False)


control.InitializeManualControl()

ss.SavePrimitives(primdir+'/primitives')

# setup feedback variables
target = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
articulation = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
area_function = np.zeros(control.area_function.shape[0]) 
control.speaker.GetAreaFcn(area_function) # grab initial area_function

pressure_function = np.zeros(control.area_function.shape[0]) 
control.speaker.GetPressureFcn(pressure_function) # grab initial area_function
lung_pressure = np.mean(pressure_function[ss.tubes['lungs']], axis=0)
print "Resting Lung Pressure:", lung_pressure

last_art = np.zeros(control.art_hist.shape[0])
past_data = np.zeros((ss._data.shape[0], ss._past))
# area function should be back filled
past_data = (past_data.T+
                np.append(articulation, 
                    np.append(area_function[ss.tubes['glottis_to_velum']], lung_pressure))).T 

_h = np.zeros((1, dim))
_h[0] = ss.EstimateState(past_data, normalize=True)

#target = ss.GetControl(_h[0])
#target[target<0] = 0.
#target[target>1] = 1.
#control.speaker.SetArticulation(target)

while control.speaker.NotDone():

    past_data = np.roll(past_data, -1, axis=1) # roll to the left
    
    past_data[:, -1] = np.append(articulation, np.append(area_function[ss.tubes['glottis_to_velum']], lung_pressure)) # put data in last column

    h = ss.EstimateState(past_data, normalize=True)
    v = np.zeros(h.shape)
    v[0] = 5 
    target = ss.GetControl(h+v)
    print h
    #plt.plot(target)
    #plt.show()
    _h = np.append(_h, h.reshape((1, dim)), axis=0)

    # reset variables so they will get loaded again
    last_art = np.copy(articulation)

    area_function = np.zeros(control.area_function.shape[0]) 
    lung_pressure=0
    for t in range(sample_period*8):
        if control.speaker.NotDone():

            # interpolate to target in order to make smooth motions
            for k in range(target.size):
                articulation[k] = np.interp(t, [0, sample_period*8-1], [last_art[k], target[k]])
                #if last_art[k] - articulation[k] < -max_delta:
                #    articulation[k] = last_art[k]-max_delta

                #elif articulation[k] - last_art[k] > max_delta:
                #    articulation[k] = last_art[k]+max_delta
                    
                if articulation[k] < 0.:
                    articulation[k] = 0.
                elif articulation[k] > 1.:
                    articulation[k] = 1.


            #print articulation
            
            #if control.speaker.Now() > 100*sample_period*8:
                # pass the current articulation in
                #control.speaker.SetArticulation(articulation)

            control.speaker.SetArticulation(articulation)
            control.speaker.IterateSim()

            control.SaveOutputs()
            # Save sound data point

            #area_function += control.area_function[:, control.speaker.Now()-1]/down_sample/8.
            #last_art += articulation/down_sample/8.

            control.art_hist[:, control.speaker.Now()-1] = articulation

        area_function += control.area_function[:, control.speaker.Now()-1]/sample_period/8.
        lung_pressure += np.mean(control.pressure_function[ss.tubes['lungs'], control.speaker.Now()-1], axis=0)/sample_period/8.


control.Save()
#plt.plot(_h)
PlotTraces(_h.T, np.arange(_h.shape[1]), _h.shape[0], sample_period, highlight=0)
plt.show()

plt.figure()
PlotTraces(control.art_hist, np.arange(control.art_hist.shape[0]), control.art_hist.shape[1], sample_period, highlight=0)
#plt.plot(control.art_hist.T)
plt.show()

