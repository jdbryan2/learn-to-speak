
import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
from primitive.RandExcite import RandExcite
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
from primitive.ActionSequence import ActionSequence
from primitive.DataHandler import DataHandler

import pylab as plt

from genfigures.plot_functions import *

NO_THRESHOLD = False   
loops = 1000 
utterance_length = 1. #10.0
#full_utterance = loops*utterance_length

#savedir = 'data/rand_steps_full'
savedir = 'data/rand_prim_1sec'

#prim_filename = 'round550'
#prim_dirname = 'data/batch_random_1_1'
#full_filename = os.path.join(prim_dirname, prim_filename)

prim_dirname = 'data/batch_random_20_5'
ind= get_last_index(prim_dirname, 'round')
prim_filename = 'round%i.npz'%ind

#true_dim = prim._dim 


loop_start = get_last_index(savedir, 'data')+1
print loop_start
k = loop_start
failed_attempts = 0

#for k in range(loop_start, loop_start+loops):
while k < loop_start+loops:

    prim = PrimitiveUtterance()
    #prim.LoadPrimitives(full_filename)
    prim.LoadPrimitives(prim_filename, prim_dirname) # updated loading function

    initial_control = np.random.random(prim._dim)*2. - 1.
    #initial_control[:true_dim] = np.random.random(true_dim)*2. - 1.
    #end_control = np.zeros(prim._dim)
    #end_control[:true_dim] = np.random.random(true_dim)*2 - 1.

    prim.utterance = Utterance(directory = savedir, 
                               utterance_length=utterance_length, 
                               loops=loops,
                               addDTS=False)
                               #initial_art = prim.GetControlMean(),



# compute sample period in seconds
    sample_period = prim._sample_period/prim.utterance.sample_freq

# setup action sequence
    rand = ActionSequence(dim=prim._dim,
                          initial_action=initial_control,
                          sample_period=sample_period,
                          random=True,
                          min_increment=0.1, # 20*sample_period, 
                          max_increment=0.1, # 20*sample_period,
                          max_delta_target=1.)

    # all factors over 3 to be constant zero
    #for factor in range(prim._dim):
    #    rand.SetManualTarget(factor, initial_control[factor], 0.)
    #    rand.SetManualTarget(factor, initial_control[factor], 0.25)
    #    rand.SetManualTarget(factor, end_control[factor], 0.5)
    #    rand.SetManualTarget(factor, end_control[factor], 1.0)

    prim.InitializeControl(initial_art = prim.GetControl(rand.GetAction(time=0.0)))

    #handler = DataHandler()
    #handler.params = prim.GetParams()


    prim._act = rand
    prim.Simulate()
    #while prim.NotDone():
    #    action = rand.GetAction(prim.NowSecondsLooped())
    #    #print action
    #    #print prim.NowSecondsLooped()
    #    prim.SimulatePeriod(control_action=action)

    #plt.figure()
    #plt.plot(prim.state_hist.T)
    #plt.figure()
    #plt.plot(prim.action_hist.T)
    #plt.show()

    #handler.raw_data = prim.GetOutputs()
    #handler.SaveAnimation(directory=prim.utterance.directory,fname="vid"+str(k))

    sound = prim.GetSoundWave()
    total_energy = np.sum((sound[1:] - sound[:-1])**2)


    print "*"*50
    print "Total energy: %f"%total_energy

    if total_energy > 10**(-3) or NO_THRESHOLD:
        prim.SaveOutputs(fname=str(k))
        #rand.SaveSequence(fname='sequence'+str(k), directory=sequence_dir)
        print "Saved k=%i"%k
        k = k+1

        #energy = (sound[1:] - sound[:-1])**2
        #average_energy = moving_average(energy.reshape((1,-1)), 100)
        #plt.plot(average_energy.flatten())
        #plt.show()

    else:
        failed_attempts= failed_attempts + 1
        print "Utterance below sound threshold: %i"%failed_attempts
    print "*"*50
    #prim.SaveOutputs(fname=str(k))


    
#    save_data = {}
#    save_data['state_hist'] = prim.state_hist
#    save_data['action_hist'] = prim.action_hist
#    np.savez(os.path.join(savedir, 'state_action_'+str(k)), **save_data)
#    print prim.state_hist.shape

    #load_data = np.load(os.path.join(savedir, 'state_action.npz'))
    #plt.plot(load_data['state_hist'].T)
