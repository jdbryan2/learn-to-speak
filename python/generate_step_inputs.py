
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

#def get_last_index(directory):
#    index_list = []  # using a list for simplicity
#    if os.path.exists(directory):
#        for filename in os.listdir(directory):
#            if filename.startswith('data') and filename.endswith(".npz"):
#                index = filter(str.isdigit, filename)
#                if len(index) > 0:
#                    index_list.append(int(index))
#
#    if len(index_list):
#        return max(index_list)
#    else:
#        return 0
    
loops = 10 
utterance_length = 0.5 #10.0
#full_utterance = loops*utterance_length

savedir = 'data/rand_steps_threshold2'
#savedir = 'data/rand_full'

#prim_filename = 'round411'
#prim_dirname = 'data/batch_random_12_12'

#prim_filename = 'primitives.npz'
#prim_dirname = 'data/art3D'
prim_dirname = 'data/batch_random_20_10'
ind= get_last_index(prim_dirname, 'round')
prim_filename = 'round%i.npz'%ind
#full_filename = os.path.join(prim_dirname, prim_filename)

#print true_dim
#exit()


loop_start = get_last_index(savedir)+1
print loop_start

prim = PrimitiveUtterance()
prim.LoadPrimitives(prim_filename, prim_dirname)

true_dim = prim.K.shape[0]
dim = 3

#for k in range(loop_start, loop_start+loops):
#for k in range(loops):

k = loop_start
failed_attempts = 0
while k < loop_start + loops:

    # random steps
    initial_control = np.zeros(true_dim)
    initial_control[:dim] = np.random.random(dim)*2. - 1.
    end_control = np.zeros(true_dim)
    end_control[:dim] = np.random.random(dim)*2 - 1.

    prim.utterance = Utterance(directory = savedir, 
                               utterance_length=utterance_length, 
                               loops=loops,
                               addDTS=False)
                               #initial_art = prim.GetControlMean(),



    # compute sample period in seconds
    sample_period = 1./8000 #prim.control_period/prim.utterance.sample_freq

    # setup action sequence
    rand = ActionSequence(dim=dim,
                          initial_action=initial_control,
                          sample_period=sample_period,
                          random=False,
                          min_increment=0.1, # 20*sample_period, 
                          max_increment=0.1, # 20*sample_period,
                          max_delta_target=0.8)

    # set intial and final targets
    for factor in range(dim):
        #print factor, initial_control[factor], end_control[factor]
        rand.SetManualTarget(factor, initial_control[factor], 0.)
        rand.SetManualTarget(factor, initial_control[factor], 0.1)
        rand.SetManualTarget(factor, end_control[factor], 0.2) # need to look at effect of transition time too
        rand.SetManualTarget(factor, end_control[factor], 1.0)

    prim.InitializeControl(initial_art = prim.GetControl(rand.GetAction(time=0.0)))

    handler = DataHandler()
    handler.params = prim.GetParams()


    while prim.NotDone():
        action = rand.GetAction(prim.NowSecondsLooped())
        #print action
        #print prim.NowSecondsLooped()
        prim.SimulatePeriod(control_action=action)

    # debug outputs, plot state, actions, and sound
    plt.figure()
    plt.plot(prim.state_hist.T)
    plt.figure()
    plt.plot(prim.action_hist.T)
    plt.figure()
    plt.plot(prim.utterance.data['sound_wave'])
    plt.show()

    ## manually save state action history
    #save_data = {}
    #save_data['state_hist'] = prim.state_hist
    #save_data['action_hist'] = prim.action_hist
    #np.savez(os.path.join(savedir, 'state_action_'+str(k)), **save_data)
    #print prim.state_hist.shape
    
    sound = prim.GetSoundWave()
    energy = np.sum((sound[1:] - sound[:-1])**2)/sound.size

    print "*"*50
    print "Average energy: %f"%energy
    if energy > 0.002:
        prim.SaveOutputs(fname=str(k))
        print "Saved k=%i"%k
        k = k+1

    else:
        failed_attempts= failed_attempts + 1
        print "Utterance below sound threshold: %i"%failed_attempts
    print "*"*50

    #load_data = np.load(os.path.join(savedir, 'state_action.npz'))
    #plt.plot(load_data['state_hist'].T)
