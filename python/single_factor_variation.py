
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

from genfigures.plot_functions import *

import pylab as plt

DEBUG = False

loops = 59
utterance_length = 1. #10.0
#full_utterance = loops*utterance_length

savedir = 'data/covariance/D'

#prim_filename = 'round411'
prim_dirname = 'data/batch_random_12_12'

ind= get_last_index(prim_dirname, 'round')
prim_filename = 'round%i.npz'%ind


# load 3D primitives from file
##################################################
prim = PrimitiveUtterance()
prim.LoadPrimitives(fname=prim_filename, directory = prim_dirname)
dim = prim.K.shape[0]


for act_dim in range(dim):

    loop_start = get_last_index(savedir+str(act_dim))+1

    for k in range(loop_start, loop_start+loops):
        print "Dimension: ", act_dim, " Loop: ", k

        initial_control = np.zeros(dim)
        initial_control[act_dim] = np.random.random(1)*2. - 1.
        #print initial_control
        end_control = np.zeros(dim)
        #end_control[:dim] = np.random.random(dim)*2 - 1.

        prim.utterance = Utterance(directory = savedir+str(act_dim), 
                                   utterance_length=utterance_length, 
                                   loops=loops,
                                   addDTS=False)

        rand = ActionSequence(dim=dim,
                              initial_action=initial_control,
                              sample_period=1./8000, #sample_period,
                              random=True, # following parameters are only used if set to true
                              min_increment=0.1, # 20*sample_period, 
                              max_increment=0.1, # 20*sample_period,
                              max_delta_target=0.8)

        # everything but the factor of iterest is set to constant 0
        for factor in range(dim):
            if factor != act_dim:
                #print factor, initial_control[factor], end_control[factor]
                rand.SetManualTarget(factor, 0.,  0.) #primitive, target, time,
                rand.SetManualTarget(factor, 0., 1.0)

        prim.InitializeControl(initial_art = prim.GetControl(rand.GetAction(time=0.0)))

        handler = DataHandler()
        handler.params = prim.GetParams()


        while prim.NotDone():
            action = rand.GetAction(prim.NowSecondsLooped())
            #print action
            #print prim.NowSecondsLooped()
            prim.SimulatePeriod(control_action=action)

        #plt.figure()
        #plt.plot(prim.state_hist.T)
        #plt.figure()
        #plt.plot(prim.action_hist.T)
        #plt.show()

        #save_data = {}
        #save_data['state_hist'] = prim.state_hist
        #save_data['action_hist'] = prim.action_hist
        #np.savez(os.path.join(savedir+str(act_dim), 'state_action_'+str(k)), **save_data)
        #print prim.state_hist.shape
        prim.SaveOutputs(str(k))

        if DEBUG == True:
            plt.plot(prim.state_hist.T)
            plt.figure()
            plt.plot(prim.action_hist.T)
            plt.show()
