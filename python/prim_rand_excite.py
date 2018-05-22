
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

def get_last_index(directory):
    index_list = []  # using a list for simplicity
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith('data') and filename.endswith(".npz"):
                index = filter(str.isdigit, filename)
                if len(index) > 0:
                    index_list.append(int(index))

    if len(index_list):
        return max(index_list)
    else:
        return 0
    
loops = 10 
utterance_length = 5. #10.0
#full_utterance = loops*utterance_length

savedir = 'data/rand_steps_full'
savedir = 'data/rand_prim_5sec'

prim_filename = 'round550'
prim_dirname = 'data/batch_random_1_1'
full_filename = os.path.join(prim_dirname, prim_filename)

true_dim = 10#prim._dim 


loop_start = get_last_index(savedir)+1
print loop_start

for k in range(loop_start, loop_start+loops):
    prim = PrimitiveUtterance()
    prim.LoadPrimitives(full_filename)

    initial_control = np.zeros(prim._dim)
    initial_control[:true_dim] = np.random.random(true_dim)*2. - 1.
    end_control = np.zeros(prim._dim)
    end_control[:true_dim] = np.random.random(true_dim)*2 - 1.

    prim.utterance = Utterance(directory = savedir, 
                               utterance_length=utterance_length, 
                               loops=loops,
                               addDTS=False)
                               #initial_art = prim.GetControlMean(),



# compute sample period in seconds
    sample_period = prim.control_period/prim.utterance.sample_freq

# setup action sequence
    rand = ActionSequence(dim=prim._dim,
                          initial_action=initial_control,
                          sample_period=sample_period,
                          random=True,
                          min_increment=0.1, # 20*sample_period, 
                          max_increment=0.3, # 20*sample_period,
                          max_delta_target=0.8)

    # all factors over 3 to be constant zero
    #for factor in range(prim._dim):
    #    rand.SetManualTarget(factor, initial_control[factor], 0.)
    #    rand.SetManualTarget(factor, initial_control[factor], 0.25)
    #    rand.SetManualTarget(factor, end_control[factor], 0.5)
    #    rand.SetManualTarget(factor, end_control[factor], 1.0)

    prim.InitializeControl(initial_art = prim.GetControl(rand.GetAction(time=0.0)))

    handler = DataHandler()
    handler.params = prim.GetParams()


    prim._act = rand
    prim.Simulate()
    #while prim.NotDone():
    #    action = rand.GetAction(prim.NowSecondsLooped())
    #    #print action
    #    #print prim.NowSecondsLooped()
    #    prim.SimulatePeriod(control_action=action)

    plt.figure()
    plt.plot(prim.state_hist.T)
    plt.figure()
    plt.plot(prim.action_hist.T)
    plt.show()

    handler.raw_data = prim.GetOutputs()
    #handler.SaveAnimation(directory=prim.utterance.directory,fname="vid"+str(k))
    #prim.SaveOutputs(fname=str(k))


    
#    save_data = {}
#    save_data['state_hist'] = prim.state_hist
#    save_data['action_hist'] = prim.action_hist
#    np.savez(os.path.join(savedir, 'state_action_'+str(k)), **save_data)
#    print prim.state_hist.shape

    #load_data = np.load(os.path.join(savedir, 'state_action.npz'))
    #plt.plot(load_data['state_hist'].T)
