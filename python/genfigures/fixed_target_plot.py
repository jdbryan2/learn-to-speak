
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

DEBUG = False

def get_last_index(directory):
    index_list = []  # using a list for simplicity
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith('state_action') and filename.endswith(".npz"):
                index = filter(str.isdigit, filename)
                if len(index) > 0:
                    index_list.append(int(index))

    if len(index_list):
        return max(index_list)
    else:
        return 0
    
loops = 200 
utterance_length = 1. #10.0
#full_utterance = loops*utterance_length

savedir = '../data/fixed_target/D1'


dim = 3

# load 3D primitives from file
##################################################
prim = PrimitiveUtterance()

loop_end = get_last_index(savedir)
#loop_end = 20
state = [[] for k in range(dim)]
action = [[] for k in range(dim)]
for k in range(1, loop_end):
    load_data = np.load(os.path.join(savedir, 'state_action_'+str(k)+'.npz'))
    for n in range(dim):
        if len(state[n]) == 0:
            state[n] = load_data['state_hist'][n, :].reshape((1, -1))
            action[n] = load_data['action_hist'][n, :].reshape((1, -1))
        else:
            state[n] = np.append(state[n], load_data['state_hist'][n, :].reshape((1, -1)), axis=0)
            action[n] = np.append(action[n], load_data['action_hist'][n, :].reshape((1, -1)), axis=0)

    

for d in range(dim):
    plt.figure()
    plt.plot(state[d].T)
    plt.title('State '+str(d))
    plt.figure()
    plt.plot(action[d].T)
    plt.title('Action ' + str(d))
plt.show()
