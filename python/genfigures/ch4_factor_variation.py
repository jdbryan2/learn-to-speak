
import os
import time
import numpy as np
import config
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

savedir = '../data/covariance_20_10/D'

#prim_filename = 'round411'
#prim_dirname = 'data/batch_random_12_12'
#
#ind= get_last_index(prim_dirname, 'round')
#prim_filename = 'round%i.npz'%ind


# load 3D primitives from file
##################################################
dim = 10

handler = DataHandler()

#dim = 

tubes = config.TUBES['all']
action_cov = np.zeros((dim, tubes.size))
state_cov = np.zeros((dim, dim))

for act_dim in range(dim):
    handler.LoadDataDir(directory=savedir+str(act_dim), min_index=0, max_index=60)
    state = handler.raw_data['state_hist_1']
    action = handler.raw_data['action_hist_1']
    area = handler.raw_data['area_function']

    # interpolate action up to same sample freq as area function
    control_time = np.arange(0., action.shape[1])/action.shape[1]
    area_time = np.arange(0., area.shape[1])/area.shape[1]
    action = np.interp(area_time, control_time, action[act_dim, :])

    up_state = np.zeros((dim, area_time.size))
    for d in range(dim): 
        up_state[d, :] = np.interp(area_time, control_time, state[d, :])

    # subtract the mean
    area = (area.T-np.mean(area, axis=1)).T
    action = (action.T-np.mean(action)).T
    up_state = (up_state.T-np.mean(up_state, axis=1)).T

    for k in range(tubes.size):
        action_cov[act_dim, k] = np.dot(area[tubes[k], :], action)/(action.size-1)
    for d in range(dim):
        state_cov[act_dim, d] = np.dot(up_state[d, :], action)/(action.size-1)

    #plt.plot(state_cov[act_dim, :])
    #plt.show()
    #plt.plot(action_cov[act_dim, :])
    #plt.show()

    
np.savez('covariance.npz', action_cov=action_cov, state_cov=state_cov)
plt.figure()
plt.plot(action_cov)
plt.figure()
plt.plot(state_cov)
plt.show()
    #plt.plot(handler.raw_data['state_hist_1'].T)
    #plt.show()
    
