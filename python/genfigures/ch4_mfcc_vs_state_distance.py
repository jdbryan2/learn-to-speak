
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
from features.functions import MFCC
import pylab as plt

#from sklearn.neighbors import NearestNeighbors
#from sklearn.cluster import DBSCAN

from plot_functions import *

from features.BaseFeatures import moving_average as moving_average

handler = DataHandler()

directory = '../data/steps_threshold_20_10'

Y = np.array([])

index_list = handler.GetIndexList(directory=directory)

print "Loading data from: " + directory
E = np.array([])
M = np.array([])
for index in index_list[:100]:
    handler.LoadDataDir(directory=directory, min_index=index, max_index=index)
    state = handler.raw_data['state_hist_1']
    #state = handler.raw_data['action_hist_1']
    sound = handler.raw_data['sound_wave'][0]

    # find states where the energy is above average
    energy = (sound[1:] - sound[:-1])**2
    # average window corresponds to history length (20ms)
    energy_threshold = np.mean(energy)
    average_energy = moving_average(energy.reshape((1,-1)), 20*8).flatten()
    #print average_energy.shape
    average_energy = average_energy[::5*8]
    average_energy = np.append(np.zeros(4), average_energy)
    energetic = np.where(average_energy>energy_threshold) # indeces of energetic states

    # compute mfcc 
    mfcc, e = MFCC(sound, ncoeffs=13, nfilters=26, nfft=512, nperseg=8*20,
                noverlap=8*(20-5), low_freq=0)#133.3)
    mfcc = mfcc.T
    mfcc = np.append(np.zeros((13, 3)), mfcc, axis=1)
    energetic = np.where(average_energy>energy_threshold) # indeces of energetic states

    if E.size == 0:
        E = state[:, energetic].reshape((10, -1))
        M = mfcc[:, energetic].reshape((13, -1))
    else: 
        E = np.append(E, state[:, energetic].reshape((10, -1)), axis=1)
        M = np.append(M, mfcc[:, energetic].reshape((13, -1)), axis=1)


    #plt.plot(average_energy)
    ##plt.show()
    #plt.hold(True)
    #plt.show(block=False)

    #plt.figure()
    #plt.imshow(np.abs(y[:, 1:].T), aspect=3, interpolation='none')
    #plt.show()

    

CE = np.cov(E, M, rowvar=True)
for k in range(CE.shape[1]-10):
    plt.plot(CE[k+10, 10:])

plt.show()
