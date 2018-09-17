
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
for index in index_list[:100]:
    handler.LoadDataDir(directory=directory, min_index=index, max_index=index)
    state = handler.raw_data['state_hist_1']
    sound = handler.raw_data['sound_wave'][0]

    energy = (sound[1:] - sound[:-1])**2
    # average window corresponds to history length (20ms)
    average_energy = moving_average(energy.reshape((1,-1)), 20*8).flatten()
    print average_energy.shape
    average_energy = average_energy[::5*8]
    average_energy = np.append(np.zeros(4), average_energy)

    energetic = np.where(average_energy>0.002)

    if E.size == 0:
        E = state[:, energetic].reshape((10, -1))
    else: 
        E = np.append(E, state[:, energetic].reshape((10, -1)), axis=1)


    plt.plot(average_energy)
    #plt.show()
    plt.hold(True)
    plt.show(block=False)

    #plt.figure()
    #plt.imshow(np.abs(y[:, 1:].T), aspect=3, interpolation='none')
    #plt.show()

    
plt.figure()
plt.scatter(E[0, :], E[1, :])
#plt.plot(E)
plt.show()
