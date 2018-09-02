
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

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from plot_functions import *

handler = DataHandler()

directory = '../data/rand_steps'

Y = np.array([])

index_list = handler.GetIndexList(directory=directory)

print "Loading data from: " + directory
E = np.array([])
for index in index_list:
    handler.LoadDataDir(directory=directory, min_index=index, max_index=index)
    sound = handler.raw_data['sound_wave'][0]

    #plt.figure()
    #plt.plot(sound)

    #y, e = MFCC(sound, ncoeffs=13, nfilters=26, nfft=512, nperseg=3*160,
    #            noverlap=3*160-80, low_freq=0)#133.3)
    E = np.append(E, np.sum((sound[1:] - sound[:-1])**2)/sound.size)
    if E[-1] > 0.002:
        plt.figure()
        PlotTraces(handler.raw_data['action_hist_1'], np.arange(3), handler.raw_data['action_hist_1'].shape[1], 12*8)
    

    
    #plt.figure()
    #plt.imshow(np.abs(y[:, 1:].T), aspect=3, interpolation='none')
    #plt.show()

print 1.*np.sum(E>1)/np.sum(E<1)
#plt.plot(E)
plt.show()
