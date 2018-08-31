
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

from sklearn.neighbors import NearestNeighbors

import pylab as plt


FRESH_LOAD = False
directory = "data/rand_steps"
handler = DataHandler()

index_list = handler.GetIndexList(directory=directory)

Y = np.array([])

if FRESH_LOAD:
    for index in index_list:
        handler.LoadDataDir(directory=directory, min_index=index, max_index=index)
        sound = handler.raw_data['sound_wave'][0]

        #plt.figure()
        #plt.plot(sound)

        y, e = MFCC(sound, ncoeffs=13, nfilters=26, nfft=512, nperseg=3*160,
                    noverlap=3*160-80, low_freq=0)#133.3)

        if Y.size == 0:
            Y = y
        else:
            Y = np.append(Y, y, axis=0)

        print Y.shape
        
        #plt.figure()
        #plt.imshow(np.abs(y[:, 1:].T), aspect=3, interpolation='none')
        #plt.show()
    np.savez(directory+'/mfcc_precomp', Y=Y) 
else:
    data = np.load(directory+'/mfcc_precomp.npz')
    Y = data['Y']

#ref = np.random.randint(Y.shape[0])
#dist = np.zeros(Y.shape[0])
#print 'Computing distances from ', ref
#for k in range(0, Y.shape[0]):
#    dist[k] = np.sqrt(np.sum(np.abs(Y[ref]-Y[k])**2))
#
#plt.plot(dist)
#plt.show()
#from sklearn.neighbors import kneighbors_graph
#A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
nn = NearestNeighbors(radius=1)
nn.fit(Y)
A = nn.radius_neighbors_graph(Y, mode='distance')

