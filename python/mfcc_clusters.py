
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
from sklearn.cluster import DBSCAN

import pylab as plt

def mfcc_cluster(directory="data/rand_steps", full_reload = False):

    full_reload = False
    #directory = "data/rand_steps"
    handler = DataHandler()


    Y = np.array([])

    if not full_reload and os.path.exists(directory+'/mfcc_precomp.npz'):
        data = np.load(directory+'/mfcc_precomp.npz')
        Y = data['Y']

    else:
        index_list = handler.GetIndexList(directory=directory)

        print "Loading data from: " + directory
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


    # trim off first mfcc because it's mostly just energy
    Y = Y[:, 1:]

    nn = NearestNeighbors(radius=1)
    nn.fit(Y)
    A = nn.radius_neighbors_graph(Y, mode='distance')

    print 'Clustering'
    db = DBSCAN(eps=0.3, min_samples=5).fit(Y)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    print('Estimated number of clusters: %d' % n_clusters_)
    return Y, labels

if __name__ == '__main__':
    Y, L = mfcc_cluster()
#    Y_ = np.copy(Y[labels==1, :])
#    for k in range(20, n_clusters_):
#        Y_ = np.append(Y_, Y[labels==k, :], axis=0)
#        #plt.figure()
#        #plt.imshow(Y[labels==k, :].T, interpolation=None, aspect=np.sum(labels==k)/12.)
#        #plt.title(np.sum(labels==k))
#        #plt.show()
#
#    plt.imshow(Y_[:, :].T, interpolation=None, aspect=Y_.shape[0]/12.)
#    plt.show()


