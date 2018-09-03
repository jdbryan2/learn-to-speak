# Script for evaluating primitives by looking at distributions of
# ipa utterances in the low-dimensional space.

from primitive.IncrementalDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
from test_params import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mfcc_clusters import *
from plot_functions import *

def Plot3DSamples(index_list):
# Select which three primitives to view

# ipa logs that will be plotted
    colors = ['b','g','r','c','m','y','k','0.75']
    markers = ['o','o','o','o','x','x','x','x',]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    H = np.array([])

    for index, c, m in zip(index_list,colors,markers):
        
        # TODO: Make a member function that will clear this
        # Required to make sure ss doesn't append files together
        ss._data = np.array([])
        #ss.LoadDataFile("data/ipa" + str(ipa_num) + "/data1.npz")#, sample_period=sample_period)

        # shift to zero mean and  normalize by standard deviation
        #data = ((ss._data.T-ss._ave)/ss._std).T
        #ss.EstimateStateHistory(data)
        data = ss.ExtractDataFile(data_dir+"/rand_steps/data%i.npz"%index)
        h = ss.EstimateStateHistory(data)

        xp = h[0][past:]
        yp = h[1][past:]
        zp = h[2][past:]
        ax.plot(xp, yp, zp, color=c, marker=m)
        if H.size == 0:
            H = h[:3, :].T
        else:
            H = np.append(H,h[:3, :].T,axis=0)

    ax.set_xlabel('Primitive 0')
    ax.set_ylabel('Primitive 1')
    ax.set_zlabel('Primitive 2')

    plt.show()

    fig = plt.figure()
    plt.plot(H)
    plt.show()


data_dir = '../data'
#dirname = '../data/batch_random_20_5'
#dirname = '../data/batch_random_1_1'
dirname = data_dir+'/batch_random_12_12'

ind= get_last_index(dirname, 'round')
load_fname = 'round%i.npz'%ind

ss = SubspaceDFA()
ss.LoadPrimitives(fname=load_fname, directory = dirname)
# Set feature extractor to be same as one that we used to learn primitives
#ss.Features = ArtFeatures() # set feature extractor

past = ss._past
future = ss._future
print data_dir+'/rand_steps'
mfcc_data, labels = mfcc_cluster(directory = data_dir+'/rand_steps_threshold', full_reload=False)


clusters = set(labels)

for c in clusters:
    if c != -1:
        time_index, = np.where(labels==c)
        file_index = set(np.floor(time_index/45.).astype('int'))

        print file_index
    Plot3DSamples(file_index)



#fig = plt.figure()
#plt.plot(H)
#plt.show()
