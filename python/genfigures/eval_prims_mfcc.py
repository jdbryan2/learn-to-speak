# Script for evaluating primitives by looking at distributions of
# ipa utterances in the low-dimensional space.

from primitive.IncrementalDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
from test_params import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dim = 8
sample_period_ms = 20 # in milliseconds
#sample_period_ms = 1 # in milliseconds
sample_period=sample_period_ms*8 # # (*8) -> convert to samples ms
dirname0 = 'data/batch_random_1_1'
dirname1 = 'data/mfcc_random_1_1_20'
#load_fname = 'round550.npz' # class points toward 'data/' already, just need the rest of the path

s1 = SubspaceDFA() 
s1.LoadPrimitives(fname='round1005.npz', directory = dirname1)
print s1._downpointer_fname,  s1._downpointer_directory

s0 = SubspaceDFA()
#s0.LoadPrimitives(fname='round550.npz', directory = dirname0)
s0.LoadPrimitives(fname=s1._downpointer_fname, directory = s1._downpointer_directory)




# Select which three primitives to view
prims = [0,1,2]
# ipa logs that will be plotted
ipa_nums = [101, 132,134,140,142,301,304,305,316]
colors = ['b','g','r','c','m','y','k','0.75']
markers = ['o','o','o','o','x','x','x','x',]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

H = np.array([])

for ipa_num, c, m in zip(ipa_nums,colors,markers):
    
    # TODO: Make a member function that will clear this
    # Required to make sure ss doesn't append files together
    s0._data = np.array([])
    #ss.LoadDataFile("data/ipa" + str(ipa_num) + "/data1.npz")#, sample_period=sample_period)

    # shift to zero mean and  normalize by standard deviation
    #data = ((ss._data.T-ss._ave)/ss._std).T
    #ss.EstimateStateHistory(data)
    data = s0.ExtractDataFile("data/utterances/ipa" + str(ipa_num) + "/data1.npz")
    h = s0.EstimateStateHistory(data)

    v = s0.EstimateControlHistory(data)

    s1.ExtractDataFile("data/utterances/ipa" + str(ipa_num) + "/data1.npz", raw=True)
    s1.raw_data["state_hist_1"] = h[:, 20:]
    s1.raw_data["action_hist_1"] = v[:, 20:]

    data1 = s1.Features.Extract(s1.raw_data, sample_period=s1.sample_period)

    h1 = s1.EstimateStateHistory(data1)
    v1 = s1.EstimateControlHistory(data1)

    


    xp = h1[prims[0]][past:]
    yp = h1[prims[1]][past:]
    zp = h1[prims[2]][past:]
    ax.plot(xp, yp, zp, color=c, marker=m)
    if H.size == 0:
        H = h1[:3, :].T
    else:
        H = np.append(H,h1[:3, :].T,axis=0)

ax.set_xlabel('Primitive '+str(prims[0]))
ax.set_ylabel('Primitive '+str(prims[1]))
ax.set_zlabel('Primitive '+str(prims[2]))

plt.show()

fig = plt.figure()
plt.plot(H)
plt.show()
