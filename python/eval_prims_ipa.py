# Script for evaluating primitives by looking at distributions of
# ipa utterances in the low-dimensional space.

from primitive.SubspaceDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
from test_params import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ss = SubspaceDFA()
ss.LoadPrimitives(load_fname)
# Set feature extractor to be same as one that we used to learn primitives
# TODO: This should be made more general so that we don't have to keep track
#       of which extractor was being used
ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor

# Select which three primitives to view
prims = [0,1,2]
# ipa logs that will be plotted
ipa_nums = [132,134,140,142,301,304,305,316]
colors = ['b','g','r','c','m','y','k','w']
markers = ['o','o','o','o','x','x','x','x',]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for ipa_num, c, m in zip(ipa_nums,colors,markers):
    
    ss.LoadDataFile("data/ipa" + str(ipa_num) + "/data1.npz")

    # shift to zero mean and  normalize by standard deviation
    data = ((ss._data.T-ss._ave)/ss._std).T
    ss.EstimateStateHistory(data)

    xp = ss.h[prims[0]][100:]
    yp = ss.h[prims[1]][100:]
    zp = ss.h[prims[2]][100:]
    ax.scatter(xp, yp, zp, color=c, marker=m)

ax.set_xlabel('Primitive '+str(prims[0]))
ax.set_ylabel('Primitive '+str(prims[1]))
ax.set_zlabel('Primitive '+str(prims[2]))

plt.show()
