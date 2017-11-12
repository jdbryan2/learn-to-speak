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

ipa_name = "ipa132"
ss.LoadDataFile("data/" + ipa_name + "/data1.npz")

# shift to zero mean and  normalize by standard deviation
data = ((ss._data.T-ss._ave)/ss._std).T
ss.EstimateStateHistory(data)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

prims = [0,1,2]
xp = ss.h[prims[0]][100:]
yp = ss.h[prims[1]][100:]
zp = ss.h[prims[2]][100:]
c = 'r'
m = 'o'
ax.scatter(xp, yp, zp, c=c, marker=m)
ax.set_xlabel('Primitive '+str(prims[0]))
ax.set_ylabel('Primitive '+str(prims[1]))
ax.set_zlabel('Primitive '+str(prims[2]))

plt.show()