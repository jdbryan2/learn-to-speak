# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
from primitive.SubspaceDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures
## pretty sure these aren't used ##
#import scipy.signal as signal
#import numpy.linalg as ln
#import os

#dim = 8
#sample_period = 10
#dirname = 'full_random_500'
#past = 50
#future = 50

# call in all the necessary global variables
from test_params import *

ss = SubspaceDFA()
#ss.LoadDataDir(dirname)
#ss.ConvertData(sample_period)
ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
#ss.Features = SpectralAcousticFeatures(tubes=ss.tubes) # set feature extractor
ss.LoadDataDir(dirname, sample_period=sample_period, verbose=True)
ss.PreprocessData(past, future)
ss.SubspaceDFA(dim)

ss.EstimateStateHistory(ss._data)
plt.plot(ss.h.T)
plt.show()

ss.SavePrimitives(dirname+'/primitives')

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.K[k, :].reshape(ss._past, 88))
#    plt.title('Input: '+str(k))

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.O[:, k].reshape(ss._future, 88), aspect=2)
#    plt.title('Output: '+str(k))

##for k in range(dim):
##    plt.figure();
##    K = ss.K[k,:].reshape(ss._past, 88)
##    for p in range(ss._past):
##        plt.plot(K[p, :], 'b-', alpha=1.*(p+1)/(ss._past+1))
##    plt.title('Input: '+str(k))
##
##for k in range(dim):
##    plt.figure();
##    O = ss.O[:, k].reshape(ss._future, 88)
##    for f in range(ss._future):
##        dat = O[f, :]
##        dat = ((dat.T*ss._std)+ss._ave).T
##
##        plt.plot(dat, 'b-', alpha=1.*(ss._future-f+1)/(ss._future+1))
##    plt.title('Output: '+str(k))
##
##plt.show()

