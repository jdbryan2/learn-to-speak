# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
#from primitive.SubspaceDFA import SubspaceDFA
from primitive.IncrementalDFA import SubspaceDFA 
from primitive.RandExcite import RandExcite
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures

import Artword as aw

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
#from test_params import *

dim = 8
sample_period = 10 # in milliseconds
sample_period=sample_period*8 # # (*8) -> convert to samples ms
dirname = 'data/test'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
past = 10
future = 10
v_ = 5

rounds = 1
loops_per_round = 10
utterance_length = 1.0




ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)

ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor

for k in range(1, 11):
    ss.LoadDataDir(directory=dirname+"/breathe_rand_init_"+str(k))
    

ss.SubspaceDFA(dim)
ss.SavePrimitives(directory="data/test/breathe_rand_init_prim")


#plt.figure()
state_history = ss.StateHistoryFromFile(dirname+"/breathe_rand_init_1/data1.npz")
plt.plot(state_history.T)
plt.title("State History")
plt.figure()
#plt.plot(error[1:])
plt.title("Magnitude of error in estimate of F")
plt.show()

