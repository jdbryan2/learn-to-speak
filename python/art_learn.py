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
from test_params import *

loops = 10
utterance_length = 1.0
full_utterance = loops*utterance_length

rando = RandExcite(dirname=dirname+"_"+str(loops), 
                   utterance_length=utterance_length,
                   initial_art=np.random.random((aw.kArt_muscle.MAX, )), 
                   max_increment=0.3, min_increment=0.01, max_delta_target=0.2)

#rando.InitializeAll()


dim = 8
sample_period = 10*8 # (*8) -> ms


ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)

ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
#ss.SetFeatures(SpectralAcousticFeatures)

error = np.zeros(50)
for k in range(error.size):
    
    print "#"*20
    print "Round " + str(k+1)
    print "#"*20
    ss.GenerateData(rando, loops)

    ss.SubspaceDFA(dim)

    if k>0:
        error[k] = np.sum(np.sum(np.abs(F-ss.F)**2))
        print "Update Delta: " + str(error[k])

    F = np.copy(ss.F)

    #plt.figure()
    #plt.imshow(np.abs(ss.F))
    #plt.figure()
    #plt.imshow(np.abs(ss.O))
    #plt.figure()
    #plt.imshow(np.abs(ss.K))
    #plt.show()

#plt.figure()
state_history = ss.StateHistoryFromFile(rando.directory+"data1.npz")
plt.plot(state_history.T)
plt.figure()
plt.plot(error[1:])
plt.show()

