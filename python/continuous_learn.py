# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
from primitive.SubspaceDFA import SubspaceDFA
from primitive.IncrementalDFA import SubspaceDFA as IncrementalDFA
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures
from primitive.RandExcite import RandExcite
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

################################################################################
loops = 1000
utterance_length = 1.0
full_utterance = loops*utterance_length

rando = RandExcite(dirname="data/continuous_"+str(loops), 
                   method="gesture",
                   loops=loops,
                   utterance_length=utterance_length,
                   max_increment=0.3, min_increment=0.02, max_delta_target=0.2)
                   #initial_art=np.zeros((aw.kArt_muscle.MAX, )),

################################################################################

inc_ss = IncrementalDFA(sample_period=sample_period, past=past, future=future)
#inc_ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)
#ss.LoadDataDir(dirname)
#ss.ConvertData(sample_period)
inc_ss.Features = ArtFeatures(tubes=inc_ss.tubes) # set feature extractor
#inc_ss.Features = SpectralAcousticFeatures(tubes=inc_ss.tubes,
                                       #sample_period=sample_period) # set feature extractor

inc_ss.GenerateData(rando, loops)
inc_ss.SubspaceDFA(dim)


inc_ss.SavePrimitives(rando.directory+'primitives')


#ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)
#ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
#ss.LoadDataDir(rando.directory, verbose=True)
#
#inc_ss.EstimateStateHistory(ss._data) # estimate state history off other data
#
#plt.figure()
#plt.plot(inc_ss.h.T)
#plt.show()
#
