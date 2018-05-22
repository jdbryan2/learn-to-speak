import numpy as np
import pylab as plt
#from primitive.SubspaceDFA import SubspaceDFA
from primitive.IncrementalDFA import SubspaceDFA 
from primitive.RandExcite import RandExcite
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance


dim = 8
sample_period_ms = 20 # in milliseconds
sample_period=sample_period_ms*8 # # (*8) -> convert to samples ms
dirname = 'data/rand_prim_5sec'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
past = 1 
future =1 
#v_ = 5

rounds = 1
loops_per_round = 10
utterance_length = 1.0


ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)

# this set of primitives was computed with 1ms sample period
ss.Features = SpectralAcousticFeatures(control_action='action_hist_1', 
                                       control_sample_period=8,
                                       periodsperseg=1) # set feature extractor

for k in range(0, 20):
    ss.LoadDataDir(directory=dirname, min_index=k, max_index=k)
    ss.ResetDataVars()
    

ss.SubspaceDFA(dim)
#ss.SavePrimitives(directory="data/test/rand_init_prim")
#plt.figure()
state_history = ss.StateHistoryFromFile(dirname+"/data20.npz")
plt.plot(state_history.T)
plt.title("State History")
#plt.figure()
#plt.plot(error[1:])
#plt.title("Magnitude of error in estimate of F")
plt.show()
