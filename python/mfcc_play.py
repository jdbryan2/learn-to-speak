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
#dirname = 'data/rand_prim_5sec'
dirname = 'data/batch_random_1_1'
load_fname = 'round588.npz' 
past = 1 
future =1 

from genfigures.plot_functions import tikz_save

#ss= SubspaceDFA()
#ss.LoadPrimitives(fname=load_fname, directory=dirname)
#data = ss.ExtractDataFile("data/test/data0.npz")#, sample_period=sample_period)
#h = ss.EstimateStateHistory(data)
#v = ss.EstimateControlHistory(data)
#plt.figure()
#plt.plot(h.T)
#plt.figure()
#plt.plot(ss.raw_data['state_hist_1'].T)
#plt.show()
#
#plt.figure()
#plt.plot(v.T)
#plt.figure()
#plt.plot(ss.raw_data['action_hist_1'].T)
#plt.show()

#exit()

#v_ = 5

rounds = 1
loops_per_round = 10
utterance_length = 1.0


ss = SubspaceDFA(sample_period=sample_period, past=past, future=future)

# this set of primitives was computed with 1ms sample period
ss.Features = SpectralAcousticFeatures(control_action='action_hist_1', 
                                       control_sample_period=8,
                                       periodsperseg=1) # set feature extractor

dirname = 'data/rand_prim_5sec'
#for k in range(0, 20):
ss.LoadDataDir(directory=dirname, max_index=20)
#ss.ResetDataVars()
    
print "Estimating Primitives"
ss.SubspaceDFA(dim)
#ss.SavePrimitives(directory="data/test/rand_init_prim")
#plt.figure()
state_history = ss.StateHistoryFromFile(dirname+"/data20.npz")
#plt.plot(state_history.T)
#plt.title("State History")

plt.figure()
plt.imshow(np.abs(ss.K),interpolation="none")
tikz_save('/home/jacob/Projects/Dissertation/Doc/tikz/acoustic_input_operator.tikz')
#plt.title("Input operator for acoustic primitive")

plt.figure()
plt.imshow(np.abs(ss.O), interpolation="none")
tikz_save('/home/jacob/Projects/Dissertation/Doc/tikz/acoustic_output_operator.tikz')
#plt.title("Input operator for acoustic primitive")

#plt.figure()
#plt.plot(error[1:])
#plt.title("Magnitude of error in estimate of F")
plt.show()
