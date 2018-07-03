from primitive.IncrementalDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
import matplotlib.pyplot as plt
import numpy as np

from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
from primitive.ActionSequence import ActionSequence
from primitive.DataHandler import DataHandler
#import Artword as aw

import scikits.talkbox.features as tb


dim = 3
primitive_dir = 'data/art3D'

# Load up goal output
dh = DataHandler(directory='data/utterances/seq3D_1')
dh.LoadDataDir()
sound_wave = dh.raw_data['sound_wave']
baseline_mfcc = tb.mfcc(sound_wave, nwin=256, nfft=512, nceps=13)
#plt.imshow(baseline_mfcc[0].T)
#plt.show()

# load 3D primitives from file
##################################################
prim = PrimitiveUtterance()
prim.LoadPrimitives(fname='primitives.npz', directory = primitive_dir)

utterance = Utterance(directory='data/utterances/seq3D_1_immitate',
                      utterance_length=2.,
                      addDTS=False)

prim.SetUtterance(utterance)

# load up control sequence
##################################################
# TODO: Add loading function for action sequence class.

# load control sequence file
control_input = np.genfromtxt('control_sequences/dx.csv', delimiter=",", skip_header=1)

# pass sequence into ActionSequence Class
# not sure why 'sample_period' is needed at all. I think it may be from an old version of random excitation
act_seq = ActionSequence(dim=dim, initial_action=np.zeros(dim), sample_period=1./8000, random=False)

for k in range(control_input.shape[0]):
    for n in range(dim):
        #                       dim, target, time
        if k < 2:
            act_seq.SetManualTarget(n, control_input[k, n+1], control_input[k, 0])
        else:   
            pass

# get initial art based on first control_input entry
##################################################
initial_art = prim.GetControl(act_seq.GetAction(0.))

prim.InitializeControl(initial_art=initial_art)


while prim.NotDone():
    action = act_seq.GetAction(prim.NowSecondsLooped())
    print prim.NowSecondsLooped()
    prim.SimulatePeriod(control_action=action)


y = tb.mfcc(prim.utterance.data['sound_wave'], nwin=256, nfft=512, nceps=13) 

#plt.imshow(np.abs(y[0].T))
#plt.show()

error = np.abs(y[0]-baseline_mfcc[0]).T
plt.imshow(error)
plt.show()

print "Total error: ", np.sum(error)
prim.SaveOutputs()
