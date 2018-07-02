
from primitive.IncrementalDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
import matplotlib.pyplot as plt
import numpy as np

from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
import Artword as aw

RECOMPUTE_PRIMITIVES = True

dim = 3
primitive_dir = '../data/art3D'

# if new primitives must be computed:
##################################################
if RECOMPUTE_PRIMITIVES:
    # load up primitives from latest round
    learn = SubspaceDFA()
    learn.LoadPrimitives(fname='round411.npz', directory = '../data/batch_random_12_12')
    
    # compute 3D version of primitives
    learn.SubspaceDFA(dim) # argument is dimension of primitive space
    
    # save to new primitive file
    learn.SavePrimitives(directory = primitive_dir)

# load 3D primitives from file
##################################################
prim = PrimitiveUtterance()
prim.LoadPrimitives(fname='primitives.npz', directory = primitive_dir)

utterance = Utterance(directory='../data/utterances/seq3D_1',
                      utterance_length=2.,
                      addDTS=False)

prim.SetUtterance(utterance)

# load control sequence
##################################################
control_input = np.genfromtxt('../control_sequences/dx.csv', delimiter=",", skip_header=1)

# get initial art based on first control_input entry
##################################################
if control_input[0, 0] == 0.:
    initial_art = prim.GetControl(control_input[0, 1:dim+1])
else:
    initial_art = prim.GetControlMean()

prim.InitializeControl(initial_art=initial_art)

# simulate utterance
##################################################
target_index = 0 
prev_target = initial_art
prev_time = 0.
next_target = control_input[target_index, 1:dim+1]
next_time = control_input[target_index, 0]

while prim.NotDone():
    _now = prim.NowSecondsLooped()
    if control_input[target_index, 0] < _now:
        prev_target = control_input[target_index, 1:dim+1]
        prev_time = control_input[target_index, 0]
        target_index += 1
        next_target = control_input[target_index, 1:dim+1]
        next_time = control_input[target_index, 0]


    #control_action = -1.*current_state
    control_action = np.zeros(prim.current_state.shape)
    for k in range(control_action.size):
        control_action[k] = np.interp(_now,
                                      [prev_time, next_time], 
                                      [prev_target[k], next_target[k]])
    

    current_state = prim.SimulatePeriod(control_action=control_action) 


# save utterance to file
##################################################
prim.SaveOutputs()

