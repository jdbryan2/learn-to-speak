from primitive.IncrementalDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
from primitive.ActionSequence import ActionSequence
from primitive.DataHandler import DataHandler
#import Artword as aw

import scikits.talkbox.features as tb

def get_acoustic_output(prim, sequence):
    print "Running simulation"
    initial_art = prim.GetControl(sequence.GetAction(0.))

    #prim.utterance.Reset()
    prim.InitializeControl(initial_art=initial_art)


    while prim.NotDone():
        action = sequence.GetAction(prim.NowSecondsLooped())
        prim.SimulatePeriod(control_action=action)

    y = tb.mfcc(prim.utterance.data['sound_wave'], nwin=256, nfft=512, nceps=13) 
    return y[0].T
    # return mfcc output


dim = 3
primitive_dir = 'data/art3D'

# Load up goal output
dh = DataHandler(directory='data/utterances/seq3D_1')
dh.LoadDataDir()
sound_wave = dh.raw_data['sound_wave']
baseline_mfcc = (tb.mfcc(sound_wave, nwin=256, nfft=512, nceps=13))[0].T
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

MANUAL_LOAD = True
act_seq = ActionSequence(dim=dim, initial_action=np.zeros(dim), sample_period=1./8000, random=False)

if MANUAL_LOAD == True:
    #TODO: Add functionality to load from CSV into ActionSequence class
    # load control sequence file
    control_input = np.genfromtxt('control_sequences/dx.csv', delimiter=",", skip_header=1)

    # pass sequence into ActionSequence Class
    # not sure why 'sample_period' is needed at all. I think it may be from an old version of random excitation

    for k in range(control_input.shape[0]):
        for n in range(dim):
            #                       dim, target, time
            #if k < 2:
                act_seq.SetManualTarget(n, control_input[k, n+1], control_input[k, 0])
            #else:   
                #pass

else:
    act_seq.LoadSequence(directory='control_sequences/learn')


# Generate action sequence and get output
##################################################

# size of time step between control input commands 

# initialize ActionSequence class
random_sequence = ActionSequence(dim=dim, initial_action=np.zeros(dim), sample_period=1./8000, random=False)

step_size = 0.2
global_time = 1*step_size
global_sequence = ActionSequence(dim=dim, intial_action=np.zeros(dim), sample_period=1./8000, random=False)

# callback function for optimizer
def error_callback(target):
    global global_sequence

    for n, x in enumerate(target):
        global_sequence.SetManualTarget(n, x, global_time)

    print global_sequence.manual_targets
    y = get_acoustic_output(prim, global_sequence)

    error = np.sum(np.abs(y-baseline_mfcc)**2)
    print "Error: ", error
    return error

x0 = np.zeros(dim)
while global_time <= 2.0:
    print "Time step: ", global_time
    #print global_sequence.manual_targets
    res = minimize(error_callback, x0, method='Nelder-Mead', tol=1e-2, 
                   options={'initial_simplex':[[1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.3, -0.3, -0.3]]})
    if res.status == 0:
        x0 = res.x
        global_time += step_size

        global_sequence.SaveSequence(fname=str(global_time), directory='control_sequences/learn')


## loop over time steps and pass control targets to ActionSequence class
#for time in np.arange(step_size, 2.0, step_size):
#    for k in range(dim):
#        # set random action sequence
#        random_sequence.SetManualTarget(k, np.random.random()/2., time)

#output1 = get_acoustic_output(prim, random_sequence)
output1 = get_acoustic_output(prim, global_sequence)

output2 = get_acoustic_output(prim, act_seq)


error = np.abs(output2-baseline_mfcc)
plt.figure()
plt.imshow(output1)
plt.figure()
plt.imshow(output2)
plt.figure()
plt.imshow(error)
plt.show()

print "Total error: ", np.sum(error)
prim.SaveOutputs()
