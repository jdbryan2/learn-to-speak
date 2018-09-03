# load primitives from file
# control vocal tract 
# generate figures with constant control inputs at various values
# save vocal tract data of utterance to file, and save output audio to file


# for forcing floating point division
#from __future__ import division
import os
import numpy as np
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance 
import Artword as aw
from matplotlib2tikz import save as tikz_save

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    #if data.shape[1] < max_length:
    #    max_length= data.shape[1]

    print max_length

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/8000.))

#import test_params
from test_params import *
primdir = "data/batch_random_12_12/"
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals

rnd = 400 

factor_index = 1
factor = -0.9


control = PrimitiveUtterance( prim_fname=primdir+"round%i"%rnd)
control.utterance = Utterance(directory="data/single_factor_r%i_f%i"%(rnd, factor_index), utterance_length=1.5)

print control.K
initial_art = control._ave[control.Features.pointer[control.Features.control_action]]
#initial_art = np.zeros(initial_art.shape)
#initial_art[0] = 0.2
print initial_art
control.InitializeControl(initial_art=initial_art)

# Setup state variables
current_state = control.current_state
control_action = np.zeros(current_state.shape)

# Perform Control
j = 0 
while control.NotDone():

    ## Step Simulation
    #print control.current_state
    #plt.plot(control.current_state)
    #plt.show()
    #print control.Now()
    #current_state = control.SimulatePeriod(hold=(control.Now()<1000)) #control_action=control_action, control_period=0.)

    #if control.Now() > 8000*j: 
        #factor = j*0.2
        #print factor
        #j += 1

    #control_action = -1.*current_state
    control_action = np.zeros(current_state.shape)
    if control.Now() > 3000 and control.Now() < 6000:
        control_action[factor_index] = factor*current_state[factor_index]
    
    current_state = control.SimulatePeriod(control_action=control_action) #control_action=control_action, control_period=0.)
    #plt.plot(current_state)
    #plt.show()

plt.plot(control.utterance.data['sound_wave'])
plt.show()
plt.plot(control.state_hist.T)
plt.show()
control.SaveOutputs()
#plt.plot(_h)

#savedir = 'data/' + primdir + '/figures/in_out/'
#if not os.path.exists(savedir):
#    os.makedirs(savedir)

_h = control.state_hist

PlotTraces(_h, np.arange(_h.shape[0]), _h.shape[1], sample_period, highlight=0)
plt.grid(True)
#tikz_save(savedir + 'state_history.tikz',
#    figureheight = '\\figureheight',
#    figurewidth = '\\figurewidth')

#plt.show()
plt.figure()
art_hist = control.utterance.data['art_hist']
PlotTraces(art_hist, np.arange(art_hist.shape[0]), art_hist.shape[1], sample_period, highlight=0)
plt.show()

#prim_nums = np.arange(0,dim)
#prim_nums = np.arange(3)
##prim_nums = np.array([test_dim])
#colors = ['b','g','r','c','m','y','k','0.75']
#markers = ['o','o','o','o','x','x','x','x']
#fig = plt.figure()
#for prim_num, c, m in zip(prim_nums,colors,markers):
#    plt.plot(_h[prim_num][:],color=c)
#    plt.plot(control.action_hist[prim_num][0:-1], color=str(0.5+0.2*prim_num/prim_nums[-1]))
#
#
## Remove last element from plot because we didn't perform an action
## after the last update of the state history.
##plt.plot(control.action_hist[test_dim][0:-1], color="0.5")
#plt.show()
#



