# load primitives from file
# control vocal tract 
# generate figures with constant control inputs at various values
# save vocal tract data of utterance to file, and save output audio to file


# for forcing floating point division
#from __future__ import division
import os
import numpy as np
import numpy.linalg as la
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance 
import Artword as aw
from matplotlib2tikz import save as tikz_save

from genfigures.plot_functions import *


#import test_params
from test_params import *
#primdir = dirname+'_prim'
#primdir = "data/batch_zeros_100_10/"
prim_dirname = '../data/batch_random_20_10'
ind= get_last_index(prim_dirname, 'round')
prim_filename = 'round%i.npz'%ind

feedback_dir = "../data/steps_20_10"
feedback_dir = "../data/rand_prim_5sec"

#fname = "round1"
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals

rnd = 411

#load learned feedback params
feedback = np.load(feedback_dir+'/feedback.npz')
#feedback = np.load('data/rand_full/feedback.npz')
gain = feedback['K']
A = feedback['A']
B = feedback['B']
I = np.eye(A.shape[0])


control = PrimitiveUtterance()
control.LoadPrimitives(prim_filename, prim_dirname)
#control = PrimitiveUtterance( prim_fname=primdir+"round%i"%rnd)
control.utterance = Utterance(directory="../data/%i_out"%rnd, utterance_length=3.)
#control.SetUtterance(utterance)

print control.K
initial_action = np.random.random(control._dim)
initial_action = np.ones(control._dim)
initial_art = control.GetControl(initial_action)
print initial_art
#initial_art = np.zeros(initial_art.shape)
#initial_art=np.ones((aw.kArt_muscle.MAX, ))*0.36
#initial_art[0] = 0.2
control.InitializeControl(initial_art=initial_art)

#plt.figure()
#plt.show()

Ts = 1000/(sample_period)


# Setup state variables
current_state = control.current_state
#desired_state = np.zeros(current_state.shape)
desired_state = -0.5*np.ones(current_state.shape)
#desired_state[0] = -0.5

delta_action = np.zeros(initial_action.shape)
current_action = np.copy(initial_action)

# Perform Control
j=0
max_inc = 0.5
#target_action = np.dot(gain, current_state) 
scale = 0.2
while control.NotDone():
    
    #delta_action = np.dot(la.pinv(B), desired_state-np.dot(A, current_state))
    #delta_action = 0.3*np.dot(gain, desired_state-current_state) 
    target_action = scale*np.dot(gain, desired_state-current_state) 
    #print target_action
    delta_action = target_action #- current_action
    print delta_action
    
    #if np.max(np.abs(delta_action)) < 0.0001:
        #scale = 0.4


    #delta_action = np.dot(gain, current_state)

    #delta_action[delta_action>max_inc] = max_inc
    #delta_action[delta_action<-max_inc] = -max_inc

    print delta_action
    current_action += delta_action
    ## Step Simulation
    #print control.current_state
    #plt.plot(control.current_state)
    #plt.show()
    #print control.Now()
    #current_state = control.SimulatePeriod(hold=(control.Now()<1000)) #control_action=control_action, control_period=0.)
    current_state = control.SimulatePeriod(control_action = current_action) #control_action=control_action, control_period=0.)
    #plt.plot(current_state)
    #plt.show()
    j+=1

plt.plot(control.utterance.data['sound_wave'])
plt.show()
plt.plot(control.state_hist[:, :].T)
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

plt.show()
plt.figure()
art_hist = control.utterance.data['art_hist']
PlotTraces(art_hist, np.arange(art_hist.shape[0]), art_hist.shape[1], sample_period, highlight=0)
#plt.plot(art_hist.T)
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



