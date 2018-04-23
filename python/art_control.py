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
#primdir = dirname+'_prim'
primdir = "data/batch/"
fname = "round109"
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals

rnd = 6


control = PrimitiveUtterance( prim_fname="data/batch/round%i"%rnd)
control.utterance = Utterance(directory="data/%i_out"%rnd, utterance_length=5.)
#control.SetUtterance(utterance)

print control.K
#initial_art=np.random.random((aw.kArt_muscle.MAX, ))
#initial_art=np.zeros((aw.kArt_muscle.MAX, ))
initial_art = control._ave[control.Features.pointer[control.Features.control_action]]
print initial_art
initial_art = np.zeros(initial_art.shape)
#initial_art=np.ones((aw.kArt_muscle.MAX, ))*0.36
#initial_art[0] = 0.2
control.InitializeControl(initial_art=initial_art)

plt.figure()
plt.show()

Ts = 1000/(sample_period)

# Good values for individual primitive controllers
Kp = np.array([1/3,1/3,0,0,0,0,0,0])
Ki = np.array([0.5/3,1/3,0,0,0,0,0,0])
Kd = np.array([10/3,5/3,0,0,0,0,0,0])
I_lim = np.array([100,150,0,0,0,0,0,0])

#Integral gain only
Kp = np.array([0.0,0,0,0,0,0,0,0])
Ki = np.array([0.1,0.05,0.2,0,0,0,0,0])
Kd = np.array([0,0,0,0,0,0,0,0])
I_lim = np.array([150,150,150,0,0,0,0,0])
# leaky integrator constant
a = 1
#Ki[0]=0
#Ki[2]=0

# Setup state variables
current_state = control.current_state
desired_state = np.zeros(current_state.shape)

## Test Controller
# Setpoint for controller
desired_state[0] = -1
desired_state[1] = 1
desired_state[2] = 2

"""
## Tune Controller # Comment out this block for testing.
# Override all gains to 0 for tuning PID controllers
Kp = np.array([0.0,0,0,0,0,0,0,0])
Ki = np.array([0.0,0,0,0,0,0,0,0])
Kd = np.array([0,0,0,0,0,0,0,0])
I_lim = np.array([0.0,0,0,0,0,0,0,0])

# Tune test_dim
test_dim = 0
Kp[test_dim] = 0
Ki[test_dim] = 0.1
Kd[test_dim] = 0
I_lim[test_dim] = 150
# Setpoint for controller
desired_state[test_dim] = 1
"""

# Setup PID History variables
E_prev = desired_state - current_state
I_prev = np.zeros(current_state.shape)

# Perform Control
j=0
while control.NotDone():
#    ## Compute control action
#    # Introduce a disturbance
#    #if j == 100:
#        #desired_state[test_dim] = 1
#    E = desired_state - current_state
#    # Proporational Contribution
#    P = Kp*E
#    # Integral Contribution (Leaky Trapezoidal)
#    # TODO: make a scaled by Ts
#    I = (a ) * I_prev + Ki * (E_prev + E)*Ts / 2
#    #print (a ) * I_prev + np.array([0.9,0,0,0,0,0,0,0]) * (E_prev + E) * Ts/2
#    # Peform anti-windup
#    for prim_num in np.arange(0,dim):
#        i = I[prim_num]
#        i_lim = I_lim[prim_num]
#        if i > i_lim:
#            print ("hit limit pos")
#            I[prim_num] = i_lim
#        elif i < -i_lim:
#            print ("hit limit neg")
#            I[prim_num] = -i_lim
#        if j < past:
#            I[prim_num]=0
#
#    """
#    # For running tests. helps to show system is non-linear and single state
#    # doesn't capture state of system.
#    prim_num = test_dim
#    if j == 50:
#        I[prim_num] = -5
#    elif j == 100:
#        I[prim_num] = 0
#    elif j == 150:
#        I[prim_num] = -5
#    elif j ==200:
#        I[prim_num] = 0
#    """
#
#    # Derivative Contribution
#    D = Kd * (E - E_prev) / Ts
#    # Set PID history variables
#    E_prev = E
#    I_prev = I
#    # Combine Terms
#    control_action = P + I + D
#    #control_action[test_dim] = -20

    ## Step Simulation
    #print control.current_state
    #plt.plot(control.current_state)
    #plt.show()
    #print control.Now()
    #current_state = control.SimulatePeriod(hold=(control.Now()<1000)) #control_action=control_action, control_period=0.)
    current_state = control.SimulatePeriod(hold=(control.Now()<1000)) #control_action=control_action, control_period=0.)
    #plt.plot(current_state)
    #plt.show()
    j+=1

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



