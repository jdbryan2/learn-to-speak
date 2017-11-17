# load primitives from file
# control vocal tract 
# generate figures with constant control inputs at various values
# save vocal tract data of utterance to file, and save output audio to file


# for forcing floating point division
from __future__ import division
import os
import numpy as np
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
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
primdir = dirname+'_prim'
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals




control = PrimitiveUtterance(dir_name=primdir,
                             prim_fname=load_fname,
                             loops=1,
                             utterance_length=2)

#control.InitializeDir(dirname=primdir, addDTS=False)

print control.K
control.InitializeControl()


test_dim = 0
Ts = 1000/(sample_period)
#Kp = np.array([1/3,0,0,0,0,0,0,0])
#Ki = np.array([0.2/3,0,0,0,0,0,0,0])
#Kd = np.array([10/3,0,0,0,0,0,0,0])

Kp = np.array([1/3,0,0,0,0,0,0,0])
Ki = np.array([0.5/3,0,0,0,0,0,0,0]) #0.5/3
Kd = np.array([10/3,0,0,0,0,0,0,0])
I_lim = np.array([100,0,0,0,0,0,0,0])
# leaky integrator constant
a = 1
#I_lim = Ki*100
#Kp[test_dim] = 0
# Get initial state
current_state = control.current_state
desired_state = np.zeros(current_state.shape)
desired_state[test_dim] = 2

E_prev = desired_state - current_state
I_prev = np.zeros(current_state.shape)

while control.speaker.NotDone():
    E = desired_state - current_state
    # Proporational Contribution
    P = Kp*E
    # Integral Contribution (Leaky Trapezoidal)
    # TODO: make a scaled by Ts
    I = (a ) * I_prev + Ki * (E_prev + E)*Ts / 2
    # Peform anti-windup
    for prim_num in np.arange(0,dim):
        i = I[prim_num]
        i_lim = I_lim[prim_num]
        if i > i_lim:
            print ("hit limit pos")
            I[prim_num] = i_lim
        elif i < -i_lim:
            print ("hit limit neg")
            I[prim_num] = -i_lim
    # Derivative Contribution
    D = Kd * (E - E_prev) / Ts
    # Set PID history variables
    E_prev = E
    I_prev = I
    # Combine Terms
    control_action = P + I + D
    #control_action[test_dim] = -20
    current_state = control.SimulatePeriod(control_action=control_action, control_period=0.)

control.Save()
#plt.plot(_h)
savedir = 'data/' + primdir + '/figures/in_out/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

_h = control.state_hist

"""
PlotTraces(_h, np.arange(_h.shape[0]), _h.shape[1], sample_period, highlight=0)
plt.grid(True)
tikz_save(savedir + 'state_history.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')

plt.show()
"""
#plt.figure()
#PlotTraces(control.art_hist, np.arange(control.art_hist.shape[0]), control.art_hist.shape[1], sample_period, highlight=0)
#plt.plot(control.art_hist.T)
#plt.show()

prim_nums = np.arange(0,dim)
colors = ['b','g','r','c','m','y','k','0.75']
markers = ['o','o','o','o','x','x','x','x']
fig = plt.figure()
for prim_num, c, m in zip(prim_nums,colors,markers):
    plt.plot(_h[prim_num][:],color=c)

# Remove last element from plot because we didn't perform an action
# after the last update of the state history.
plt.plot(control.action_hist[test_dim][0:-1], color="0.5")
plt.show()




