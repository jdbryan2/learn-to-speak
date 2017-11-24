# load primitives from file
# control vocal tract in primitive space
# perform "look-up-table" Q-learning to system
# to a fixed desired state from a random starting state


# for forcing floating point division
from __future__ import division
import os
import numpy as np
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
import Artword as aw
import learners.Learner

#import test_params
from test_params import *
primdir = dirname+'_prim'

max_seconds = 3.0

initial_art=np.random.random((aw.kArt_muscle.MAX, ))

control = PrimitiveUtterance(dir_name=primdir,
                             prim_fname=load_fname,
                             loops=1,
                             utterance_length=max_seconds,
                             initial_art = initial_art)
                             #initial_art=np.random.random((aw.kArt_muscle.MAX, )))
#print initial_art

control.InitializeControl()

Ts = 1000/(sample_period)

# Setup state variables
current_state = control.current_state
desired_state = np.zeros(current_state.shape)

# Initialize q learning class
num_state_bins = 10
num_action_bins = 10
q_learn = Learner(states = np.linspace(-10.0,10.0,num=num_state_bins),
                  goal_state = np.zeros((1,num_state_bins)),
                  goal_width = np.ones(1,num_state_bins)*0.1,
                  goal_reached_steps = 10,
                  max_steps = max_seconds*Ts,
                  actions = np.linspace(-10.0,10.0,num=num_action_bins),
                  alpha = 0.99)

## Test Controller
# Setpoint for controller
desired_state[0] = 1
test_dim =0

# Perform Control
while control.speaker.NotDone():
    ## Compute control action
    control_action = current_state
    #control_action[test_dim] = -20
    ## Step Simulation
    current_state = control.SimulatePeriod(control_action=control_action)

control.Save()
#plt.plot(_h)
savedir = 'data/' + primdir + '/figures/in_out/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

_h = control.state_hist
#print("_h")
#print _h

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




