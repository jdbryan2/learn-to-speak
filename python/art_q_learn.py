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
from learners.Learner import Learner

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

Ts = 1000/(sample_period_ms)


# Initialize q learning class
num_state_bins = 3
num_action_bins = 10
# ensure that actions and states are 2d arrays
states = np.linspace(-10.0,10.0,num=num_state_bins)
states = states.reshape(1,states.shape[0])
print states
goal_state = np.zeros((1))
goal_state = goal_state.reshape(1,goal_state.shape[0])
goal_width = np.ones((1))*.1
goal_width = goal_width.reshape(1,goal_width.shape[0])
actions = np.linspace(-10.0,10.0,num=num_action_bins)
actions = actions.reshape(1,actions.shape[0])

q_learn = Learner(states = states,
                  goal_state = goal_state,
                  goal_width = goal_width,
                  goal_reached_steps = 1,
                  max_steps = np.floor(max_seconds*Ts),
                  actions = actions,
                  alpha = 0.99)

# Perform Q learning Control
num_episodes = 20
# TODO: Change to condition checking some change between Q functions
for e in range(num_episodes):
    # Reset/Initialize Prim Controller and Simulation
    control.InitializeControl()
    # Setup state variables
    current_state = control.current_state
    desired_state = np.zeros(current_state.shape)
    
    # TODO: Change with each episode
    exploit_prob  = 0.1
    # TODO: Change with each episode
    learning_rate = 0.9
    # Get initial state
    state = control.current_state
    i = 0
    while not q_learn.episodeFinished():
        ## Compute control action
        d_action = q_learn.exploit_and_explore(state=state,p_=exploit_prob)
        ## Step Simulation
        next_state = control.SimulatePeriod(control_action=d_action)
        ## Update the estimate of Q
        q_learn.updateQ(state=state,action=d_action,next_state=next_state)
        ## Update state
        state = next_state
        i+=1
    print("Episode"+str(e))
    print q_learn.Q
    print "number of loops"
    print i
    print q_learn.actions
    print q_learn.states
    q_learn.resetEpisode()


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
plt.plot(control.action_hist[0][0:-1], color="0.5")
plt.show()




