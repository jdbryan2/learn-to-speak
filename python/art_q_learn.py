# load primitives from file
# control vocal tract in primitive space
# perform "look-up-table" Q-learning to system
# to a fixed desired state from a random starting state


# for forcing floating point division
from __future__ import division
import os
import numpy as np
import pylab as plt
import matplotlib.pyplot as pltm
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
num_state_bins = 10
num_action_bins = 10
# ensure that actions and states are 2d arrays
#states = np.linspace(-10.0,10.0,num=num_state_bins)
goal_state = 1
goal_width = .2
states = np.linspace(-10.0,goal_state-goal_width/2.0,num=np.floor(num_state_bins/2.0))
states = np.append(states,np.linspace(goal_state+goal_width/2.0,10,num=np.ceil(num_state_bins/2.0)))
states = states.reshape(1,states.shape[0])
print states
goal_state_index = np.array([np.floor(num_state_bins/2.0)])
print("Goal State")
ind2d = np.zeros((2,1))
ind2d[1,0] = goal_state_index
print states[tuple(ind2d.astype(int))]
actions_inc = np.linspace(-1.0,1.0,num=num_action_bins)
actions_inc = actions_inc.reshape(1,actions_inc.shape[0])
print actions_inc

q_learn = Learner(states = states,
                  goal_state_index = goal_state_index,
                  max_steps = np.floor(max_seconds*Ts),
                  actions = actions_inc,
                  alpha = 0.99)

# Perform Q learning Control
num_episodes = 40
num_view_episodes = 2
num_tests = 5
# TODO: Change to condition checking some change between Q functions
for e in range(num_episodes+num_tests):
    # Reset/Initialize Prim Controller and Simulation
    control.InitializeControl()
    # Setup state variables
    current_state = control.current_state
    desired_state = np.zeros(current_state.shape)
    
    # TODO: Change with each episode
    #1-1.0/(e+1.0)
    exploit_prob  = 0.1
    # TODO: Change with each episode
    learning_rate = 0.1
    # Get initial state
    state = control.current_state
    i = 0
    # Cumulative action command
    action = np.zeros(actions_inc.shape[0])
    while not q_learn.episodeFinished():
        ## Compute control action
        if e >= num_episodes:
            action_inc = q_learn.exploit_and_explore(state=state,p_=1)
        else:
            action_inc = q_learn.exploit_and_explore(state=state,p_=exploit_prob)
        action = action+action_inc
        ## Step Simulation
        next_state = control.SimulatePeriod(control_action=action)
        
        # Don't update Q if we are just using the policy
        if e >= num_episodes:
            q_learn.incrementSteps()
        else:
            ## Update the estimate of Q
            q_learn.updateQ(state=state,action=action_inc,next_state=next_state,epsilon=learning_rate)

        ## Update state
        state = next_state
        i+=1
    print("Episode"+str(e))
    print q_learn.Q
    #pltm.imshow(q_learn.Q)
    #pltm.show()
    q_learn.resetEpisode()

    if e >= num_episodes-num_view_episodes:
        _h = control.state_hist

        prim_nums = np.arange(0,dim)
        colors = ['b','g','r','c','m','y','k','0.75']
        markers = ['o','o','o','o','x','x','x','x']
        fig = plt.figure()
        for prim_num, c, m in zip(prim_nums,colors,markers):
            plt.plot(_h[prim_num][:],color=c)
        
        print("states")
        print states
        print("Goal State")
        print states[tuple(ind2d.astype(int))]
        # Remove last element from plot because we didn't perform an action
        # after the last update of the state history.
        plt.plot(control.action_hist[0][0:-1], color="0.5")
        plt.show()

        pltm.imshow(q_learn.Q)
        pltm.show()


control.Save()
#plt.plot(_h)
savedir = 'data/' + primdir + '/figures/in_out/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

_h = control.state_hist

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


