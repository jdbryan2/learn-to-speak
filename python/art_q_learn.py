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
from primitive.Utterance import Utterance
import Artword as aw
from learners.Learner import Learner

# Hacky way to get initial articulation/primitive value from some desired vocal tract shape
# Default initial_art is all zeros
ipa101 = Utterance(dirname="ipa101_q_test",
                      loops=1,
                      utterance_length=0.5,
                      addDTS=False)

ipa101.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa101.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5],[1.0, 1.0])
ipa101.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1],[0.2, 0.0])
ipa101.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.25], [0.7])
ipa101.SetManualArticulation(aw.kArt_muscle.ORBICULARIS_ORIS, [0.25], [0.2])




#import test_params
from test_params import *
primdir = dirname+'_prim'

# With no reset and using simple reward
# this scheme works well to oscillate right about goal state
# although some wierdness happens at the beginning/middle of an episode
#max_seconds = 10.0
#num_episodes = 50

# With reset the policy learned in the end seems to
# enable full control of the state, but it is oscillatory with large amplitude.
# So I wonder if the intialzation is really what is giving us trouble...
# Maybe we shouldn't train off of it, or do initialize differently, or just make
# episodes longer so it doesn't have as much weight in the update.
#max_seconds = 3.0
#num_episodes = 20

# Without integral, just 10 actions in [-1,1] and 10 5 sec trials and goal width 0.5
# get really good result.

# With the first two primitives, each 10 possible values, 11 actions, no transience,
# no reset, and reward based only on state 0, and control for state two set to 0,
# we get good results. It oscilates about the goal, with occasional disturbances popping
# up.

max_seconds =   10.0
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
num_action_bins = 11
reset_action = 100
# ensure that actions and states are 2d arrays
#states = np.linspace(-10.0,10.0,num=num_state_bins)
goal_state = 1
goal_width = .2
states = np.linspace(-2.0,goal_state-goal_width/2.0,num=np.floor(num_state_bins/2.0))
states = np.append(states,np.linspace(goal_state+goal_width/2.0,2.0,num=np.ceil(num_state_bins/2.0)))
states = states.reshape(1,states.shape[0])
states = np.concatenate((states,states),axis=0)

print states
goal_state_index = np.array([np.floor(num_state_bins/2.0),-1])
print("Goal State")
ind2d = np.zeros((2,dim))
#ind2d[1,0] = goal_state_index
ind2d[1,:] = goal_state_index
print states[tuple(ind2d.astype(int))]
actions_inc = np.linspace(-1.0,1.0,num=num_action_bins)
#actions_inc = np.append(actions_inc,reset_action)
actions_inc = actions_inc.reshape(1,actions_inc.shape[0])
print actions_inc

q_learn = Learner(states = states,
                  goal_state_index = goal_state_index,
                  max_steps = np.floor(max_seconds*Ts),
                  actions = actions_inc,
                  alpha = 0.99)

# Perform Q learning Control
num_episodes = 10
num_view_episodes = 0
num_tests = 2
# TODO: Change to condition checking some change between Q functions
for e in range(num_episodes+num_tests):
    # Reset/Initialize Prim Controller and Simulation
    control.InitializeControl()
    # Setup state variables
    current_state = control.current_state
    
    # TODO: Change with each episode
    #1-1.0/(e+1.0)
    exploit_prob  = 0.1
    #exploit_prob  = 0.0
    
    # TODO: Change with each episode
    #learning_rate = 0.1
    learning_rate = 1.0/(e+1.0)
    # Get initial state
    state = control.current_state
    i = 0
    # Cumulative action command
    action = np.zeros(actions_inc.shape[0])
    while not q_learn.episodeFinished():
        ## Compute control action
        if e >= num_episodes :
            action_inc = q_learn.exploit_and_explore(state=state,p_=1)
        else:
            action_inc = q_learn.exploit_and_explore(state=state,p_=exploit_prob)
    
        # Reset the acumulated action command to 0 if the reset action is taken
        # or if we are still in the transient due to initialization of past in primitives
        if action_inc == reset_action or i < past + 10:
            action = 0
        else:
            # Make action state incremental
            action = action+action_inc
            # Make action state absolute
            #action = action_inc
        
        ## Step Simulation
        # Currently give uncontrolled state zero command
        next_state = control.SimulatePeriod(control_action=np.append(action,0))
        
        # Don't update Q if we are just using the policy or
        # if we are in the initial period of transience from the
        # initialization of the primitives.
        no_train_samples = past + 10
        #no_train_samples = 0
        if e >= num_episodes or i < no_train_samples:
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
        for k in range(actions_inc.shape[1]):
            fig = plt.figure()
            pltm.imshow(q_learn.Q[:,:,k])

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


