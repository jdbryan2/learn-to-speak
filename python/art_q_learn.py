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
# For getting tracking
from primitive.SubspaceDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures

# Hacky way to get initial articulation/primitive value from some desired vocal tract shape





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

max_seconds =   5.0
# RANDOM INIT
#initial_art=np.random.random((aw.kArt_muscle.MAX, )) #------
# STATIC INIT
# From a randomly generated art that worked well
"""
initial_art = np.array([ 0.52779292,  0.32185364,  0.86558425,  0.33471684,  0.65112661,  0.79903379,
 0.11987483,  0.24855711,  0.31139851,  0.24787388,  0.19895598,  0.05290729,
 0.32252938,  0.63632587,  0.50026815,  0.98682582,  0.05327814,  0.57564972,
 0.09049773,  0.92107307,  0.25266528,  0.10626182,  0.59077853,  0.83012719,
 0.65740627,  0.65219443,  0.7657366,   0.66722533,  0.49950773,])
"""
# Does well for state 0 with goal btw [.6,1.3)
initial_art=np.zeros((aw.kArt_muscle.MAX, ))
# Does ok for state 1 with goal btw [-2,-1.33) using 1-10.0/(e-exploit_offset+10.0) for exploit
#initial_art=np.ones((aw.kArt_muscle.MAX, ))

# initialize from ipa 305. see ipa305.py
"""
initial_art=np.zeros((aw.kArt_muscle.MAX, ))
initial_art[aw.kArt_muscle.INTERARYTENOID] = 0.5
initial_art[aw.kArt_muscle.LUNGS] = 0.3
initial_art[aw.kArt_muscle.MYLOHYOID] = 0.1
initial_art[aw.kArt_muscle.SPHINCTER] = 0.7
initial_art[aw.kArt_muscle.HYOGLOSSUS] = 0.3
"""

print "Initial Articulation"
print initial_art

control = PrimitiveUtterance(dir_name=primdir,
                             prim_fname=load_fname,
                             loops=1,
                             utterance_length=max_seconds,
                             initial_art = initial_art)

Ts = 1000/(sample_period_ms)


# Initialize q learning class
num_state_bins = 10
num_action_bins = 3
num_int_state_bins = 10
reset_action = 100
# ensure that actions and states are 2d arrays
#states = np.linspace(-10.0,10.0,num=num_state_bins)
goal_state = 1
goal_width = .2
states = np.linspace(-2.0,goal_state-goal_width/2.0,num=np.floor(num_state_bins/2.0))
states = np.append(states,np.linspace(goal_state+goal_width/2.0,2.0,num=np.ceil(num_state_bins/2.0)))
states = states.reshape(1,states.shape[0])
# 2DSTATE
"""
first_state = np.linspace(-4,2,num=num_state_bins)
first_state = first_state.reshape(1,first_state.shape[0])
other_state = np.linspace(-4,2,num=num_state_bins)
other_state = other_state.reshape(1,other_state.shape[0])
states = np.concatenate((first_state,other_state),axis=0)
"""

# INTSTATE
#action_int_state = np.linspace(-20,20,num=num_int_state_bins)
#action_int_state = action_int_state.reshape(1,action_int_state.shape[0])
#prev_state = np.linspace(-1,1,num=num_int_state_bins)
#prev_state = prev_state.reshape(1,prev_state.shape[0])
#prev_state = states
#states = np.concatenate((states,action_int_state))
#states = np.concatenate((states,prev_state))

print states
# 1DSTATE
goal_state_index = np.array([np.floor(num_state_bins/2.0)])
# 2DSTATE and INTSTATE
#goal_state_index = np.array([np.floor(num_state_bins/2.0),np.floor(num_state_bins/3.0)])
#goal_state_index = np.array([np.floor(num_state_bins/2.0),-1])
#goal_state_index = np.array([np.floor(num_state_bins/2.0),np.floor(num_state_bins/3.0),-1,-1])
#goal_state_index = np.array([7,5])
#1.2,-1.7 # For IPA 305
#goal_state_index = np.array([8,4])
print("Goal State")
print goal_state_index.shape
# 1DSTATE OR 2DSTATE
ind2d = np.zeros((2,dim))
# INTSTATE
#ind2d = np.zeros((2,dim+1))
#ind2d = np.zeros((2,dim*2))
#ind2d[1,0] = goal_state_index
ind2d[1,:] = goal_state_index
print states[tuple(ind2d.astype(int))]

#1DACTION
actions_inc = np.linspace(-0.5,0.5,num=num_action_bins)
#actions_inc = np.append(actions_inc,reset_action)
actions_inc = actions_inc.reshape(1,actions_inc.shape[0])
# 2DACTION
#actions_inc = np.concatenate((actions_inc,actions_inc),axis=0)
print actions_inc

q_learn = Learner(states = states,
                  goal_state_index = goal_state_index,
                  max_steps = np.floor(max_seconds*Ts),
                  actions = actions_inc,
                  alpha = 0.99)

# Perform Q learning Control
num_episodes = 80 #----10
num_view_episodes = 2
num_tests = 2
# TODO: Change to condition checking some change between Q functions
for e in range(num_episodes+num_tests):
    print("--------------Episode"+str(e)+"--------------")
    # Reset/Initialize Prim Controller and Simulation
    # Was using same intialzation for each episode before
    # STATIC INIT
    control.InitializeControl(initial_art = initial_art)
    # RANDOM INIT
    #control.InitializeControl(initial_art = np.random.random((aw.kArt_muscle.MAX, )))
    
    learning_offset = 5 #------0
    if e<learning_offset:
        learning_rate = 1
    else:
        learning_rate = 20.0/(e-(learning_offset-1)+20.0)
        #learning_rate = 0.1
        #learning_rate = 1.0/(e+1.0) #--------
        #learning_rate = 10.0/(e+10.0)
        #learning_rate = 20.0/(e+20.0)

    print "Learning Rate = " + str(learning_rate)
    
    #1-1.0/(e+1.0)
    exploit_offset = 10 #----0
    if e<min(exploit_offset,num_episodes):
        exploit_prob = 0
    elif  e >= num_episodes:
        exploit_prob = 1
    else:
        #exploit_prob = 1-1.0/(0.1*e**(1/10.0)+1.0)
        #exploit_prob = 1-1.0/(e**(1/10.0)+1.0) #------------
        # This worked ish for two states and actions and 20 10 sectiond trials
        #exploit_prob = 1-1.0/(0.02*(e-exploit_offset)+1.0)
        #exploit_prob = 1-1.0/(0.01*(e-exploit_offset)+1.0)
        #exploit_prob = 1-learning_rate
        #exploit_prob = 1-10.0/(e-exploit_offset+10.0)
        #exploit_prob = 1- 20.0/(e+20.0)
        exploit_prob = 1 - learning_rate

    #overide exploit
    #exploit_prob = .1

    print "Exploitation Probability = " +str(exploit_prob)
    #exploit_prob  = 0.1
    #exploit_prob  = 0.0

    # Boltzmann Temperature
    #T = 10-e
    T =  -0.1304*e+4.91
    if T < 0.1:
        T = 0.1
    if  e >= num_episodes:
        T = 0.1
    #T = 0.1

    #print "Boltzman Temperature = " +str(T)

    i = 0
    # Cumulative action command
    action = np.zeros(actions_inc.shape[0])
    action.reshape(1,action.shape[0])
    # Get initial state
    state = control.current_state
    # INTSTATE
    #state = np.concatenate((state,action),axis=0)
    #state = np.concatenate((state,state),axis=0)
    while not q_learn.episodeFinished():
        ## Compute control action
        # For epsilon Greedy
        action_inc = q_learn.exploit_and_explore(state=state,p_=exploit_prob)
        # For Boltzman Exploration
        #action_inc = q_learn.boltzmann_exploration(state=state,T=T)
    
        # Reset the acumulated action command to 0 if the reset action is taken
        # or if we are still in the transient due to initialization of past in primitives
        if action_inc.all() == reset_action or i < past + 10:
            action = np.zeros(action.shape)
        else:
            # Make action state incremental
            action = action+action_inc.flatten()
            # Make action state absolute
            #action = action_inc

        ## Step Simulation
        # Currently give uncontrolled state zero command
        # 1DSTATE
        next_state = control.SimulatePeriod(control_action=action)
        # 2DSTATE
        #next_state = control.SimulatePeriod(control_action=np.append(action,0))
        # INTSTATE
        #next_state = np.concatenate((next_state,action),axis=0)
        #next_state = np.concatenate((next_state,state[dim:]),axis=0)
        
        # Don't update Q if we are just using the policy or
        # if we are in the initial period of transience from the
        # initialization of the primitives.
        no_train_samples = past + 10
        #no_train_samples = 0
        if e >= num_episodes or i < no_train_samples:
            q_learn.incrementSteps(state=state,action=action_inc.flatten())
        else:
            ## Update the estimate of Q
            q_learn.updateQ(state=state,action=action_inc.flatten(),next_state=next_state,epsilon=learning_rate)

        ## Update state
        state = next_state
        i+=1
    #print q_learn.Q
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
        for k in range(actions_inc.shape[0]):
            plt.plot(control.action_hist[k][0:-1], color=str(0.5+k/(2*actions_inc.shape[0])))
        # 1DSTATE
        fig = plt.figure()
        pltm.imshow(q_learn.Q)
        # 2DSTATE and INTSTATE
        #for k in range(actions_inc.shape[1]):
        #    fig = plt.figure()
        #    pltm.imshow(q_learn.Q[:,:,k])
        control.Save()
        pltm.show()

print q_learn.Q
#plt.plot(_h)
savedir = 'data/' + primdir + '/figures/in_out/'
if not os.path.exists(savedir):
    os.makedirs(savedir)



