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
#from primitive.SubspaceDFA import SubspaceDFA
#from features.ArtFeatures import ArtFeatures

# Hacky way to get initial articulation/primitive value from some desired vocal tract shape





from genfigures.plot_functions import *
#import test_params
#from test_params import *
#primdir = dirname+'_prim'
primdir = '../data/batch_random_20_10'
ind= get_last_index(primdir, 'round')
prim_fname = 'round%i.npz'%ind
print "Loading from: %s/%s"%(primdir, prim_fname)
save_dir = '../data/qlearn/output'

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
initial_art=np.random.random((aw.kArt_muscle.MAX, )) #------


control = PrimitiveUtterance()
control.LoadPrimitives(prim_fname, primdir)
control.utterance = Utterance(directory=save_dir, utterance_length=max_seconds)

# pull in relevant stuff from PrimitiveUtterance
sample_period = control._sample_period
dim = 3 #control._dim # dimension of states we will try to control
target_dim = 3 # dimension of states we will try to control
full_dim = control._dim
past = control._past
future = control._future

initial_action = np.random.random(control._dim)
initial_art = control.GetControl(initial_action)
print "Initial Articulation"
print initial_art
control.InitializeControl(initial_art=initial_art)

sample_period_ms = sample_period/8.
Ts = 1000/(sample_period_ms)
print "Sample Period (ms): ", sample_period_ms


# Initialize q learning class
# states centered on -0.1, 0.0, 0.1 # target is 0
state_lims = 0.1 
num_state_bins = 3
num_action_bins = 3
#num_int_state_bins = 11
reset_action = 100

action_lims = 0.1
# 1DState
"""
goal_state = 1
goal_width = .2
states = np.linspace(-2.0,goal_state-goal_width/2.0,num=np.floor(num_state_bins/2.0))
states = np.append(states,np.linspace(goal_state+goal_width/2.0,2.0,num=np.ceil(num_state_bins/2.0)))
#states = np.linspace(-2.0,2.0,num=num_state_bins)
states = states.reshape(1,states.shape[0])
"""
# 2DSTATE
#goal_state = 1
#goal_width = .2
#first_state = np.linspace(-2.0,goal_state-goal_width/2.0,num=np.floor(num_state_bins/2.0))
#first_state = np.append(first_state,np.linspace(goal_state+goal_width/2.0,2.0,num=np.ceil(num_state_bins/2.0)))
#first_state = first_state.reshape(1,first_state.shape[0])
#states = np.concatenate((first_state,first_state),axis=0)

first_state = np.linspace(-state_lims,state_lims,num=num_state_bins)
states = np.tile(first_state, (dim, 1))
#first_state = first_state.reshape(1,first_state.shape[0])
#other_state = np.linspace(-state_lims,state_lims,num=num_state_bins)
#other_state = other_state.reshape(1,other_state.shape[0])
#states = np.concatenate((first_state,other_state),axis=0)


print states
goal_state_index = np.ones(dim) #np.array([1,1])
print("Goal State")
#print goal_state_index
# 1DSTATE OR 2DSTATE
ind2d = np.zeros((2,target_dim))
ind2d[1,:] = goal_state_index
print states[tuple(ind2d.astype(int))]

#1DACTION
actions_inc = np.linspace(-action_lims, action_lims,num=num_action_bins)
actions_inc = actions_inc.reshape(1,actions_inc.shape[0])
# 2DACTION
actions_inc = np.tile(actions_inc, (dim, 1)) # full dimension action
print actions_inc

q_learn = Learner(states = states,
                  goal_state_index = goal_state_index,
                  max_steps = np.floor(max_seconds*Ts),
                  actions = actions_inc,
                  alpha = 0.99)


print "Q matrix: ", q_learn.Q.shape
#plt.plot(_h)
#savedir = 'data/' + primdir + '/figures/in_out/'
#if not os.path.exists(savedir):
#    os.makedirs(savedir)


# Perform Q learning Control
num_episodes = 100 #----10
num_view_episodes = 2 
num_tests = 1
rewards = np.zeros(num_episodes+num_tests)
# TODO: Change to condition checking some change between Q functions
for e in range(num_episodes+num_tests):
    print("--------------Episode"+str(e)+"--------------")
    # Reset/Initialize Prim Controller and Simulation
    # Was using same intialzation for each episode before
    # STATIC INIT
    #control.InitializeControl(initial_art = initial_art)
    # RANDOM INIT
    #initial_action = np.random.random(control._dim)
    initial_action = np.random.randint(-10, 10, size=control._dim)/10.
    initial_action[dim:] = 0.
    initial_art = control.GetControl(initial_action)
    control.InitializeControl(initial_art = initial_art)
    
    learning_offset = 5 #------0
    if e<learning_offset:
        learning_rate = 1
    else:
        learning_rate = 10.0/(e-(learning_offset-1)+10.0)
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
    action = np.copy(initial_action[:actions_inc.shape[0]])
    print action
    #action = np.zeros(actions_inc.shape[0])
    #action.reshape(1,action.shape[0])

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
        #if action_inc.all() == reset_action or i < past + 10:
        if action_inc.all() == reset_action: # don't worry about transient at initialization
            action = np.zeros(action.shape)
        else:
            # Make action state incremental
            action = action+action_inc.flatten()
            # Make action state absolute
            #action = action_inc

        #action_padded = np.append(action, np.zeros(control._dim-dim))
        action_padded = np.append(action, initial_action[dim:])

        ## Step Simulation
        # Currently give uncontrolled state zero command
        # 1DSTATE
        next_state = control.SimulatePeriod(control_action=action_padded)
        # 2DSTATE with 1D control
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
            rewards[e] += q_learn.incrementSteps(state=state,action=action_inc.flatten())
        else:
            ## Update the estimate of Q
            rewards[e] += q_learn.updateQ(state=state,action=action_inc.flatten(),next_state=next_state,epsilon=learning_rate)

        ## Update state
        state = next_state
        i+=1
    #print q_learn.Q
    #pltm.imshow(q_learn.Q)
    #pltm.show()
    q_learn.resetEpisode()
 
    print e, num_episodes, num_view_episodes
    if e >= num_episodes-num_view_episodes:
        _h = control.state_hist

        #prim_nums = np.arange(0,dim)
        #colors = ['b','g','r','c','m','y','k','0.75']
        #markers = ['o','o','o','o','x','x','x','x']
        ##for prim_num, c, m in zip(prim_nums,colors,markers):
        #    #plt.plot(_h[prim_num][:],color=c)
        

        fig = plt.figure()
        MultiPlotTraces(control.state_hist, np.arange(control.state_hist.shape[0]), control.state_hist.shape[1],
                        control._sample_period, highlight=[0, 1])
        plt.title("Trial Run - Exploiting Q")
        plt.xlabel("Time (s)")
        plt.ylabel("Primitive State")

        print("states")
        print states
        print("Goal State")
        print states[tuple(ind2d.astype(int))]
        # Remove last element from plot because we didn't perform an action
        # after the last update of the state history.
        #for k in range(actions_inc.shape[0]):
            #plt.plot(control.action_hist[k][0:-1], color=str(0.5+k/(2*actions_inc.shape[0])))

        plt.figure()
        MultiPlotTraces(control.action_hist, np.arange(control.action_hist.shape[0]), control.action_hist.shape[1],
                        control._sample_period, highlight=[1])

        plt.title("Trial Run - Exploiting Q")
        plt.xlabel("Time (s)")
        plt.ylabel("Primitive Inputs")
        # 1DSTATE
        #fig = plt.figure()
        #pltm.imshow(q_learn.Q)
        #pltm.colorbar()
        #plt.title("Q Function")
        #plt.xlabel("Discrete Action Index")
        #plt.ylabel("Discrete State Index")
        # 2DSTATE and INTSTATE
        """
        for k in range(actions_inc.shape[1]):
            fig = plt.figure()
            pltm.imshow(q_learn.Q[:,:,k])
            plt.title("Q Function")
            plt.xlabel("Discrete Action Index")
            plt.ylabel("Discrete State Index")
        """

        # Plot Reward
        fig = plt.figure()
        plt.plot(rewards)
        plt.title("Undiscounted Accumulated Reward throughout Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        control.SaveOutputs(fname="episode_"+str(e))
        plt.show()

print q_learn.Q.shape
#plt.plot(_h)
#savedir = 'data/' + primdir + '/figures/in_out/'
#if not os.path.exists(savedir):
#    os.makedirs(savedir)



