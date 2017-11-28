import numpy as np

# Implements the infrastructure for performing discrete episodic Q-Learning
# States and actions passed in can be continous and are discretized in the class.

# Note: This construction requires each state to have the same number
# of discretized values, and each action to have the same number of
# discretized values.

# TODO: Make a base class that defines a RL interface for different approaches

class Learner:

    def __init__(self, **kwargs):
        
        # Row is continuous state and Column is discrete values of that state
        # State is discretized in range defined by intervals of
        # ( states[i,j-1], states[i,j] ].
        # If x[i] <= states[i,0] then state is states[i,0]
        # If x[i] > states[i,N] then state is states[i,N]
        # 1D ex: System is in state states[0,j] when continous state x is
        #        in ( states[0,j-1], states[0,j] ]
        # TODO: Make this into list of arrays or dict for states to have
        #       different numbers of discrete values.
        default_states = np.linspace(-10.0,10.0,num=10)
        default_states = default_states.reshape(1,default_states.shape[0])
        self.states = kwargs.get("states", default_states)
        
        
        self.state_shape = self.states.shape
        self.goal_state_index = kwargs.get("goal_state_index", np.zeros((self.state_shape[0])))
        
        # Max number of timesteps for episode
        self.max_steps = kwargs.get("max_steps", 200)
        
        # Row is continuous action and Column is discrete values of that action
        # TODO: Make this into list of arrays or dict for actions to have
        #       different numbers of discrete values.
        self.actions = np.zeros((1,10))
        self.action_shape = self.actions.shape
        for action in range(self.action_shape[0]):
            self.actions[action,:] = np.linspace(-5.0,5.0,num = self.action_shape[1])
        
        self.actions = kwargs.get("actions", self.actions)
        self.action_shape = self.actions.shape
        
        # Keep track of timesteps for computing discounted reward
        self.steps = 0

        # Discount factor
        self.alpha = kwargs.get("alpha",0.99)

        # let N=(state_shape[0]+action_shape[0])
        # Then Q function is a Nd array with
        # each dimension having state_shape[1] or action_shape[1] elements
        # Access elements by providing N coordinates
        q_shape = []
        for s in range(self.state_shape[0]):
            q_shape.append(self.state_shape[1])
        for a in range(self.action_shape[0]):
            q_shape.append(self.action_shape[1])
        self.Q = np.zeros(q_shape)
        
    # Will return true if goal_reached_steps has been hit false otherwise if true.
    def reachedGoal(self):
        
        if self.goal_steps >= self.goal_reached_steps:
            return True
        # else:
        return False

    # Will return true if the max_steps has been hit
    def episodeFinished(self):

        maxed_out = (self.steps >= self.max_steps)
        if maxed_out:
            print("Hit Max Number of Steps")
        
        return maxed_out


    # Check to see if state is in goal state
    def inGoalState(self,state):
    
        d_state = self.getDiscreteState(state)
        ingoal = np.zeros(self.state_shape[0])
        for s in np.arange(self.state_shape[0]):
            if d_state[s] == self.goal_state_index[s]:
                ingoal[s] = 1
    
        if ingoal.all():
            print("In Goal State")
            return True

        return False

    # Return undiscounted reward for this state, action pair
    def getReward(self, state, action):
        
        """
        d_state = self.getDiscreteState(state)
        reward = np.zeros(self.state_shape[0])
        for s in np.arange(self.state_shape[0]):
            reward[s] = 1/(np.fabs(d_state[s] - self.goal_state_index[s])+1)
        
        
        #print("d_state")
        #print("reward")
        #print d_state
        #print np.sum(reward)
        
        
        return np.sum(reward)
        
        """
        # Only give reward if in goal state
        if not self.inGoalState(state):
            return 0.0

        # Else compute reward
        # Could compute reward dependent on how close state is to goal
        # and size of action, but for now just return constant.
        return 1.0

    
    # Return discounted reward for this state, action pair, for the entire episode
    def getDiscountedReward(self,state,action):
        
        return (self.alpha ** self.steps)* self.getReward(state,action)

    # Reset member variables to start new episode
    def resetEpisode(self):
        
        self.steps = 0
        self.goal_steps = 0
        self.inrange = 0
    
    # Discretize state
    def getDiscreteState(self,state):

        d_state = np.zeros((self.state_shape[0],1))
        for s in np.arange(self.state_shape[0]):
            for d in np.arange(self.state_shape[1]):
                if d == 0 and state[s] <= self.states[s,d]:
                    d_state[s] = d
                    break
                elif d == self.state_shape[1]-1 and state[s] > self.states[s,d]:
                    d_state[s] = d
                    break
                elif self.states[s,d-1] < state[s] <= self.states[s,d]:
                    d_state[s] = d
                    break
        return d_state

    # Discretize action
    def getDiscreteAction(self,action):

        d_action = np.zeros((self.action_shape[0],1))
        for a in np.arange(self.action_shape[0]):
            for d in np.arange(self.action_shape[1]):
                if d == 0 and action[a] <= self.actions[a,d]:
                    d_action[a] = d
                    break
                elif d == self.action_shape[1]-1 and action[a] > self.actions[a,d]:
                    d_action[a] = d
                    break
                elif self.actions[a,d-1] < action[a] <= self.actions[a,d]:
                    d_action[a] = d
                    break
    
        return d_action
    
    # Find maximum value of Q over all actions for a given discrete state
    # Also return the corresponding action
    def getMaxQ_and_u(self,d_state):

        Q_max = 0.0
        # Loop through all possible actions to find maximum
        for a in np.arange(self.action_shape[0]):
            for d in np.arange(self.action_shape[1]):
                d_action = np.zeros((self.action_shape[0],1))
                for ai in np.arange(self.action_shape[0]):
                    d_action[ai] = d
            
                # Access element of Q
                # Note: Tentatively working, but not tested with more than one
                #       action and state. Also, a mess...
                Q_u = self.Q[tuple(np.concatenate((d_state,d_action)).astype(int))]
                if Q_u >= Q_max:
                   Q_max = Q_u
                   u_max = d_action
        
        
        """
        # Loop through all possible actions to find maximum
        # This is confusing.
        # Just do triple for loop instead
        #self.action_shape = (2,5)
        all_actions = np.dstack(np.meshgrid(*[np.arange(0, x) for x in np.ones(self.action_shape[0])*self.action_shape[1]]))
        all_actions = all_actions.reshape((self.action_shape[1]**self.action_shape[0],self.action_shape[0]))
        ###########STILL NOT RIGHT
        for d_action in all_actions:
            index = np.zeros(4)
            if self.state_shape[0] ==1:
                index[0] = 1
                index[1] = d_state[0]
            else:
                index[0] = d_state[0]
                index[1] = d_state[1]
            if self.action_shape[0] == 1:
                index[2] = 1
                index[3] = d_action[0]
            else:
                index[2] = d_action[0]
                index[3] = d_action[1]
            print index
            #Q_u = self.Q[np.concatenate((d_state,d_action.reshape((-1,1))))]
            #if Q_u >= Q_max:
            #   Q_max = Q_u
            #   u_max = d_action
        """
        """
        # TODO: Is there a way to loop through all of the actions without
        #       creating this massive (exponentially growing) meshgrid? Would this require something
        #       like recursion to account for different numbers of continuous actions?
        # Loop through all possible actions to find maximum
        self.action_shape = (2,5)
        for d in np.arange(self.action_shape[1]):
            for a in np.arange(self.action_shape[0]):
                d_action = np.zeros((self.action_shape[0],1))
                for a1 in np.arange(self.action_shape[0]):
                    d_action = (d,d1)
                    print d_action
                    #Q_u = self.Q[np.concatenate((d_state,d_action))]
                    #if Q_u >= Q_max:
                    #   Q_max = Q_u
                    #   u_max = d_action
        """
        
        return Q_max, u_max
    
                    
    # Choose a discrete action uniformly at random
    def explore(self):
        
        d_action = np.zeros((self.action_shape[0],1))
        for a in np.arange(self.action_shape[0]):
            d_action[a] = np.random.choice(self.action_shape[1],1)
        
        return d_action

    # Update the Q function asynchronously
    # epsilon is the learning rate. Plan is to have user of this class
    # update after each episode to something like 1/(number of episodes)
    def updateQ(self,state,action,next_state,epsilon = 0.9):

        d_state = self.getDiscreteState(state)
        d_action = self.getDiscreteAction(action)
        d_next_state = self.getDiscreteState(next_state)
        
        # Find max Q and action over next state
        Q_next_max, v = self.getMaxQ_and_u(d_next_state)
        
        # Get reward for state action pair
        r = self.getReward(state,action)

        # Update Q for this state action pair
        index = tuple(np.concatenate((d_state,d_action)).astype(int))
        
        """
        print ("Divider-----------------------Divider")
        print ("state ind")
        print d_state
        print ("action ind")
        print d_action
        print ("next state ind")
        print d_next_state
        print ("next max action ind")
        print v
        print ("Divider-----------------------Divider")
        """
        
        self.Q[index] = ((1-epsilon)*self.Q[index] +
                        epsilon*(r + self.alpha*Q_next_max) )
    
        # Increment steps to keep track of the time
        self.steps += 1
    
    # Call this instead of updateQ if you just want to use the learned policy
    def incrementSteps(self):
        # Increment steps to keep track of the time
        self.steps += 1

    # Choose the action greedily with probability p_ and
    # choose randomly with probability 1-p_
    def exploit_and_explore(self,state,p_):

        d_state = self.getDiscreteState(state)
        d_action = np.zeros((self.action_shape[0],1))
        # 0 is exploit, 1 is explore
        exp = np.random.choice(2,1,p=[p_,1.0-p_])
        
        if exp == 0:
            Q,d_action = self.getMaxQ_and_u(d_state)
        else:
            d_action = self.explore()

        # Cheat to make indexing work when only one dimension
        if self.action_shape[0]==1:
            ind2d = np.zeros((2,1))
            ind2d[1,0] = d_action
            return self.actions[tuple(ind2d.astype(int))]
        else:
            return self.actions[tuple(d_action.astype(int))]

