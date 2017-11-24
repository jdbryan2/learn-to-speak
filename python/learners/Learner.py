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
        self.states = kwargs.get("states", np.linspace(-10.0,10.0,num=10))
        
        self.state_shape = self.states.shape
        self.goal_state = kwargs.get("goal_state", np.zeros((1,self.state_shape[0])))
        
        # If one wants the goal_range to be uniform around goal_state just
        # provide the goal_width for each state and the range will be
        # (-0.5 * width + goal_state,0.5 * width + goal_state]
        self.goal_width = kwargs.get("goal_width",np.ones(self.state_shape))
        
        # Row is state and Column is the low and high values of the acceptable ragne
        self.goal_range = np.array([self.goal_state-self.goal_width,
                                    self.goal_state+self.goal_width]).T
        self.goal_range = kwargs.get("goal_range", self.goal_range)
        
        # Number of timesteps required for state to be within goal_range before
        # we say that the goal has been reached.
        self.goal_reached_steps = kwargs.get("state_shape", 10)
        
        # Used to keep track of how many timesteps that the state has been within
        # the goal_range
        self.goal_steps = 0
                                     
        # Max number of timesteps for episode
        self.max_steps = kwargs("max_steps", 200)
        
        # Row is continuous action and Column is discrete values of that action
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

        # Q function is a 4d array with Q(i,j,k,l) referring to
        # continous state i, discretized value of that state  j,
        # continous action k, discretized value of that action l
        self.Q = np.zeros(self.state_shape+self.action_shape)
        
    # Will return true if goal_reached_steps has been hit false otherwise if true.
    def reachedGoal(self):

        if self.goal_steps >= self.goal_reached_steps:
            return True
        # else:
        return False

    # Will return true if either the goal has been reached or the max_steps has been hit
    def episodeFinshed(self):

        return self.reachedGoal() or (self.steps >= self.max_steps)

    # Will see if all states are within the goal_range and increment goal_steps
    # Important that this is called only once per step of the trial
    def incrementSteps(self,state):
        
        for s in np.arange(self.state_shape[0]):
        
            if self.goal_range[s,0] < state[s] <= self.goal_range[s,1]:
                continue
            elif self.goal_range[s,0] >= state[s]:
                break
            elif state[s] > self.goal_range[s,1]:
                break
            
            # All states are within goal_range so increment goal_steps
            self.goal_steps += 1

        # Increment steps to keep track of the time
        self.steps += 1

    # Return undiscounted reward for this state, action pair
    def getReward(self, state, action):
        
        # Only give reward if goal has been reached
        if not reachedGoal(state, increment_step = True):
            return 0.0

        # Else compute reward
        # Could compute reward dependent on how close state is to goal,
        # but for now just return constant.
        return 1.0
    
    # Return discounted reward for this state, action pair
    def getDiscountedReward(self,state,action):
        
        return (self.alpha ** self.steps)* self.getReward(state,action)

    # Reset member variables to start new episode
    def resetEpisode(self):
        
        self.steps = 0
        self.goal_steps = 0
    
    # Discretize state
    def getDiscreteState(self,state):

        d_state = np.zeros((state_shape[0],1))
        for s in np.arange(self.state_shape[0]):
            for d in np.arange(self.state_shape[1]):
                if d == 0 and state[s] <= self.states[s,0]:
                    d_state[s] = d
                elif d == self.state_shape[0]-1 and state[s] > self.states[s,d]:
                    d_state[s] = d
                elif self.states[s,d-1] < state[s] <= self.states[s,d]:
                    d_state[s] = d
                else:
                    print("I messed up the logic here.")
        return d_state

    # Discretize action
    def getDiscreteAction(self,action):

        d_action = np.zeros((action_shape[0],1))
        for a in np.arange(self.action_shape[0]):
            for d in np.arange(self.action_shape[1]):
                if d == 0 and action[a] <= self.actions[a,0]:
                    d_action[a] = d
                elif d == self.action_shape[0]-1 and action[a] > self.actions[a,d]:
                        d_action[s] = d
                elif self.actions[a,d-1] < action[a] <= self.actions[a,d]:
                    d_actions[a] = d
                else:
                    print("I messed up the logic here.")
        return d_action
    
    # Find maximum value of Q over all actions for a given discrete state
    # Also return the corresponding action
    def getMaxQ_and_u(self,d_state):

        Q_max = 0.0
        # Loop through all possible actions to find maximum
        for a in np.arange(self.action_shape[0]):
            for d in np.arange(self.action_shape[1]):
                d_action = self.actions[a,d]
                Q_u = Q[np.concatenate((d_state,d_action))]
                if Q_u >= Q_max:
                    Q_max = Q_u
                    u_max = d_action

        return Q_max, u_max
                    
    # Choose a discrete action uniformly at random
    def explore(self):
        
        d_action = np.zeros((action_shape[0],1))
        for a in np.arange(self.action_shape[0]):
            d = np.random.choice(self.action_shape[1],1)
            d_action[a] = self.actions[a,d]
                    
        return d_action

    # Update the Q function asynchronously
    # epsilon is the learning rate. Plan is to have user of this class
    # update after each episode to something like 1/(number of episodes)
    def updateQ(self,state,action,next_state,epsilon = 0.9):

        d_state = self.getDiscreteState(state)
        d_action = self.getDiscreteAction(state)
        d_next_state = self.getDiscreteState(state)
        
        # Find max Q and action over next state
        Q_next_max, v = self.getMaxQ_and_u(d_next_state)
        
        # Get reward for state action pair
        r = self.getReward(state,action)

        # Update Q for this state action pair
        Q[np.concatenate((d_state,d_action))] = (
                        (1-epsilon)*Q[np.concatenate((d_state,d_action))] +
                        epsilon*(r + Q_next_max) )

    # Choose the action greedily with probability p_ and
    # choose randomly with probability 1-p_
    def exploit_and_explore(state,p_):

        d_state = self.getDiscreteState(state)
        d_action = np.zeros((action_shape[0],1))
        # 0 is exploit, 1 is explore
        exp = np.random.choice(2,1,p=[p_,1.0-p_])
        
        if exp == 0:
            Q,d_action = self.getMaxQ_and_u(d_state)
        else:
            d_action = self.explore()

        return d_action

