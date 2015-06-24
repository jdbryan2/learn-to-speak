
# Hidden Markov Models for Autoregressive signals
# using linear prediction coefficients
#
# Written by Jacob Bryan
# April 2013

import random
import math 
import numpy as np
import scipy.linalg as lin
import pylab as pl
#from scipy.linalg import toeplitz

# Baum algorithm
# Inputs: Observations (autocorrelation function of windowed signal), 
#			number of states (N=2-7), 
#			number of LPCs (P)
#			convergence ratio
# Outputs: State Transition Matrix (A)
#			Sigma_j for each state (bj(O_t)
#			LPCs for each state (c_jk)
#			Initial State Vector (pi)
#			Probability of each state at each time (alpha(t, j)beta(t, j))

class HMM:
	def __init__(self, observation_sequence, N):
		# observation sequence in a numpy array
		self.O = observation_sequence
		self.V = list(set(self.O)) # vocabulary
		self.V.sort() # put in alphabetical order because it makes it easier to read
		
		self.T = len(self.O) # length of the observation sequence
		
		self.M = len(self.V) # number of elements in stochastic matrix
		self.N = N # model order (integer)
		
		# other variables to deal with
		self.convergence = float('Inf'); # sum of diffs between reestimates
		
		# initalize the state transition matrix and stochastic process matrix
		self.A = np.random.rand(self.N, self.N)
		self._old_A = np.copy(self.A)
		#self.A = np.array([[1.0, 0.0000001, 0.0000001], [0.0000001, 1.0, 0.0000001],[0.0000001, 0.0000001, 1.0],])
		self.A.dtype = 'float64' # make sure we're at hight precision
		
		self.B = np.random.rand(self.N, self.M)
		self.B.dtype = 'float64'
		
		# normalize each row to sum to 1 (like good probabilities should)
		for i in range(self.N):
			self.A[i, :] = self.A[i, :]/sum(self.A[i, :])
			self.B[i, :] = self.B[i, :]/sum(self.B[i, :])
			
		self.pi = np.zeros((self.N, 1))
		self.pi[0] = 1.0;
		
		self.Prob = 0.0; # probability of the total observation sequence
		self.Prob_hist = []
		
		self.alpha = np.zeros((self.T, self.N), dtype='float64')
		self.beta = np.zeros((self.T, self.N), dtype='float64')
		self.c = np.ones(self.T, dtype='float64')
	

	
	def ObservationProbability(self, j, t):
		return self.B[j, self.V.index(self.O[t])]
		
	
	def ForwardBackward(self, scale = True):
		# compute alpha and beta for the forward-backward algorithm
		
		# reset alpha and beta
		self.alpha = np.zeros((self.T, self.N), dtype='float64')
		self.beta = np.zeros((self.T, self.N), dtype='float64')		
		
		# forward
		############################################
		
		# initialize alpha_1(j)
		for j in range(self.N):			
			self.alpha[0, j] = self.pi[j]*self.ObservationProbability(j, 0);
		
		# scale factor
		if scale == True:
			self.c[0] = 1/sum(self.alpha[0,:])
			self.alpha[0, :] = self.alpha[0, :]*self.c[0]
		
		#print self.alpha[0,:] #debug
		# compute alpha_t(j)
		for t in range(self.T-1):

			for j in range(self.N):
				self.alpha[t+1, j] = np.dot(self.alpha[t, :], self.A[:, j]) * self.ObservationProbability(j, t+1);
				
			# scale factor
			if scale == True:
				#print self.alpha.shape #debug
				self.c[t+1] = 1/sum(self.alpha[t+1, :])
				self.alpha[t+1, :] = self.alpha[t+1, :]*self.c[t+1]
			
				
		# backward
		############################################
		
		# initialize beta_T(j)
		for j in range(self.N):
			self.beta[self.T-1, j] = 1.0;
			
		# scale factor
		if scale == True:
			self.beta[self.T-1, :] = self.beta[self.T-1, :]*self.c[t]			
			
		# compute beta_t(i)
		for t in range(self.T-2, -1, -1): # loop from T-1 to 0
			for i in range(self.N):
				self.beta[t, i] = 0.0
				for j in range(self.N):
					self.beta[t, i] += self.A[i, j]*self.beta[t+1, j]*self.ObservationProbability(j, t+1)
					
			# scale factor
			if scale == True:
				self.beta[t, :] = self.beta[t, :]*self.c[t]
		
		
		# compute the probability of the observation
		############################################
		
		if scale == True:
			self.Prob = -sum(np.log(self.c))
		else:
			self.Prob = sum(self.alpha[self.T-1, :])
			
		self.Prob_hist.append(self.Prob)
				
		return 0
		
	def EstimateTransitionMatrix(self):
		#print 'A'
		self._old_A = np.copy(self.A)
		
		gamma = np.zeros((self.N, self.N))
		
		for i in range(self.N):	
			for j in range(self.N):
				for t in range(self.T-1):
					# compute upper sum
					gamma[i, j] += self.alpha[t, i]*self.A[i, j]*self.ObservationProbability(j, t+1)*self.beta[t+1, j]
			
			gamma_i = sum(gamma[i, :])

			for j in range(self.N):
				# add to convergence sum and save new a_ij
				self.convergence += abs(self.A[i, j] - 1.0*gamma[i, j]/gamma_i)
				self.A[i, j] = 1.0*gamma[i, j]/gamma_i
				
		return 0
		
	def EstimateStochasticMatrix(self):
	
		denominator = np.zeros(self.N)
		numerator = np.zeros((self.N, self.M))
		
		for j in range(self.N):
			for i in range(self.N):
				for t in range(self.T-1):
					numerator[j, self.V.index(self.O[t+1])] += self.alpha[t, i] * self.A[i, j] * self.ObservationProbability(j, t+1) * self.beta[t+1, j]
					
		# Loop through B matrix, check convergence and re-estimate
		for j in range(self.N): # commented here, indented the contents of the loop
			denominator[j] = sum(numerator[j, :])
			
			for k in range(self.M):
				# add to convergence factor
				self.convergence += abs(self.B[j, k] - 1.0*numerator[j, k]/denominator[j])
				self.B[j, k] = 1.0*numerator[j, k]/denominator[j]


	def TuringGoode(self, A=True, B=True):
		
		if A: # can turn on and off if necessary
			
			# apply the turing-good estimate to A and B
			epsilon = 1.0/(self.T+1)
			print epsilon
			
			# A first
			for i in range(self.N):
				# count how many zeros
				l = 0
				row_sum = 0
				
				for j in range(self.N):
					if self.A[i, j] < epsilon: #== 0:
						#print "FOUND ONE!"
						#print self.A[i,j]
						l += 1
					else:
						row_sum += self.A[i, j]
				#print row_sum
				# compute the sum for that row (denominator for formula)
				if l > 0:
									
					# replace each zero with new estimate
					for j in range(self.N):
						if self.A[i, j] < epsilon: #== 0:
							
							self.A[i, j] = epsilon
						else:
							self.A[i, j] = (self.A[i, j] / row_sum) * (1-l*epsilon)
			
							
		if B:				
			# apply the turing-good estimate to A and B
			epsilon = 1.0/(self.T+1)
			
			# compute B next
			for j in range(self.N):
				l = 0
				row_sum = 0
				for k in range(self.M):
					if self.B[j, k] < epsilon: #== 0:
						l += 1
					else:
						row_sum += self.B[j, k]
									
				if l > 0:
					#row_sum = sum(self.B[j, :])
					#print row_sum
					
					for k in range(M):
						if self.B[j, k] < epsilon: #== 0:
							self.B[j, k] = epsilon
						else:
							self.B[j, k] = (self.B[j, k] / row_sum) * (1-l*epsilon)
		
	def Converging(self, tolerance):
	  # check convergence sum and return logic value
		if self.convergence > tolerance:
			self.convergence = 0
			return True
		else:
			# we have converged, end the re-estimation loop
			return False
		
	def Baum(self, tolerance):

		while self.Converging(tolerance):
			self.ForwardBackward(scale=True)
			self.EstimateTransitionMatrix()
			self.EstimateStochasticMatrix()
			self.TuringGoode()
			print "%i: %f (%f)"%(len(self.Prob_hist), self.convergence, self.Prob_hist[-1])

	def EstimateStateSequence(self, plot = True):
		state_sequence_probability = self.alpha*self.beta
		state_sequence = np.zeros((len(state_sequence_probability), 1))
		for t in range(self.T):
			state_sequence[t] = np.argmax(state_sequence_probability[t])

		if plot:
			pl.plot(state_sequence)
			pl.show()

	def Generator(self, length, initial_state = 0):
		
		#print "Doesn't actually generate yet..."
		# random vectors for state transition
		state_driver = np.random.rand(length, 1)
		# and observation generation
		observation_driver = np.random.rand(length, 1)
		
		output = ""
		#output_raw = []
		#state_raw = []
		
		state = initial_state
		state_raw.append(initial_state)
		for l in range(length):
		
			k = 0;
			#print "threshold: %f\n"%(observation_driver[l, 0])
			while observation_driver[l, 0] > np.sum(self.B[state, 0:k+1]) and k <= self.M:
				#print np.sum(self.B[state, 0:k+1])
				k=k+1
			#print k
			output = output + self.V[k]
			#output_raw.append(k)
			
			#print self.V[k]
			
			j = 0;
			#print "state: %f\n"%(state_driver[l, 0])
			while state_driver[l, 0] > np.sum(self.A[state, 0:j+1]) and j <= self.N:
				#print np.sum(self.A[state, 0:j+1])
				j=j+1
			#print j
			state = j#self.A[state, j]
			#state_raw.append(state)
			
		#print output_raw
		#print state_raw
		print output
		
		

