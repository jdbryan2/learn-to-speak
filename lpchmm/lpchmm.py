
# Hidden Markov Models for Autoregressive signals
# using linear prediction coefficients
#
# Written by Jacob Bryan
# April 2013

import random
import math 
import cmath
import numpy as np
import scipy.linalg as lin
from hmm import HMM

class LPCHMM(HMM):
	#################################################################
	# Init - prep all relevant HMM/LPC variables
	#################################################################
	def __init__(self, observation_sequence, N, P = 2):
	
		
		# initialize all other HMM variables through parent class
		# observation sequence in a numpy array
		self.O = observation_sequence
		
		self.T = len(self.O) # length of the observation sequence
		
		self.N = N # model order (integer) (A = NxN matrix)
		
		# number of poles in the linear predictive filter
		self.P = P
		
		# initialize LPC variables
		self.L = np.zeros((self.N, self.P))
		# compute random zeros within the unit circuit and then generate the filter using them
		for j in range(self.N):
			
			zeros = []
			
			if (self.P)%2 == 0:
				zeros.append(np.random.random_sample()*2.0 - 1)
			
			if self.P > 2:
				coords = np.random.rand(np.floor((self.P-1)/2), 2)
				for p in range(int((self.P-1)/2)):
					zeros.append(coords[p, 0]*cmath.exp(cmath.pi*1j*coords[p, 1]))
					zeros.append(coords[p, 0]*cmath.exp(-cmath.pi*1j*coords[p, 1]))
			
			self.L[j, :] = np.poly(zeros)
			#self.L[j, :] = self.L[j, :]/self.L[j, self.P-1]
			

		print "#####################################"
		print self.L
		print "#####################################"

		self._old_L = np.copy(self.L)
#		self.L = np.array([[1, 1.0/10], [1, -1.0/10], [1, 2.0/3], [1, -2.0/3]])#np.array([[1, -1.0/100], [1, -2.0/3], [1, -1.0/7]])
		self.sigma2 = np.random.rand(self.N) # sigma squared...
		self._old_sigma2 = np.copy(self.sigma2)
		#self.sigma2 = np.array([4.0, 4.0, 4.0, 4.0])
		self.R = np.zeros((self.N, self.P))

		# other variables to deal with
		self.convergence = float('Inf'); # sum of diffs between reestimates
		
		# initalize the state transition matrix and stochastic process matrix
		self.A = np.random.rand(self.N, self.N)
#		a = 0.2 # a < 1 must hold
#		self.A = np.identity(self.N)*(1-a*self.N/(self.N-1))+np.ones((self.N, self.N))*a/self.N
#		self.A = np.array([[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]])
		self.A.dtype = 'float64' # make sure we're at high precision
		
		# normalize each row to sum to 1 (like good probabilities should)
		for i in range(self.N):
			self.A[i, :] = self.A[i, :]/sum(self.A[i, :])

		self._old_A = np.copy(self.A)
			
		# B_j(O_t) is now it's own matrix
		self.B = np.zeros((self.N, self.T), dtype='float64')
		
		# initialize pi as [1, 0, 0, 0, ... ] - reestimation unnecessary
		self.pi = np.zeros((self.N, 1))
		self.pi[0] = 1.0;
		
		self.Prob = 0.0; # probability of the total observation sequence
		self.Prob_hist = []
		
		self.alpha = np.zeros((self.T, self.N), dtype='float64')
		self.beta = np.zeros((self.T, self.N), dtype='float64')
		self.c = np.ones(self.T, dtype='float64')

	#################################################################
	# ObservationProbability 
	#################################################################
	# return the probability of each observation given the state
	def ObservationProbability(self, j, t):
		return self.B[j, t]
		
	# End ObservationProbability 
	
	#################################################################
	# PrepObservations 
	#################################################################
	# Compute the array of the observation probabilities so they only 
	# need to be computed once per loop. 
	def PrepObservations(self):
		for j in range(self.N):
			coeff = (2/(np.sqrt(2*math.pi*self.sigma2[j])))
			for t in range(self.T):
				R = lin.toeplitz(self.O[t, 0:self.P])
			
				self.B[j, t] = coeff*math.exp(-np.dot(np.dot(self.L[j,:], R), self.L[j,:])/(2*self.sigma2[j]))
#				print coeff*math.exp(-np.dot(np.dot(self.L[j,:], R), self.L[j,:])/(2*self.sigma2[j]))
#				print coeff
#				print -np.dot(np.dot(self.L[j,:], R), self.L[j,:])/(2*self.sigma2[j])
#				print R
#				print self.B[j, t]

	#################################################################
	# EstimateSigma2(self):
	#################################################################
	# Compute sigma squared based on reestimation formula derived from 
	# autoregressive model.
	def EstimateSigma2(self):

		self._old_sigma2 = np.copy(self.sigma2)
		
		
		for j in range(self.N):
			R = lin.toeplitz(self.R[j])
			self.sigma2[j] = np.dot(np.dot(self.L[j,:], R), self.L[j,:])/np.sum(self.alpha[:,j]*self.beta[:,j])
			
#			print "Version 1 (sigma):"
#			print np.dot(np.dot(self.L[j,:], R), self.L[j,:])
#			print np.sum(self.alpha[:,j]*self.beta[:,j])
#			denominator = 0
#			numerator = 0
#			
#			
#			for t in range(self.T):
#				R = lin.toeplitz(self.O[t, 0:self.P]) # I should consider converting the observation matrix to this format on initialization
#				numerator = numerator + self.alpha[t,j]*self.beta[t,j]*np.dot(np.dot(self.L[j,:], R), self.L[j,:])
#				denominator = denominator + self.alpha[t,j]*self.beta[t,j]
#			
#			self.sigma2[j] = numerator/denominator
#			print "Version 2 (sigma):"
#			print numerator
#			print denominator

	#################################################################
	# EstimateR(self)
	#################################################################
	# Compute the weighted sum of autocorrelation functions for each state
	def EstimateR(self):

		for j in range(self.N):
			for p in range(self.P):
				self.R[j, p] = np.sum(self.alpha[:, j]*self.beta[:, j]*self.O[:, p])


	#################################################################
	# EstimateLPCs(self):
	#################################################################
	# Compute linear prediction coefficients based on reestimation formula derived from 
	# autoregressive model.
	def EstimateLPCs(self):
		self._old_L = np.copy(self.L)
		for j in range(self.N):
			X = lin.toeplitz(self.R[j, 0:self.P-1])
			self.L[j, 1:self.P+1] = -np.dot(lin.inv(X), self.R[j, 1:self.P])


	def MeasureConvergence(self):
		self.convergence = np.sum(np.fabs(self.A-self._old_A)/np.fabs(self._old_A)) \
							+ np.sum(np.fabs(self.L-self._old_L)/np.fabs(self._old_L)) \
							+ np.sum(np.fabs(self.sigma2-self._old_sigma2)/np.fabs(self._old_sigma2))

	#################################################################
	# Baum
	#################################################################
	# Iterate the reestimations until convergence. The tolerance is 
	# set in the function parameters. The function can also be called
	# multiple times in a row and it picks up where it left off.
	def Baum(self, tolerance):

		while self.Converging(tolerance):
			
			
			self.PrepObservations()
			#break
			self.ForwardBackward(scale=True)

			self.EstimateTransitionMatrix()
			self.EstimateR()
			self.EstimateLPCs()
			self.EstimateSigma2()		
	
			self.TuringGoode(A=True, B=False) # only do this for A
			
			self.MeasureConvergence()
			print "%i: %f"%(len(self.Prob_hist), self.convergence)
			print "State Transition Matrix:"
			print self.A
			print "Linear Prediction Coefficients:"
			print self.L
			print "Sigma Squared:"
			print self.sigma2
			print "Weighted Autocorrelation:"
			print self.R
# if __name__ == '__main__':
	
