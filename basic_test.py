from lpchmm import *
import numpy as np
import pylab as pl
import scipy.linalg as lin
#from scipy.linalg import toeplitz


signal_length = 50000
state_length = 1000

excitation = np.array([0.0 for t in range(signal_length)])
signal = np.array([0.0 for t in range(signal_length)])
for t in range(signal_length):
	if t%50 == 0:
		excitation[t] = 1.0
	
lpc = np.array([[1, -1.0/2, -1.0/2, -1.0/2], [1, -1.0/4, -1.0/4, -1.0/4]])
p = len(lpc[0])

for state in range(signal_length/state_length):
	for t in range(state*state_length, state_length*(state+1)):
		#if state%2:
		for i in range(0, p):
			#print lpc[state%3]
			signal[t] = signal[t] + signal[t-i]*lpc[state%len(lpc), i]
		signal[t] = signal[t] + excitation[t]
		#else:
		#	for i in range(p):
		#		signal[t] = signal[t] + signal[t-i]*lpc[1, i]
		#	signal[t] = signal[t] + excitation[t-p]		
	

step_size = 50
window_size = 200
observations = []


# walk along the signal 
for i in range(signal_length/step_size):
	window_start = i*step_size
	if window_start+window_size+1 < signal_length:
		# grab the autocorrelation function for each window
		acf = np.correlate(signal[window_start:window_start+window_size], signal[window_start:window_start+window_size], "full")
		acf = acf[window_size-1:]/window_size


		# append to observation array
		observations.append(acf)
		
		R = lin.toeplitz(acf[range(3)])

		r = acf[range(1, 4)]
		a = np.dot(lin.inv(R), r)
		a_ = np.mat(a)
		
		print "-----------------------"
		print acf[0]
#		if i > 100:
#			break
		print np.dot(a, np.dot(R, a)), a_*R*a_.T
		print np.dot(lin.inv(R), r)
		print i

if True:
	#convert to numpy array for hmm input
	observations = np.array(observations)

	[Prob, A, C, sigma, pi, iterations, state_probabilities] = LPCHMM(observations, N=2, P=4, convergence_ratio = 10**(-2))
	pl.plot(Prob)
	pl.show()
	print A
	print ' '
	print C
	print ' '
	print sigma
	print ' '
	print pi
	print ' '
	#print state_probabilities



