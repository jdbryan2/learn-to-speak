from lpchmm import *
import numpy as np
import pylab as pl
import scipy.linalg as lin
#from scipy.linalg import toeplitz


signal_length = 500000
state_length = 1000

excitation = np.array([0.0 for t in range(signal_length)])
signal = np.array([0.0 for t in range(signal_length)])
for t in range(signal_length):
	if t%50 == 0:
		excitation[t] = 1.0
	
lpc = np.array([[-1.0/10], [1.0/10], [-2.0/3], [2.0/3]])
lpc = np.array([[0.0, 0.25], [1.0/2.0, -1.0/4]])
print lpc
p = len(lpc[0])

for state in range(signal_length/state_length):
	for t in range(state*state_length, state_length*(state+1)):
		#if state%2:
		for i in range(0, p):
			#print lpc[state%3]
			signal[t] = signal[t] + signal[t-i-1]*lpc[state%len(lpc), i]
		signal[t] = signal[t] + excitation[t]
		#else:
		#	for i in range(p):
		#		signal[t] = signal[t] + signal[t-i]*lpc[1, i]
		#	signal[t] = signal[t] + excitation[t-p]		
	

step_size = 20
window_size = 100
observations = []


# walk along the signal 
for i in range(signal_length/step_size):
	window_start = i*step_size
	if window_start+window_size+1 < signal_length:
		# grab the autocorrelation function for each window
		acf = np.correlate(signal[window_start:window_start+window_size], signal[window_start:window_start+window_size], "full")
		#print acf
		acf = acf[window_size-1:]#/window_size # wtf is this?????
		#print acf[0]

		# append to observation array
		observations.append(acf)
		
		R = lin.toeplitz(acf[range(p)])
		RR = lin.toeplitz(acf[range(p+1)])
		#print RR

		r = acf[range(1, p+1)]
		a = -np.dot(lin.inv(R), r)
		a = np.insert(a, 0, 1)
		a_ = np.mat(a)
		#print a
		#print np.dot(a, np.dot(RR, a))#, a_*R*a_.T
		#print np.dot(lin.inv(R), r)
		#print i

	
if True:
	#convert to numpy array for hmm input
	observations = np.array(observations)

	lpchmm = LPCHMM(observations, N=2, P=3)#p+1)
	lpchmm.Baum(0.0001)
	print lpchmm.A
	np.save("A", lpchmm.A)
	np.save("alpha", lpchmm.alpha)
	np.save("beta", lpchmm.beta)
	np.save("B", lpchmm.B)
	np.save("LPCs", lpchmm.L)
	np.save("sigma2", lpchmm.sigma2)
	
	



