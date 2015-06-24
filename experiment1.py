from lpchmm import *

import numpy as np
import pylab as pl
import scipy.linalg as lin
import scipy.signal as sig
import scipy.io.wavfile as wav
import os

# load sound file
# pull in recording from a wav file
x = wav.read("harvard-sentences.wav")

# decimate to 8kHz 
print "Original Sample Rate: %i" % x[0]
sample_freq = x[0]/6;
signal = sig.decimate(np.array(x[1]), 6);
signal = signal/np.max(signal)

# setup window and step size variables
step_size = 80#64
window_size = 240# 256
observations = []
signal_length = len(signal)

#pl.plot(signal)
#pl.show()

# Loop through the signal
# walk along the signal 
print "Stepping through signal and computing autocorrelation..."
for i in range(signal_length/step_size):
	window_start = i*step_size
	if window_start+window_size+1 < signal_length:
		# grab the autocorrelation function for each window
		acf = np.correlate(signal[window_start:window_start+window_size], signal[window_start:window_start+window_size], "full")

		acf = acf[window_size-1:]#/window_size # wtf is this?????

		# append to observation array
		observations.append(acf)
		
print "Number of Observations: %i"%len(observations)
    
# initialize LPCHMM class object
if True:
	#convert to numpy array for hmm input
	observations = np.array(observations)
	
	n = 6 #number of states
	p = 4 # number of poles in LPC filter

	save_dir = "Experiment1"
	save_dir = save_dir+"_n"+str(n)+"_p"+str(p)

	# create directory to save in if none exists
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_dir = save_dir+"/numpy_files/"


	# create directory to save in if none exists
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	lpchmm = LPCHMM(observations, N=n, P=p+1)
	lpchmm.Baum(0.0001)
	print lpchmm.A
	np.save(save_dir+"A_n"+str(n)+"_p"+str(p), lpchmm.A)
	np.save(save_dir+"alpha_n"+str(n)+"_p"+str(p), lpchmm.alpha)
	np.save(save_dir+"beta_n"+str(n)+"_p"+str(p), lpchmm.beta)
	np.save(save_dir+"B_n"+str(n)+"_p"+str(p), lpchmm.B)
	np.save(save_dir+"LPCs_n"+str(n)+"_p"+str(p), lpchmm.L)
	np.save(save_dir+"sigma2_n"+str(n)+"_p"+str(p), lpchmm.sigma2)	
	np.save(save_dir+"c_n"+str(n)+"_p"+str(p), lpchmm.c) # scale factor
	np.save(save_dir+"prob_n"+str(n)+"_p"+str(p), lpchmm.Prob_hist)
	np.save(save_dir+"sig_params_n"+str(n)+"_p"+str(p), np.array([sample_freq, window_size, step_size]))

