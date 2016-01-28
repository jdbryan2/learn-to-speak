#from lpchmm import *

import numpy as np
import pylab as pl
#import scipy.linalg as lin
import scipy.signal as sig
import scipy.io.wavfile as wav
import os

####################
# USEFUL FUNCTIONS
####################

def latexify(matrix):
    output = '';

    # build the tabular tag
    output = "\begin{tabular}{" # }\n"
    
    # all columns are defaulted to centered
    for c in range(len(matrix[0])):
        output = output+"c"
        if c < len(maxtrix[0])-1:
            output = output+"|"
        else:
            output = output+"}\n"


    for r in range(len(matrix)):
        output = output + "\\hline\n"
        for c in range(len(matrix[r]) - 1):
            output = output + "%2.5f &" % matrix[r][c]
            
        output = output +  "%2.5f " % matrix[r][len(matrix[r])-1]
        output = output + "\\\\ \n"
    
    output = output+"\end{tabular}\n"
    return output

# experiment parameters
n = 6
p = 4

# setup window and step size variables
step_size = 64


# load sound file
# pull in recording from a wav file
x = wav.read("harvard-sentences.wav")

# decimate to 8kHz 
print "Original Sample Rate: %i" % x[0]
sample_freq = x[0]/6;
signal = sig.decimate(np.array(x[1]), 6);
signal = signal/np.max(signal)


################################### 
#Load all the results parameters
################################### 

save_dir = "Experiment1"
save_dir = save_dir+"_n"+str(n)+"_p"+str(p)+"/"
print save_dir

# create directory to save in if none exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

A = np.load(save_dir+"numpy_files/A_n"+str(n)+"_p"+str(p)+".npy")
alpha =	np.load(save_dir+"numpy_files/alpha_n"+str(n)+"_p"+str(p)+".npy")
beta = np.load(save_dir+"numpy_files/beta_n"+str(n)+"_p"+str(p)+".npy")
B =np.load(save_dir+"numpy_files/B_n"+str(n)+"_p"+str(p)+".npy")
lpc = np.load(save_dir+"numpy_files/LPCs_n"+str(n)+"_p"+str(p)+".npy")
sigma2 = np.load(save_dir+"numpy_files/sigma2_n"+str(n)+"_p"+str(p)+".npy")
sample_freq, window_size, step_size = np.load(save_dir+"numpy_files/sig_params_n"+str(n)+"_p"+str(p)+".npy")

# plot frequency responses of the lpcs
for j in range(n):

    w,h = sig.freqz(sigma2[j], lpc[j])
    phase = np.unwrap(np.angle(h))
    h = np.abs(h)

    pl.figure(figsize=(7,3.5))

    pl.plot(w, h)
#    pl.hold(True)
#    pl.plot(w, phase)

    pl.grid(True)
    pl.xlabel("$\omega$")
    pl.ylabel(r"$\|H(j\omega)\|$")
    pl.subplots_adjust(right=0.84, bottom=0.15)

    pl.savefig(save_dir+"lpc"+str(j) + "_n"+str(n)+"_p"+str(p)+".png")


total_likelihood = np.sum(alpha*beta, 1)
state_prob = np.zeros((n,len(total_likelihood)))
for j in range(n):
    state_prob[j] = alpha[:, j]*beta[:,j]/total_likelihood

start = 10 # in seconds
length = 10 # in seconds

start = start*sample_freq/step_size
length = length*sample_freq/step_size

print "Start: " + str(start)
print "Stop: " + str(start+length)


# plot a segment of the state history
for j in range(n):
    pl.figure(figsize = (7, 3.5))
    pl.plot(state_prob[j, start:start+length])
    #pl.show()
    pl.subplots_adjust(right=0.84, bottom=0.15)
    
    pl.savefig(save_dir+"statehist"+str(j) + "_n"+str(n)+"_p"+str(p)+".png")
    



# plot and save the figure now
pl.figure(figsize = (7,3.5))
pl.plot(signal[start*step_size:start*step_size+length*step_size])
pl.subplots_adjust(right=0.84, bottom=0.15)
pl.savefig(save_dir+"sig_seg_n"+str(n)+"_p"+str(p)+".png")

pl.close('all')



# axes is one dim array
fig, axes = pl.subplots(n+1, sharex=True)
time = np.arange(0, length)

signal_time = np.arange(0, length*step_size)
#axes[0].set_autoscaley_on(False)
axes[0].plot(signal_time/step_size, signal[start*step_size:start*step_size+length*step_size])
#pl.ylim([0,1])

for j in range(n):
    axes[j+1].set_autoscaley_on(False)
    axes[j+1].plot(time, alpha[start:start+length, j]*beta[start:start+length,j]/total_likelihood[start:start+length])
    pl.ylim([0, 1])
                      
pl.figure()
pl.plot(signal[start:start+length*step_size/4])

pl.show()
signal_state_prob = np.repeat(state_prob, step_size, axis=1)
print len(signal_state_prob[0])
#original_signal = np.array(x[1])
print len(signal)

for j in range(n):
    state_selector = signal_state_prob[j] > 0.8
    pl.close('all')
    wav_sig = np.asarray(signal[state_selector]*3000, dtype = np.int16) # make ready for saving as .wav
    #pl.plot(wav_sig)
    #pl.show()
    wav.write(save_dir+"state"+str(j) + "_n"+str(n)+"_p"+str(p)+".wav", sample_freq, wav_sig)



