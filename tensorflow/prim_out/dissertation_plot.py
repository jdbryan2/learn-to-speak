import numpy as np
import os
import pylab as plt
import scipy.signal as signal
from genfigures.plot_functions import *

def moving_average(a, n=3) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    ret[n-1:, :] = ret[n - 1:, :] / n
    for k in range(n-1):
        ret[k, :] = ret[k, :]/(k+1)
    return ret
    #return ret / n


directory = 'primtest3_1'
filename = 'b77'
#if os.path.exists(directory):
#    for filename in os.listdir(directory):
print filename
data = np.load(directory+'/'+filename+'/model_data.npz')
print data.keys()

# load data into local variables
y = data['y']
x = data['x']
h = data['h']
y_hat = data['y_hat']
x_hat = data['x_hat']
h_hat = data['h_hat']

h_std = data['h_std_inst']
h_hat_std = data['h_hat_std_inst']

data = np.load(directory+'/'+filename+'/data0.npz')
sound_hat = data['sound_wave']

data = np.load('../../python/data/rand_prim_1sec/data'+filename[1:]+'.npz')
sound = data['sound_wave']

# compute spectrograms
nfft = 512
window = np.hamming(200)
f,t,spectrum= signal.stft(sound,
                        fs=8000,
                        window=window,
                        nperseg=window.size,
                        noverlap=int(window.size*0.95),
                        nfft=nfft)

f,t,spectrum_hat = signal.stft(sound_hat,
                        fs=8000,
                        window=window,
                        nperseg=window.size,
                        noverlap=int(window.size*0.95),
                        nfft=nfft)

plt.figure()
plt.subplot(2,1,1)
plt.pcolormesh(t, f/1000., np.log(np.abs(spectrum)), cmap='bone_r')
plt.ylabel('Freq (Hz)')
#plt.xlabel('Time (sec)')
plt.xlim([0, 1.0])

plt.subplot(2,1,2)
plt.pcolormesh(t, f/1000., np.log(np.abs(spectrum_hat)), cmap='bone_r')
plt.ylabel('Freq (kHz)')
plt.xlabel('Time (sec)')
plt.xlim([0, 1.0])

# make directory if it doesn't exist
if not os.path.exists(directory+'/'+filename+'/figures'):
    os.makedirs(directory+'/'+filename+'/figures')

# save tikz figure
#tikz_save(directory+'/'+filename+'/figures/spectrum_'+filename+'.tikz',
#            data_path='tikz/ICE/')
plt.close()



time = np.arange(0.03, 1.01, 0.01)
plt.figure()
for k in range(10):
    plt.subplot(5, 2, k+1)
    plt.plot(time, x[:, k], 'b')
    plt.plot(time, x_hat[:, k], 'r')
    plt.title("Input "+str(k))
    plt.ylim([-1, 1])
    if k > 7:
        plt.xlabel("Time (sec)")
    plt.tight_layout()

#tikz_save(directory+'/'+filename+'/figures/control_input_'+filename+'.tikz',
#            data_path='tikz/ICE/')
plt.close()


h_std = moving_average(h_std, n=10) # moving average
h_std = np.exp(h_std)#**2

h_hat_std = moving_average(h_hat_std, n=10) # moving average
h_hat_std = np.exp(h_hat_std)#**2
plt.figure()
for k in range(10):
    plt.subplot(5, 2, k+1)
    plt.plot(time, h[:, k], 'b')
    #plt.plot(time, h[:, k] - h_std[:, k], 'b--', alpha=0.5)
    #plt.plot(time, h[:, k] + h_std[:, k], 'b--', alpha=0.5)
    plt.plot(time, h_hat[:, k], 'r')
    plt.plot(time, h_hat[:, k] - h_hat_std[:, k], 'r--', alpha=0.5)
    plt.plot(time, h_hat[:, k] + h_hat_std[:, k], 'r--', alpha=0.5)
    plt.title("Primitive "+str(k))
    #plt.ylim([-1, 1])
    if k > 7:
        plt.xlabel("Time (sec)")
    plt.tight_layout()


tikz_save(directory+'/'+filename+'/figures/primitives_'+filename+'.tikz',
            data_path='tikz/ICE/')
#plt.close()
plt.show()



