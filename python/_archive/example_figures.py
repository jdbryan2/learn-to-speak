
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib2tikz import save as tikz_save
import scipy.signal as signal

from DataHandler import DataHandler

directory = 'click'
# click requires some zooming to be able to see it

directory = 'apa'
directory = 'sigh'
directory = 'ejective'
dh = DataHandler()
dh.LoadDataDir(directory)
#dh.SaveAnimation(dirname=directory)

print dh.data.keys()
print dh.data['sound_wave'][0].size

sound = dh.data['sound_wave'][0]
time = np.arange(sound.size)/8000.
#plt.plot(time[0:1600], sound[1600:3200])
plt.plot(time, sound)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
#plt.xlim([0., 0.2])
plt.xlim([0., 0.5])
tikz_save(
    'data/'+directory+'/amplitude.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth'
    )

nfft = 512
window = np.hamming(200)
f,t,Sxx = signal.stft(sound, fs=8000, window=window, nperseg=window.size,noverlap=int(window.size*0.95), nfft=nfft)
plt.figure()
#plt.pcolormesh(t[0:161]-t[0], f, np.log(np.abs(Sxx[:, 141:302])), cmap='bone_r')
plt.pcolormesh(t, f, np.log(np.abs(Sxx)), cmap='bone_r')
plt.ylabel('Freq (Hz)')
plt.xlabel('Time (sec)')
plt.xlim([0, 0.5])
#plt.colorbar()
tikz_save(
    'data/'+directory+'/spectrum.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth'
    )


plt.show()
