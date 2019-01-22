
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib2tikz import save as tikz_save
import scipy.signal as signal

from scipy.io import wavfile
directory='../data/utterances/syllables'
fnames = ['di_syllable', 'du_syllable']

for fname in fnames:
    fs, sound = wavfile.read(directory+'/'+fname+'.wav')
    sound = 1.*sound/np.max(sound) # normalize just because

    nfft = 512
    window = np.hamming(200)
    f,t,Sxx = signal.stft(sound, fs=fs, window=window, nperseg=window.size,noverlap=int(window.size*0.90), nfft=nfft)
    plt.figure()
    plt.pcolormesh(t, f, np.log(np.abs(Sxx)), cmap='bone_r')
    plt.ylabel('Freq (Hz)')
    plt.xlabel('Time (sec)')
    #plt.xlim([0, 0.5])
    #plt.colorbar()
    tikz_save(directory+'/'+fname+'.tikz',
              figureheight = '\\figureheight',
              figurewidth = '\\figurewidth')


    plt.show()
