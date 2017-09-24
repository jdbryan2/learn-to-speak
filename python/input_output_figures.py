

import os
import numpy as np
import pylab as plt
from primitive.SubspacePrim import PrimLearn
from matplotlib2tikz import save as tikz_save

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    if data.shape[1] < max_length:
        max_length= data.shape[1]

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/1000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/1000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/1000.))

def PlotDistribution(ave, std, rows):
    plt.plot(ave[rows], 'bo-', linewidth=2)
    plt.plot(ave[rows]-std[rows], 'ro--', linewidth=1)
    plt.plot(ave[rows]+std[rows], 'ro--', linewidth=1)


dim = 8
sample_period = 10
dirname = 'full_random_50'
dirname = 'full_random_50_prim'
savedir = 'data/' + dirname + '/figures/in_out/'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals

if not os.path.exists(savedir):
    os.makedirs(savedir)

ss = PrimLearn()
ss.ConvertDataDir(dirname, sample_period=sample_period)
#ss.LoadDataDir(dirname)
#ss.ConvertData(sample_period=sample_period)
#ss.PreprocessData(50, 10, sample_period=down_sample)
#ss.SubspaceDFA(dim)
ss.LoadPrimitives(load_fname)

ss.EstimateStateHistory(ss._data)

plt.figure()
PlotTraces((ss._data.T*ss._std+ss._ave).T, ss.features['art_hist'], 1000, sample_period, highlight=0)
#PlotTraces((ss._data.T*ss._std+ss._ave).T, ss.features['lung_pressure'], 1000, sample_period, highlight=0, highlight_style='r--')
plt.xlabel('Time (s)')
plt.ylabel('Articulator Value')
plt.grid(True)
tikz_save(savedir + 'articulator_traces.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')
plt.show()

plt.figure()
PlotDistribution(ss._ave, ss._std, ss.features['art_hist'])
plt.xlabel('Articulator')
plt.ylabel('$\\mu \\pm \\sigma$')
plt.grid(True)
tikz_save(savedir + 'articulator_stats.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')
plt.show()

plt.figure()
PlotTraces((ss._data.T*ss._std+ss._ave).T/(ATM), ss.features['lung_pressure'], 1000, sample_period, highlight=1)
lung_ave = ss._ave[ss.features['lung_pressure']]
lung_std = ss._std[ss.features['lung_pressure']]
plt.plot([0, 10], np.ones(2)*lung_ave/ATM, 'b-', linewidth=2)
plt.plot([0, 10], np.ones(2)*(lung_ave-lung_std)/ATM, 'r--')
plt.plot([0, 10], np.ones(2)*(lung_ave+lung_std)/ATM, 'r--')
plt.xlabel('Time (s)')
plt.ylabel('Lung Pressure (atm)')
plt.grid(True)
tikz_save(savedir + 'lung_traces.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')
plt.show()

plt.figure()
PlotDistribution(ss._ave, ss._std, ss.features['area_function'])
plt.xlabel('Area Function')
plt.ylabel('$\\mu \\pm \\sigma$')
plt.grid(True)
tikz_save(savedir + 'area_stats.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')
plt.show()



################################################################################
if ss.h.shape[1] < 1000:
    hist_len = ss.h.shape[1]
else:
    hist_len = 1000
print hist_len
plt.plot(np.arange(ss._past, hist_len)*sample_period/1000., ss.h[:, ss._past:hist_len].T)
plt.xlim((0, hist_len*sample_period/1000.))
plt.xlabel('Time (s)')
plt.ylabel('$h_t$')
plt.title('State History (not saved by this script)')

plt.show()
