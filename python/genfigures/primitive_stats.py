
import os
import numpy as np
import pylab as plt
from matplotlib2tikz import save as tikz_save
from primitive.PrimitiveHandler import PrimitiveHandler
from plot_functions import *

import config

tubes = config.TUBES

def PlotDistribution(ave, std, rows, offset=0):
    x = np.arange(rows.size)+offset
    plt.plot(x, ave[rows], 'bo-', linewidth=2)
    plt.plot(x, ave[rows]-std[rows], 'ro--', linewidth=1)
    plt.plot(x, ave[rows]+std[rows], 'ro--', linewidth=1)


dirname = '../data/batch_random_20_10'

ind= get_last_index(dirname, 'round')

ind= get_last_index(dirname, 'round')
load_fname = 'round%i.npz'%ind

savedir = dirname + '/figures/stats/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

ph = PrimitiveHandler()
ph.LoadPrimitives(fname=load_fname, directory=dirname)
plt.figure()
PlotDistribution(ph._mean, ph._std, ph.Features.pointer[ph.Features.control_action])
plt.xlabel('Articulator')
plt.ylabel('$\\mu \\pm \\sigma$')
plt.grid(True)
tikz_save(savedir + 'articulator_stats.tikz')
plt.show()

glottis = 29
plt.figure()
PlotDistribution(10000.*ph._mean, 10000.*ph._std, ph.Features.pointer['area_function'][:glottis])
plt.ylabel('Area Function (cm$^2$)')
plt.xlabel('Tube Segment')
#plt.xlabel('Area Function')
#plt.ylabel('$\\mu \\pm \\sigma$')
plt.grid(True)
tikz_save(savedir + 'lung_area_stats.tikz')
plt.show()

plt.figure()
PlotDistribution(10000.*ph._mean, 10000.*ph._std, ph.Features.pointer['area_function'][glottis:-1])
PlotDistribution(10000.*ph._mean, 10000.*ph._std,
                 np.array([ph.Features.pointer['area_function'][-1],ph.Features.pointer['area_function'][-1]]),
                 offset=29)
plt.ylabel('Area Function (cm$^2$)')
plt.xlabel('Tube Segment')
#plt.xlabel('Area Function')
#plt.ylabel('$\\mu \\pm \\sigma$')
plt.grid(True)
tikz_save(savedir + 'tract_area_stats.tikz')
plt.show()
