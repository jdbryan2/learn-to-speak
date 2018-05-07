
import numpy as np
import pylab as plt

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    if data.shape[1] < max_length:
        max_length= data.shape[1]

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/8000.))

def PlotDistribution(ave, std, rows):
    plt.plot(ave[rows], 'bo-', linewidth=2)
    plt.plot(ave[rows]-std[rows], 'ro--', linewidth=1)
    plt.plot(ave[rows]+std[rows], 'ro--', linewidth=1)


