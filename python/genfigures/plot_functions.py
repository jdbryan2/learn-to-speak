
import numpy as np
import pylab as plt
import os
from matplotlib2tikz import save as _tikz_save

def tikz_save(fname):
    _tikz_save(fname, 
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth')

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    if data.shape[1] < max_length:
        max_length= data.shape[1]

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/8000.))

def MultiPlotTraces(data, rows, max_length, sample_period, highlight=[0], highlight_style='b-'):
    if data.shape[1] < max_length:
        max_length= data.shape[1]

    for k in range(rows.size):
        if k in highlight: 
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/8000.))

def PlotDistribution(ave, std, rows):
    plt.plot(ave[rows], 'bo-', linewidth=2)
    plt.plot(ave[rows]-std[rows], 'ro--', linewidth=1)
    plt.plot(ave[rows]+std[rows], 'ro--', linewidth=1)


def get_last_index(directory, base_name = 'data'):
    index_list = []  # using a list for simplicity
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith(base_name) and filename.endswith(".npz"):
                index = filter(str.isdigit, filename)
                if len(index) > 0:
                    index_list.append(int(index))

    if len(index_list):
        return max(index_list)
    else:
        return 0
