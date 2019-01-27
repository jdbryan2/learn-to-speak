

import os
import numpy as np
import scipy.signal as sig
from scikits.talkbox import segment_axis

from genfigures.plot_functions import get_index_list

def reshape_data(data,history):
    max_len = int(np.floor(data.shape[0]/history)*history)
    return data[:max_len, :].reshape((-1, data.shape[1]*history))


def LoadData(directory='speech_io', outputs_name='mfcc', inputs_name='action_segs', states_name='state_segs', shuffle=True, seq_length=1):

    ind_list = np.array(get_index_list(directory=directory, base_name='data'))

    ind_list = np.sort(ind_list)

    outputs = np.array([])
    inputs = np.array([])
    states = np.array([])
    print "Loading Data..."
    for k in ind_list:
        data = np.load(directory+"/data%i.npz"%k)

        if outputs.size:
            outputs = np.append(outputs, data[outputs_name], axis=0)
            inputs = np.append(inputs, data[inputs_name][:, :, -1], axis=0)
            states = np.append(states, data[states_name][:, :, -1], axis=0)
            #spec = np.append(spec, data['mel_spectrum'], axis=0)
        else:
            outputs = data[outputs_name]
            inputs = data[inputs_name][:, :, -1]
            states = data[states_name][:, :, -1]
            #spec = data['mel_spectrum']

    if seq_length > 1:

        print outputs.shape
        outputs = segment_axis(outputs, length=seq_length, overlap=seq_length-1, axis=0, end='cut')
        print outputs.shape
        outputs = outputs.reshape((outputs.shape[0], outputs.shape[1]*outputs.shape[2]))

        inputs = segment_axis(inputs, length=seq_length, overlap=seq_length-1, axis=0, end='cut')
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1]*inputs.shape[2]))

        states = segment_axis(states, length=seq_length, overlap=seq_length-1, axis=0, end='cut')
        states = states.reshape((states.shape[0], states.shape[1]*states.shape[2]))

    #plt.imshow(spec.T, aspect=3*20)
    #plt.show()
    remove = ~np.isnan(outputs)
    outputs = outputs[remove].reshape((-1, outputs.shape[1]))
    #remove.reshape((-1, 13))
    inputs = inputs[remove[:, 0], :]
    states = states[remove[:, 0], :]
    print "Tossing out NaN rows: ", np.sum(np.sum(remove, axis=1)==0), " of ", remove.shape[0]

    #E_m = np.mean(outputs[:, 0])
    #E_sd = np.std(outputs[:, 0])
    #print E_m, E_sd
    #m_count = np.sum(outputs[:, 0]>E_m)
    #sd_count = np.sum(outputs[:, 0]>E_m+E_sd)
    #count = outputs.shape[0]

    #print 1.*m_count/count, 1.*sd_count/count

    #MultiPlotTraces(actions.T, np.arange(actions.shape[1]), actions.shape[0], 80)
    #plt.show()

    if shuffle:
        new_index = np.arange(0, inputs.shape[0]).astype('int')
        np.random.shuffle(new_index)

        return inputs[new_index, :], outputs[new_index, :], states[new_index,:]
    else:
        return inputs, outputs, states


class PyraatDataset:
    def __init__(self, actions, features):
        self.actions = actions
        self.features = features

    def num_batches(self, batch_size):
        return int(np.floor(self.features.shape[0]/batch_size))

    def parameter_sizes(self):
        return self.features.shape[1], self.actions.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]

    def data(self):
        start = 0
        stop = self.actions.shape[0]
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]

if __name__=='__main__':

    inputs, outputs, states = LoadData(directory='speech_io',
                                       inputs_name='art_segs',
                                       outputs_name='mfcc',
                                       states_name='art_segs',
                                       shuffle=True,
                                       seq_length=3)

    import pylab as plt
    plt.figure()
    plt.imshow(outputs[:30, :])
    plt.show()

