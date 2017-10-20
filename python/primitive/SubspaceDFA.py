import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

def moving_average(a, n=3, axis=0) :
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# TODO: create simple class that loads data files, use as parent for prim class

class PrimLearn(DataHandler):

    def __init__(self, **kwargs):

        #super(PrimLearn, self).__init__(**kwargs)
        DataHandler.__init__(self, **kwargs)

        # initialize the variables
        self.Xf = np.array([])
        self.Xp = np.array([])

        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0

        # data vars
        self.art_hist = np.array([])
        self.area_function = np.array([])
        self.sound_wave = np.array([])
        self._data = np.array([]) # internal data var
        self.features = {}


        # downsample data such that it's some number of ms
        # period at which controller can make observations
        self._observation_period = 1 # in ms

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")

    def LoadRawData(self, raw_data):
        # directly pass array into class
        self._data = np.copy(raw_data)


    #def LoadDataFile(self, fname):
    #    # overload parent function so that we save on some memory
    #    # BE CAREFUL: Data files must not have fractions of a ms of data 
    #    #             i.e. 100ms per file == good
    #    #                  100.3ms per file == bad

    #    # load the data from fname, store in class variable
    #    file_data = np.load(fname)

    #    # load the data from file and append the dictionary to internal
    #    # dictionary
    #    for key, value in file_data.iteritems():
    #        if key in self.data:
    #            if len(value.shape) < 2:
    #                # reshape if it's audio
    #                self.data[key] = np.append(self.data[key], value.reshape((1, -1)), axis=1)
    #            else:
    #                self.data[key] = np.append(self.data[key], value, axis=1)
    #        else:
    #            if len(value.shape) < 2:
    #                # reshape if it's audio
    #                self.data[key] = value.reshape((1, -1))
    #            else:
    #                self.data[key] = value

    def ConvertData_chunk(self, sample_period=1):
        # loop over indeces and stack them into data array

        area_function = self.data['area_function'][self.tubes['glottis_to_velum'], :]
        lung_pressure = np.mean(self.data['pressure_function'][self.tubes['lungs'], :], axis=0)
        nose_pressure = np.mean(self.data['pressure_function'][self.tubes['nose'], :], axis=0)


        start = 0
        _data = self.data['art_hist']
        self.features['art_hist'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, area_function, axis=0)
        self.features['area_function'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]

        _data = np.append(_data, lung_pressure.reshape((1, -1)), axis=0)
        self.features['lung_pressure'] = np.arange(start, _data.shape[0])
        start=_data.shape[0]
        
        #self._data = np.append(self._data, nose_pressure.reshape((1, -1)), axis=0)
        #self.features['nose_pressure'] = np.arange(start, self._data.shape[0])
        #start=self._data.shape[0]

        self.features['all'] = np.arange(0, _data.shape[0])
        self.features['all_out'] = np.arange(self.features['art_hist'][-1], _data.shape[0])

        # decimate to 1ms sampling period
        _data = signal.decimate(_data, 8, axis=1, zero_phase=True) 

        # decimate to 5ms sampling period
        if sample_period > 1:
            _data = signal.decimate(_data, sample_period, axis=1, zero_phase=True)


        return _data

    def ConvertDataDir(self, dirname, sample_period=1, normalize=True):
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # append home_dir to the front of dirname
        dirname = os.path.join(self.home_dir, dirname)

        # load up data parameters before anything else
        self.params = {}
        params = np.load(os.path.join(dirname, 'params.npz'))
        self.params['gender'] = params['gender'].item()
        self.params['sample_freq'] = params['sample_freq'].item()
        self.params['glottal_masses'] = params['glottal_masses'].item()
        #self.params['method'] = params['method'].item()
        self.params['loops'] = params['loops'].item()
        #self.params['initial_art'] = params['initial_art']
        #self.params['max_increment'] = params['max_increment'].item()
        #self.params['min_increment'] = params['min_increment'].item()
        #self.params['max_delta_target'] = params['max_delta_target'].item()

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(dirname):
            if filename.startswith('data') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

        # sort numerically and load files in order
        index_list = sorted(index_list)

        #print index_list
        std_vals = np.zeros(len(index_list))
        ave_vals = np.zeros(len(index_list))

        for k, index in enumerate(index_list):
            print os.path.join(dirname, 'data'+str(index)+'.npz')
            self.LoadDataFile(os.path.join(dirname, 'data'+str(index)+'.npz'))
            if k==0:
                self._data = self.ConvertData_chunk(sample_period)
                self.data.clear() # clear out the raw data to save space
            else: 
                data = self.ConvertData_chunk(sample_period)
                self._data = np.append(self._data, data, axis=1)
                self.data.clear()
                
        self._std = np.std(self._data, axis=1)
        self._ave = np.mean(self._data, axis=1)

        if normalize:
            # shift to zero mean and  normalize by standard deviation
            self._data = ((self._data.T-self._ave)/self._std).T

        if len(index_list) == 0:
            print "No data has been loaded."
            return 0

    def ConvertData(self, sample_period=1):
        # convert data dictionary into useable data array  
        # perform necessary normalizations and trim out unnecessary parts
        # function still not complete...
        if len(self.data) == 0:
            print "No data to convert."
            return 0

        # round down for decimation rate
        #dec_rate = int(np.floor(self.params['sample_freq']/sample_rate))

        # TODO: finish preprocess of data
        # compute moving_average over time dimension and down sample

        # this should include a more detailed process with downsampling
        # and feature extraction

        # loop over indeces and stack them into data array

        area_function = self.data['area_function'][self.tubes['glottis_to_velum'], :]
        lung_pressure = np.mean(self.data['pressure_function'][self.tubes['lungs'], :], axis=0)
        nose_pressure = np.mean(self.data['pressure_function'][self.tubes['nose'], :], axis=0)


        start = 0
        self._data = self.data['art_hist']
        self.features['art_hist'] = np.arange(start, self._data.shape[0])
        start=self._data.shape[0]

        self._data = np.append(self._data, area_function, axis=0)
        self.features['area_function'] = np.arange(start, self._data.shape[0])
        start=self._data.shape[0]

        self._data = np.append(self._data, lung_pressure.reshape((1, -1)), axis=0)
        self.features['lung_pressure'] = np.arange(start, self._data.shape[0])
        start=self._data.shape[0]
        
        #self._data = np.append(self._data, nose_pressure.reshape((1, -1)), axis=0)
        #self.features['nose_pressure'] = np.arange(start, self._data.shape[0])
        #start=self._data.shape[0]

        self.features['all'] = np.arange(0, self._data.shape[0])
        self.features['all_out'] = np.arange(self.features['art_hist'][-1], self._data.shape[0])

        # decimate to 1ms sampling period
        self._data = signal.decimate(self._data, 8, axis=1, zero_phase=True) 

        # decimate to 5ms sampling period
        if sample_period > 1:
            self._data = signal.decimate(self._data, sample_period, axis=1, zero_phase=True)

        # shift to zero mean and  normalize by standard deviation
        self._ave = np.mean(self._data, axis=1)
        self._std = np.std(self._data, axis=1)
        self._data = ((self._data.T-self._ave)/self._std).T



    def PreprocessData(self, past, future, overlap=False, sample_period=1):
        # note: axis 0 is parameter dimension, axis 1 is time
        #       sample_period is measured in milliseconds

        # save past, future values for later use
        self._past = past
        self._future = future

        # check if _data is populated first (data dictionary is ignored if so)
        #if len(self._data) == 0:
        #    self.ConvertData(sample_period)

        if (len(self._data) == 0) and (len(self.data) == 0):
            print "No data has been loaded."
            return 0

        
        dim = self._data.shape[0]

        if overlap==False:
            chunks = int(np.floor(self._data.shape[1]/((past+future))))
            # format data into Xf and Xp matrices
            Xl = self._data[:, :chunks*(past+future)].T.reshape(-1, (past+future)*dim).T  # reshape into column vectors of length past+future
            print Xl.shape
        else:
            print ''
            chunks = int(np.floor(self._data.shape[1]/(past+future)))
            Xl = self._data[:, :(chunks-1)*(past+future):future]
            print Xl.shape
            for k in range(1, past+future):
                Xl = np.append(Xl, self._data[:, k:(chunks-1)*(past+future)+k:future], axis=0)
                print Xl.shape


        self.Xf = Xl[(past*dim):((past+future)*dim), :]
        self.Xp = Xl[0:(past*dim), :]



    def SubspaceDFA(self, k):
        """Decompose linear prediction matrix into O and K matrices"""

        self._dim = k

        # compute predictor matrix
        self.F = np.dot(self.Xf, ln.pinv(self.Xp))

#       #gamma_f = ln.cholesky(np.cov(Xf))
#       #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(self.F)
        self._U = np.copy(U)
        self._Vh = np.copy(Vh)
        self._S = np.copy(S)

        U = U[:, 0:k]
        # plt.plot(S)
        # plt.show()
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

#       #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
        self.K = np.dot(np.sqrt(S), Vh)

#       #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
        self.O = np.dot(U, np.sqrt(S))

        # return [O, K]

    def SavePrimitives(self, fname=None):

        if fname==None:
            fname = 'primitives'

        np.savez(os.path.join(self.home_dir, str(fname)),
                 K=self.K,
                 O=self.O,
                 ave=self._ave,
                 std=self._std,
                 past=self._past,
                 future=self._future)

    def LoadPrimitives(self, fname=None):

        if fname==None:
            fname = 'primitives.npz'

        primitives = np.load(os.path.join(self.home_dir, fname))

        self.K = primitives['K']
        self.O = primitives['O']
        self._ave = primitives['ave']
        self._std = primitives['std']
        self._past = primitives['past'].item()
        self._future = primitives['future'].item()

    def EstimateStateHistory(self, data):
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()

    
    # controller functions 
    #######################
    # should set this up so I can feed it area and art individually
    #def EstimateState(self, articulators, area_function): 
    def EstimateState(self, data, normalize=True):
        data_length = data.shape[1]

        if normalize:
            # convert data by mean and standard deviation
            _data = data
            _data = ((_data.T-self._ave)/self._std).T
        
        if data_length < self._past:
            _data = np.append(np.zeros((data.shape[0], self._past-data_length)), _data, axis=1)
            data_length = self._past
            
        _Xp = np.reshape(_data[:, data_length-self._past:data_length].T, (-1, 1))
        current_state = np.dot(self.K, _Xp).flatten()
        return current_state

    def GetControl(self, current_state):

        _Xf = np.dot(self.O, current_state)
        _Xf = _Xf.reshape((self._data.shape[0], self._future))
        predicted = _Xf[:, 0]
        predicted = ((predicted.T*self._std)+self._ave).T
        return predicted[self.features['art_hist']]


if __name__ == "__main__":
    print "Do stuff"
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them

    dim = 8

    #down_sample = 5
    #ss = PrimLearn()
    #ss.LoadDataDir('full_random_30')
    #ss.PreprocessData(50, 10, sample_period=down_sample)

    down_sample = 10
    ss = PrimLearn()
    ss.LoadDataDir('full_random_10')
    ss.PreprocessData(50, 10, sample_period=down_sample)
    ss.SubspaceDFA(dim)

    ss.EstimateStateHistory(ss._data)
    plt.plot(ss.h.T)
    plt.show()

    #for k in range(dim):
    #    plt.figure();
    #    plt.imshow(ss.K[k, :].reshape(ss._past, 88))
    #    plt.title('Input: '+str(k))

    #for k in range(dim):
    #    plt.figure();
    #    plt.imshow(ss.O[:, k].reshape(ss._future, 88), aspect=2)
    #    plt.title('Output: '+str(k))

    feature_dim = ss.features['all'].size

    for k in range(dim):
        # pull out the kth primitive dimension
        K = ss.K[k,:].reshape(ss._past, feature_dim)
        O = ss.O[:, k].reshape(ss._future, feature_dim)

        _K =((K)+ss._ave) 
        plt.figure();
        for p in range(ss._past):
            plt.plot(_K[p, ss.features['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))

        plt.plot( ss._ave[ss.features['art_hist']], 'r-')
        plt.title('Articulators Input: '+str(k))

        plt.figure();
        for p in range(ss._past):
            plt.plot(K[p, ss.features['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))
        plt.title('Articulators Raw Input: '+str(k))

        #plt.figure()
        #plt.imshow(K[:, ss.features['art_hist']])
        #plt.title('Articulators Input: '+str(k))


        #plt.figure();
        #for p in range(ss._past):
        #    plt.plot(K[p, ss.features['area_function']], 'b-', alpha=1.*(p+1)/(ss._past+1))
        #plt.title('Area Function Input: '+str(k))

        #plt.figure()
        #plt.imshow(K[:, ss.features['area_function']])
        #plt.title('Area Function Input: '+str(k))

        #plt.figure();
        #plt.plot(K[:, ss.features['lung_pressure']], 'b-')
        #plt.title('Lungs Input: '+str(k))
        #plt.figure();
        #plt.plot(K[:, ss.features['nose_pressure']], 'b-')
        #plt.title('Nose Input: '+str(k))

        ##for p in range(ss._past):
        ##    plt.plot(K[p, :], 'b-', alpha=1.*(p+1)/(ss._past+1))
        ##plt.title('Input: '+str(k))

        #plt.show()

    #for# k in range(dim):
        #plt.figure();
        #for f in range(ss._future):
        #    dat = O[f, :]
        #    dat = ((dat.T*ss._std)+ss._ave).T

        #    plt.plot(dat, 'b-', alpha=1.*(ss._future-f+1)/(ss._future+1))
        #plt.title('Output: '+str(k))
    
    plt.show()


