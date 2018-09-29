import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

class SubspaceDFA(DataHandler):
    """ Subspace Method for inferring primitive operators

    TODO:

    """

    def __init__(self, **kwargs):
        """ Class initialization

        Arguments: 
            past --  number of past samples used in history
            future -- number of future predicted samples
            dim -- internal dimension of primitive space
            sample_period -- period of downsampling operator
            homedir -- base directory 

        Outputs:
            N/A

        """

        super(SubspaceDFA, self).__init__(**kwargs)
        #DataHandler.__init__(self, **kwargs)

        self.InitVars()
        self.DefaultParams()
        self.InitParams(**kwargs)


    def InitVars(self):
        """ Initialize internal variables

        Arguments: 
            N/A

        Outputs:
            N/A

        """
        # initialize the variables
        super(SubspaceDFA, self).InitVars()

        # preprocessed data matrices
        self.Xf = np.array([])
        self.Xp = np.array([])

        # primitive model operators
        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # data vars
        #self.art_hist = np.array([])
        #self.area_function = np.array([])
        #self.sound_wave = np.array([])
        self.feature_data = np.array([]) # internal data var
        self.Features = None # init to nothing
        self.raw_data.clear() # do I want or need this here?


    def DefaultParams(self):
        """ Set default parameters of subspace method

        Arguments: 
            N/A

        Outputs:
            N/A

        """
        super(SubspaceDFA, self).DefaultParams()
        # internal paramters 
        # dimensions of past and future histories and internal state dimension
        # sample period of downsampling operator
        self._past = 0
        self._future = 0
        self._dim = 0
        self.sample_period=1
        return 0

    def InitParams(self, **kwargs):
        """ Parameter initialization

        Arguments: 
            past -- number of past samples used in history
            future -- number of future predicted samples
            dim -- internal dimension of primitive space
            sample_period -- period of downsampling operator

        Outputs:
            N/A

        """

        super(SubspaceDFA, self).UpdateParams(**kwargs) # parent class had this method name changed, too lazy to update this outdated class
        self._past = kwargs.get('past', self._past)
        self._future = kwargs.get('future', self._future)
        self._dim = kwargs.get('dim', self._dim)
        # period over which data is downsampled
        self.sample_period = kwargs.get('sample_period', self.sample_period)

    def SetFeatures(self, feature_class):
        """ Set feature extractor class

        Arguments: 
            feature_class -- feature extractor class that inherits from BaseFeatures
        Outputs:
            N/A

        """
        self.Features = feature_class(tubes=self.tubes)

    def LoadRawData(self, raw_data):
        """ Pass raw data into internal data array without feature extraction or downsampling. Used only for external
        processing of data.

        Arguments: 
            raw_data -- data array that is copied into internal _data attribute

        Outputs: 
            N/A

        """
        # directly pass array into class
        self.feature_data = np.copy(raw_data)

    def LoadDataFile(self, fname, **kwargs):
        """ Load data file, extract features, and downsample. Automatically called by LoadDataDir.

        Arguments: 
            fname -- name of data file
            **kwargs -- key word arguments passed to InitParams

        Output: 
            _data -- data matrix generated by the specified file

        """ 
        #print fname
        self.InitParams(**kwargs)

        # clear internal data dictionary and 
        # load data file according to parent class
        ## No longer clearing before we start, leftovers are now important
        #self.raw_data.clear()

        # if soundwave key does not exist
        # create it and fill with zero padding
        if 'sound_wave' not in self.raw_data:
            self.raw_data['sound_wave'] = np.zeros((1, self.Features.min_sound_length))

        # load data and append to self.raw_data dictionary
        super(SubspaceDFA, self).LoadDataFile(fname)
        
        # count number of sample periods in data dictionary
        #total_periods = (self.raw_data['sound_wave'].shape[1] - 
        #                 self.Features.min_sound_length) / self.sample_period
        #total_periods = int(np.floor(total_periods))
        
        _data = self.Features.Extract(self.raw_data, sample_period=self.sample_period)
        # features needs function that returns the size of the used data
        #print total_periods, _data.shape

        # TODO: Need to decide what this was intially intended for
        # Disabled because it breaks loading of multiple unrelated files
        if self.feature_data.size==0:
            self.feature_data = _data
        else:
            self.feature_data = np.append(self.feature_data, _data, axis=1)

        #print _data.shape

        # clear out the raw data dictionary to save space
        #self.raw_data.clear() 
        
        # remove all but the necessary bits
        self.ClearExcessData(size=_data.shape[1]*self.sample_period)

        return _data

    def ClearExcessData(self, size=None):
        """ Remove all data that has already been used to generate feature vectors. Always leaves 

        Arguments: 
            size -- length of data that has already been used, default assumes all data has been used

        """
        # clear excess data

        # set size to the full length of data if not set
        if size == None: 
            size = self.raw_data['sound_wave'].shape[1]-self.Features.min_sound_length

        # always leave behind the minimum sound length (needed for spectral features)
        if self.Features.min_sound_length > 0:
            size = max(size-self.Features.min_sound_length, 0)

        # loop over each key in the dictionary and remove all data used to
        # generate features
        for key in self.raw_data:
            if key == 'sound_wave':
                self.raw_data[key] = self.raw_data[key][size:]
            else:
                self.raw_data[key] = self.raw_data[key][:, size:]


    def PreprocessData(self, past, future, overlap=False, normalize=True):
        """ Reshape data into chunks for subspace method

        Arguments: 
            past -- number of past samples used in history
            future -- number of future predicted samples
            overlap -- number of samples that overlap between windows of size (past+future)
            normalize -- normalize data to zero mean and unit variance

        Outputs:
            N/A
            
        Affected Attributes: 
            _past -- number of past samples used in history
            _future -- number of future predicted samples
            _std -- estimated standard deviation of _data
            _ave -- estimated mean of _data
            Xf, Xp -- used to compute primitive operators

        """
        # note: axis 0 is parameter dimension, axis 1 is time
        #       sample_period is measured in milliseconds

        # save past, future values for later use
        self._past = past
        self._future = future
        chunks = int(np.floor(self.feature_data.shape[1]/((self._past+self._future))))
        data_chunks = self.feature_data[:, :chunks*(self._past+self._future)]

        if (len(self.feature_data) == 0) and (len(self.raw_data) == 0):
            print "No data has been loaded."
            return 0

        if normalize:
            self._std = np.std(data_chunks, axis=1)
            self._ave = np.mean(data_chunks, axis=1)

            # shift to zero mean and  normalize by standard deviation
            data_chunks = ((data_chunks.T-self._ave)/self._std).T
        #if normalize:
        #    self._std = np.std(self.feature_data, axis=1)
        #    self._ave = np.mean(self.feature_data, axis=1)

        #    # shift to zero mean and  normalize by standard deviation
        #    self.feature_data = ((self.feature_data.T-self._ave)/self._std).T

        
        # get dimension of input/output feature space
        dim = data_chunks.shape[0] #self.feature_data.shape[0]

        if overlap==False:
            # format data into Xf and Xp matrices
            #Xl = self.feature_data[:, :chunks*(self._past+self._future)].T.reshape(-1, (self._past+self._future)*dim).T  # reshape into column vectors of length past+future
            Xl = data_chunks.T.reshape(-1, (self._past+self._future)*dim).T  # reshape into column vectors of length past+future
            #print Xl.shape
        else:
            print ''
            #chunks = int(np.floor(self.feature_data.shape[1]/(self._past+future)))
            Xl = self.feature_data[:, :(chunks-1)*(self._past+self._future):self._future]
            print Xl.shape
            for k in range(1, past+future):
                Xl = np.append(Xl, self.feature_data[:, k:(chunks-1)*(self._past+self._future)+k:self._future], axis=0)
                #print Xl.shape


        self.Xf = Xl[(self._past*dim):((self._past+self._future)*dim), :]
        self.Xp = Xl[0:(self._past*dim), :]

    def SubspaceDFA(self, k):
        """Estimate linear prediction matrix and decompose into primitive operators 

        Arguments: 
            k -- internal dimension of primitive subspace

        Outputs:
            N/A
            
        Affected Attributes: 
            F -- linear prediction operator
            K -- primitive input operator
            O -- primitive output operator
            _dim -- internal dimension of primitive space

        """

        self._dim = k

        # compute predictor matrix
        self.F = np.dot(self.Xf, ln.pinv(self.Xp))

        #gamma_f = ln.cholesky(np.cov(Xf))
        #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(self.F)
        #self._U = np.copy(U)
        #self._Vh = np.copy(Vh)
        #self._S = np.copy(S)

        U = U[:, 0:k]
        # plt.plot(S)
        # plt.show()
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

        #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
        self.K = np.dot(np.sqrt(S), Vh)

        #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
        self.O = np.dot(U, np.sqrt(S))

    def SavePrimitives(self, fname=None):
        """Save primitive operators to file

        Arguments: 
            fname -- output file name, default to 'primitives'

        Outputs:
            N/A
            
        Affected Attributes: 
            N/A

        """

        if fname==None:
            fname = 'primitives'

        np.savez(os.path.join(self.home_dir, str(fname)),
                 K=self.K,
                 O=self.O,
                 ave=self._ave,
                 std=self._std,
                 past=self._past,
                 future=self._future, 
                 control_period=self.sample_period, 
                 features=self.Features.__class__.__name__, # feature extractor parameters (don't like this way of passing them through)
                 feature_pointer=self.Features.pointer,
                 feature_tubes=self.Features.tubes,
                 control_action=self.Features.control_action)

    def LoadPrimitives(self, fname=None):
        """Load primitive operators from file

        Arguments: 
            fname -- input file name, default to 'primitives.npz'

        Outputs:
            N/A
            
        Affected Attributes: 
            N/A

        """

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
        """Estimate primitive state history from data

        Arguments: 
            data -- input data matrix (after feature extraction and downsampling)

        Outputs:
            N/A
            
        Affected Attributes: 
            h -- primitive state history array, first dim is primitive index, second is time

        """

        # normalize
        data= ((data.T-self._ave)/self._std).T

        # initilize state history matrix
        # first dimension is primitive index, second dimension is time
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()


if __name__ == "__main__":
    print "Do stuff"
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them
    from features.ArtFeatures import ArtFeatures
    from features.SpectralAcousticFeatures import SpectralAcousticFeatures

    dim = 8

    #down_sample = 5
    #ss = PrimLearn()
    #ss.LoadDataDir('full_random_30')
    #ss.PreprocessData(50, 10, sample_period=down_sample)

    down_sample = 10*8
    ss = SubspaceDFA(home_dir='../data')
    print "loading features class"

    ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
    #ss.SetFeatures(SpectralAcousticFeatures)

    print "loading data dir"
    ss.LoadDataDir('full_random_10', sample_period=down_sample, verbose=True)
    ss.PreprocessData(50, 10)
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

    feature_dim = ss.Features.pointer['all'].size

    for k in range(dim):
        # pull out the kth primitive dimension
        K = ss.K[k,:].reshape(ss._past, feature_dim)
        O = ss.O[:, k].reshape(ss._future, feature_dim)

        _K =((K)+ss._ave) 
        plt.figure();
        for p in range(ss._past):
            plt.plot(_K[p, ss.Features.pointer['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))

        plt.plot( ss._ave[ss.Features.pointer['art_hist']], 'r-')
        plt.title('Articulators Input: '+str(k))

        plt.figure();
        for p in range(ss._past):
            plt.plot(K[p, ss.Features.pointer['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))
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


