import numpy as np
import scipy.signal as signal
import scipy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

class SubspaceDFA(DataHandler):
    """ Subspace Method for inferring primitive operators

    Methods: 
        __init__ -- initialize class
            InitVars -- initialize internal variables
            DefaultParams -- defaults for all DFA parameters
            InitParams -- initialize DFA parameters
        SetFeatures -- initialize feature extractor class

        LoadDataFile -- routine for loading data from file
            ClearExcessData -- clear excess preprocessed data from memory
            PreprocessData -- extract features, compute intermediate matrices for subspace DFA

        GenerateData -- generate data from utterance object
            LoadDataChunk -- routine for loading data from dictionary of arrays

        LoadRawData -- pass data in without preprocessing (depreciated)

        SubspaceDFA -- compute primitive operators
        LoadPrimitives -- load primitive operators from file
        SavePrimitives -- save primitive operators to file
        EstimateStateHistory -- estimate state history from raw data based on current primitive operators


    Attributes: 
        _past --  number of past samples used in history
        _future -- number of future predicted samples
        _dim -- internal dimension of primitive space
        sample_period -- period of downsampling operator
        homedir -- base directory 
        Features -- feature extractor class object (must inherit from BaseFeatures)

        _data -- array of feature vectors extracted from raw data
        _std -- standard deviation of _data
        _ave -- mean of _data
        _sum -- sum over all time samples of _data
        _sum_f -- 
        _sum_p -- 
        _sum2 -- sum over all time samples of _data**2
        _sum2_f -- 
        _sum2_p -- 
        _count -- number of time samples of _data
        phi -- intermediate matrix for computing primitive operators
        psi -- intermediate matrix for computing primitive operators

        F -- linear prediction matrix
        O -- primitive output operator
        K -- primitive input operator




    TODO:
        - Add check in EstimateStateHistory so that it returns error or prints out when no primitive operators exist

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
        #self.Xf = np.array([])
        #self.Xp = np.array([])

        # intermediate data for computing prediction matrix F without saving all data into memory directly
        # all parameters are saved without any data normalization 
        # mean and variance are computed after (from the sums) and used to adjust phi and psi
        self.phi = np.array([])
        self.psi = np.array([])
        self._sum = np.array([]) # running sum of data
        self._sum2 = np.array([]) # running sum of squared data
        self._count = 0. # total number of data points added to sums

        self._mean = 0.
        self._2nd_moment = 1.
        self._std = 1.

        # primitive model operators
        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # data vars
        self._data = np.array([]) # internal data var
        self.Features = None # init to nothing
        self.data.clear() # do I want or need this here?

    def DefaultParams(self):
        """ Set default parameters of subspace method

        Arguments: 
            N/A

        Outputs:
            N/A

        """
        super(SubspaceDFA, self).DefaultParams()

        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0
        self.sample_period=1
        self._verbose = True
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
        super(SubspaceDFA, self).InitParams(**kwargs)

        self._past = kwargs.get('past', self._past)
        self._future = kwargs.get('future', self._future)
        self._dim = kwargs.get('dim', self._dim)
        self._verbose = kwargs.get('verbose', self._verbose)
        # period over which data is downsampled
        self.sample_period = kwargs.get('sample_period', self.sample_period)

    def SetFeatures(self, feature_class, **kwargs):
        """ Set feature extractor class

        Arguments: 
            feature_class -- feature extractor class that inherits from BaseFeatures
        Outputs:
            N/A

        """
        self.Features = feature_class(**kwargs)

    def LoadRawData(self, raw_data):
        """ Pass raw data into internal data array without feature extraction or downsampling. Used only for external
        processing of data.

        Arguments: 
            raw_data -- data array that is copied into internal _data attribute

        Outputs: 
            N/A

        """
        # directly pass array into class
        self._data = np.copy(raw_data)

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

        # if soundwave key does not exist
        # create it and fill with zero padding
        if 'sound_wave' not in self.data:
            self.data['sound_wave'] = np.zeros((1, self.Features.min_sound_length))

        # load data and append to self.data dictionary
        super(SubspaceDFA, self).LoadDataFile(fname)
        
        # compute intermediate matrices for inferring primitive operators
        self.PreprocessData()

    def LoadDataChunk(self, data, **kwargs):
        """ Load data file, extract features, and downsample. Automatically called by LoadDataDir.

        Arguments: 
            data -- dictionary of data 
            **kwargs -- key word arguments passed to InitParams

        Output: 
            _data -- data matrix generated by the specified file

        """ 
        #print fname
        self.InitParams(**kwargs)

        # if soundwave key does not exist
        # create it and fill with zero padding
        if 'sound_wave' not in self.data:
            self.data['sound_wave'] = np.zeros((1, self.Features.min_sound_length))

        # load data and append to self.data dictionary
        self.AppendData(data)
        
        # compute intermediate matrices for inferring primitive operators
        self.PreprocessData()

    def ClearExcessData(self, size=None):
        """ Remove all data that has already been used to generate feature vectors. Always leaves 

        Arguments: 
            size -- length of data that has already been used, default assumes all data has been used

        TODO: 
            make sure minimum sound length cannot be clipped
        """
        # clear excess data

        ################################################################################################################
        # NOTE: 
        # sound_wave gets padded with extra zeros at start of LoadDataFile, since we trim off the same length from 
        # all data arrays, the extra padding length always stays there. No need to treat sound_wave different.
        ################################################################################################################

        # Set size to the full length of data if not set.
        # This will remove everything from all arrays except the extra padding in sound_wave.
        if size == None: 
            size = self.data['sound_wave'].shape[1]-self.Features.min_sound_length

        # loop over each key in the dictionary and remove all data used to
        # generate features
        for key in self.data:
            if key=='sound_wave':
                self.data[key] = self.data[key][size:]
            else:
                self.data[key] = self.data[key][:, size:]

    def PreprocessData(self, overlap=False):
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

        # NOTE: 
        #    _data axis 0 is parameter dimension, axis 1 is time
        #    sample_period is measured in milliseconds

        if overlap:
            print "Overlap is currently not supported. No overlap used."

        if (len(self._data) == 0) and (len(self.data) == 0):
            print "No data has been loaded."
            return 0


        # extract features and throw into _data array
        _data = self.Features.Extract(self.data, sample_period=self.sample_period)

        # append to or initialize data array
        if self._data.size==0:
            self._data = _data
        else: 
            self._data = np.append(self._data, _data, axis=1)
        
        # remove all but the necessary bits from self.data (not self._data)
        # TODO: Fix confusing naming convention here
        #       currently removes data from self.data but does not touch self._data
        self.ClearExcessData(size=_data.shape[1]*self.sample_period)

        # count how many chunks of data we can use
        # data_chunks points to the useable chunks   
        # used data removed from self._data
        chunks = int(np.floor(self._data.shape[1]/(self._past+self._future)))
        data_chunks = self._data[:, :chunks*(self._past+self._future)]
        self._data = self._data[:, chunks*(self._past+self._future):] # remove indeces for chunks we're using

        # get dimension of feature space
        dim = data_chunks.shape[0]

        # compute delta mean, save previous mean for some computations
        delta_mean = np.sum((data_chunks.T-self._mean).T, axis=1)/(self._count+data_chunks.shape[1])
        #old_mean = np.copy(self._mean) # don't think it's actually necessary
        self._mean += delta_mean 
        self._delta_mean = delta_mean

        # compute second moment or variance update
        delta_second_moment = np.sum(data_chunks, axis=1)/(self._count+data_chunks.shape[1])

        # normalize data chunks with new mean estimate
        data_chunks = (data_chunks.T-self._mean).T

        # format data into Xf and Xp matrices
        Xl = data_chunks.T.reshape(-1, (self._past+self._future)*dim).T  # reshape into column vectors of length past+future
        Xf = Xl[(self._past*dim):((self._past+self._future)*dim), :]
        Xp = Xl[0:(self._past*dim), :]

        # save summation matrices:
        # use to compute normalization at start of SubspaceDFA
        #  mean -> sum(data_chunks, axis=1)
        #  var -> sum(data_chunks**2, axis=1)
        #  total_count += data_chunks.shape[1]
        #  sum over Xf and Xp 
        #  sum over Xf**2 and Xp**2
        #  save phi=dot(Xf, Xp.T) and psi=dot(Xp, Xp.T)
        if self._count == 0:

            self.phi = np.dot(Xf, Xp.T)
            self.psi = np.dot(Xp, Xp.T)

            self._sum_f = np.sum(Xf, axis=1)
            self._sum_p = np.sum(Xp, axis=1)
            self._count_fp = Xl.shape[1]

            # stuff for computing variance update
            self._var = np.sum(data_chunks**2, axis=1)/(data_chunks.shape[1])
            self._sum = np.sum(data_chunks, axis=1) # zero mean sum

        else:
            # new incremental updates
            delta_mean_f = np.tile(delta_mean, self._future)
            delta_mean_p = np.tile(delta_mean, self._past)

            self.phi += np.dot(Xf, Xp.T) 
            #print  np.dot(delta_mean_f, self._sum_p.T) 
            self.phi -= np.outer(delta_mean_f, self._sum_p) 
            self.phi -= np.outer(self._sum_f, delta_mean_p)
            self.phi += self._count_fp*np.outer(delta_mean_f, delta_mean_p)

            #print "Incrents of Phi"
            #print np.dot(Xf, Xp.T), np.outer(delta_mean_f, self._sum_p), np.outer(self._sum_f, delta_mean_p), self._count_fp*np.outer(delta_mean_f, delta_mean_p)

            self.psi += np.dot(Xp, Xp.T)
            self.psi -= np.outer(delta_mean_p, self._sum_p) 
            self.psi -= np.outer(self._sum_p, delta_mean_p)
            self.psi += self._count_fp*np.outer(delta_mean_p, delta_mean_p)

            self._sum_f += np.sum(Xf, axis=1) - self._count_fp*delta_mean_f 
            self._sum_p += np.sum(Xp, axis=1) - self._count_fp*delta_mean_p
            self._count_fp += Xl.shape[1]

            # stuff for computing variance update
            self._var += (np.sum(data_chunks**2, axis=1)-data_chunks.shape[1]*self._var)/(data_chunks.shape[1]+self._count)
            self._var += self._count/(self._count+data_chunks.shape[1])*delta_mean**2
            self._var -= 2.*delta_mean/(self._count+data_chunks.shape[1])*self._sum
            self._sum += np.sum(data_chunks, axis=1) - self._count*delta_mean # zero mean sum

        self._count += data_chunks.shape[1]
        
        print self._count, self._count_fp

        # debug printouts to make sure indexing is correct
        #print self._count, self._count_fp
        #print Xf.shape
        #print Xp.shape
        if self._verbose:
            print "Preprocessing: %i chunks (%i total)" % (Xl.shape[1], self._count_fp)

    def GenerateData(self, utterance, loops, save_data=True, fname=None):
        # utterance should be completely initialized before getting passed in
        # data is simply generated for some number of loops and passed to SubspaceDFA
        utterance.ResetOutputVars()
        for k in range(loops):
            if self._verbose:
                msg = "\nSimulating iteration %i / %i"% (k+1, loops)
                print msg
                print "-"*len(msg)
            utterance.Simulate()

            #data = {'sound_wave': utterance.sound_wave,
            #        'area_function':utterance.area_function,
            #        'pressure_function':utterance.pressure_function,
            #        'art_hist':utterance.art_hist}

            self.LoadDataChunk(utterance.data)

            if save_data:
                if not fname==None:
                    utterance.SaveData(fname)
                else: 
                    utterance.SaveData()

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

        # compute mean and variance
        self._ave = self._mean
        self._std = np.sqrt(self._var)

        # Normalization works when I use these values
        #_mean_f = np.tile(self._ave, self._future)
        #_mean_p = np.tile(self._ave, self._past)
        _std_f = np.tile(self._std, self._future)
        _std_p = np.tile(self._std, self._past)

        # note: phi and psi must not be changed by this function call so that this can be called multiple times while
        # additional data is loaded.

        # compute predictor matrix
        if self._verbose:
            print "Computing F(%i, %i)..."%(self.phi.shape[0], self.psi.shape[1])
        
        # use scipy.linalg.pinvh to speed up inverting symmetric matrix
        self.F = np.dot(self.phi/np.outer(_std_f, _std_p), ln.pinvh(self.psi/np.outer(_std_p, _std_p)))

        if self._verbose: 
            print "Decomposing F into O and K with rank %i..." % self._dim

        [U, S, Vh] = ln.svd(self.F)
        self._U = np.copy(U)
        self._Vh = np.copy(Vh)
        self._S = np.copy(S)

        U = U[:, 0:k]
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

        self.K = np.dot(np.sqrt(S), Vh)
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

        kwargs = {}
        kwargs['K']=self.K
        kwargs['O']=self.O
        kwargs['ave']=self._ave
        kwargs['std']=self._std
        kwargs['past']=self._past
        kwargs['future']=self._future
        kwargs['control_period']=self.sample_period
        kwargs['features']=self.Features.__class__.__name__ # feature extractor parameters (don't like this way of passing them through)
        kwargs['feature_pointer']=self.Features.pointer
        kwargs['feature_tubes']=self.Features.tubes
        kwargs['control_action']=self.Features.control_action

        np.savez(fname, **kwargs)
                 #K=self.K,
                 #O=self.O,
                 #ave=self._ave,
                 #std=self._std,
                 #past=self._past,
                 #future=self._future, 
                 #control_period=self.sample_period, 
                 #features=self.Features.__class__.__name__, # feature extractor parameters (don't like this way of passing them through)
                 #feature_pointer=self.Features.pointer,
                 #feature_tubes=self.Features.tubes,
                 #control_action=self.Features.control_action)

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

    def StateHistoryFromFile(self, fname):
        """Estimate primitive state history from data file

        Arguments: 
            fname -- name of data file (must include directory)

        Outputs:
            N/A
            
        Affected Attributes: 
            h -- primitive state history array, first dim is primitive index, second is time

        """
        
        # reset internal data vars
        self.data.clear() # do I want or need this here?
        self._data = np.array([]) # internal data var


        super(SubspaceDFA, self).LoadDataFile(fname)

        self._data = self.Features.Extract(self.data, sample_period=self.sample_period)


        # has to be normalized before processing state history
        data= ((self._data.T-self._ave)/self._std).T

        h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            h[:, t] = np.dot(self.K, _Xp).flatten()

        return h

    def EstimateStateHistory(self, data):
        """Estimate primitive state history from data

        Arguments: 
            data -- input data matrix (after feature extraction and downsampling)

        Outputs:
            N/A
            
        Affected Attributes: 
            h -- primitive state history array, first dim is primitive index, second is time

        """
        # has to be normalized before processing state history
        data= ((data.T-self._ave)/self._std).T

        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()


if __name__ == "__main__":
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them
    from features.ArtFeatures import ArtFeatures
    from features.SpectralAcousticFeatures import SpectralAcousticFeatures
    from RandExcite import RandExcite

    #loops = 5 
    #utterance_length = 1.0
    #full_utterance = loops*utterance_length

    #rando = RandExcite(dirname="../data/IDFA_test", 
    #                   utterance_length=utterance_length,
    #                   initial_art=np.random.random((aw.kArt_muscle.MAX, )), 
    #                   max_increment=0.3, min_increment=0.01, max_delta_target=0.2)

    ##rando.InitializeAll()


    dim = 8
    sample_period = 10*8 # (*8) -> ms


    #ss = SubspaceDFA(sample_period=sample_period, past=10, future=10)

    #ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
    ##ss.SetFeatures(SpectralAcousticFeatures)

    #ss.GenerateData(rando, 5)

    #ss.SubspaceDFA(dim)

    #plt.figure()
    #plt.imshow(np.abs(ss.F))
    #plt.figure()
    #plt.imshow(np.abs(ss.O))
    #plt.figure()
    #plt.imshow(np.abs(ss.K))
    #plt.show()

    ss = SubspaceDFA(sample_period=sample_period, past=50, future=10)

    ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
    #ss.SetFeatures(SpectralAcousticFeatures)

    #ss.LoadDataDir(directory="../data/apa")
    #ss.LoadDataDir(directory="../data/click")
    ss.LoadDataDir(directory="../data/random_10")
    ss.LoadDataDir(directory="../data/random_1000", max_index=10)

    ss.SubspaceDFA(dim)


    #ss.EstimateStateHistory(ss._data)
    #plt.plot(ss.h.T)
    #plt.show()

