import config 
import numpy as np
# import numpy.linalg as ln
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io.wavfile import write
# import scipy.signal as signal
import os




class DataHandler(object): # inherit from "object" declares DataHandler as a "new-style-class"
#""" Load data files from simulator output 
#
#TODO: 
#    Fix tubes so that it is global variable to all classes
#"""
    def __init__(self, **kwargs):
        # define tube sections from PRAAT
        self.tubes = config.TUBES # defined in config/constants.py

        # initialize the variables
        self.raw_data = {}

        self.directory = kwargs.get("directory", "data")
        self.directory = kwargs.get("home_dir", self.directory)

        self.InitVars()
        self.DefaultParams()
        self.InitParams(**kwargs)

    def InitVars(self):
        self.tubes = config.TUBES # defined in config/constants.py

        # initialize the variables
        self.raw_data = {}

    def DefaultParams(self):
        self.directory = "data"
        self._verbose = True
        self._params_loaded = False

    def InitParams(self, **kwargs):
        # stupid backward compatibility because I can't decide on a variable name
        self.directory = kwargs.get("directory", self.directory)
        self.directory = kwargs.get("home_dir", self.directory)
        self.directory = kwargs.get("dirname", self.directory)
        self._verbose = kwargs.get("verbose", self._verbose)
        

    def LoadDataFile(self, fname, sample_period=1):
        # note: sample_period is only used by child classes
        

        # load the data from fname, store in class variable
        file_data = np.load(fname)

        # TODO: fix how this works - currently returns false if everything went well...
        _error = self.AppendData(file_data)

        # print error message before we're done.
        if _error:
            print "Warning: No data found in file."

    def AppendData(self, data_dict):
        _error = True

        # load the data from file and append the dictionary to internal dictionary
        for key, value in data_dict.iteritems():

            # if any data is found, we don't print an error message
            if len(value) > 0:
                _error = False


            # handle data according to whether the dictionary has the key or not
            if key in self.raw_data:
                if len(value.shape) < 2:
                    # reshape if it's audio
                    #self.data[key] = np.append(self.data[key], value.reshape((1, -1)), axis=1)
                    self.raw_data[key] = np.append(self.raw_data[key], value)
                else:
                    self.raw_data[key] = np.append(self.raw_data[key], value, axis=1)
            else:
                if len(value.shape) < 2:
                    # reshape if it's audio
                    self.raw_data[key] = value.reshape((1, -1))
                else:
                    self.raw_data[key] = value

        return _error


    def LoadDataParams(self, dirname):

        if not self._params_loaded:
            if self._verbose: 
                print "Loading data paramters."

            # load up data parameters before anything else
            self.params = {}
            params = np.load(os.path.join(dirname, 'params.npz'))

            for key in params.keys():
                if not params[key].shape:
                    self.params[key] = params[key].item()
                else:
                    self.params[key] = params[key]

            self._params_loaded = True
            return True

        else:

            if self._verbose: 
                print "Verifying compatible data parameters."
            # load up data parameters before anything else
            params = np.load(os.path.join(dirname, 'params.npz'))

            for key in self.params.keys():
                value = None 
                if key in params:
                    if not params[key].shape:
                        value = params[key].item()
                    else:
                        value = params[key]

                if not value == self.params[key]:
                    if self._verbose: 
                        print "Data parameter mismatch in %s (%s, %s)"%(str(key), str(self.params[key]), str(value))
                    return False

            return True


    def LoadDataDir(self, **kwargs):#dirname, sample_period=None, verbose = False):
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # clear unused data from last directory
        # prevents erroneous predictions based on discontinuous data
        self.data = {}
        self._data = np.array([])


        self.InitParams(**kwargs)

        # minimum and maximum index values for data files to be loaded from directory
        # files with indexes equal to the min and max will be included in the loading
        _min = kwargs.get('min_index', 0)
        _max = kwargs.get('max_index', np.infty)

        # load up data parameters before anything else
        if not self.LoadDataParams(self.directory):
            if self._verbose: 
                print "Data not loaded from %s" % self.directory

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(self.directory):
            if filename.startswith('data') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

        # sort numerically and load files in order
        index_list = np.array(sorted(index_list))

        # trim off the indexes outside of min and max index range
        index_list = index_list[index_list >= _min]
        index_list = index_list[index_list <= _max]

        if self._verbose: 
            print "Loading data files:"

        for index in index_list:
            if self._verbose:
                print os.path.join(self.directory, 'data'+str(index)+'.npz')

            self.LoadDataFile(os.path.join(self.directory, 'data'+str(index)+'.npz'))

    def SaveAnimation(self, **kwargs):
        # should probably clean this up...
        fname = kwargs.get('fname', 'video')
        self.directory = kwargs.get('dirname', self.directory)
        self.directory = kwargs.get('directory', self.directory)
        #dirname = os.path.join(self.home_dir, dirname)
        fname = os.path.join(self.directory, fname)

        # lungs are far larger than the rest of the apparatus
        # scale them by this factor to make the animation look better
        lung_scale = 0.010

        # variables to pass to animation callback
        fig = plt.figure() # figure handle
        # evidently the hold function is obsolete, default is hold(on)
        vt, = plt.plot([], [], 'b-')
        _vt, = plt.plot([], [], 'b-')
        lungs, = plt.plot([], [], 'r-')
        _lungs, = plt.plot([], [], 'r-')
        nose, = plt.plot([], [], 'g-')
        _nose, = plt.plot([], [], 'g-')

        # cheap downsample, might be better to implement decimate function
        area_function = self.raw_data['area_function'][:, ::100]

        # vertical offset for nasal tubes
        nasal_offset = np.amax(area_function[self.tubes['nose'], :])*1.5

        # scale the lungs down to something reasonable
        area_function[self.tubes['lungs'][:-1]] *= lung_scale

        # set some limits
        plt.xlim(0, self.tubes['all'][-1]+5)
        plt.ylim(-np.amax(area_function[self.tubes['all_no_lungs'], :]),
                 np.amax(area_function[self.tubes['all_no_lungs'], :]) +
                 2*nasal_offset)

        # ffmpeg animation writter
        writer = animation.FFMpegWriter(fps=80,
                                        metadata=dict(artist='PyRAAT'),
                                        bitrate=1800)

        # callback function for updating plots
        def update_figure(num, data, vt, _vt, lungs, _lungs, nose, _nose):

            lungs.set_data(self.tubes['lungs'],
                           data[self.tubes['lungs'], num])
            _lungs.set_data(self.tubes['lungs'],
                            -data[self.tubes['lungs'], num])
            vt.set_data(self.tubes['all_no_lungs'],
                        data[self.tubes['all_no_lungs'], num])
            _vt.set_data(self.tubes['all_no_lungs'],
                         -data[self.tubes['all_no_lungs'], num])
            nose.set_data(self.tubes['nose'] - 15,
                          data[self.tubes['nose'], num]+2*nasal_offset)
            _nose.set_data(self.tubes['nose'] - 15,
                           -data[self.tubes['nose'], num]+2*nasal_offset)

            return vt, _vt, lungs, _lungs, nose, _nose

        # define animation function
        line_ani = animation.FuncAnimation(
            fig,  # figure
            update_figure,  # call back
            area_function.shape[1],  # first arg
            fargs=(area_function, vt, _vt, lungs, _lungs, nose, _nose),  # args
            interval=10,  # not sure what this does
            blit=True)  # blit is a form of video optimization

        # save in the same folder as data was loaded from (see top of function)
        line_ani.save(fname+'.mp4', writer=writer)

    def SaveWav(self, **kwargs):
        fname = kwargs.get('fname', 'concat')
        self.directory = kwargs.get('dirname', self.directory)
        self.directory = kwargs.get('directory', self.directory)
        #dirname = os.path.join(self.home_dir, dirname)
        fname = os.path.join(self.directory, fname)


        # nanmax ignores any nan values that may have occured
        scaled = np.int16(self.raw_data['sound_wave']/np.nanmax(np.abs(self.raw_data['sound_wave']))*32767)
        write(str(fname) + '.wav', self.params['sample_freq'], scaled)


    def IsValid(self):
        _valid = True
        for key in self.raw_data:
            _valid = _valid and (np.any(np.isfinite(self.raw_data[key])))

        return _valid


    #def ClearData(self):
    def ResetDataVars(self):
        self.raw_data = {}


if __name__ == "__main__":
    #print "Do stuff"

    # directory = 'gesture_2017-05-10-20-16-18'
    #directory = '../data/exploded'
    for k in range(1, 10):
        dh = DataHandler(directory='../data/test/breathe_zero_init_'+str(k))
        dh.LoadDataDir(verbose=True, max_index=10)
        dh.SaveAnimation()
        #print 'done'
        dh.SaveWav()

    #area_std = np.std(dh.data['area_function'], axis=1)
    #pressure_std = np.std(dh.data['pressure_function'], axis=1)
    #art_std = np.std(dh.data['art_hist'], axis=1)

    #area_ave = np.mean(dh.data['area_function'], axis=1)
    #pressure_ave = np.mean(dh.data['pressure_function'], axis=1)
    #art_ave = np.mean(dh.data['art_hist'], axis=1)

    #area_dr = np.max(dh.data['area_function'], axis=1)-np.min(dh.data['area_function'], axis=1)
    #pressure_dr = np.max(dh.data['pressure_function'], axis=1)-np.min(dh.data['pressure_function'], axis=1)
    #art_dr = np.max(dh.data['art_hist'], axis=1)-np.min(dh.data['art_hist'], axis=1)

    #area = dh.data['area_function'][dh.tubes['all'], :]
    #area = (area.T-area_ave[dh.tubes['all']]).T
    #area = (area.T/area_std[dh.tubes['all']]).T

    #for k in dh.tubes['all']:
    #    plt.figure()
    #    plt.plot(dh.data['area_function'][k])
    #    plt.title(k)
    #    plt.figure()
    #    plt.plot(dh.data['pressure_function'][k])
    #    plt.title(k)
    #    plt.show()

    #plt.figure()
    #plt.plot(area_std)
    #plt.figure()
    #plt.plot(art_std)
    #plt.figure()
    #plt.plot(pressure_std)

    #plt.show()


