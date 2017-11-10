import numpy as np
# import numpy.linalg as ln
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import scipy.signal as signal
import os


# TODO: create simple class that loads data files, use as parent for prim class


class DataHandler(object): # inherit from "object" declares DataHandler as a "new-style-class"

    def __init__(self, **kwargs):
        # define tube sections from PRAAT
        self.tubes = {}
        self.tubes['lungs'] = np.append(np.arange(6, 17), 23)
        self.tubes['full_lungs'] = np.arange(6, 23)
        self.tubes['bronchi'] = np.arange(23, 29)
        self.tubes['trachea'] = np.arange(29, 35)
        self.tubes['glottis'] = np.arange(35, 37)
        self.tubes['tract'] = np.arange(37, 64)
        self.tubes['glottis_to_velum'] = np.arange(35, 65)
        self.tubes['nose'] = np.arange(64, 78)
        self.tubes['all'] = np.arange(6, 65)  # include velum, exclude nasal cavity
        self.tubes['all_no_lungs'] = np.arange(23, 65)  # include velum, exclude nasal cavity

        # initialize the variables
        self.data = {}

        self.home_dir = kwargs.get("home_dir", "data")

    def LoadDataFile(self, fname, sample_period=1):
        # note: sample_period is only used by child classes
        
        _error = True

        # load the data from fname, store in class variable
        file_data = np.load(fname)

        # load the data from file and append the dictionary to internal dictionary
        for key, value in file_data.iteritems():

            # if any data is found, we don't print an error message
            if len(value) > 0:
                _error = False

            # handle data according to whether the dictionary has the key or not
            if key in self.data:
                if len(value.shape) < 2:
                    # reshape if it's audio
                    self.data[key] = np.append(self.data[key], value.reshape((1, -1)), axis=1)
                else:
                    self.data[key] = np.append(self.data[key], value, axis=1)
            else:
                if len(value.shape) < 2:
                    # reshape if it's audio
                    self.data[key] = value.reshape((1, -1))
                else:
                    self.data[key] = value

        # print error message before we're done.
        if _error:
            print "Warning: No data found in file."

    def LoadParams(self, dirname):
        
        # append home_dir to the front of dirname
        full_dirname = os.path.join(self.home_dir, dirname)

        # load up data parameters before anything else
        self.params = {}
        params = np.load(os.path.join(full_dirname, 'params.npz'))

        for key in params.keys():
            if not params[key].shape:
                self.params[key] = params[key].item()
            else:
                self.params[key] = params[key]

    def LoadDataDir(self, dirname, sample_period=None, verbose = False):
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # append home_dir to the front of dirname
        full_dirname = os.path.join(self.home_dir, dirname)

        # load up data parameters before anything else
        self.LoadParams(dirname)

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(full_dirname):
            if filename.startswith('data') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

        # sort numerically and load files in order
        index_list = sorted(index_list)
        #print index_list

        if verbose: 
            print "Loading data files:"

        for index in index_list:
            if verbose:
                print os.path.join(full_dirname, 'data'+str(index)+'.npz')
            if sample_period == None:
                self.LoadDataFile(os.path.join(full_dirname, 'data'+str(index)+'.npz'))
            else: 
                self.LoadDataFile(os.path.join(full_dirname, 'data'+str(index)+'.npz'), sample_period)



    def SaveAnimation(self, **kwargs):
        # should probably clean this up...
        fname = kwargs.get('fname', 'video')
        dirname = kwargs.get('dirname', '')
        dirname = os.path.join(self.home_dir, dirname)
        fname = os.path.join(dirname, fname)

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
        area_function = self.data['area_function'][:, ::100]

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


    def ClearData(self):
        self.data = {}


if __name__ == "__main__":
    #print "Do stuff"

    # directory = 'gesture_2017-05-10-20-16-18'
    directory = 'full_random_100_primv5'
    dh = DataHandler(home_dir='../data')
    dh.LoadDataDir(directory)
    #dh.SaveAnimation(dirname=directory)
    print 'done'

    area_std = np.std(dh.data['area_function'], axis=1)
    pressure_std = np.std(dh.data['pressure_function'], axis=1)
    art_std = np.std(dh.data['art_hist'], axis=1)

    area_ave = np.mean(dh.data['area_function'], axis=1)
    pressure_ave = np.mean(dh.data['pressure_function'], axis=1)
    art_ave = np.mean(dh.data['art_hist'], axis=1)

    area_dr = np.max(dh.data['area_function'], axis=1)-np.min(dh.data['area_function'], axis=1)
    pressure_dr = np.max(dh.data['pressure_function'], axis=1)-np.min(dh.data['pressure_function'], axis=1)
    art_dr = np.max(dh.data['art_hist'], axis=1)-np.min(dh.data['art_hist'], axis=1)

    area = dh.data['area_function'][dh.tubes['all'], :]
    area = (area.T-area_ave[dh.tubes['all']]).T
    area = (area.T/area_std[dh.tubes['all']]).T


    plt.figure()
    plt.plot(area_std)
    plt.figure()
    plt.plot(art_std)
    plt.figure()
    plt.plot(pressure_std)

    plt.show()


