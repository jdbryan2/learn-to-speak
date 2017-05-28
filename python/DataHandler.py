import numpy as np
# import numpy.linalg as ln
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
x = np.array([])

# TODO: create simple class that loads data files, use as parent for prim class

class DataHandler:

    def __init__(self, **kwargs):
        # define tube sections from PRAAT
        self.lungs = np.arange(6, 23)
        self.bronchi = np.arange(23, 29)
        self.trachea = np.arange(29, 35)
        self.glottis = np.arange(35,37)
        self.tract = np.arange(37, 64)
        self.nose = np.arange(64, 78)

        self.all_tubes = self.lungs[:]
        self.all_tubes = np.append(self.all_tubes, self.bronchi)
        self.all_tubes = np.append(self.all_tubes, self.trachea)
        self.all_tubes = np.append(self.all_tubes, self.glottis)
        self.all_tubes = np.append(self.all_tubes, self.tract)
        # nose may only need to be the nasopharangeal port
        self.all_tubes = np.append(self.all_tubes, 64)
        #self.all_tubes = np.append(self.all_tubes, self.nose)

        # initialize the variables
        self.data = {}

        self.home_dir = kwargs.get("home_dir", "data")

    def LoadDataFile(self, fname):
        # load the data from fname, store in class variable
        file_data = np.load(fname)

        # load the data from file and append the dictionary to internal
        # dictionary
        for key, value in file_data.iteritems():
            if key in self.data:
                print key, value.shape
                if len(value.shape) < 2:
                    # reshape if it's audio
                    self.data[key] = np.append(self.data[key], value.reshape((1, -1)), axis=1)
                else:
                    self.data[key] = value
                    self.data[key] = np.append(self.data[key], value, axis=1)
                print key, self.data[key].shape
            else:
                if len(value.shape) < 2:
                    # reshape if it's audio
                    self.data[key] = value.reshape((1, -1))
                else:
                    self.data[key] = value

    def LoadDataDir(self, dirname):
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # append home_dir to the front of dirname
        dirname = os.path.join(self.home_dir, dirname)

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(dirname):
            if filename.startswith('data') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

        # sort numerically and load files in order
        index_list = sorted(index_list)
        print index_list
        for index in index_list:
            print os.path.join(dirname, 'data'+str(index)+'.npz')
            self.LoadDataFile(os.path.join(dirname, 'data'+str(index)+'.npz'))

        if not self.data:
            print "No data has been loaded."
            return 0

    def SaveAnimation(self, **kwargs):
        fname = kwargs.get('fname', 'video')
        lung_scale = 0.01

        fig = plt.figure()
        vt, = plt.plot([], [], 'b-')
        lungs, = plt.plot([], [], 'r-')


        plt.xlim(0, self.data['area_function'].shape[0])
        plt.ylim(0, 1.1*lung_scale*np.amax(self.data['area_function']))
        #ax = plt.axes(xlim=(0, self.data['area_function'].shape[0]), ylim=(0, np.amax(self.data['area_function'])))

        writer = animation.FFMpegWriter(fps=100,
                                        metadata=dict(artist='PyRAAT'),
                                        bitrate=1800)

        def update_figure(num, data, vt, lungs):
            lungs.set_data(self.lungs,lung_scale*data[self.lungs, num*100])
            #vt[0].set_data(np.arange(data.shape[0]),data[:, num*100])
            vt.set_data(self.all_tubes[self.lungs.size:], data[self.all_tubes[self.lungs.size:], num*100])
            #return vt,
            return vt, lungs

        line_ani = animation.FuncAnimation(
            fig,
            update_figure,
            self.data['area_function'].shape[1]/100,
            fargs=(self.data['area_function'], vt, lungs),
            interval=10,
            blit=True)

        line_ani.save(fname+'.mp4', writer=writer)


    def ClearData(self):
        self.data = {}


if __name__ == "__main__":
    print "Do stuff"

    dh = DataHandler()
    dh.LoadDataDir('gesture_2017-05-10-20-16-18')
    dh.SaveAnimation()
