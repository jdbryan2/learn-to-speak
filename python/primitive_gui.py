import wx
import os

import numpy as np
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.DataHandler import DataHandler
from primitive.Utterance import Utterance 
import Artword as aw
from matplotlib2tikz import save as tikz_save
import plot_functions as pf

class MyFrame(wx.Frame):
    """ We simply drive a new class of Frame."""
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(800, 600))
        #self.control = wx.TextCtrl(self, pos=(300, 20), size=(200, 300), style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.InitUI()
        self.InitSimulator()
        self.Centre()

    def InitUI(self):
        self.statusbar = self.CreateStatusBar() # A statusbar at the bottom of the window

        # main panel
        ############
        main_panel = wx.Panel(self)
        main_panel.SetBackgroundColour('gray')

        # left panel
        ############
        left_panel = wx.Panel(main_panel)
        left_panel.SetBackgroundColour('white')

        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # [select primtive] /static/text/to/file.npz
        # ------------------------------------------
        prim_box = wx.BoxSizer(wx.HORIZONTAL)
        prim_file = wx.Button(left_panel, id=wx.ID_ANY, label="Select Primitive", name="prim_file")
        self.prim_text = wx.StaticText(left_panel, label="(no file selected)")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSelectPrimitive, prim_file)

        # add to sizer
        prim_box.Add(prim_file)
        prim_box.Add(self.prim_text, 2, border=5)
        left_sizer.Add(prim_box)


        # properties: textbox (past, future, dimension, sample rate, features, etc)
        #               note: includes simulation properties too
        # ------------------------------------------
        self.param_text = wx.StaticText(left_panel, label="Parameters:")
        self.param_view = wx.TextCtrl(left_panel, size=(300, 300), style=wx.TE_MULTILINE|wx.TE_READONLY)

        # add to sizer
        left_sizer.AddSpacer(10)
        left_sizer.Add(self.param_text)
        left_sizer.Add(self.param_view)

        # plot primitive operators (button), checkbox for save to tikz
        # ------------------------------------------
        operator_plot_box = wx.BoxSizer(wx.HORIZONTAL)
        operator_plot = wx.Button(left_panel, id=wx.ID_ANY, label="Plot Operators", name="operator_plot")
        self.operator_tikz = wx.CheckBox(left_panel, label="Generate Tikz Files")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnOperatorPlot, operator_plot)

        operator_plot_box.Add(operator_plot, 1)
        operator_plot_box.Add(self.operator_tikz, 2, border=3)
        left_sizer.AddSpacer(10)
        left_sizer.Add(operator_plot_box, 1)


        # Set sizer for panel
        left_panel.SetSizer(left_sizer)

        # right panel
        #############
        right_panel = wx.Panel(main_panel)
        right_panel.SetBackgroundColour('white')

        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # name: textbox (default to primitive name, updates if path with simulation parameters is chosen)
        # [path select button] /static/text/path/ (default to data/) 
        # ------------------------------------------
        dir_box = wx.BoxSizer(wx.HORIZONTAL)
        dir_button = wx.Button(right_panel, id=wx.ID_ANY, label="Set Directory", name="dir_button")
        self.dir_text = wx.StaticText(right_panel, label="(no directory selected)")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSelectDir, dir_button)

        # add to sizer
        dir_box.Add(dir_button)
        dir_box.Add(self.dir_text, 2, border=5)
        right_sizer.Add(dir_box)

#        name_box = wx.BoxSizer(wx.HORIZONTAL)
#        #name_label = wx.StaticText(right_panel, id=wx.ID_ANY, label="Name: ")
#        name_button = wx.Button(right_panel, id=wx.ID_ANY, label="Choose Name:", name="name_button")
#        self.save_name = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="save_name", value="")

#        self.Bind(wx.EVT_BUTTON, self.OnSelectFilename, name_button)
#
#        # add to sizer
#        name_box.Add(name_button, 1, wx.EXPAND|wx.ALIGN_RIGHT)
#        name_box.Add(self.save_name, 5,wx.EXPAND|wx.ALIGN_LEFT)
#        right_sizer.Add(name_box)


        # [control policy] /static/text/to/file.txt
        # ------------------------------------------
        control_box = wx.BoxSizer(wx.HORIZONTAL)
        control_file = wx.Button(right_panel, id=wx.ID_ANY, label="Control Policy", name="control_file")
        self.control_text = wx.StaticText(right_panel, label="(no file selected)")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSelectPolicy, control_file)

        # add to sizer
        control_box.Add(control_file)
        control_box.Add(self.control_text, 2, border=5)
        right_sizer.AddSpacer(10)
        right_sizer.Add(control_box)

        # [initial art |v] (dropdown menu)
        # ------------------------------------------
        art_box = wx.BoxSizer(wx.HORIZONTAL)
        art_label = wx.StaticText(right_panel, id=wx.ID_ANY, label="Initial Art: ")
        art_choices= ["Zeros", "Random", "Mean Primitive"]
        self.art_option = wx.ComboBox(right_panel, id=wx.ID_ANY, name="art_option", value=art_choices[2], choices=art_choices)

        art_box.Add(art_label)
        art_box.Add(self.art_option)
        right_sizer.AddSpacer(10)
        right_sizer.Add(art_box)
        

        # utterance length: textbox X textbox (default to 1x10 second)
        # ------------------------------------------
        utterance_box = wx.BoxSizer(wx.HORIZONTAL)
        utterance_label = wx.StaticText(right_panel, id=wx.ID_ANY, label="Utterance Length (seconds): ")
        self.utterance_length = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="utterance_length", value="1.0")

        # add to sizer
        utterance_box.Add(utterance_label, 1, wx.EXPAND|wx.ALIGN_RIGHT)
        utterance_box.Add(self.utterance_length, 1,wx.EXPAND|wx.ALIGN_LEFT)
        right_sizer.AddSpacer(10)
        right_sizer.Add(utterance_box)

        # [simulate] 
        # ------------------------------------------
        simulate_box = wx.BoxSizer(wx.HORIZONTAL)
        simulate_button = wx.Button(right_panel, id=wx.ID_ANY, label=" ------ Run Simulation ------ ", name="run_simulation")
        self.save_raw_data   = wx.CheckBox(right_panel, label="Save Data")
        simulate_box.Add(simulate_button)
        simulate_box.Add(self.save_raw_data)
        


        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSimulate, simulate_button)

        right_sizer.AddSpacer(10)
        right_sizer.Add(simulate_box)

        # [Load From File]
        # ------------------------------------------

        load_box = wx.BoxSizer(wx.HORIZONTAL)
        self.min_load = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="max_load", value="")
        to = wx.StaticText(right_panel, id=wx.ID_ANY, label=" to ")
        self.max_load = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="max_load", value="")
        load_button = wx.Button(right_panel, id=wx.ID_ANY, label="Load From Dir", name="load_button")

        self.Bind(wx.EVT_BUTTON, self.OnLoad, load_button)

        # add to sizer
        load_box.Add(self.min_load)
        load_box.Add(to)
        load_box.Add(self.max_load)
        load_box.Add(load_button)

        right_sizer.Add(load_box)

        # checkboxes: primitive states, articulators, features
        # [generate Tikz] [generate plots]
        # ------------------------------------------
        plot_box1 = wx.BoxSizer(wx.HORIZONTAL)

        self.plot_state   = wx.CheckBox(right_panel, label="State")
        self.plot_art     = wx.CheckBox(right_panel, label="Articulator")
        self.plot_sound = wx.CheckBox(right_panel, label="Sound")
        self.plot_control = wx.CheckBox(right_panel, label="Control")
        self.plot_prediction_error = wx.CheckBox(right_panel, label="Error")

        
        plot_box2 = wx.BoxSizer(wx.HORIZONTAL)
        trajectory_plot = wx.Button(right_panel, id=wx.ID_ANY, label="Plot Trajectories", name="trajectory_plot")
        trajectory_tikz = wx.Button(right_panel, id=wx.ID_ANY, label="Save to Tikz", name="trajectory_tikz")
        #self.trajectory_tikz = wx.CheckBox(right_panel, label="Generate Tikz Files")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnTrajectoryPlot, trajectory_plot)
        self.Bind(wx.EVT_BUTTON, self.OnTrajectoryTikz, trajectory_tikz)

        plot_box1.Add(self.plot_state)
        plot_box1.Add(self.plot_art)
        plot_box1.Add(self.plot_sound)
        plot_box1.Add(self.plot_control)
        plot_box1.Add(self.plot_prediction_error)

        plot_box2.Add(trajectory_plot)
        plot_box2.AddSpacer(3)
        plot_box2.Add(trajectory_tikz)

        right_sizer.AddSpacer(20)
        right_sizer.Add(plot_box1)
        right_sizer.Add(plot_box2)



        # [generate movie]
        # ------------------------------------------
        movie_button = wx.Button(right_panel, id=wx.ID_ANY, label="Generate Video File", name="generate_movie")

        # set binding 
        self.Bind(wx.EVT_BUTTON, self.OnMovieButton, movie_button)

        right_sizer.AddSpacer(20)
        right_sizer.Add(movie_button)

        

        # set panel sizer
        right_panel.SetSizer(right_sizer)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        padding = 10
        hbox.Add(left_panel, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)
        hbox.Add(right_panel, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)

        #padding = 10
        #vbox = wx.BoxSizer(wx.VERTICAL)
        ##vbox.Add(hbox, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)
        #vbox.Add(hbox, 10, wx.EXPAND|wx.ALL, padding)
        #vbox.Add(bottom_panel, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, padding)

        main_panel.SetSizer(hbox)




        # file menu
        ############
        filemenu = wx.Menu()
        _about = filemenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        filemenu.AppendSeparator()
        _exit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # bindings
        self.Bind(wx.EVT_MENU, self.OnAbout, _about)
        self.Bind(wx.EVT_MENU, self.OnExit, _exit)

        # create menubar
        ############
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "&File")
        self.SetMenuBar(menuBar)
        self.Show(True)

    def InitSimulator(self):
        self.prim = PrimitiveUtterance()
        self.handler = DataHandler()
        self._loaded = False

    def OnAbout(self, event):
        dlg = wx.MessageDialog(self, "A simple GUI for simulating primitive vocal tract controls.", "About Sample Editor", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnExit(self, event):
        self.Close(True)

    def OnSelectPrimitive(self, event):
        if self.dir_text.GetLabel() == "(no directory selected)":
            self.prim_dirname = 'data/'
        else: 
            self.prim_dirname = self.prim_text.GetLabel()

        dlg = wx.FileDialog(self, "Choose a file", self.prim_dirname, "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.prim_filename = dlg.GetFilename()
            self.prim_dirname = dlg.GetDirectory()
            full_filename = os.path.join(self.prim_dirname, self.prim_filename)

            # change static text to show selected file
            self.prim_text.SetLabel(".../"+self.prim_filename) 

            self.prim.LoadPrimitives(full_filename)

            if self.prim._downpointer_fname != None: 
                self.prim.utterance = PrimitiveUtterance()
                self.prim.utterance.LoadPrimitives(directory = self.prim._downpointer_directory, fname = self.prim._downpointer_fname)

            params_list = "Dimension: %i\n"%self.prim._dim
            params_list += "Past: %i\n" %self.prim._past
            params_list += "Future: %i\n" %self.prim._future
            params_list += "Controller Period: %i ms\n" %(self.prim._sample_period/8)
            params_list += "Features: %s \n"%self.prim.Features.__class__.__name__
            params_list += "Downpointer: %s/%s \n"%(self.prim._downpointer_directory, self.prim._downpointer_fname)
            self.param_view.SetValue(params_list)


            if self.dir_text.GetLabel() == "(no directory selected)":
                self.savedir = self.prim_dirname
                self.dir_text.SetLabel(self.savedir) 
            #if self.save_name.GetValue() == "":
                #self.save_name.SetValue(self.prim_filename[:-4]+"_utterance")


        dlg.Destroy()

    def OnSelectDir(self, event):
        if self.dir_text.GetLabel() == "(no directory selected)":
            self.savedir = 'data/'
        #else: 
        #    self.savedir = self.dir_text.GetLabel()

        dialog = wx.DirDialog(None, "Choose a destination directory:",
                              defaultPath=self.savedir,
                              style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)

        if dialog.ShowModal() == wx.ID_OK:
            self.savedir = dialog.GetPath()
            self.dir_text.SetLabel(self.savedir)

            indeces = self.handler.GetIndexList(directory=self.savedir)
            if len(indeces) > 0:
                self.min_load.SetValue(str(np.min(indeces)))
                self.max_load.SetValue(str(np.max(indeces)))

            
        dialog.Destroy()

    def OnLoad(self, event):
        self.handler.LoadDataDir(directory=self.savedir, 
                                 min_index=int(self.min_load.GetValue()),
                                 max_index=int(self.max_load.GetValue()))

        #self.state_hist = self.handler.GetStateHistory()
        #self.art_hist =   self.handler.GetControlHistory(level=0)
        #self.sound_wave = self.handler.GetSoundWave()
        #self.ctrl_hist =  self.handler.GetControlHistory(level=1)

        self._loaded = True

    def OnSelectFilename(self, event):
        if self.dir_text.GetLabel() == "(no directory selected)":
            self.output_dirname = 'data/'
        else: 
            self.output_dirname = self.dir_text.GetLabel()

        dlg = wx.FileDialog(self, "Choose a file", self.output_dirname, "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.output_filename = dlg.GetFilename()
            self.output_dirname = dlg.GetDirectory()
            full_filename = os.path.join(self.output_dirname, self.utterance_filename)

            # change static text to show selected file
            self.dir_text.SetLabel(self.output_dirname) 
            self.save_name.SetValue(self.output_filename[:-4]) 


        dlg.Destroy()

    def OnSelectPolicy(self, event):
        #if self.dir_text.GetLabel() == "(no directory selected)":
        #    self.control_dirname = 'data/'
        #else: 
        #    self.control_dirname = self.prim_text.GetLabel()

        self.control_dirname = 'control_sequences/'

        dlg = wx.FileDialog(self, "Choose a file", self.control_dirname, "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.control_filename = dlg.GetFilename()
            self.control_dirname = dlg.GetDirectory()
            full_filename = os.path.join(self.control_dirname, self.control_filename)

            # change static text to show selected file
            self.control_text.SetLabel(".../"+self.control_filename) 

            self._control_input = np.genfromtxt(full_filename, delimiter=",", skip_header=1)

        dlg.Destroy()

    def OnOperatorPlot(self, event):
        #K = ss.K[k,:].reshape(ss._past, feature_dim)
        #O = ss.O[:, k].reshape(ss._future, feature_dim)
        dlg = wx.MessageDialog(self, "Plot Primitive Operators", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def CreatePlots(self):
        if self.plot_state.GetValue():
            state_hist = self.handler.GetStateHistory()
            plt.figure()
            pf.PlotTraces(state_hist, np.arange(state_hist.shape[0]), state_hist.shape[1], self.prim._sample_period, highlight=0)

        if self.plot_art.GetValue():
            art_hist = self.handler.GetControlHistory(level=0)
            plt.figure()
            pf.PlotTraces(art_hist, np.arange(art_hist.shape[0]), art_hist.shape[1], self.prim._sample_period, highlight=0)

        if self.plot_sound.GetValue():
            sound_wave = self.handler.GetSoundWave()
            plt.figure()
            plt.plot(np.arange(len(sound_wave))/8000.,sound_wave)
            
        if self.plot_control.GetValue():
            ctrl_hist = self.handler.GetControlHistory(level=1)
            plt.figure()
            pf.PlotTraces(ctrl_hist, np.arange(ctrl_hist.shape[0]), ctrl_hist.shape[1], self.prim._sample_period, highlight=0)


        #print state_hist.shape,sound_wave[::self.prim._sample_period].shape, self.prim._sample_period
        if self.plot_prediction_error.GetValue():
            state_hist = self.handler.GetStateHistory()

            ctrl_hist = self.handler.GetControlHistory(level=1)
            
            output = self.prim.Features.Extract(self.handler.raw_data, sample_period=self.prim._sample_period)
            output = ((output.T-self.prim._ave)/self.prim._std).T

            predicted1 = np.zeros(output.shape)
            predicted2 = np.zeros(output.shape)
            predicted3 = np.zeros(output.shape)
            predicted4 = np.zeros(output.shape)
            predicted5 = np.zeros(output.shape)

            print state_hist.shape
            for t in range(state_hist.shape[1]):
                _Xf = np.dot(self.prim.O, state_hist[:, t])
                _Xf = np.mean(_Xf.reshape((self.prim._ave.size, -1)), axis=1)

                predicted1[:, t] = _Xf[:self.prim._ave.size]
                _Xf = np.dot(self.prim.O, ctrl_hist[:, t])
                _Xf = _Xf.reshape((self.prim._ave.size, -1))
                predicted2[:, t] = _Xf[:, 0]
                predicted3[:, t] = _Xf[:, 1]
                predicted4[:, t] = _Xf[:, 2]
                predicted5[:, t] = _Xf[:, 3]

            error1 = np.mean(np.abs(output-predicted1)**2, axis=0)
            error2 = np.mean(np.abs(output-predicted2)**2, axis=0)
            error3 = np.mean(np.abs(output-predicted3)**2, axis=0)
            error4 = np.mean(np.abs(output-predicted4)**2, axis=0)
            error5 = np.mean(np.abs(output-predicted5)**2, axis=0)

            plt.figure()
    
            plt.plot(error1)
            plt.plot(error2)
            plt.plot(error3)
            plt.plot(error4)
            plt.plot(error5)
            plt.show()
            #for k in range(output.shape[0]):
            #    plt.figure()
            #    plt.plot(output[k], 'b-o')
            #    plt.plot(predicted1[k], 'g-*')
            #    plt.plot(predicted2[k], 'r-*')
            #    plt.plot(predicted3[k], 'r-*', alpha=0.9)
            #    plt.plot(predicted4[k], 'r-*', alpha=0.8)
            #    plt.plot(predicted5[k], 'r-*', alpha=0.7)
            #    plt.plot(predicted2[k]+predicted1[k], 'c-*')

            #    #plt.plot(error[k], 'r--') 
            #    plt.title(k)
            #    plt.show()
            #plt.figure()
            #pf.PlotTraces(output, np.arange(output.shape[0]), output.shape[1], self.prim._sample_period, highlight=0)

            #plt.figure()
            #pf.PlotTraces(predicted, np.arange(predicted.shape[0]), predicted.shape[1], self.prim._sample_period, highlight=0)

            #plt.figure()
            #pf.PlotTraces(error, np.arange(error.shape[0]), error.shape[1], self.prim._sample_period, highlight=0)

    def OnTrajectoryPlot(self, event):
        #if self.plot_state.GetValue():
        self.CreatePlots()
            
        plt.show()
        #dlg = wx.MessageDialog(self, "Plot Primitive Trajectories", "Button Clicked", wx.OK)
        #dlg.ShowModal()
        #dlg.Destroy()
        
    def OnTrajectoryTikz(self, event):
        dlg = wx.MessageDialog(self, "Plot Primitive Trajectories", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnMovieButton(self, event):

        if not self._loaded:
            self.handler.LoadDataDir(directory=self.savedir, 
                                     min_index=int(self.min_load.GetValue()),
                                     max_index=int(self.max_load.GetValue()))

        self.handler.SaveAnimation()
        self.handler.SaveWav()
        
        self.statusbar.SetStatusText("Saved Movie and Audio File to "+self.savedir)

        #dlg = wx.MessageDialog(self, "Generate Video File", "Button Clicked", wx.OK)
        #dlg.ShowModal()
        #dlg.Destroy()
        
    def OnSimulate(self, event):
        # set utterance
        base_utterance = Utterance(directory=self.savedir,
                                        utterance_length=float(self.utterance_length.GetValue()), 
                                        addDTS=False)
        self.prim.SetUtterance(base_utterance)
        print "Directory:"
        print base_utterance.directory
        print self.savedir

        # select initial articulation
        art_option = self.art_option.GetValue()
        initial_art = self.prim.GetControlMean(level=1)

        if art_option == "Zeros":
            initial_art = np.zeros(initial_art.shape)
        elif art_option == "Random":
            initial_art = np.random.random(initial_art.shape)

        #initialize controller
        self.prim.InitializeControl(initial_art=initial_art)

        # TODO: Adjust controller so that time is actually delta time between inputs
        # should make it easier to do repetative stuff and not worry about when it is
        target_index = 0 
        prev_target = initial_art
        prev_time = 0.
        next_target = self._control_input[target_index, 1:]
        next_time = self._control_input[target_index, 0]

        while self.prim.NotDone():
            _now = self.prim.NowSecondsLooped()
            if self._control_input[target_index, 0] < _now:
                prev_target = self._control_input[target_index, 1:]
                prev_time = self._control_input[target_index, 0]
                target_index += 1
                next_target = self._control_input[target_index, 1:]
                next_time = self._control_input[target_index, 0]


            #control_action = -1.*current_state
            control_action = np.zeros(self.prim.current_state.shape)
            for k in range(control_action.size):
                control_action[k] = np.interp(_now,
                                              [prev_time, next_time], 
                                              [prev_target[k], next_target[k]])
            

            current_state = self.prim.SimulatePeriod(control_action=control_action) 
            
            if _now%0.1 == 0.:
                self.statusbar.SetStatusText("Simulating: %.1fs..."%_now)
        
        if self.save_raw_data.GetValue():
            self.statusbar.SetStatusText("Saving Data...")
            self.prim.SaveOutputs()

        self.statusbar.SetStatusText("Done, Simulated %.1fs"%_now)

        # update the index limiters so that we can just load from dir with one click
        indeces = self.handler.GetIndexList(directory=self.savedir)
        if len(indeces) > 0:
            self.min_load.SetValue(np.min(indeces))
            self.max_load.SetValue(np.max(indeces))

        # tuck variables into current class so that we can plot easier
        #self.state_hist = self.prim.GetStateHistory()
        #self.art_hist = self.prim.GetControlHistory(level=0)
        #self.sound_wave = self.prim.GetSoundWave()
        #self.ctrl_hist = self.prim.GetControlHistory(level=1)
        self.handler.raw_data = self.prim.GetOutputs()
        self.handler.params = self.prim.GetParams()
        self._loaded = True


        return 0
        
    def FileSelect(self, event):
        button = event.GetEventObject()
        dlg = wx.MessageDialog(self, "Button: " + button.GetName(), button.GetLabel(), wx.OK)
        dlg.ShowModal()
        dlg.Destroy()



app = wx.App(False)
frame = MyFrame(None, 'Pyraat primitive simulator GUI')
app.MainLoop()
