import wx
import os

class MyFrame(wx.Frame):
    """ We simply drive a new class of Frame."""
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(800, 600))
        #self.control = wx.TextCtrl(self, pos=(300, 20), size=(200, 300), style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.InitUI()
        self.Centre()

    def InitUI(self):
        self.CreateStatusBar() # A statusbar at the bottom of the window

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
        name_box = wx.BoxSizer(wx.HORIZONTAL)
        name_label = wx.StaticText(right_panel, id=wx.ID_ANY, label="Name: ")
        self.save_name = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="save_name")

        # add to sizer
        name_box.Add(name_label, 1, wx.EXPAND|wx.ALIGN_RIGHT)
        name_box.Add(self.save_name, 5,wx.EXPAND|wx.ALIGN_LEFT)
        right_sizer.Add(name_box)

        # [control policy] /static/text/to/file.txt
        # ------------------------------------------
        control_box = wx.BoxSizer(wx.HORIZONTAL)
        control_file = wx.Button(right_panel, id=wx.ID_ANY, label="Control Policy", name="control_file")
        self.control_text = wx.StaticText(right_panel, label="(no file selected)")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSelectPolicy, prim_file)

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
        self.art_option = wx.ComboBox(right_panel, id=wx.ID_ANY, name="art_option", value=art_choices[0], choices=art_choices)

        art_box.Add(art_label)
        art_box.Add(self.art_option)
        right_sizer.AddSpacer(10)
        right_sizer.Add(art_box)
        

        # utterance length: textbox X textbox (default to 1x10 second)
        # ------------------------------------------
        utterance_box = wx.BoxSizer(wx.HORIZONTAL)
        utterance_label = wx.StaticText(right_panel, id=wx.ID_ANY, label="Utterance Length: ")
        self.utterance_length = wx.TextCtrl(right_panel, id=wx.ID_ANY, name="utterance_length")

        # add to sizer
        utterance_box.Add(utterance_label, 1, wx.EXPAND|wx.ALIGN_RIGHT)
        utterance_box.Add(self.utterance_length, 1,wx.EXPAND|wx.ALIGN_LEFT)
        right_sizer.AddSpacer(10)
        right_sizer.Add(utterance_box)

        # [simulate] 
        # ------------------------------------------
        simulate = wx.Button(right_panel, id=wx.ID_ANY, label=" ------ Run Simulation ------ ", name="run_simulation")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnSelectPolicy, prim_file)

        right_sizer.AddSpacer(10)
        right_sizer.Add(simulate)

        # checkboxes: primitive states, articulators, features
        # [generate Tikz] [generate plots]
        # ------------------------------------------
        plot_box1 = wx.BoxSizer(wx.HORIZONTAL)

        self.plot_state   = wx.CheckBox(right_panel, label="State")
        self.plot_art     = wx.CheckBox(right_panel, label="Articulator")
        self.plot_feature = wx.CheckBox(right_panel, label="Features")

        
        plot_box2 = wx.BoxSizer(wx.HORIZONTAL)
        trajectory_plot = wx.Button(right_panel, id=wx.ID_ANY, label="Show Trajectories", name="trajectory_plot")
        trajectory_tikz = wx.Button(right_panel, id=wx.ID_ANY, label="Save Trajectories (tikz)", name="trajectory_tikz")
        #self.trajectory_tikz = wx.CheckBox(right_panel, label="Generate Tikz Files")

        # set binding function
        self.Bind(wx.EVT_BUTTON, self.OnTrajectoryPlot, trajectory_plot)
        self.Bind(wx.EVT_BUTTON, self.OnTrajectoryTikz, trajectory_tikz)

        plot_box1.Add(self.plot_state)
        plot_box1.Add(self.plot_art)
        plot_box1.Add(self.plot_feature)

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

    def OnAbout(self, event):
        dlg = wx.MessageDialog(self, "A simple GUI for simulating primitive vocal tract controls.", "About Sample Editor", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnExit(self, event):
        self.Close(True)

    def OnSelectPrimitive(self, event):
        dlg = wx.MessageDialog(self, "Select Primitive File", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnSelectPolicy(self, event):
        dlg = wx.MessageDialog(self, "Select Control Policy File", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnOperatorPlot(self, event):
        dlg = wx.MessageDialog(self, "Plot Primitive Operators", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnTrajectoryPlot(self, event):
        dlg = wx.MessageDialog(self, "Plot Primitive Trajectories", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
        
    def OnTrajectoryTikz(self, event):
        dlg = wx.MessageDialog(self, "Plot Primitive Trajectories", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def OnMovieButton(self, event):
        dlg = wx.MessageDialog(self, "Generate Video File", "Button Clicked", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
        
        
    def FileSelect(self, event):
        button = event.GetEventObject()
        dlg = wx.MessageDialog(self, "Button: " + button.GetName(), button.GetLabel(), wx.OK)
        dlg.ShowModal()
        dlg.Destroy()



app = wx.App(False)
frame = MyFrame(None, 'Pyraat primitive simulator GUI')
app.MainLoop()
