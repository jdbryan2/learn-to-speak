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


        left_panel = wx.Panel(main_panel)
        left_panel.SetBackgroundColour('white')

        # primitive - file select

        # properties: textbox (past, future, dimension, sample rate, features, etc)
        #               note: includes simulation properties too

        # plot primitive operators (button), checkbox for save to tikz

        right_panel = wx.Panel(main_panel)
        right_panel.SetBackgroundColour('white')
        # name: textbox (default to primitive name, updates if path with simulation parameters is chosen)
        # [path select button] /static/text/path/ (default to data/) 

        # [control policy] /static/text/to/file.txt
        # [initial art |v] (dropdown menu)
        # utterance length: textbox X textbox (default to 1x10 second)
        # [simulate] 

        # checkboxes: primitive states, articulators, features
        # [generate Tikz] [generate plots]



        bottom_panel = wx.Panel(main_panel)
        bottom_panel.SetBackgroundColour('white')
        # simulate - button
        # generate plots - button
        # generate movie

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        padding = 10
        hbox.Add(left_panel, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)
        hbox.Add(right_panel, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)

        padding = 10
        vbox = wx.BoxSizer(wx.VERTICAL)
        #vbox.Add(hbox, wx.ID_ANY, wx.EXPAND|wx.ALL, padding)
        vbox.Add(hbox, 10, wx.EXPAND|wx.ALL, padding)
        vbox.Add(bottom_panel, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, padding)

        main_panel.SetSizer(vbox)




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


app = wx.App(False)
frame = MyFrame(None, 'Pyraat primitive simulator GUI')
app.MainLoop()
