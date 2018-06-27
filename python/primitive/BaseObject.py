class BaseObject(object):
    def __init__(self, **kwargs):
        super(BaseObject, self).__init__()

    def InitVars(self):
        pass

    def DefaultParams(self):
        pass

    def UpdateParams(self, **kwargs):
        pass
