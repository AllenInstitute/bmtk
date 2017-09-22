class IFormat(object):
    def __init__(self, network):
        self._network = network

    @property
    def format(self):
        raise NotImplementedError()