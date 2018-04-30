class Simulator(object):
    def __init__(self):
        self._sim_mods = []

    def add_mod(self, module):
        self._sim_mods.append(module)

    def run(self):
        raise NotImplementedError()