class SimulatorMod(object):
    """Class for writing custom bionet functions that will be called during the simulation. To use overwrite one or
    more of the following methods in a subclass, and bionet will call the function at the appropiate time.

    To call during a simulation:
      ...
      sim = Simulation(...)
      mymod = MyModule(...)
      sim.add_mod(mymod)
      sim.run()

    """

    def initialize(self, sim):
        """Will be called once at the beginning of the simulation run, after the network and simulation parameters have
        all been finalized.

        :param sim: Simulation object
        """
        pass

    def step(self, sim, tstep, rel_time):
        """Called on every single time step (dt).

        The step method is used for anything that should be recorded or changed continously. dt is determined during
        the setup, and the sim parameter can be used to access simulation, network and individual cell properties

        :param sim: Simulation object.
        :param tstep: The decrete time-step
        :param rel_time: The real time
        """
        pass

    def block(self, sim, block_interval):
        """This method is called once after every block of time, as specified by the configuration.

        Unlike the step method which is called during every time-step, the block method will typically be called only a
        few times over the entire simulation. The block method is preferable for accessing and saving to the disk,
        summing up existing data, or simular functionality

        :param sim: Simulation object
        :param block_interval: The time interval (tstep_start, tstep_end) for which the block is being called on.
        """
        pass

    def finalize(self, sim):
        """Call once at the very end of the simulation.

        :param sim: Simulation object
        """
        pass
