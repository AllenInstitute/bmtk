# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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

    def step(self, sim, tstep):
        """Called on every single time step (dt).

        The step method is used for anything that should be recorded or changed continously. dt is determined during
        the setup, and the sim parameter can be used to access simulation, network and individual cell properties

        :param sim: Simulation object.
        :param tstep: The decrete time-step
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
