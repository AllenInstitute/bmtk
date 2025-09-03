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
from bmtk.simulator.bionet.pyfunction_cache import synapse_model, synaptic_weight, cell_model, add_weight_function, model_processing, \
    spikes_generator
from bmtk.simulator.bionet.config import Config
from bmtk.simulator.bionet.bionetwork import BioNetwork
from bmtk.simulator.bionet.biosimulator import BioSimulator
from bmtk.simulator.bionet.nrn import reset

import sys
import argparse
import copy


class ArgumentParser(argparse.ArgumentParser):
    """A Helper class for using argparse when calling script through nrniv, eg
      $ nrniv -python run_bionet.py

    or
      $ mpirun -np 2 nrniv -mpi -python run_bionet.py
      
    
    """
    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            args = copy.copy(sys.argv)
        args = ArgumentParser.parse_nrniv_arg(args)[1:]

        return super().parse_known_args(args, namespace)

    @staticmethod
    def parse_nrniv_arg(sys_argv):
        if sys_argv[0].endswith('nrniv') or sys_argv[0].endswith('nrniv.exe'):
            for i, cmd_opt in enumerate(sys_argv):
                if cmd_opt == '-python':
                    return sys_argv[i+1:]
        else:
            return sys_argv