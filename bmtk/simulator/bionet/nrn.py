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
import sys
import os
import glob
import neuron
from neuron import h

from bmtk.simulator.bionet.pyfunction_cache import py_modules
from bmtk.simulator.bionet.pyfunction_cache import load_py_modules
from bmtk.simulator.bionet.pyfunction_cache import synapse_model, synaptic_weight, cell_model


pc = h.ParallelContext()


def quit_execution(): # quit the execution with a message
    pc.done()
    sys.exit()
    return


def clear_gids():
    pc.gid_clear()
    pc.barrier()


def load_neuron_modules(mechanisms_dir, templates_dir, default_templates=True):
    """

    :param mechanisms_dir:
    :param templates_dir:
    :param default_templates:
    """
    h.load_file('stdgui.hoc')

    bionet_dir = os.path.dirname(__file__)
    h.load_file(os.path.join(bionet_dir, 'import3d.hoc').replace("\\","/"))  # customized import3d.hoc to supress warnings
    h.load_file(os.path.join(bionet_dir,'default_templates',  'advance.hoc').replace("\\","/"))

    if isinstance(mechanisms_dir, list):
        for mdir in mechanisms_dir:
            neuron.load_mechanisms(str(mdir))

    elif mechanisms_dir is not None:
        neuron.load_mechanisms(str(mechanisms_dir))

    if default_templates:
        load_templates(os.path.join(bionet_dir, 'default_templates'))

    if templates_dir:
        load_templates(templates_dir)


def load_templates(template_dir):
    """Load all templates to be available in the hoc namespace for instantiating cells"""
    cwd = os.getcwd()
    os.chdir(template_dir)

    hoc_templates = glob.glob("*.hoc")

    for hoc_template in hoc_templates:
        h.load_file(str(hoc_template))

    os.chdir(cwd)
