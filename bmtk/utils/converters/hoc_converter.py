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
import json
import os.path
import re
from collections import defaultdict
from itertools import groupby
from lxml import etree
import bluepyopt.ephys as ephys
from tqdm import tqdm
import utils

XML_NS = '{http://www.neuroml.org/schema/neuroml2}'
MECHANISMS = [
    'channelDensity', 'channelDensityNernst', 'specificCapacitance', 'species',
    'resistivity', 'concentrationModel'
]

LOCATION_MAP = {
    'apic': 'apical',
    'soma': 'somatic',
    'dend': 'basal',
    'axon': 'axonal',
    'all': 'all'
}


def map_location_name(name):
    return LOCATION_MAP[name]


def load_json(json_path):
    params = json.load(open(json_path))

    scalar = ephys.parameterscalers.NrnSegmentLinearScaler()
    mechanisms = {}
    sections_lookup = {'soma': 'somatic', 'dend': 'basal', 'axon': 'axonal', 'apic': 'apical'}
    def getNrnSeclist(loc_name):
        return ephys.locations.NrnSeclistLocation(loc_name, seclist_name=loc_name)

    parameters = []
    for d in params['genome']:
        section = sections_lookup[d['section']]
        value = d['value']
        name = d['name']
        mech = 'pas' if name == 'g_pass' else d['mechanism']
        mech_name = 'CaDynamics' if mech == 'CaDynamics' else '{}.{}'.format(name, d['section'])
        p_name = '{}_{}'.format(name, section) if name == 'g_pass' else name

        if mech_name not in mechanisms:
            nrn_mech = ephys.mechanisms.NrnMODMechanism(name=mech_name, mod_path=None, suffix=mech,
                                                        locations=[getNrnSeclist(section)])
            mechanisms[mech_name] = nrn_mech

        parameters.append(ephys.parameters.NrnSectionParameter(name=p_name, param_name=name, value_scaler=scalar,
                                                               value=value, locations=[getNrnSeclist(section)]))

    parameters.append(ephys.parameters.NrnSectionParameter(name='erev_na', param_name='ena', value_scaler=scalar,
                                                           value=params['conditions'][0]['erev'][0]['ena'],
                                                           locations=[getNrnSeclist('somatic')]))
    parameters.append(ephys.parameters.NrnSectionParameter(name='erev_k', param_name='ek', value_scaler=scalar,
                                                           value=params['conditions'][0]['erev'][0]['ek'],
                                                           locations=[getNrnSeclist('somatic')]))
    parameters.append(ephys.parameters.NrnSectionParameter(name='erev_pas', param_name='e_pas', value_scaler=scalar,
                                                           value=params['conditions'][0]['v_init'],
                                                           locations=[getNrnSeclist('somatic'), getNrnSeclist('axonal'),
                                                                      getNrnSeclist('basal'), getNrnSeclist('apical')]))

    parameters.append(ephys.parameters.NrnSectionParameter(name='erev_Ih', param_name='ehcn', value_scaler=scalar,
                                                           value=-45.0,
                                                           locations=[getNrnSeclist('somatic')]))

    parameters.append(ephys.parameters.NrnSectionParameter(name='res_all', param_name='Ra', value_scaler=scalar,
                                                           value=params['passive'][0]['ra'],
                                                           locations=[getNrnSeclist('somatic')]))
    for sec in params['passive'][0]['cm']:
        parameters.append(
            ephys.parameters.NrnSectionParameter(name='{}_cap'.format(sec['section']), param_name='cm',
                                                 value_scaler=scalar,
                                                 value=sec['cm'],
                                                 locations=[getNrnSeclist(sec['section'])]))

    parameters.append(
        ephys.parameters.NrnSectionParameter(name='ca', param_name='depth_CaDynamics', value_scaler=scalar,
                                             value=0.1, locations=[getNrnSeclist('somatic')]))
    parameters.append(
        ephys.parameters.NrnSectionParameter(name='ca', param_name='minCai_CaDynamics', value_scaler=scalar,
                                             value=0.0001, locations=[getNrnSeclist('somatic')]))

    return mechanisms.values(), parameters


def load_neuroml(neuroml_path):
    root = etree.parse(neuroml_path).getroot()
    biophysics = defaultdict(list)
    for mechanism in MECHANISMS:
        xml_mechanisms = root.findall('.//' + XML_NS + mechanism)
        for xml_mechanism in xml_mechanisms:
            biophysics[mechanism].append(xml_mechanism.attrib)

    return biophysics


def define_mechanisms(biophysics):
    def keyfn(x):
        return x['segmentGroup']

    channels = biophysics['channelDensity'] + biophysics[
        'channelDensityNernst']
    segment_groups = [(k, list(g))
                      for k, g in groupby(
                          sorted(
                              channels, key=keyfn), keyfn)]
    mechanisms = []
    for sectionlist, channels in segment_groups:
        loc_name = map_location_name(sectionlist)
        seclist_loc = ephys.locations.NrnSeclistLocation(
            loc_name, seclist_name=loc_name)
        for channel in channels:
            # print 'mechanisms.append(ephys.mechanisms.NrnMODMechanism(name={}.{}, mod_path=None, suffix={}, locations=[{}]))'.format(channel['ionChannel'], loc_name, channel['ionChannel'], seclist_loc)
            mechanisms.append(
                ephys.mechanisms.NrnMODMechanism(
                    name='%s.%s' % (channel['ionChannel'], loc_name),
                    mod_path=None,
                    suffix=channel['ionChannel'],
                    locations=[seclist_loc], ))
    for elem in biophysics['species']:
        section = map_location_name(elem['segmentGroup'])
        section_loc = ephys.locations.NrnSeclistLocation(
            section, seclist_name=section)
        # print 'mechanisms.append(ephys.mechanisms.NrnMODMechanism(name={}, mod_path=None, suffix={}, location=[{}]))'.format(elem['concentrationModel'], elem['concentrationModel'], section_loc)
        mechanisms.append(
            ephys.mechanisms.NrnMODMechanism(
                name=elem['concentrationModel'],
                mod_path=None,
                suffix=elem['concentrationModel'],
                locations=[section_loc]))

    return mechanisms


def define_parameters(biophysics):
    ''' for the time being all AIBS distribution are uniform '''
    parameters = []

    def keyfn(x):
        return x['ionChannel']

    NUMERIC_CONST_PATTERN = r'''[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'''
    rx = re.compile(NUMERIC_CONST_PATTERN, re.VERBOSE)

    def get_cond_density(density_string):
        m = re.match(rx, density_string)
        return float(m.group())

    scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
    MAP_EREV = {
        'Im': 'ek',
        'Ih': 'ehcn',  # I am not sure of that one
        'Nap': 'ena',
        'K_P': 'ek',
        'K_T': 'ek',
        'SK': 'ek',
        'SKv3_1': 'ek',
        'NaTs': 'ena',
        'Kv3_1': 'ek',
        'NaV': 'ena',
        'Kd': 'ek',
        'Kv2like': 'ek',
        'Im_v2': 'ek',
        'pas': 'e_pas'
    }
    for mech_type in ['channelDensity', 'channelDensityNernst']:
        mechanisms = biophysics[mech_type]
        for mech in mechanisms:
            section_list = map_location_name(mech['segmentGroup'])
            seclist_loc = ephys.locations.NrnSeclistLocation(
                section_list, seclist_name=section_list)

            def map_name(name):
                ''' this name has to match the name in the mod file '''
                reg_name = re.compile('gbar\_(?P<channel>[\w]+)')
                m = re.match(reg_name, name)
                if m:
                    channel = m.group('channel')
                    return 'gbar' + '_' + channel
                if name[:len('g_pas')] == 'g_pas':
                    ''' special case '''
                    return 'g_pas'
                assert False, "name %s" % name

            param_name = map_name(mech['id'])
            # print 'parameters.append(ephys.parameters.NrnSectionParameter(name={}, param_name={}, value_scalar={}, value={}, locations=[{}]))'.format(mech['id'], param_name, scaler, get_cond_density(mech['condDensity']), seclist_loc)
            parameters.append(
                ephys.parameters.NrnSectionParameter(
                    name=mech['id'],
                    param_name=param_name,
                    value_scaler=scaler,
                    value=get_cond_density(mech['condDensity']),
                    locations=[seclist_loc]))
            if mech_type != 'channelDensityNernst':
                # print 'parameters.append(ephys.parameters.NrnSectionParameter(name={}, param_name={}, value_scalar={}, value={}, locations=[{}]))'.format('erev' + mech['id'], MAP_EREV[mech['ionChannel']], scaler, get_cond_density(mech['erev']), seclist_loc)
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name='erev' + mech['id'],
                        param_name=MAP_EREV[mech['ionChannel']],
                        value_scaler=scaler,
                        value=get_cond_density(mech['erev']),
                        locations=[seclist_loc]))

    # print '<specificCapacitance, resistivity>'
    PARAM_NAME = {'specificCapacitance': 'cm', 'resistivity': 'Ra'}
    for b_type in ['specificCapacitance', 'resistivity']:
        for elem in biophysics[b_type]:
            section = map_location_name(elem['segmentGroup'])
            section_loc = ephys.locations.NrnSeclistLocation(
                section, seclist_name=section)

            # print 'parameters.append(ephys.parameters.NrnSectionParameter(name={}, param_name={}, value_scalar={}, value={}, locations=[{}]))'.format(elem['id'], PARAM_NAME[b_type], scaler, get_cond_density(elem['value']), seclist_loc)
            parameters.append(
                ephys.parameters.NrnSectionParameter(
                    name=elem['id'],
                    param_name=PARAM_NAME[b_type],
                    value_scaler=scaler,
                    value=get_cond_density(elem['value']),
                    locations=[section_loc]))
            concentrationModel = biophysics['concentrationModel'][0]

    # print '<species>'
    for elem in biophysics['species']:
        section = map_location_name(elem['segmentGroup'])
        section_loc = ephys.locations.NrnSeclistLocation(
            section, seclist_name=section)
        for attribute in ['gamma', 'decay', 'depth', 'minCai']:
            # print 'parameters.append(ephys.parameters.NrnSectionParameter(name={}, param_name={}, value_scalar={}, value={}, locations=[{}]))'.format(elem['id'], attribute + '_' + elem['concentrationModel'], scaler, get_cond_density(concentrationModel[attribute]), seclist_loc)
            parameters.append(
                ephys.parameters.NrnSectionParameter(
                    name=elem['id'],
                    param_name=attribute + '_' + elem['concentrationModel'],
                    value_scaler=scaler,
                    value=get_cond_density(concentrationModel[attribute]),
                    locations=[section_loc]))

    return parameters


def create_hoc(neuroml_path, neuroml, morphologies, incr, output_dir):
    if neuroml_path.endswith('json'):
        mechanisms, parameters = load_json(neuroml_path)

    else:
        biophysics = load_neuroml(neuroml_path)
        mechanisms = define_mechanisms(biophysics)
        parameters = define_parameters(biophysics)

    for morphology in morphologies:
        ccell_name = utils.name_ccell(neuroml, morphology)
        hoc = ephys.create_hoc.create_hoc(
            mechs=mechanisms,
            parameters=parameters,
            template_name='ccell' + str(incr),
            template_filename='cell_template_compatible.jinja2',
            template_dir='.',
            morphology=morphology + '.swc', )
        with open(os.path.join(output_dir, ccell_name + '.hoc'), 'w') as f:
            f.write(hoc)


def convert_to_hoc(config, cells, output_dir):
    to_convert = cells[['dynamics_params', 'morphology', 'neuroml']]
    to_convert = to_convert.drop_duplicates()
    neuroml_config_path = config['components']['biophysical_neuron_models_dir']
    incr = 0
    for name, g in tqdm(to_convert.groupby('dynamics_params'), 'creating hoc files'):
        neuroml_path = os.path.join(neuroml_config_path, name)
        create_hoc(neuroml_path,
                   list(g['neuroml'])[0],
                   set(g['morphology']), incr, output_dir)
        incr += 1