class NMLTree(object):
    nml_ns = '{http://www.neuroml.org/schema/neuroml2}'
    element_registry = {}

    def __init__(self, nml_path):
        from xml.etree import ElementTree
        self._nml_path = nml_path
        self._nml_root = ElementTree.parse(nml_path).getroot()
        #self._relevant_elements = {
        #    NMLTree.ns_name('channelDensity'): ChannelDensity,
        #    NMLTree.ns_name('resistivity'): Resistivity
        #}

        # For each section store a list of all the NML elements include
        self._soma_props = {}
        self._axon_props = {}
        self._dend_props = {}
        self._apic_props = {}
        # For lookup by segmentGroup attribute, include common synonyms for diff sections
        self._section_maps = {
            'soma': self._soma_props, 'somatic': self._soma_props,
            'axon': self._axon_props, 'axonal': self._axon_props,
            'dend': self._dend_props, 'basal': self._dend_props, 'dendritic': self._dend_props,
            'apic': self._apic_props, 'apical': self._apic_props
        }

        self._parse_root(self._nml_root)

    @classmethod
    def ns_name(cls, name):
        return '{}{}'.format(cls.nml_ns, name)

    @staticmethod
    def common_name(elem):
        if '}' in elem:
            return elem.split('}')[-1]
        else:
            return elem

    @staticmethod
    def parse_value(value):
        val_list = value.split(' ')
        if len(val_list) == 2:
            return float(val_list[0]), val_list[1]
        elif len(val_list) == 1:
            return float(val_list[0]), 'NONE'
        else:
            raise Exception('Cannot parse value {}'.format(value))

    @classmethod
    def register_module(cls, element_cls):
        cls.element_registry[cls.ns_name(element_cls.element_tag())] = element_cls
        return element_cls

    def _parse_root(self, root):
        for elem in root.iter():
            if elem.tag in NMLTree.element_registry:
                nml_element = NMLTree.element_registry[elem.tag](elem)
                self._add_param(nml_element)

    def _add_param(self, nml_element):
        seggroup_str = nml_element.section
        if seggroup_str is None:
            raise Exception('Error: tag {} in {} is missing segmentGroup'.format(nml_element.id, self._nml_path))
        elif seggroup_str.lower() == 'all':
            sections = ['soma', 'axon', 'apic', 'dend']
        else:
            sections = [seggroup_str.lower()]

        for sec_name in sections:
            param_table = self._section_maps[sec_name]
            if sec_name in param_table:
                raise Exception('Error: {} already has a {} element in {}.'.format(nml_element.id, sec_name,
                                                                                   self._nml_path))

            self._section_maps[sec_name][nml_element.id] = nml_element

    def __getitem__(self, section_name):
        return self._section_maps[section_name]


class NMLElement(object):
    def __init__(self, nml_element):
        self._elem = nml_element
        self._attribs = nml_element.attrib

        self.tag_name = NMLTree.common_name(self._elem.tag)
        self.section = self._attribs.get('segmentGroup', None)
        self.id = self._attribs.get('id', self.tag_name)

    @staticmethod
    def element_tag():
        raise NotImplementedError()


@NMLTree.register_module
class ChannelDensity(NMLElement):
    def __init__(self, nml_element):
        super(ChannelDensity, self).__init__(nml_element)
        self.ion = self._attribs['ion']
        self.ion_channel = self._attribs['ionChannel']

        if 'erev' in self._attribs:
            v_list = NMLTree.parse_value(self._attribs['erev'])
            self.erev = v_list[0]
            self.erev_units = v_list[1]
        else:
            self.erev = None

        v_list = NMLTree.parse_value(self._attribs['condDensity'])
        self.cond_density = v_list[0]
        self.cond_density_units = v_list[1]

    @staticmethod
    def element_tag():
        return 'channelDensity'


@NMLTree.register_module
class ChannelDensityNernst(ChannelDensity):

    @staticmethod
    def element_tag():
        return 'channelDensityNernst'


@NMLTree.register_module
class Resistivity(NMLElement):
    def __init__(self, nml_element):
        super(Resistivity, self).__init__(nml_element)
        v_list = NMLTree.parse_value(self._attribs['value'])
        self.value = v_list[0]
        self.value_units = v_list[1]

    @staticmethod
    def element_tag():
        return 'resistivity'


@NMLTree.register_module
class SpecificCapacitance(NMLElement):
    def __init__(self, nml_element):
        super(SpecificCapacitance, self).__init__(nml_element)
        v_list = NMLTree.parse_value(self._attribs['value'])
        self.value = v_list[0]
        self.value_units = v_list[1]

    @staticmethod
    def element_tag():
        return 'specificCapacitance'


@NMLTree.register_module
class ConcentrationModel(NMLElement):
    def __init__(self, nml_element):
        super(ConcentrationModel, self).__init__(nml_element)
        self.type = self._attribs['type']
        v_list = NMLTree.parse_value(self._attribs['decay'])
        self.decay = v_list[0]
        self.decay_units = v_list[1]

        v_list = NMLTree.parse_value(self._attribs['gamma'])
        self.gamma = v_list[0]
        self.gamma_units = v_list[1]

    @staticmethod
    def element_tag():
        return 'concentrationModel'
