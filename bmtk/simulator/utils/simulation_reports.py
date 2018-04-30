import os


class SimReport(object):
    default_dir = '.'
    registry = {}  # Used by factory to keep track of subclasses

    def __init__(self, name, module, params):
        self.report_name = name
        self.module = module
        self.params = params

        # Not part of standard, just want a quick way to turn off modules
        if 'enabled' in params:
            self.enabled = params['enabled']
            del params['enabled']
        else:
            self.enabled = True

        # Set default parameter values (when not explicity stated). Should occur on a module-by-module basis
        self._set_defaults()

    @property
    def node_set(self):
        return self.params.get('cells', 'all')

    def _set_defaults(self):
        for var_name, default_val in self._get_defaults():
            if var_name not in self.params:
                self.params[var_name] = default_val

    def _get_defaults(self):
        """Should be overwritten by subclass with list of (var_name, default_val) tuples."""
        return []

    @staticmethod
    def avail_modules():
        # Return a string (or list of strings) to identify module name for each subclass
        raise NotImplementedError

    @classmethod
    def build(cls, report_name, params):
        """Factory method to get the module subclass, using the params (particularlly the 'module' value, which is
        required). If there is no registered subclass a generic SimReport object will be returned

        :param report_name: name of report
        :param params: parameters of report
        :return: A SimReport (or subclass) object with report parameters parsed out.
        """
        params = params.copy()
        if 'module' not in params:
            raise Exception('report {} does not specify the module.'.format(report_name))

        module_name = params['module']
        del params['module']
        module_cls = SimReport.registry.get(module_name, SimReport)
        return module_cls(report_name, module_name, params)

    @classmethod
    def register_module(cls, subclass):
        # For factory, register subclass based on the module name(s)
        assert(issubclass(subclass, cls))
        mod_registry = cls.registry
        mod_list = subclass.avail_modules()
        modules = mod_list if isinstance(mod_list, list) else [mod_list]
        for mod_name in modules:
            if mod_name in mod_registry:
                raise Exception('Multiple modules named {}'.format(mod_name))
            mod_registry[mod_name] = subclass

        return subclass


@SimReport.register_module
class MembraneReport(SimReport, object):
    def __init__(self, report_name, module, params):
        super(MembraneReport, self).__init__(report_name, module, params)
        # Want variable_name option to allow for singular of list of params
        variables = params['variable_name']
        if isinstance(variables, list):
            self.params['variable_name'] = variables
        else:
            self.params['variable_name'] = [variables]
        self.variables = self.params['variable_name']

        self.params['buffer_data'] = self.params.pop('buffer')

        if self.params['transform'] and not isinstance(self.params['transform'], dict):
            self.params['transform'] = {var_name: self.params['transform'] for var_name in self.variables}


    def _get_defaults(self):
        tmp_dir = os.path.dirname(os.path.realpath(self.params['file_name'])) if 'file_name' in self.params else \
            self.default_dir
        file_name = os.path.join(tmp_dir, 'cell_vars.h5')
        return [('cells', 'biophysical'), ('sections', 'all'), ('tmp_dir', tmp_dir), ('file_name', file_name),
                ('buffer', True), ('transform', {})]

    def add_variables(self, var_name, transform):
        self.params['variable_name'].extend(var_name)
        self.params['transform'].update(transform)

    def can_combine(self, other):
        def param_eq(key):
            return self.params.get(key, None) == other.params.get(key, None)

        return param_eq('cells') and param_eq('sections') and param_eq('file_name') and param_eq('buffer')
        #return self.cells == other.cells and self.sections == other.sections and self.file_name == other.file_name and \
        #       self.buffer == other.buffer

    @staticmethod
    def avail_modules():
        return 'membrane_report'

    @classmethod
    def build(cls, name, params):
        report = cls(name)
        report.cells = params.get('cells', 'biophysical')
        report.sections = params.get('sections', 'all')

        if 'file_name' in params:
            report.file_name = params['file_name']
            report.tmp_dir = os.path.dirname(os.path.realpath(report.file_name))
        else:
            report.file_name = os.path.join(cls.default_dir, 'cell_vars.h5')
            report.tmp_dir = cls.default_dir

        variables = params['variable_name']
        if isinstance(variables, list):
            report.variables = variables
        else:
            report.variables = [variables]

        return report


@SimReport.register_module
class SpikesReport(SimReport):
    def __init__(self, report_name, module, params):
        super(SpikesReport, self).__init__(report_name, module, params)

    @classmethod
    def build(cls, name, params):
        return None

    @staticmethod
    def avail_modules():
        return 'spikes_report'

    @classmethod
    def from_output_dict(cls, output_dict):
        params = {
            'spikes_file': output_dict.get('spikes_file', None),
            'spikes_file_csv': output_dict.get('spikes_file_csv', None),
            'spikes_file_nwb': output_dict.get('spikes_file_nwb', None),
            'spikes_sort_order': output_dict.get('spikes_sort_order', None),
            'tmp_dir': output_dict.get('output_dir', cls.default_dir)
        }
        if not (params['spikes_file'] or params['spikes_file_csv'] or params['spikes_file_nwb']):
            # User hasn't specified any spikes file
            params['enabled'] = False

        return cls('spikes_report', 'spikes_report', params)


@SimReport.register_module
class SEClampReport(SimReport):
    def __init__(self, report_name, module, params):
        super(SEClampReport, self).__init__(report_name, module, params)

    @staticmethod
    def avail_modules():
        return 'SEClamp'


@SimReport.register_module
class ECPReport(SimReport):
    def __init__(self, report_name, module, params):
        super(ECPReport, self).__init__(report_name, module, params)
        self.tmp_dir = self.default_dir
        self.positions_file = None
        self.file_name = None

    @staticmethod
    def avail_modules():
        return 'extracellular'

    def _get_defaults(self):
        tmp_dir = os.path.dirname(os.path.realpath(self.params['ecp_file'])) if 'ecp_file' in self.params else \
            self.default_dir
        file_name = os.path.join(tmp_dir, 'ecp.h5')
        return [('tmp_dir', tmp_dir), ('ecp_file', file_name)]

    @classmethod
    def build(cls, name, params):
        report = cls(name)

        if 'file_name' in params:
            report.file_name = params['file_name']
            report.tmp_dir = os.path.dirname(os.path.realpath(report.file_name))
        else:
            report.file_name = os.path.join(cls.default_dir, 'ecp.h5')
            report.tmp_dir = cls.default_dir

        report.contributions_dir = params.get('contributions_dir', cls.default_dir)
        report.positions_file = params['electrode_positions']
        return report

'''
module_lookup = {
    'membrane_report': MembraneReport,
    'extracellular': ECPReport,
    'SEClamp': SEClampReport,
    'spikes_report': SpikesReport
}
'''


def from_config(cfg):
    SimReport.default_dir = cfg.output_dir

    reports_list = []
    membrane_reports = []
    has_spikes_report = False
    for report_name, report_params in cfg.reports.items():
        # Get the Report class from the module_name parameter
        if not report_params.get('enabled', True):
            # not a part of the standard but will help skip modules
            continue

        report = SimReport.build(report_name, report_params)

        '''
        print report_params
        if 'module' not in report_params:
            raise Exception('module not specified in report {}.'.format(report_name))

        module = report_params['module']
        report = module_lookup.get(module, SimReport).build(report_name, report_params)
        if report is None:
            print('Warning. Report module {} not implemented yet. Skipping'.format(module))
            continue

        # keep track of an explicitly defined Spike Report
        has_spikes_report = has_spikes_report or isinstance(report, SpikesReport)
        '''

        if isinstance(report, MembraneReport):
            # When possible for membrane reports combine multiple reports into one module if all the parameters
            # except for the variable name differs.
            for existing_report in membrane_reports:
                if existing_report.can_combine(report):
                    existing_report.add_variables(report.variables, report.params['transform'])
                    break
            else:
                reports_list.append(report)
                membrane_reports.append(report)

        else:
            reports_list.append(report)

    if not has_spikes_report:
        report = SpikesReport.from_output_dict(cfg.output)
        if report is None:
            # TODO: Log exception or possibly warning
            pass
        else:
            reports_list.append(report)

    return reports_list
