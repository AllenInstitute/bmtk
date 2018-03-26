import os


class SimReport(object):
    default_dir = '.'

    def __init__(self, name, module):
        self.report_name = name
        self.module = module
        self.params = {}


    @classmethod
    def build(cls, name, params):
        return None


class MembraneReport(SimReport):
    def __init__(self, name):
        super(MembraneReport, self).__init__(name, 'SpikesMod')
        self.cells = None
        self.sections = None
        self.file_name = None
        self.tmp_dir = None
        self.buffer = True
        self.soma_report = True
        self.variables = []

    def can_combine(self, other):
        return self.cells == other.cells and self.sections == other.sections and self.file_name == other.file_name and \
               self.buffer == other.buffer

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


class SpikesReport(SimReport):
    def __init__(self, name):
        super(SpikesReport, self).__init__(name, 'SpikesReport')
        self.h5_file = None
        self.csv_file = None
        self.nwb_file = None
        self.sort_order = None
        self.tmp_dir = self.default_dir

    @classmethod
    def build(cls, name, params):
        return None

    @classmethod
    def from_output_dict(cls, output_dict):
        report = cls('spikes_report')
        report.h5_file = output_dict.get('spikes_file', None)
        report.csv_file = output_dict.get('spikes_file_csv', None)
        report.nwb_file = output_dict.get('spikes_file_nwb', None)
        report.sort_order = output_dict.get('spikes_sort_order', None)
        report.tmp_dir = output_dict.get('output_dir', cls.default_dir)

        if not (report.h5_file or report.csv_file or report.nwb_file):
            return None
        else:
            return report


class VClampReport(SimReport):
    @classmethod
    def from_output_dict(cls, output_dict):
        return None


class ECPReport(SimReport):
    def __init__(self, name):
        super(ECPReport, self).__init__(name, 'EcpMod')
        self.tmp_dir = self.default_dir
        self.positions_file = None
        self.file_name = None

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


module_lookup = {
    'membrane_report': MembraneReport,
    'extracellular': ECPReport,
    'SEClamp': VClampReport,
    'spikes': SpikesReport
}


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

        if 'module' not in report_params:
            raise Exception('module not specified in report {}.'.format(report_name))

        module = report_params['module']
        report = module_lookup.get(module, SimReport).build(report_name, report_params)
        if report is None:
            print('Warning. Report module {} not implemented yet. Skipping'.format(module))
            continue

        # keep track of an explicitly defined Spike Report
        has_spikes_report = has_spikes_report or isinstance(report, SpikesReport)

        if isinstance(report, MembraneReport):
            # When possible for membrane reports combine multiple reports into one module if all the parameters
            # except for the variable name differs.
            for existing_report in membrane_reports:
                if existing_report.can_combine(report):
                    existing_report.variables.extend(report.variables)
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
