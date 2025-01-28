import pytest

from bmtk.simulator.utils.simulation_reports import SimReport, MembraneReport, SpikesReport, ECPReport, NetconReport
from bmtk.simulator.utils.simulation_reports import from_config
from bmtk.utils.sonata.config import SonataConfig


def test_sim_report():
    sreport = SimReport.build(
        report_name='my_report',
        params={
            'module': 'my_mod',
            'arg_one': 'v1',
            'arg_two': 100,
            'arg_three': False,
            'arg_four': ['a', 'b', 'c'],
            'cells': 'my_cells'
        }
    )

    assert(sreport.report_name == 'my_report')
    assert(sreport.params == {'arg_one': 'v1', 'arg_two': 100, 'arg_three': False, 'arg_four': ['a', 'b', 'c'],
                              'cells': 'my_cells'})
    assert(sreport.node_set == 'my_cells')
    assert(sreport.enabled is True)


def test_sim_report_disabled():
    sreport = SimReport.build(
        report_name='my_report',
        params={
            'module': 'my_mod',
            'arg_one': 'v1',
            'enabled': False
        }
    )

    assert(sreport.report_name == 'my_report')
    assert(sreport.params == {'arg_one': 'v1'})
    assert(sreport.enabled is False)
    assert(sreport.node_set == 'all')


def test_registry():
    class CustomReport(SimReport):
        def __init__(self, *params):
            super(CustomReport, self).__init__(*params)
            self.report_name = 'CUSTOM_REPORT'

        @staticmethod
        def avail_modules():
            return ['cus_rep', 'custom']

    SimReport.register_module(CustomReport)
    input_mod = SimReport.build(
        'my_custom_report',
        params={
            "module": "cus_rep",
            "sections": "soma",
            "variable_name": "v",
            "enabled": True
        }
    )
    assert(isinstance(input_mod, CustomReport))
    assert(input_mod.report_name == 'CUSTOM_REPORT')


def test_from_config():
    config = {
        'reports': {
            "membrane_potential": {
                "cells": 'some',
                "variable_name": "v",
                "module": "membrane_report",
                "sections": "soma",
                "enabled": True
            },
            "syn_report": {
                "cells": [0, 1],
                "variable_name": "tau1",
                "module": "netcon_report",
                "sections": "soma",
                "syn_type": "Exp2Syn"
            },
            "ecp": {
                "cells": 'all',
                "variable_name": "v",
                "module": "extracellular",
                "electrode_positions": "linear_electrode.csv",
                "file_name": "ecp.h5",
                "electrode_channels": "all",
                "contributions_dir": "ecp_contributions"
            },
            "spikes": {
                'cells': 'all',
                'module': 'spikes_report',
                'spikes_file': 'my_spikes.h5',
                'cache_to_disk': False
            }
        }
    }

    config_dict = SonataConfig.from_dict(config)
    reports = from_config(config_dict)

    assert(len(reports) == 4)
    assert({r.report_name for r in reports} == {'spikes', 'ecp', 'membrane_potential', 'syn_report'})
    for report in reports:
        if report.report_name == 'spikes':
            assert(isinstance(report, SpikesReport))
            assert(report.params == {'cells': 'all', 'spikes_file': 'my_spikes.h5', 'cache_to_disk': False})

        elif report.report_name == 'ecp':
            assert(isinstance(report, ECPReport))
            assert(report.params == {'cells': 'all', 'variable_name': 'v',
                                     'electrode_positions': 'linear_electrode.csv',
                                     'file_name': 'ecp.h5', 'electrode_channels': 'all',
                                     'contributions_dir': 'ecp_contributions', 'tmp_dir': '.'})

        elif report.report_name == 'membrane_potential':
            assert(isinstance(report, MembraneReport))
            assert(report.params == {'cells': 'some', 'variable_name': ['v'], 'sections': 'soma', 'tmp_dir': '.',
                                     'file_name': 'membrane_potential.h5', 'transform': {}, 'buffer_data': True})

        elif report.report_name == 'syn_report':
            assert(isinstance(report, MembraneReport))
            assert(report.params == {'cells': [0, 1], 'variable_name': ['tau1'], 'sections': 'soma',
                                     'syn_type': 'Exp2Syn', 'tmp_dir': '.', 'file_name': 'syn_report.h5',
                                     'transform': {}, 'buffer_data': True})


if __name__ == '__main__':
    test_sim_report()
    test_sim_report_disabled()
    test_registry()
    test_from_config()
