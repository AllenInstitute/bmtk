import logging

from . import env_builder


logger = logging.getLogger(__name__)


environment_builders = {
    'bionet': env_builder.BioNetEnvBuilder,
    'biophysical': env_builder.BioNetEnvBuilder,
    'bio': env_builder.BioNetEnvBuilder,
    'pointnet': env_builder.PointNetEnvBuilder,
    'point': env_builder.PointNetEnvBuilder,
    'filternet': env_builder.FilterNetEnvBuilder,
    'filter': env_builder.FilterNetEnvBuilder,
    'popnet': env_builder.PopNetEnvBuilder,
    'pop': env_builder.PopNetEnvBuilder
}


def create_environment(simulator,
                       base_dir='.',
                       network_dir=None,
                       components_dir=None,
                       node_sets_file=None,
                       output_dir=None,
                       overwrite=False,
                       run_script=True,
                       config_file=None,
                       config_name=None,
                       split_configs=False,
                       network_filter=None,
                       report_vars=[],
                       report_nodes=None,
                       spikes_inputs=None,
                       rates_inputs=None,
                       clamp_reports=[],
                       current_clamp=None,
                       file_current_clamp=None,
                       se_voltage_clamp=None,
                       tstart=0.0,
                       tstop=1000.0,
                       dt=0.001,
                       dL=20.0,
                       spikes_threshold=-15.0,
                       nsteps_block=5000,
                       v_init=-80.0,
                       celsius=34.0,
                       compile_mechanisms=False,
                       use_relative_paths=True,
                       include_examples=False
                       ):
    """

    :param simulator:
    :param base_dir:
    :param network_dir:
    :param components_dir:
    :param node_sets_file:
    :param output_dir:
    :param overwrite:
    :param run_script:
    :param config_file:
    :param config_name:
    :param split_configs:
    :param network_filter:
    :param report_vars:
    :param report_nodes:
    :param spikes_inputs:
    :param rates_inputs:
    :param clamp_reports:
    :param current_clamp:
    :param file_current_clamp:
    :param se_voltage_clamp:
    :param tstart:
    :param tstop:
    :param dt:
    :param dL:
    :param spikes_threshold:
    :param nstesp_block:
    :param v_init:
    :param celsius:
    :param compile_mechanisms:
    :param use_relative_paths:
    :param include_examples:
    """
    simulator_name = simulator.lower()
    if simulator_name not in environment_builders.keys():
        logging.error('Could not find bmtk simulator {}; available options: {}'.format(
            simulator_name, list(environment_builders.keys())
        ))

    build_cls = environment_builders[simulator_name]
    env_builder = build_cls(
        base_dir=base_dir,
        network_dir=network_dir,
        components_dir=components_dir,
        output_dir=output_dir,
        node_sets_file=node_sets_file
    )

    env_builder.build(
        include_examples=include_examples,
        use_relative_paths=use_relative_paths,
        report_vars=report_vars,
        report_nodes=report_nodes,
        clamp_reports=clamp_reports,
        current_clamp=current_clamp,
        file_current_clamp=file_current_clamp,
        se_voltage_clamp=se_voltage_clamp,
        spikes_inputs=spikes_inputs,
        rates_inputs=rates_inputs,
        tstart=tstart,
        tstop=tstop,
        dt=dt,
        dL=dL,
        spikes_threshold=spikes_threshold,
        nsteps_block=nsteps_block,
        v_init=v_init,
        celsius=celsius,
        config_file=config_file,
        overwrite_config=overwrite,
        config_name=config_name,
        split_configs=split_configs,
        network_filter=network_filter,
        run_script=run_script
    )

    if compile_mechanisms:
        env_builder.compile_mechanisms()
