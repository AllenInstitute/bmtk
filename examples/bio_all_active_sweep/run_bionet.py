from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet.default_setters.cell_models import set_params_allactive
from bmtk.simulator.bionet import model_processing
from bmtk.analyzer.compartment import plot_traces


@model_processing
def aibs_allactive_fullaxon(hobj, cell, dynamics_params):
    # This is essentially the same method used to intilize cell parameters as found
    # in bmtk.simulator.bionet.default_setters.cell_models.aibs_allactive function.
    # The main difference is that in the original the axon is cut and replaced by a 
    # stub. Here we leave the full axon intact 
    io.log_info('Initializing Cell Model Params')
    set_params_allactive(hobj, dynamics_params)
    return hobj


def run(config_path):
    conf = bionet.Config.from_json(config_path, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    
    plot_traces(config_file=config_path, report_name='membrane_potential')
    # bionet.nrn.quit_execution()


if __name__ == '__main__':
    parser = bionet.ArgumentParser(description='Run BioNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default='config.simulation.491766131_fullaxon.sweep35.json', help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)
