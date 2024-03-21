from bmtk.builder.networks import NetworkBuilder
from download_data import download_ephys_data, download_model, compile_mechanisms


def build_network(specimen_id=488683425, network_dir='network', axon_type='full', fetch_data=True):
    processing_func = 'aibs_allactive' if axon_type == 'stub' else 'aibs_allactive_fullaxon'

    # Get data from Allen CellTypes database.
    if fetch_data:
        model_dir = download_model(specimen_id=specimen_id)
        download_ephys_data(specimen_id=specimen_id)
    else:
        model_dir='models/neuronal_model_48868368425'

    # Builds a SONATA network consisting of a single cell
    net = NetworkBuilder(f'bio_{axon_type}axon')
    net.add_nodes(
        N=1,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        model_processing=processing_func,
        morphology=f'reconstruction.swc',
        dynamics_params=f'fit_parameters.json',

        specimen_id=specimen_id
    )
    net.build()
    net.save(output_dir=network_dir)


if __name__ == '__main__':
    build_network(axon_type='full')
    build_network(axon_type='stub')
