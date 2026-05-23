import argparse

from bmtk.simulator import dpointnet


def run(config_path):
    config = dpointnet.Config.from_json(config_path)
    config.build_env()

    rnn_network = dpointnet.RNN.from_config(config)
    rnn_network.run()
    # rnn_network.save_weights()
    # rnn_network.predict()
    # results = rnn_network.inference()
    # results.spikes.to_csv('output/spikes.csv', split_batches=True)
    # results.spikes.to_csv('output/spikes.csv')
    # results.spikes.to_sonata('output/spikes.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str, 
        nargs='?', 
        default='configs/config.training.target_fr.json'
    )

    args, _ = parser.parse_known_args()
    run(args.config_path)
