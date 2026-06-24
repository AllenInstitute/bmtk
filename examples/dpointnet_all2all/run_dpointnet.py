import argparse

from bmtk.simulator import dpointnet


def run(config_path):
    config = dpointnet.Config.from_json(config_path)
    config.build_env()

    rnn_network = dpointnet.RNN.from_config(config)
    results = rnn_network.run()
    results.spikes.raster(batch_nums=0, show=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str, 
        nargs='?', 
        default='config.inference.json'
    )

    args, _ = parser.parse_known_args()
    run(args.config_path)
