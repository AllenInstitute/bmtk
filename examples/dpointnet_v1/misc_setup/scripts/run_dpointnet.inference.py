import argparse
import os
import sys
import traceback

from bmtk.simulator import dpointnet


def run(config_path):
    rnn_network = None
    config = dpointnet.Config.from_json(config_path)
    config.build_env()

    try:
        rnn_network = dpointnet.RNN.from_config(config)
        rnn_network.build()
        results = rnn_network.run_inference()
        print(results)
    finally:
        if rnn_network is not None:
            rnn_network.cleanup()
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
    try:
        run(args.config_path)
    except Exception:
        traceback.print_exc()
        dpointnet.cleanup_tensorflow()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        dpointnet.cleanup_tensorflow()
