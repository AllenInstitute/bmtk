import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from bmtk.simulator import dpointnet


RULES = {
    'eprop': {
        'name': 'eprop',
        'surfaces': ['<recurrent>'],
        'edge_chunk_size': 4096,
    },
    'three_factor': {
        'name': 'three_factor',
        'signal': 'spike',
        'surfaces': ['<recurrent>'],
        'edge_chunk_size': 4096,
    },
    'modprop': {
        'name': 'modprop',
        'surfaces': ['<recurrent>'],
        'filter_taps': 3,
        'mean_activity': 0.5,
        'edge_chunk_size': 4096,
    },
}

TASKS = {
    'voltage_control': {
        'target_rate_hz': None,
        'description': 'move normalized membrane voltage toward threshold',
        'voltage_cost': 1.0,
        'voltage_penalty_mode': 'threshold',
        'n_epochs': 5,
        'steps_per_epoch': 10,
        'learning_rate': 20.0,
        'minimum_improvement_percent': 1.0,
    },
    'silence': {
        'target_rate_hz': 0.0,
        'description': 'suppress recurrent firing',
        'voltage_cost': 0.25,
        'voltage_penalty_mode': 'range',
        'n_epochs': 10,
        'steps_per_epoch': 10,
        'learning_rate': 0.001,
    },
    'rate_control': {
        'target_rate_hz': 20.0,
        'description': 'raise recurrent firing toward 20 Hz',
        'voltage_cost': 0.25,
        'voltage_penalty_mode': 'range',
        'n_epochs': 2,
        'steps_per_epoch': 4,
        'learning_rate': 0.01,
    },
}


def evaluation_batch(network):
    parameter = network.training_engine.parameters[0]
    generators = [
        module.create_generator(seq_len=parameter.seq_len)
        for module in parameter.input_generators
    ]
    batches = [next(iter(generator.batch(parameter.batch_size))) for generator in generators]
    spikes = tf.concat([batch[0] for batch in batches], axis=2)
    signatures = [batch[1] for batch in batches]
    return spikes, signatures


def evaluate(network, input_spikes, signatures):
    parameter = network.training_engine.parameters[0]
    initial_state = network.training_engine.init_state.get_state()
    output = network.run_extractor(input_spikes, initial_state)
    spikes, voltages = output[0]
    losses = {
        name: float(loss(
            spikes=spikes,
            voltages=voltages,
            model_state=output[1:],
            y=signatures,
        ).numpy())
        for name, loss in parameter.loss_functions.items()
    }
    simulation_seconds = parameter.seq_len * network.dt / 1000.0
    mean_rate = float(tf.reduce_mean(
        tf.reduce_sum(spikes, axis=1) / simulation_seconds
    ).numpy())
    losses['total'] = sum(losses.values())
    losses['mean_rate_hz'] = mean_rate
    return losses


def run(rule_name, task_name='voltage_control', batch_size=None):
    example_dir = Path(__file__).resolve().parent
    with open(example_dir / 'config.base.json') as config_file:
        config_data = json.load(config_file)

    config_data['manifest']['$BASE_DIR'] = example_dir.as_posix()
    rule_config = copy.deepcopy(RULES[rule_name])
    if rule_name == 'three_factor' and task_name == 'voltage_control':
        rule_config['signal'] = 'voltage'
    config_data['training']['learning_rule'] = rule_config
    losses = config_data['training']['parameters'][0]['loss_functions']
    target_rate_hz = TASKS[task_name]['target_rate_hz']
    if target_rate_hz is None:
        losses.pop('target_rate')
    else:
        losses['target_rate']['firing_rate'] = target_rate_hz
    losses['voltage_regularizer'].update({
        'voltage_cost': TASKS[task_name]['voltage_cost'],
        'penalty_mode': TASKS[task_name]['voltage_penalty_mode'],
    })
    for option in ('n_epochs', 'steps_per_epoch', 'learning_rate'):
        config_data['training'][option] = TASKS[task_name][option]
    config_data['output']['output_dir'] = (
        example_dir / 'output' / rule_name
    ).as_posix()
    if batch_size is not None:
        config_data['run']['batch_size'] = batch_size

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', dir=example_dir, delete=False) as config_file:
        json.dump(config_data, config_file, indent=2)
        generated_config = Path(config_file.name)

    try:
        config = dpointnet.Config.from_json(generated_config.as_posix())
        config.build_env()
        network = dpointnet.RNN.from_config(config)
        network.build()
        initial_weights = network.cell.recurrent_weight_values.numpy().copy()
        initial_signs = np.sign(initial_weights)
        fixed_spikes, fixed_signatures = evaluation_batch(network)
        initial_metrics = evaluate(network, fixed_spikes, fixed_signatures)

        network.train()

        trained_metrics = evaluate(network, fixed_spikes, fixed_signatures)
        trained_weights = network.cell.recurrent_weight_values.numpy()
        change = trained_weights - initial_weights
        objective_improvement_percent = 100.0 * (
            initial_metrics['total'] - trained_metrics['total']
        ) / initial_metrics['total']
        expected_improvement = TASKS[task_name].get('minimum_improvement_percent')
        verification_passed = (
            expected_improvement is None
            or objective_improvement_percent >= expected_improvement
        )
        print(f'rule={rule_name}')
        print(f'task={task_name}')
        print(f'task_description={TASKS[task_name]["description"]}')
        print(f'batch_size={network.batch_size}')
        print(f'mean_abs_weight_change={np.mean(np.abs(change)):.8g}')
        print(f'max_abs_weight_change={np.max(np.abs(change)):.8g}')
        signs_preserved = np.all(
            (trained_weights == 0.0)
            | (np.sign(trained_weights) == initial_signs)
        )
        print(f'dale_signs_preserved={signs_preserved}')
        for metric_name in initial_metrics:
            print(f'{metric_name}_before={initial_metrics[metric_name]:.8g}')
            print(f'{metric_name}_after={trained_metrics[metric_name]:.8g}')
        print(f'objective_improved={trained_metrics["total"] < initial_metrics["total"]}')
        print(f'objective_improvement_percent={objective_improvement_percent:.8g}')
        print(f'verification_passed={verification_passed}')
        if not verification_passed:
            raise RuntimeError(
                f'{rule_name} improved the {task_name} objective by '
                f'{objective_improvement_percent:.3f}%, below the required '
                f'{expected_improvement:.3f}%.'
            )
    finally:
        generated_config.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rule', choices=tuple(RULES))
    parser.add_argument('--task', choices=tuple(TASKS), default='voltage_control')
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    run(args.rule, task_name=args.task, batch_size=args.batch_size)
