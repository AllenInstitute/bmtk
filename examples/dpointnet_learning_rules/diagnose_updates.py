import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet.learning_rules import (
    EPropLearningRule,
    LearningRuleObservations,
    ModPropLearningRule,
    ThreeFactorLearningRule,
)

from run_learning_rule import evaluation_batch


TARGET_RATE_HZ = 20.0
NORMALIZED_UPDATE_SIZE = 0.1
PROGRESSION_STEPS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_MAX_TRAINING_STEPS = 2000
DEFAULT_INITIAL_TRAINING_UPDATE_SIZE = 0.03
PLATEAU_WINDOW = 100
PLATEAU_RELATIVE_TOLERANCE = 5e-4
PLATEAU_LEARNING_RATE_FACTOR = 0.3
PLATEAU_REDUCTIONS_BEFORE_STOP = 3


def cosine_similarity(left, right):
    return float(
        np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def projected_window_improvement(values):
    values = np.asarray(values[-PLATEAU_WINDOW:], dtype=np.float64)
    x = np.arange(values.size, dtype=np.float64)
    trend_per_update = np.polyfit(x, values, 1)[0]
    return -trend_per_update * (values.size - 1)


def objective_components(spikes, voltages, simulation_seconds):
    firing_rates = tf.reduce_sum(spikes, axis=1) / simulation_seconds
    rate_loss = tf.reduce_mean(
        tf.square((firing_rates - TARGET_RATE_HZ) / TARGET_RATE_HZ)
    )
    voltage_loss = tf.reduce_mean(
        tf.square(tf.nn.relu(tf.abs(voltages - 0.5) - 0.5))
    )
    return {
        'target_rate': rate_loss,
        'voltage_regularizer': voltage_loss,
    }


def write_figures(
        output_dir, progression, training_progression, max_training_steps,
        initial_training_update_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'loss_progression.csv'
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=progression[0])
        writer.writeheader()
        writer.writerows(progression)

    labels = {
        'eprop': 'e-prop / combined',
        'three_factor_spike': 'three-factor: spike',
        'three_factor_voltage': 'three-factor: voltage',
        'modprop': 'ModProp',
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for rule_name, label in labels.items():
        rows = [row for row in progression if row['rule'] == rule_name]
        steps = [row['normalized_step'] for row in rows]
        axes[0].plot(
            steps, [row['improvement_percent'] for row in rows],
            marker='o', label=label,
        )
        axes[1].plot(
            steps, [row['target_rate_loss'] for row in rows],
            marker='o', label=label,
        )
        axes[2].plot(
            steps, [row['voltage_loss'] for row in rows],
            marker='o', label=label,
        )
    axes[0].axhline(0.0, color='black', linewidth=0.8)
    axes[0].set_ylabel('Total loss reduction (%)')
    axes[1].set_ylabel('Relative firing-rate MSE')
    axes[2].set_ylabel('Voltage range loss')
    for axis in axes:
        axis.set_xlabel('Applied recurrent-weight update norm')
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        'Actual loss after one constrained update on the same deterministic batch'
    )
    figure.tight_layout()
    figure.savefig(output_dir / 'loss_progression.png', dpi=180)
    figure.savefig(output_dir / 'loss_progression.svg')
    plt.close(figure)

    training_csv_path = output_dir / 'training_progression.csv'
    with open(training_csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=training_progression[0])
        writer.writeheader()
        writer.writerows(training_progression)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for rule_name, label in labels.items():
        rows = [row for row in training_progression if row['rule'] == rule_name]
        iterations = [row['iteration'] for row in rows]
        line = axes[0].plot(
            iterations, [row['total_loss'] for row in rows], label=label,
        )[0]
        color = line.get_color()
        axes[1].plot(
            iterations, [row['target_rate_loss'] for row in rows], color=color,
        )
        axes[2].plot(
            iterations, [row['voltage_loss'] for row in rows], color=color,
        )
        axes[0].scatter(
            iterations[-1], rows[-1]['total_loss'], marker='s', color=color
        )
        axes[1].scatter(
            iterations[-1], rows[-1]['target_rate_loss'],
            marker='s', color=color,
        )
        axes[2].scatter(
            iterations[-1], rows[-1]['voltage_loss'], marker='s', color=color
        )
        reduction_rows = [
            row for previous, row in zip(rows, rows[1:])
            if (
                row['learning_rate_reductions']
                > previous['learning_rate_reductions']
            )
        ]
        for axis, component in zip(
                axes, ('total_loss', 'target_rate_loss', 'voltage_loss')):
            axis.scatter(
                [row['iteration'] for row in reduction_rows],
                [row[component] for row in reduction_rows],
                marker='v', color=color, s=24,
            )
    axes[0].set_ylabel('Total loss')
    axes[1].set_ylabel('Relative firing-rate MSE')
    axes[2].set_ylabel('Voltage range loss')
    for axis in axes:
        axis.set_xlabel('Learning iteration')
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        f'Loss progression over at most {max_training_steps} recomputed '
        'full-batch updates\n'
        f'(initial ||ΔW||₂ = {initial_training_update_size}; triangles mark '
        'learning-rate reductions; squares mark termination)'
    )
    figure.tight_layout()
    figure.savefig(output_dir / 'training_progression.png', dpi=180)
    figure.savefig(output_dir / 'training_progression.svg')
    plt.close(figure)

    voltages = np.linspace(-0.5, 1.5, 500)
    unscaled_penalty = np.maximum(np.abs(voltages - 0.5) - 0.5, 0.0) ** 2
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].axis('off')
    axes[0].text(
        0.5, 0.88, 'Deterministic comparison task',
        ha='center', va='center', fontsize=15, weight='bold',
    )
    boxes = [
        (0.04, 0.57, '10 Hz Poisson input\n100 ms, batch 4'),
        (0.37, 0.57, 'All-to-all GLIF3\nrecurrent network'),
        (0.70, 0.57, '20 Hz rate target\n+ voltage range loss'),
        (0.37, 0.18, 'One local-rule update\nnormalized to ||ΔW||₂ = 0.1'),
    ]
    for x, y, text in boxes:
        axes[0].text(
            x, y, text, transform=axes[0].transAxes,
            ha='left', va='center',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '#eef4fb'},
        )
    axes[0].annotate('', xy=(0.36, 0.64), xytext=(0.27, 0.64),
                     xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    axes[0].annotate('', xy=(0.69, 0.64), xytext=(0.60, 0.64),
                     xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    axes[0].annotate('', xy=(0.52, 0.42), xytext=(0.80, 0.55),
                     xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    axes[0].annotate('', xy=(0.52, 0.55), xytext=(0.52, 0.32),
                     xycoords='axes fraction', arrowprops={'arrowstyle': '->'})
    axes[0].text(
        0.5, 0.03,
        'The same input, initial state, weights, and update norm are used for every rule.',
        transform=axes[0].transAxes, ha='center', fontsize=9,
    )

    axes[1].plot(voltages, unscaled_penalty, linewidth=2)
    axes[1].axvspan(0.0, 1.0, alpha=0.15, color='green', label='zero-penalty range')
    axes[1].set_xlabel('Normalized membrane voltage v')
    axes[1].set_ylabel('Unscaled penalty')
    axes[1].set_title(r'$\max(|v-0.5|-0.5, 0)^2$')
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[1].text(
        0.5, 0.95,
        r'Dimensionless objective uses $\lambda_V=1$',
        transform=axes[1].transAxes, ha='center', va='top', fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_dir / 'task_and_voltage_loss.png', dpi=180)
    figure.savefig(output_dir / 'task_and_voltage_loss.svg')
    plt.close(figure)
    return csv_path


def run(
        output_dir=None, max_training_steps=DEFAULT_MAX_TRAINING_STEPS,
        initial_training_update_size=DEFAULT_INITIAL_TRAINING_UPDATE_SIZE):
    example_dir = Path(__file__).resolve().parent
    with open(example_dir / 'config.base.json') as config_file:
        config_data = json.load(config_file)

    config_data['manifest']['$BASE_DIR'] = example_dir.as_posix()
    config_data['training']['learning_rule'] = {
        'name': 'eprop',
        'surfaces': ['<recurrent>'],
    }
    config_data['output']['output_dir'] = (
        example_dir / 'output' / 'diagnostic'
    ).as_posix()

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', dir=example_dir, delete=False) as config_file:
        json.dump(config_data, config_file, indent=2)
        generated_config = Path(config_file.name)

    try:
        config = dpointnet.Config.from_json(generated_config.as_posix())
        config.build_env()
        network = dpointnet.RNN.from_config(config)
        network.build()
        engine = network.training_engine
        parameter = engine.parameters[0]
        input_spikes, signatures = evaluation_batch(network)
        initial_state = engine.init_state.get_state()
        recurrent_weights = network.cell.recurrent_weight_values
        simulation_seconds = parameter.seq_len * network.dt / 1000.0

        with tf.GradientTape(persistent=True) as bptt_tape:
            output = engine._run_extractor(input_spikes, initial_state)
            spikes, voltages = output[0]
            components = objective_components(
                spikes, voltages, simulation_seconds
            )
            total_loss = sum(components.values())
        bptt_gradient = bptt_tape.gradient(total_loss, recurrent_weights)
        bptt_rate_gradient = bptt_tape.gradient(
            components['target_rate'], recurrent_weights
        )
        bptt_voltage_gradient = bptt_tape.gradient(
            components['voltage_regularizer'], recurrent_weights
        )
        del bptt_tape

        detached_spikes = tf.stop_gradient(spikes)
        detached_voltages = tf.stop_gradient(voltages)
        with tf.GradientTape() as signal_tape:
            signal_tape.watch([detached_spikes, detached_voltages])
            detached_components = objective_components(
                detached_spikes, detached_voltages, simulation_seconds
            )
            local_loss = sum(detached_components.values())
        spike_signal, voltage_signal = signal_tape.gradient(
            local_loss, [detached_spikes, detached_voltages]
        )
        observations = LearningRuleObservations(
            input_spikes=input_spikes,
            spikes=detached_spikes,
            voltages=detached_voltages,
            initial_state=initial_state,
            spike_learning_signal=spike_signal,
            voltage_learning_signal=voltage_signal,
            direct_weight_gradients=(tf.zeros_like(recurrent_weights),),
        )

        common = {'surfaces': ['<recurrent>'], 'edge_chunk_size': 4096}
        rules = {
            'eprop': EPropLearningRule(**common),
            'three_factor_combined': ThreeFactorLearningRule(
                signal='combined', **common
            ),
            'three_factor_spike': ThreeFactorLearningRule(
                signal='spike', **common
            ),
            'three_factor_voltage': ThreeFactorLearningRule(
                signal='voltage', **common
            ),
            'modprop': ModPropLearningRule(
                filter_taps=3, mean_activity=0.5, **common
            ),
        }
        updates = {}
        for name, rule in rules.items():
            rule.build(network)
            updates[name] = rule.compute_updates(observations)[0][1].numpy()

        exact = bptt_gradient.numpy()
        reference_gradients = {
            'eprop': exact,
            'three_factor_combined': exact,
            'three_factor_spike': bptt_rate_gradient.numpy(),
            'three_factor_voltage': bptt_voltage_gradient.numpy(),
            'modprop': exact,
        }
        target_components = {
            'eprop': 'total',
            'three_factor_combined': 'total',
            'three_factor_spike': 'target_rate',
            'three_factor_voltage': 'voltage_regularizer',
            'modprop': 'total',
        }
        eprop = updates['eprop']
        print(f'loss={float(total_loss.numpy()):.8g}')
        print(f'spike_signal_norm={np.linalg.norm(spike_signal.numpy()):.8g}')
        print(f'voltage_signal_norm={np.linalg.norm(voltage_signal.numpy()):.8g}')
        for name, update in updates.items():
            reference = reference_gradients[name]
            alignment = cosine_similarity(reference, update)
            relative_difference = np.linalg.norm(update - eprop) / np.linalg.norm(eprop)
            predicted_descent = float(np.dot(reference, update))
            print(f'{name}.update_norm={np.linalg.norm(update):.8g}')
            print(f'{name}.reference_component={target_components[name]}')
            print(f'{name}.reference_bptt_cosine={alignment:.8g}')
            print(f'{name}.reference_bptt_inner_product={predicted_descent:.8g}')
            print(f'{name}.relative_difference_from_eprop={relative_difference:.8g}')
            if predicted_descent <= 0.0:
                raise RuntimeError(
                    f'{name} is not a descent direction for its reference component.'
                )

        initial_weights = recurrent_weights.numpy().copy()
        initial_signs = np.sign(initial_weights)
        weight_constraint = recurrent_weights.constraint

        def evaluate_current_weights():
            current_output = network.run_extractor(input_spikes, initial_state)
            current_spikes, current_voltages = current_output[0]
            components = {
                name: float(value.numpy())
                for name, value in objective_components(
                    current_spikes, current_voltages, simulation_seconds
                ).items()
            }
            return sum(components.values()), components

        initial_forward_loss, initial_components = evaluate_current_weights()
        print(f'forward_loss_before={initial_forward_loss:.8g}')
        progression = []
        for name, update in updates.items():
            unit_update = update / np.linalg.norm(update)
            for requested_step in PROGRESSION_STEPS:
                candidate = initial_weights - requested_step * unit_update
                recurrent_weights.assign(weight_constraint(candidate))
                network.cell.refresh_recurrent_weight_shadow()
                constrained_weights = recurrent_weights.numpy()
                applied_update_norm = np.linalg.norm(
                    constrained_weights - initial_weights
                )
                updated_loss, updated_components = evaluate_current_weights()
                progression.append({
                    'rule': name,
                    'normalized_step': applied_update_norm,
                    'total_loss': updated_loss,
                    'improvement_percent': 100.0 * (
                        initial_forward_loss - updated_loss
                    ) / initial_forward_loss,
                    'target_rate_loss': updated_components['target_rate'],
                    'voltage_loss': updated_components['voltage_regularizer'],
                })

            final_row = min(
                (
                    row for row in progression
                    if row['rule'] == name
                ),
                key=lambda row: abs(
                    row['normalized_step'] - NORMALIZED_UPDATE_SIZE
                ),
            )
            updated_loss = final_row['total_loss']
            updated_components = {
                'target_rate': final_row['target_rate_loss'],
                'voltage_regularizer': final_row['voltage_loss'],
            }
            improvement = final_row['improvement_percent']
            applied_update_norm = final_row['normalized_step']
            dale_signs_preserved = np.all(
                (constrained_weights == 0.0)
                | (np.sign(constrained_weights) == initial_signs)
            )
            print(f'{name}.forward_loss_after={updated_loss:.8g}')
            print(f'{name}.forward_improvement_percent={improvement:.8g}')
            print(f'{name}.applied_update_norm={applied_update_norm:.8g}')
            print(f'{name}.dale_signs_preserved={dale_signs_preserved}')
            for component, value in initial_components.items():
                print(f'{name}.{component}_before={value:.8g}')
                print(f'{name}.{component}_after={updated_components[component]:.8g}')
            reference_before = (
                initial_forward_loss
                if target_components[name] == 'total'
                else initial_components[target_components[name]]
            )
            reference_after = (
                updated_loss
                if target_components[name] == 'total'
                else updated_components[target_components[name]]
            )
            if reference_after >= reference_before:
                raise RuntimeError(
                    f'{name} did not reduce its rerun reference component.'
                )
            if not dale_signs_preserved:
                raise RuntimeError(f'{name} violated Dale constraints.')
        recurrent_weights.assign(initial_weights)
        network.cell.refresh_recurrent_weight_shadow()
        training_progression = []
        training_rules = {
            name: rule for name, rule in rules.items()
            if name != 'three_factor_combined'
        }
        for name, rule in training_rules.items():
            recurrent_weights.assign(initial_weights)
            network.cell.refresh_recurrent_weight_shadow()
            plateau_history = []
            initial_reference_value = None
            initial_raw_update_norm = None
            learning_rate = None
            learning_rate_reductions = 0
            stop_reason = f'max_steps={max_training_steps}'
            for iteration in range(max_training_steps + 1):
                current_output = engine._run_extractor(input_spikes, initial_state)
                current_spikes, current_voltages = (
                    tf.stop_gradient(value) for value in current_output[0]
                )
                with tf.GradientTape() as current_signal_tape:
                    current_signal_tape.watch(
                        [current_spikes, current_voltages]
                    )
                    current_components = objective_components(
                        current_spikes, current_voltages, simulation_seconds
                    )
                    current_loss = sum(current_components.values())
                components = {
                    component_name: float(value.numpy())
                    for component_name, value in current_components.items()
                }
                iteration_loss = float(current_loss.numpy())
                training_progression.append({
                    'rule': name,
                    'iteration': iteration,
                    'total_loss': iteration_loss,
                    'target_rate_loss': components['target_rate'],
                    'voltage_loss': components['voltage_regularizer'],
                    'raw_update_norm': '',
                    'applied_update_norm': '',
                    'learning_rate': learning_rate if learning_rate is not None else '',
                    'learning_rate_reductions': learning_rate_reductions,
                })

                reference_component = target_components[name]
                reference_value = (
                    iteration_loss
                    if reference_component == 'total'
                    else components[reference_component]
                )
                plateau_history.append(reference_value)
                if initial_reference_value is None:
                    initial_reference_value = reference_value
                if len(plateau_history) >= PLATEAU_WINDOW:
                    projected_improvement = projected_window_improvement(
                        plateau_history
                    )
                    tolerance = (
                        PLATEAU_RELATIVE_TOLERANCE
                        * max(abs(initial_reference_value), 1e-12)
                    )
                    if projected_improvement <= tolerance:
                        if (
                                learning_rate_reductions
                                >= PLATEAU_REDUCTIONS_BEFORE_STOP):
                            stop_reason = (
                                f'plateau after {learning_rate_reductions} '
                                f'learning-rate reductions: projected '
                                f'{reference_component} improvement over '
                                f'{PLATEAU_WINDOW} updates is '
                                f'{projected_improvement:.3g}, at or below '
                                f'{tolerance:.3g}'
                            )
                            break
                        if learning_rate is not None:
                            learning_rate *= PLATEAU_LEARNING_RATE_FACTOR
                            learning_rate_reductions += 1
                            plateau_history.clear()
                if iteration == max_training_steps:
                    break

                current_spike_signal, current_voltage_signal = (
                    current_signal_tape.gradient(
                        current_loss, [current_spikes, current_voltages]
                    )
                )
                current_observations = LearningRuleObservations(
                    input_spikes=input_spikes,
                    spikes=current_spikes,
                    voltages=current_voltages,
                    initial_state=initial_state,
                    spike_learning_signal=current_spike_signal,
                    voltage_learning_signal=current_voltage_signal,
                    direct_weight_gradients=(tf.zeros_like(recurrent_weights),),
                )
                current_update = rule.compute_updates(
                    current_observations
                )[0][1].numpy()
                raw_update_norm = np.linalg.norm(current_update)
                if not np.isfinite(raw_update_norm) or raw_update_norm == 0.0:
                    raise RuntimeError(
                        f'{name} produced invalid update norm {raw_update_norm}.'
                    )
                if learning_rate is None:
                    initial_raw_update_norm = raw_update_norm
                    learning_rate = (
                        initial_training_update_size / initial_raw_update_norm
                    )
                current_weights = recurrent_weights.numpy()
                candidate = current_weights - learning_rate * current_update
                recurrent_weights.assign(weight_constraint(candidate))
                network.cell.refresh_recurrent_weight_shadow()
                applied_update_norm = np.linalg.norm(
                    recurrent_weights.numpy() - current_weights
                )
                training_progression[-1].update({
                    'raw_update_norm': raw_update_norm,
                    'applied_update_norm': applied_update_norm,
                    'learning_rate': learning_rate,
                    'learning_rate_reductions': learning_rate_reductions,
                })
            print(f'{name}.training_iterations={iteration}')
            print(f'{name}.training_learning_rate={learning_rate:.8g}')
            print(
                f'{name}.training_initial_raw_update_norm='
                f'{initial_raw_update_norm:.8g}'
            )
            print(f'{name}.training_stop_reason={stop_reason}')
            print(
                f'{name}.training_learning_rate_reductions='
                f'{learning_rate_reductions}'
            )

        recurrent_weights.assign(initial_weights)
        network.cell.refresh_recurrent_weight_shadow()
        if output_dir is not None:
            csv_path = write_figures(
                Path(output_dir), progression, training_progression,
                max_training_steps, initial_training_update_size,
            )
            print(f'figures_dir={Path(output_dir).resolve()}')
            print(f'progression_csv={csv_path.resolve()}')

        np.testing.assert_allclose(
            updates['three_factor_combined'], eprop, rtol=1e-5, atol=1e-7
        )
        if np.linalg.norm(updates['modprop'] - eprop) / np.linalg.norm(eprop) < 0.5:
            raise RuntimeError('ModProp update is not distinct from e-prop.')
        print('verification_passed=True')
    finally:
        generated_config.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument(
        '--max-training-steps', type=int, default=DEFAULT_MAX_TRAINING_STEPS
    )
    parser.add_argument(
        '--initial-training-update-size', type=float,
        default=DEFAULT_INITIAL_TRAINING_UPDATE_SIZE,
    )
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        max_training_steps=args.max_training_steps,
        initial_training_update_size=args.initial_training_update_size,
    )
