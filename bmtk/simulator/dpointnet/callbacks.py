from time import time
from enum import IntEnum
from datetime import datetime
import os
import csv
import subprocess
import tensorflow as tf
import pandas as pd
from collections import namedtuple
import numpy as np
from pathlib import Path

from .io_tools import io

callback_classes = {}


def add_callbacks(cls, name=None, overwrite=True):
    global callback_classes

    name = name or cls.__name__
    if overwrite or name not in callback_classes:
        callback_classes[name] = cls


def register_callbacks(cls=None, *_, name=None, overwrite=True):
    def decorator(_cls):
        add_callbacks(_cls, name=name, overwrite=overwrite)
        return _cls

    if cls is not None:
        return decorator(cls)
    else:
        return decorator


GPUMem = namedtuple(
    "GPUMem",
    [
        "name",
        "tf_current",
        "tf_peak",
        "gpu_used",
        "gpu_free",
        "gpu_total",
        "process_used",
        "peak_scope",
    ],
    defaults=[None] * 7 + ["since allocator initialization"],
)


class Verbosity(IntEnum):
    quiet = 0
    on_train = 1
    on_epoch = 2
    on_step = 3
    full = 4


@register_callbacks
class Callbacks:
    def __init__(
        self,
        rnn,
        callbacks_dir="callbacks_outputs",
        starting_epoch=0,
        verbose="full",
        time_fmt="%d-%m-%Y %H:%M",
        epoch_store_weights="best",
        epoch_cache_weights=False,
        epoch_cach_dir="weights",
        sonata_output_dir="trained_weights",
        losses_table_csv="losses.csv",
        performance_table_csv="performance.csv",
        memory_report="epoch",
    ):
        self.rnn = rnn
        self.callbacks_dir = callbacks_dir
        self.train_start_time = None
        self.training_time = None
        self.epoch_start_time = None
        self._epoch_times = []
        self.step_start_time = None
        self._step_times = []

        self.epoch_num = 0
        self.tot_step_num = 0
        self.epoch_step_num = 0

        self.verbosity = (
            verbose if isinstance(verbose, Verbosity) else Verbosity[verbose.lower()]
        )
        self.time_fmt = time_fmt

        self._gpu_devices = tf.config.list_physical_devices("GPU")
        if memory_report not in ("epoch", "step", "off"):
            raise ValueError("memory_report must be 'epoch', 'step', or 'off'")
        self.memory_report = memory_report
        self._memory_peak_scopes = {}
        self._step_memory_usage = []
        self._step_loss_rates = []
        self._step_loss_values = []
        self._losses_header_written = False
        self._performance_header_written = False

        self.epoch_losses = []
        self.epoch_vallosses = []
        self.epoch_vallosses_best = np.inf
        self.epoch_store_weights = epoch_store_weights
        self._saved_weights = []
        self.epoch_cache_weights = epoch_cache_weights
        self.epoch_cache_dir = Callbacks.find_path(self.callbacks_dir, epoch_cach_dir)

        self.sonata_output_dir = Callbacks.find_path(
            self.callbacks_dir, sonata_output_dir
        )
        self.losses_table_csv = Callbacks.find_path(
            self.callbacks_dir, losses_table_csv
        )
        self.performance_table_csv = Callbacks.find_path(
            self.callbacks_dir, performance_table_csv
        )

    @property
    def time(self):
        return datetime.now().strftime(self.time_fmt)

    @property
    def n_epochs(self):
        return self.rnn.training_engine.n_epochs

    @property
    def steps_per_epoch(self):
        return self.rnn.training_engine.steps_per_epoch

    @staticmethod
    def find_path(callbacks_dir, path):
        if path is None or path is False:
            return None
        elif Path(path).is_absolute():
            return path
        else:
            return Path(callbacks_dir) / path

    def on_train_begin(self):
        self.train_start_time = time()
        self.tot_step_num = 0

        if self.verbosity >= Verbosity.on_train:
            io.log_info(f"> Training Started @ {self.time}")

        allocator = os.environ.get("TF_GPU_ALLOCATOR", "")
        self._append_performance_to_csv(
            "tf_gpu_allocator", None, None, allocator or "default"
        )

    def on_train_end(self, metrics=None, normalizers=None):
        self.training_time = time() - self.train_start_time
        self._append_performance_to_csv("training_time", None, None, self.training_time)

        if len(self._saved_weights) > 1:
            for epoch_num, epoch_weights in self._saved_weights:
                output_dir = f"{self.sonata_output_dir}.epoch_{epoch_num}"
                epoch_weights.to_sonata(output_dir=output_dir, overwrite=True)
        elif len(self._saved_weights) == 1:
            self._saved_weights[0].to_sonata(
                output_dir=self.sonata_output_dir, overwrite=True
            )
        else:
            pass

    def on_epoch_start(self):
        self.epoch_start_time = time()
        self.epoch_num += 1
        self.epoch_step_num = 0
        if self.memory_report != "off":
            if self.epoch_num == 1:
                startup_memory = self._gpu_mem_usage()
                self._append_gpu_memory_usage("pre_epoch1", 0, None, startup_memory)
                if self.verbosity >= Verbosity.on_epoch:
                    for gpu_mem in startup_memory:
                        io.log_info(f">> Pre-epoch 1: {self._format_gpu_mem(gpu_mem)}")
            self._reset_tf_memory_stats()

        if self.verbosity >= Verbosity.on_epoch:
            io.log_info(
                f">> Epoch {self.epoch_num}/{self.n_epochs} Started @ {self.time}"
            )

    def on_epoch_end(self, validation_losses):
        validation_losses = self._detach_loss_values(validation_losses)
        epoch_time = time() - self.epoch_start_time
        self._epoch_times.append(epoch_time)
        self.epoch_losses.append(validation_losses)
        self._append_performance_to_csv(
            "epoch_timesteps", self.epoch_num, None, epoch_time
        )

        val_loss = validation_losses["__total_loss"]
        self.epoch_vallosses.append(val_loss)
        self._store_weights(val_loss)
        self._append_losses_to_csv("validation", self.epoch_num, 0, validation_losses)
        gpu_mem_info = self._gpu_mem_usage() if self.memory_report != "off" else []
        self._append_gpu_memory_usage("epoch_end", self.epoch_num, None, gpu_mem_info)

        if self.verbosity >= Verbosity.on_epoch:
            io.log_info(f">> Epoch Finished.")
            io.log_info(
                f">>   Validation Loss: {self._format_value(val_loss)} "
                f"(Rate: {self._format_mean_rate(validation_losses)})"
            )
            for loss_line in self._format_loss_components(validation_losses):
                io.log_info(f">>                    {loss_line}")
            for gpu_mem in gpu_mem_info:
                io.log_info(f">>   {self._format_gpu_mem(gpu_mem)}")

    def on_step_start(self):
        self.tot_step_num += 1
        self.epoch_step_num += 1
        self.step_start_time = time()

    def on_step_end(self, loss_vals):
        loss_vals = self._detach_loss_values(loss_vals)
        step_time = time() - self.step_start_time
        self._step_times.append(step_time)
        gpu_mem_info = (
            self._record_memory_usage() if self.memory_report == "step" else []
        )
        self._append_performance_to_csv(
            "step_timesteps", self.epoch_num, self.epoch_step_num, step_time
        )
        if gpu_mem_info and gpu_mem_info[0].gpu_used is not None:
            self._append_performance_to_csv(
                "gpu_memory_usage_per_step",
                self.epoch_num,
                self.epoch_step_num,
                gpu_mem_info[0].gpu_used,
            )
        self._append_gpu_memory_usage(
            "step", self.epoch_num, self.epoch_step_num, gpu_mem_info
        )
        self._step_loss_values.append(loss_vals)
        self._step_loss_rates.append(loss_vals["__total_loss"])
        self._append_losses_to_csv(
            "step", self.epoch_num, self.epoch_step_num, loss_vals
        )

        if self.verbosity >= Verbosity.on_step:
            epoch_width = max(2, len(str(self.n_epochs)))
            step_width = max(2, len(str(self.steps_per_epoch)))
            io.log_info(
                f">>> Epoch {self.epoch_num:{epoch_width}d}/{self.n_epochs:{epoch_width}d}, "
                f"Step {self.epoch_step_num:{step_width}d}/{self.steps_per_epoch:{step_width}d} "
                f"(run time: {step_time:.2f} s, Rate: {self._format_mean_rate(loss_vals)})"
            )
            loss_prefix = f'Loss: {self._format_value(loss_vals["__total_loss"])} '
            loss_lines = self._format_loss_components(loss_vals)
            if loss_lines:
                io.log_info(f">>>   {loss_prefix}{loss_lines[0]}")
                for loss_line in loss_lines[1:]:
                    io.log_info(f'>>>   {" " * len(loss_prefix)}{loss_line}')
            else:
                io.log_info(f">>>   {loss_prefix.rstrip()}")

            for gpu_mem in gpu_mem_info:
                io.log_info(f">>>   {self._format_gpu_mem(gpu_mem)}")

    def _format_value(self, value):
        return f"{self._as_float(value):.4f}"

    def _format_loss_components(self, loss_vals):
        component_values = []
        for pname, pval in loss_vals.items():
            if not isinstance(pval, dict):
                continue
            values = []
            for loss_name, loss_val in pval.items():
                if loss_name.startswith("__"):
                    continue
                values.append(f"{self._as_float(loss_val):8.4f}")
            if values:
                component_values.append((pname, values))

        if not component_values:
            return []

        pname_width = max(len(pname) for pname, _ in component_values)
        return [
            f'({pname:<{pname_width}}: {", ".join(values)})'
            for pname, values in component_values
        ]

    def _format_mean_rate(self, loss_vals):
        rates = []
        for pval in loss_vals.values():
            if isinstance(pval, dict) and "__mean_rate" in pval:
                rates.append(self._as_float(pval["__mean_rate"]))
        if not rates:
            return "n/a"
        return f"{np.mean(rates):.4f}"

    @staticmethod
    def _as_float(value):
        if hasattr(value, "values"):
            value = value.values[0]
        if hasattr(value, "numpy"):
            value = value.numpy()
        return float(value)

    @classmethod
    def _detach_loss_values(cls, loss_vals):
        if isinstance(loss_vals, dict):
            return {
                name: cls._detach_loss_values(value)
                for name, value in loss_vals.items()
            }
        return cls._as_float(loss_vals)

    @staticmethod
    def _format_gpu_mem(gpu_mem):
        def quantity(value):
            return "n/a" if value is None else f"{value:.2f} GiB"

        return (
            f"{gpu_mem.name}: TF current {quantity(gpu_mem.tf_current)} | "
            f"TF peak {quantity(gpu_mem.tf_peak)} ({gpu_mem.peak_scope}) | "
            f"process {quantity(gpu_mem.process_used)} | "
            f"device used {quantity(gpu_mem.gpu_used)} | "
            f"device free {quantity(gpu_mem.gpu_free)} | "
            f"device total {quantity(gpu_mem.gpu_total)}"
        )

    def _record_memory_usage(self):
        if not self._gpu_devices:
            return []

        gpu_mem_info = self._gpu_mem_usage()
        if gpu_mem_info and gpu_mem_info[0].gpu_used is not None:
            self._step_memory_usage.append(gpu_mem_info[0].gpu_used)
        return gpu_mem_info

    def _reset_tf_memory_stats(self):
        for gpu_name in self._memory_device_names():
            try:
                tf.config.experimental.reset_memory_stats(gpu_name)
                self._memory_peak_scopes[gpu_name] = (
                    "epoch 1; includes first-step tracing"
                    if self.epoch_num == 1
                    else f"since start of epoch {self.epoch_num}"
                )
            except (ValueError, RuntimeError):
                pass

    def _append_gpu_memory_usage(self, prefix, epoch_num, step_num, gpu_mem_info=None):
        if gpu_mem_info is None:
            gpu_mem_info = self._gpu_mem_usage()

        for gpu_mem in gpu_mem_info:
            metric_prefix = f'{prefix}_{gpu_mem.name.lower().replace(":", "")}'
            metrics = {
                f"{metric_prefix}_resident_used_gib": gpu_mem.gpu_used,
                f"{metric_prefix}_resident_free_gib": gpu_mem.gpu_free,
                f"{metric_prefix}_resident_total_gib": gpu_mem.gpu_total,
                f"{metric_prefix}_tf_allocator_current_gib": gpu_mem.tf_current,
                f"{metric_prefix}_tf_allocator_peak_gib": gpu_mem.tf_peak,
                f"{metric_prefix}_process_used_gib": gpu_mem.process_used,
                f"{metric_prefix}_tf_allocator_peak_scope": gpu_mem.peak_scope,
            }
            for name, value in metrics.items():
                self._append_performance_to_csv(name, epoch_num, step_num, value)

    def _memory_device_names(self):
        if not self._gpu_devices:
            return []
        devices = [
            tf.DeviceSpec.from_string(name)
            for name in self.rnn.strategy.extended.worker_devices
        ]
        return list(
            dict.fromkeys(
                f"GPU:{device.device_index}"
                for device in devices
                if device.device_type == "GPU"
            )
        )

    @staticmethod
    def _nvidia_smi_rows(query):
        try:
            result = subprocess.run(
                ["nvidia-smi", query, "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                check=True,
                timeout=2,
            )
            return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        except (OSError, subprocess.SubprocessError):
            return []

    def _gpu_mem_usage(self):
        names = self._memory_device_names()
        if not names:
            return []
        devices = self._nvidia_smi_rows(
            "--query-gpu=uuid,memory.used,memory.free,memory.total"
        )
        processes = self._nvidia_smi_rows(
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory"
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        physical_mapping = tf.config.get_visible_devices(
            "GPU"
        ) == self._gpu_devices and all(
            not tf.config.get_logical_device_configuration(device)
            for device in self._gpu_devices
        )
        mem_info = []
        for gpu_name in names:
            current = peak = used = free = total = process_used = None
            try:
                meminfo = tf.config.experimental.get_memory_info(gpu_name)
                current = meminfo["current"] / 1024**3
                peak = meminfo["peak"] / 1024**3
            except (ValueError, RuntimeError):
                pass
            gpu_id = int(gpu_name.split(":")[-1])
            matches = []
            if (
                physical_mapping
                and len(devices) == 1
                and names == ["GPU:0"]
                and len(self._gpu_devices) == 1
                and not any(token.startswith("MIG-") for token in visible)
            ):
                matches = devices
            elif (
                physical_mapping
                and gpu_id < len(visible)
                and visible[gpu_id].startswith("GPU-")
            ):
                matches = [
                    row for row in devices if row and row[0].startswith(visible[gpu_id])
                ]
            if len(matches) == 1 and len(matches[0]) == 4:
                uuid, *values = matches[0]
                try:
                    used, free, total = [float(value) / 1024 for value in values]
                except ValueError:
                    pass
                matching_processes = [
                    row
                    for row in processes
                    if len(row) == 3 and row[0] == uuid and row[1] == str(os.getpid())
                ]
                if matching_processes:
                    try:
                        process_used = (
                            sum(float(row[2]) for row in matching_processes) / 1024
                        )
                    except ValueError:
                        pass
            mem_info.append(
                GPUMem(
                    name=gpu_name,
                    tf_peak=peak,
                    tf_current=current,
                    gpu_used=used,
                    gpu_free=free,
                    gpu_total=total,
                    process_used=process_used,
                    peak_scope=self._memory_peak_scopes.get(
                        gpu_name, "since allocator initialization"
                    ),
                )
            )
        return mem_info

    def _store_weights(self, val_loss):
        current_weights = None
        if self.epoch_store_weights == "all":
            # If set to 'all' store model weights after each epoch. After full training
            # teh _saved_weights list will be of size n_epochs.
            current_weights = self.rnn.get_model_weights(deep_copy=True)
            self._saved_weights.append(current_weights)

        elif self.epoch_store_weights == "latest":
            # Only store the weights for the latest epoch, no matter the results
            current_weights = self.rnn.get_model_weights(deep_copy=True)
            self._saved_weights = [current_weights]

        elif self.epoch_store_weights == "best":
            # Only store model weights if the validation loss is smallest value.
            if val_loss < self.epoch_vallosses_best:
                self.epoch_vallosses_best = val_loss
                current_weights = self.rnn.get_model_weights(deep_copy=True)
                self._saved_weights = [current_weights]

        elif self.epoch_store_weights == "skip":
            # Don't store weights
            pass

        else:
            raise ValueError(
                f'Invalid "epoch_store_weights" otpion {self._epoch_store_weights}'
            )

        if self.epoch_cache_weights and current_weights is not None:
            current_weights.to_pickle(
                self.epoch_cache_dir / f"saved_weights.epoch{self.epoch_num}.pkl"
            )

    def _record_losses(self):
        if self.losses_table_csv is None:
            return

        steps_per_epoch = self.rnn.training_engine.steps_per_epoch

        loss_type = []
        epoch_nums = []
        step_nums = []
        pnames = []
        loss_names = []
        loss_vals = []
        for step_count, step_losses in enumerate(self._step_loss_values):
            epoch_num = int(step_count / steps_per_epoch) + 1
            step_num = step_count % steps_per_epoch + 1
            for _pname, _pval in step_losses.items():
                if isinstance(_pval, dict):
                    for _lname, _lval in _pval.items():
                        loss_type.append("step")
                        pnames.append(_pname)
                        loss_names.append(_lname)
                        loss_vals.append(self._as_float(_lval))
                        epoch_nums.append(epoch_num)
                        step_nums.append(step_num)
                else:
                    loss_type.append("step")
                    pnames.append("")
                    loss_names.append(_pname)
                    loss_vals.append(self._as_float(_pval))
                    epoch_nums.append(epoch_num)
                    step_nums.append(step_num)

        for epoch_num, epoch_val in enumerate(self.epoch_losses):
            for _pname, _pval in epoch_val.items():
                if isinstance(_pval, dict):
                    for _lname, _lval in _pval.items():
                        loss_type.append("validation")
                        pnames.append(_pname)
                        loss_names.append(_lname)
                        loss_vals.append(self._as_float(_lval))
                        epoch_nums.append(epoch_num + 1)
                        step_nums.append(0)
                else:
                    loss_type.append("validation")
                    pnames.append("")
                    loss_names.append(_pname)
                    loss_vals.append(self._as_float(_pval))
                    epoch_nums.append(epoch_num + 1)
                    step_nums.append(0)

        Path(self.losses_table_csv).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(
            {
                "loss_type": loss_type,
                "epoch": epoch_nums,
                "step": step_nums,
                "parameter": pnames,
                "loss_function": loss_names,
                "loss_value": loss_vals,
            }
        ).to_csv(self.losses_table_csv, index=False)

    def _append_losses_to_csv(self, loss_type, epoch_num, step_num, loss_vals):
        if self.losses_table_csv is None:
            return

        records = []
        for parameter_name, parameter_vals in loss_vals.items():
            if isinstance(parameter_vals, dict):
                for loss_name, loss_val in parameter_vals.items():
                    records.append(
                        {
                            "loss_type": loss_type,
                            "epoch": epoch_num,
                            "step": step_num,
                            "parameter": parameter_name,
                            "loss_function": loss_name,
                            "loss_value": self._as_float(loss_val),
                        }
                    )
            else:
                records.append(
                    {
                        "loss_type": loss_type,
                        "epoch": epoch_num,
                        "step": step_num,
                        "parameter": "",
                        "loss_function": parameter_name,
                        "loss_value": self._as_float(parameter_vals),
                    }
                )

        Path(self.losses_table_csv).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame.from_records(records).to_csv(
            self.losses_table_csv,
            mode="a" if self._losses_header_written else "w",
            header=not self._losses_header_written,
            index=False,
        )
        self._losses_header_written = True

    def _record_performance(self):
        if self.performance_table_csv is None:
            return

        steps_per_epoch = self.rnn.training_engine.steps_per_epoch
        n_epochs = len(self._epoch_times)

        names = []
        epoch_nums = []
        step_nums = []
        values = []

        _epochs = np.repeat(range(1, n_epochs + 1), steps_per_epoch).tolist()
        _steps = np.tile(range(1, steps_per_epoch + 1), n_epochs).tolist()

        values += self._step_memory_usage
        names += ["gpu_memory_usage_per_step"] * len(self._step_memory_usage)
        epoch_nums += _epochs
        step_nums += _steps

        values += self._step_times
        names += ["step_timesteps"] * len(self._step_times)
        epoch_nums += _epochs
        step_nums += _steps

        values += self._epoch_times
        names += ["epoch_timesteps"] * len(self._epoch_times)
        epoch_nums += range(1, n_epochs + 1)
        step_nums += [None] * len(self._epoch_times)

        values += [self.training_time]
        names += ["training_time"]
        epoch_nums += [None]
        step_nums += [None]

        Path(self.performance_table_csv).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(
            {
                "name": names,
                "epoch": epoch_nums,
                "step": step_nums,
                "values": values,
            }
        ).to_csv(self.performance_table_csv, index=False)

    def _append_performance_to_csv(self, name, epoch_num, step_num, value):
        if self.performance_table_csv is None:
            return

        Path(self.performance_table_csv).parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(
            [
                {
                    "name": name,
                    "epoch": epoch_num,
                    "step": step_num,
                    "values": value,
                }
            ]
        ).to_csv(
            self.performance_table_csv,
            mode="a" if self._performance_header_written else "w",
            header=not self._performance_header_written,
            index=False,
        )
        self._performance_header_written = True


def get_gpu_memory(gpu_id=0):
    """Returns GPU memory usage in GiB using only nvidia-smi."""
    try:
        # Get GPU memory info using nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        gpu_memory_info = result.stdout.strip().split("\n")
        if gpu_id < len(gpu_memory_info):
            # Convert MiB to GiB
            used_memory = float(gpu_memory_info[gpu_id]) / 1024
            return used_memory
        else:
            return 0.0
    except:
        return 0.0
