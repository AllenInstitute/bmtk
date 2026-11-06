import inspect
from collections.abc import Mapping
from numbers import Real

import tensorflow as tf
import math


# Adapted from Javier v1_model_utils/optimizers.py at commit 2c52ec10.
# Credit: Javier. DPointNet keeps its existing optimizer slot order, constraints,
# loss scaling, and post-update shadow refresh; this helper only fuses the dense
# ExponentiatedAdam tensor update into one XLA cluster.
@tf.function(jit_compile=True)
def _dense_exponentiated_adam_update(
    weights,
    momentums,
    velocities,
    gradient,
    alpha,
    beta_1,
    beta_2,
    epsilon,
):
    new_momentums = momentums + (gradient - momentums) * (1 - beta_1)
    new_velocities = velocities + (tf.square(gradient) - velocities) * (1 - beta_2)
    adam_gradient = new_momentums / (tf.sqrt(new_velocities) + epsilon)
    signs = tf.sign(weights)
    signs = tf.where(tf.equal(signs, 0), tf.ones_like(signs), signs)
    new_weights = weights * tf.exp(-alpha * adam_gradient * signs)
    return new_weights, new_momentums, new_velocities


def _optimizer_init_accepts(argument_name):
    return argument_name in inspect.signature(tf.keras.optimizers.Optimizer.__init__).parameters


def build_learning_rate(lr_schedule, **lr_params):
    if lr_schedule is None or lr_schedule == 'none':
        return lr_params['learning_rate']
    
    elif lr_schedule == 'warmup_cosine':
        schedule = LinearWarmupCosineDecay(
            **lr_params
        )
        return schedule
    
    else:
        raise ValueError(f'Invalid lr_schedule "{lr_schedule}". Supported values: "none", "warmup_cosine".')


def optimizer_is_built(optimizer):
    """Whether ``optimizer`` has created its slots.

    A loss-scale wrapper counts as built only when its inner optimizer is built and
    the wrapper itself is not explicitly unbuilt (Keras 3 wrappers own the loss-scale
    variables and are not rebuilt idempotently).
    """
    while getattr(optimizer, 'inner_optimizer', None) is not None:
        if getattr(optimizer, 'built', None) is False:
            return False
        optimizer = optimizer.inner_optimizer
    return bool(
        getattr(optimizer, 'built', False) is True
        or getattr(optimizer, '_built', False) is True
        or getattr(optimizer, '_slots_built', False) is True
    )


def build_optimizer(optimizer, variables):
    """Create optimizer state for ``variables`` (Keras 3, TF-Keras 2 and legacy)."""
    build = getattr(optimizer, 'build', None)
    if build is not None:
        build(variables)
    else:
        optimizer._create_all_weights(variables)


def loss_scale_optimizer_for(optimizer, dtype):
    """``optimizer`` wrapped in ``LossScaleOptimizer`` for float16 training.

    Returns ``optimizer`` itself when ``dtype`` is not float16 or it already scales
    losses, so repeated calls keep one wrapper object.
    """
    if optimizer is None or dtype is None or tf.as_dtype(dtype) != tf.float16:
        return optimizer
    if optimizer_supports_loss_scaling(optimizer):
        return optimizer
    return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


def optimizer_supports_loss_scaling(optimizer):
    if hasattr(optimizer, 'get_scaled_loss') and hasattr(optimizer, 'get_unscaled_gradients'):
        return True
    # Keras 3 exposes scale_loss() on every optimizer, even when
    # loss_scale_factor is None and the method is a no-op.
    loss_scale_optimizer = getattr(
        tf.keras.mixed_precision, 'LossScaleOptimizer', ()
    )
    if isinstance(optimizer, loss_scale_optimizer):
        return True
    return getattr(optimizer, 'loss_scale_factor', None) is not None


def scale_loss_for_optimizer(optimizer, loss):
    if hasattr(optimizer, 'scale_loss'):
        return optimizer.scale_loss(loss)
    if hasattr(optimizer, 'get_scaled_loss'):
        return optimizer.get_scaled_loss(loss)
    return loss


def unscale_gradients_for_optimizer(optimizer, gradients):
    if hasattr(optimizer, 'get_unscaled_gradients'):
        return optimizer.get_unscaled_gradients(gradients)
    
    return gradients


def unscale_gradients_for_local_rule(optimizer, gradients):
    """Return physical gradients before local-rule clipping or transformations."""
    if hasattr(optimizer, 'get_unscaled_gradients'):
        return optimizer.get_unscaled_gradients(gradients)
    if hasattr(optimizer, 'dynamic_scale'):
        scale = optimizer.dynamic_scale
        return [
            None if gradient is None
            else gradient / tf.cast(scale, gradient.dtype)
            for gradient in gradients
        ]
    return gradients


def prepare_local_gradients_for_optimizer(optimizer, gradients):
    """Restore Keras 3 loss scaling before LossScaleOptimizer.apply_gradients."""
    if hasattr(optimizer, 'dynamic_scale'):
        scale = optimizer.dynamic_scale
        return [
            None if gradient is None
            else gradient * tf.cast(scale, gradient.dtype)
            for gradient in gradients
        ]
    return gradients


def create_optimizer(optimizer, learning_rate, optimizer_params=None):
    """Build a configured optimizer, clipping physical (unscaled) gradients.

    ``name`` is the training-config selector, not the Keras instance name.
    Configure clipping here on the inner optimizer, before loss-scale wrapping.
    """
    if optimizer_params is None:
        optimizer_params = {}
    if not isinstance(optimizer_params, Mapping):
        raise TypeError('optimizer_params must be a mapping.')
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        if optimizer_params:
            raise ValueError('optimizer_params cannot configure an optimizer instance.')
        return optimizer

    factories = {
        'adam': (tf.keras.optimizers.Adam, {'epsilon': 1.0e-11}),
        'chunked_adam': (ChunkedAdam, {'epsilon': 1.0e-11, 'update_chunk_size': ChunkedAdam.DEFAULT_CHUNK}),
        'exp_adam': (ExponentiatedAdam, {'epsilon': 1.0e-11, 'jit_compile': True}),
        'sgd': (tf.keras.optimizers.SGD, {'momentum': 0.0, 'nesterov': False}),
    }
    if optimizer not in factories:
        raise ValueError(f'Invalid optimizer: {optimizer}')
    optimizer_cls, params = factories[optimizer]
    clipping_modes = ('clipnorm', 'clipvalue', 'global_clipnorm')
    unsupported = set(optimizer_params) - set(params) - set(clipping_modes) - {'name'}
    if unsupported:
        raise ValueError(
            f'Unsupported optimizer_params for {optimizer}: '
            + ', '.join(sorted(str(key) for key in unsupported))
        )
    if 'name' in optimizer_params and optimizer_params['name'] != optimizer:
        raise ValueError('optimizer_params name must match the optimizer selector.')

    enabled_clipping = []
    for key in clipping_modes:
        value = optimizer_params.get(key)
        if value is not None:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be a finite positive number or None.')
            enabled_clipping.append(key)
    if len(enabled_clipping) > 1:
        raise ValueError('Only one of clipnorm, clipvalue, global_clipnorm may be set.')

    params.update({key: value for key, value in optimizer_params.items() if key != 'name'})
    return optimizer_cls(learning_rate=learning_rate, **params)


def _keras_major_version():
    return int(str(getattr(tf.keras, '__version__', '2')).split('.')[0])


def _variable_storage(variable):
    """The ``tf.Variable`` behind a Keras variable (its ``value``) or the variable itself."""
    value = getattr(variable, 'value', None)
    return variable if value is None or callable(value) else value


class ChunkedAdam(tf.keras.optimizers.Adam):
    """Keras Adam that updates large one-dimensional variables in bounded chunks.

    ``Adam.update_step`` materializes several full-size temporaries per variable
    (``g - m``, ``g * g``, ``sqrt(v) + epsilon`` and the update), and the base
    optimizer then reassigns every constrained variable whole. For billion-edge
    weight vectors each temporary is gigabytes. This optimizer applies the same
    per-element arithmetic, in the same order and dtype, to contiguous slices of
    the same slot variables, and applies a :class:`SignedConstraint` per slice, so
    results and checkpoints equal plain Adam while live temporaries are bounded by
    ``update_chunk_size`` elements. Variables at most one chunk long, sparse
    gradients and other shapes use the plain Adam update.
    """

    DEFAULT_CHUNK = 1 << 23

    def __init__(self, *args, update_chunk_size=DEFAULT_CHUNK, **kwargs):
        if _keras_major_version() < 3:
            raise NotImplementedError('chunked_adam requires Keras 3.')
        super().__init__(*args, **kwargs)
        if isinstance(update_chunk_size, bool) or int(update_chunk_size) < 1:
            raise ValueError('update_chunk_size must be a positive integer.')
        if self.amsgrad:
            raise ValueError('chunked_adam does not support amsgrad.')
        self.update_chunk_size = int(update_chunk_size)

    def _chunks(self, gradient, variable):
        if isinstance(gradient, tf.IndexedSlices) or len(variable.shape) != 1:
            return False
        size = variable.shape[0]
        return size is not None and size > self.update_chunk_size

    def _condition_key(self, variable):
        # Keras' strategy-aware key: a mirrored variable's per-replica component passed to
        # update_step and the Keras variable given to apply share it, unlike their ids.
        key = getattr(self, '_var_key', None)
        return key(variable) if key is not None else id(_variable_storage(variable))

    @staticmethod
    def _sign_condition(variable):
        constraint = getattr(variable, 'constraint', None)
        condition = getattr(constraint, 'condition', None)
        if type(constraint).__name__ != 'SignedConstraint' or condition is None:
            return None
        condition = tf.convert_to_tensor(condition)
        return condition if condition.shape == variable.shape else None

    def update_step(self, gradient, variable, learning_rate):
        if not self._chunks(gradient, variable):
            return super().update_step(gradient, variable, learning_rate)
        dtype = variable.dtype
        lr = tf.cast(learning_rate, dtype)
        gradient = tf.cast(gradient, dtype)
        local_step = tf.cast(self.iterations + 1, dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, dtype), local_step)
        index = self._get_variable_index(variable)
        momentum = _variable_storage(self._momentums[index])
        velocity = _variable_storage(self._velocities[index])
        weights = _variable_storage(variable)
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        one_minus_beta_1 = tf.cast(1 - self.beta_1, dtype)
        one_minus_beta_2 = tf.cast(1 - self.beta_2, dtype)
        epsilon = tf.cast(self.epsilon, dtype)
        # The backend may pass the underlying tf.Variable, which has no Keras constraint.
        key = self._condition_key(variable)
        condition = getattr(self, '_chunk_conditions', {}).get(key)
        if condition is None:
            condition = self._sign_condition(variable)
        else:
            self._chunk_constrained.add(key)
        size = int(variable.shape[0])
        chunk = self.update_chunk_size

        def read(target, positions):
            return tf.raw_ops.ResourceGather(resource=target.handle, indices=positions, dtype=dtype)

        def body(start):
            stop = tf.minimum(start + chunk, size)
            positions = tf.range(start, stop, dtype=tf.int64)
            g = gradient[start:stop]
            m = read(momentum, positions)
            v = read(velocity, positions)
            w = read(weights, positions)
            m = m + (g - m) * one_minus_beta_1
            v = v + (tf.square(g) - v) * one_minus_beta_2
            w = w - (m * alpha) / (tf.sqrt(v) + epsilon)
            if condition is not None:
                w = tf.where(condition[start:stop], tf.nn.relu(w), -tf.nn.relu(-w))
            for target, value in ((momentum, m), (velocity, v), (weights, w)):
                tf.raw_ops.ResourceStridedSliceAssign(
                    ref=target.handle, begin=start[None], end=stop[None], strides=[1], value=value,
                )
            return (stop,)

        tf.while_loop(
            lambda start: start < size, body, (tf.constant(0, tf.int64),), parallel_iterations=1,
        )

    def apply(self, grads, trainable_variables=None):
        # Chunked variables already received their sign constraint per chunk; hide it from
        # the base class, which would reassign each constrained variable whole.
        variables = trainable_variables if trainable_variables is not None else self._trainable_variables
        hidden, self._chunk_conditions, self._chunk_constrained = [], {}, set()
        for gradient, variable in zip(grads, variables):
            condition = self._sign_condition(variable)
            if gradient is not None and self._chunks(gradient, variable) and condition is not None:
                self._chunk_conditions[self._condition_key(variable)] = condition
                hidden.append((variable, variable.constraint))
                variable.constraint = None
        try:
            result = super().apply(grads, trainable_variables)
            missed = set(self._chunk_conditions) - self._chunk_constrained
            if missed:
                raise RuntimeError(
                    f"chunked_adam did not apply the sign constraint of {len(missed)} chunked variable(s)."
                )
            return result
        finally:
            for variable, constraint in hidden:
                variable.constraint = constraint
            self._chunk_conditions, self._chunk_constrained = {}, set()


class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear, warmpu followed by cosine decay."""
    def __init__(
        self, 
        warmup_start_lr=0.1, 
        warmup_target_lr=0.05,
        warmup_steps=100,
        cosine_steps=900,
        min_lr=0.001,
        name='LinearWarmupCosineDecay',
        **opts
    ):
        super().__init__()
        self.warmup_start_lr = float(warmup_start_lr)
        self.warmup_target_lr = float(warmup_target_lr)
        self.warmup_steps = int(warmup_steps)
        self.cosine_steps = int(cosine_steps)
        self.min_lr = float(min_lr)
        self.name = name

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")
        if self.cosine_steps <= 0:
            raise ValueError(f"cosine_steps must be > 0, got {self.cosine_steps}.")
        if self.min_lr < 0.0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}.")

        # Pre-create scalar constants once to avoid rebuilding/casting every call.
        self._warmup_start_lr_tf = tf.constant(self.warmup_start_lr, dtype=tf.float32)
        self._warmup_target_lr_tf = tf.constant(self.warmup_target_lr, dtype=tf.float32)
        self._min_lr_tf = tf.constant(self.min_lr, dtype=tf.float32)
        self._warmup_steps_tf = tf.constant(float(self.warmup_steps), dtype=tf.float32)
        self._warmup_den_tf = tf.constant(float(max(1, self.warmup_steps - 1)), dtype=tf.float32)
        self._cosine_steps_tf = tf.constant(float(self.cosine_steps), dtype=tf.float32)
        self._pi_tf = tf.constant(math.pi, dtype=tf.float32)
    
    def __call__(self, step):
        with tf.name_scope(self.name):
            step = tf.cast(step, tf.float32)

            def warmup_branch():
                if self.warmup_steps > 1:
                    warmup_progress = tf.clip_by_value(step / self._warmup_den_tf, 0.0, 1.0)
                    return self._warmup_start_lr_tf + (
                        self._warmup_target_lr_tf - self._warmup_start_lr_tf
                    ) * warmup_progress
                return self._warmup_target_lr_tf

            def cosine_branch():
                cosine_step = tf.maximum(step - self._warmup_steps_tf, 0.0)
                cosine_progress = tf.clip_by_value(cosine_step / self._cosine_steps_tf, 0.0, 1.0)
                cosine_decay = 0.5 * (1.0 + tf.cos(self._pi_tf * cosine_progress))
                return self._min_lr_tf + (self._warmup_target_lr_tf - self._min_lr_tf) * cosine_decay

            if self.warmup_steps > 0:
                return tf.cond(step < self._warmup_steps_tf, warmup_branch, cosine_branch)
            return cosine_branch()

    def get_config(self):
        return {
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_target_lr": self.warmup_target_lr,
            "warmup_steps": self.warmup_steps,
            "cosine_steps": self.cosine_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }
    

class ExponentiatedAdam(tf.keras.optimizers.Optimizer):
    r"""Adam-like optimizer with *exponentiated* gradient updates.

    This rewrites the final update from

        w <- w - alpha * m / (sqrt(v) + epsilon)

    to

        w <- w * exp(- alpha * m / (sqrt(v) + epsilon) * sign(w)),

    preserving the rest of the Adam algorithm (moments `m`, `v`, AMSGrad, etc.).
    By default, `sign(0) = +1` so zero-valued parameters can still move off zero.

    Sparse updates are applied consistently via `scatter_mul()` on the affected
    slices only.

    Reference:
      - "Brain-like learning with exponentiated gradients" (and related papers).
      - The original Adam reference:
        [Kingma et al., 2014](http://arxiv.org/abs/1412.6980).
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="ExponentiatedAdam",
        **kwargs
    ):
        """Create a new ExponentiatedAdam optimizer."""
        if not isinstance(jit_compile, bool):
            raise ValueError("jit_compile must be true or false.")
        base_kwargs = dict(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            **kwargs
        )
        if _optimizer_init_accepts('jit_compile'):
            base_kwargs['jit_compile'] = jit_compile
        try:
            super().__init__(learning_rate=learning_rate, **base_kwargs)
        except TypeError as exc:
            if 'learning_rate' not in str(exc):
                raise
            super().__init__(**base_kwargs)
            self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.jit_compile = bool(jit_compile)

    def _add_slot_variable(self, variable, name):
        try:
            return self.add_variable_from_reference(
                model_variable=variable, variable_name=name
            )
        except TypeError:
            return self.add_variable_from_reference(
                reference_variable=variable, name=name
            )

    def _get_variable_index(self, variable):
        if hasattr(tf.keras.optimizers.Optimizer, '_get_variable_index'):
            return super()._get_variable_index(variable)
        return self._index_dict[self._var_key(variable)]

    def build(self, var_list):
        """Initialize optimizer variables.

        Similar to Adam: we have slot variables for
        - m (first moment)
        - v (second moment)
        - vhat (if amsgrad=True, for the max of second moments).
        """
        if getattr(self, "_slots_built", False):
            return
        super().build(var_list)
        self._slots_built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(self._add_slot_variable(var, "m"))
            self._velocities.append(self._add_slot_variable(var, "v"))
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(self._add_slot_variable(var, "vhat"))

    def _coalesce_indexed_slices(self, grad, variable_dtype=None, index_dtype=tf.int32):
        """Coalesce duplicate indices in an IndexedSlices gradient."""
        indices = tf.cast(grad.indices, index_dtype)
        values = grad.values
        if variable_dtype is not None and values.dtype != variable_dtype:
            values = tf.cast(values, variable_dtype)

        unique_idx, inverse_pos = tf.unique(indices, out_idx=tf.int32)
        num_unique = tf.shape(unique_idx, out_type=tf.int32)[0]
        coalesced_values = tf.math.unsorted_segment_sum(values, inverse_pos, num_unique)

        dense_shape = grad.dense_shape
        if dense_shape is None:
            dense_shape = tf.cast(tf.shape(values)[0:1], index_dtype)
        else:
            dense_shape = tf.cast(dense_shape, index_dtype)

        return tf.IndexedSlices(coalesced_values, unique_idx, dense_shape)

    def update_step(self, gradient, variable, learning_rate=None):
        """Update step given gradient and the associated model variable."""
        # if isinstance(gradient, tf.IndexedSlices):
        #     gradient = tf.convert_to_tensor(gradient)
        # Get current iteration
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        # Compute powers of beta_1 and beta_2
        beta_1_t = tf.cast(self.beta_1, variable.dtype)
        beta_2_t = tf.cast(self.beta_2, variable.dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        # Get the current learning rate (supports schedules)
        if learning_rate is None:
            learning_rate = self.learning_rate
        lr = tf.cast(learning_rate, variable.dtype)
        # Standard Adam alpha correction
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Fetch the slot variables for this parameter
        var_index = self._get_variable_index(variable)
        m = self._momentums[var_index]
        v = self._velocities[var_index]

        # ------------------------
        # Sparse vs. Dense branch
        # ------------------------
        if isinstance(gradient, tf.IndexedSlices):
            gradient = self._coalesce_indexed_slices(
                gradient,
                variable_dtype=variable.dtype,
                index_dtype=tf.int32,
            )

            # Sparse gradient
            # 1) Update m (first moment)
            m.assign_add(-m * (1 - beta_1_t))  # Decay existing m
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - beta_1_t),
                    gradient.indices,
                )
            )
            # 2) Update v (second moment)
            v.assign_add(-v * (1 - beta_2_t))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - beta_2_t),
                    gradient.indices,
                )
            )
            # 3) AMSGrad if needed
            if self.amsgrad:
                vhat = self._velocity_hats[var_index]
                vhat.assign(tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update for only the slices that changed
            # Gather the relevant slices for var, m, v
            var_slices = tf.gather(variable, gradient.indices)
            m_slices = tf.gather(m, gradient.indices)
            v_slices = tf.gather(v_used, gradient.indices)

            # adam_grad = m / (sqrt(v)+eps)
            adam_grad_slices = m_slices / (tf.sqrt(v_slices) + self.epsilon)

            # sign(w) for these slices, fallback sign(0)=+1
            sign_w_slices = tf.sign(var_slices)
            sign_w_slices = tf.where(
                tf.equal(sign_w_slices, 0), tf.ones_like(sign_w_slices), sign_w_slices
            )

            # exponent = - alpha * adam_grad_slices * sign_w_slices
            exponent_slices = -alpha * adam_grad_slices * sign_w_slices

            # multiplier = exp(exponent_slices)
            multiplier_slices = tf.exp(exponent_slices)

            # var_slices_new = var_slices * multiplier_slices
            # We can do partial update with scatter_mul:
            #   new_var[i] = old_var[i] * multiplier[i]
            variable.scatter_mul(
                tf.IndexedSlices(multiplier_slices, gradient.indices)
            )

        else:
            if self.jit_compile and not self.amsgrad:
                new_weights, new_m, new_v = _dense_exponentiated_adam_update(
                    tf.convert_to_tensor(variable),
                    tf.convert_to_tensor(m),
                    tf.convert_to_tensor(v),
                    gradient,
                    alpha,
                    beta_1_t,
                    beta_2_t,
                    tf.cast(self.epsilon, variable.dtype),
                )
                m.assign(new_m)
                v.assign(new_v)
                variable.assign(new_weights)
                return

            # Dense gradient
            # 1) Update m
            m.assign_add((gradient - m) * (1 - beta_1_t))
            # 2) Update v
            v.assign_add((tf.square(gradient) - v) * (1 - beta_2_t))
            # 3) AMSGrad
            if self.amsgrad:
                vhat = self._velocity_hats[var_index]
                vhat.assign(tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update
            adam_grad = m / (tf.sqrt(v_used) + self.epsilon)

            # sign(w), fallback sign(0)=+1
            sign_w = tf.sign(variable)
            sign_w = tf.where(tf.equal(sign_w, 0), tf.ones_like(sign_w), sign_w)

            exponent = -alpha * adam_grad * sign_w
            multiplier = tf.exp(exponent)
            variable.assign(variable * multiplier)

    def get_config(self):
        config = super().get_config()
        if hasattr(self, '_serialize_hyperparameter') and hasattr(self, '_learning_rate'):
            learning_rate = self._serialize_hyperparameter(self._learning_rate)
        else:
            learning_rate = config.get('learning_rate', self.learning_rate)
        config.update(
            {
                "learning_rate": learning_rate,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "jit_compile": self.jit_compile,
            }
        )
        return config
