import inspect

import tensorflow as tf
import math


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


def optimizer_supports_loss_scaling(optimizer):
    return (
        hasattr(optimizer, 'scale_loss') or
        (hasattr(optimizer, 'get_scaled_loss') and hasattr(optimizer, 'get_unscaled_gradients'))
    )


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


def create_optimizer(optimizer, learning_rate, optimizer_params=None):
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer

    _optimizer = None
    optimizer_params = optimizer_params or {}
    if optimizer == 'adam':
        _optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=optimizer_params.get('epsilon', 1.0e-11),
        )
    elif optimizer == 'exp_adam':
        _optimizer = ExponentiatedAdam(
            learning_rate=learning_rate,
            epsilon=optimizer_params.get('epsilon', 1.0e-11),
        )
    elif optimizer == 'sgd':
        _optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=optimizer_params.get('momentum', 0.0),
            nesterov=optimizer_params.get('nesterov', False),
        )
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}')
    
    return _optimizer


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
            }
        )
        return config
