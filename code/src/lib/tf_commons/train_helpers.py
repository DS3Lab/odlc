import tensorflow as tf

# ====================================================
#  default values used in FGVC classification training
# ====================================================
_ADADELTA_RHO = 0.95
_ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0.1
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_OPT_EPSILON = 1.0
_FTRL_LEARNING_RATE_POWER = -0.5
_FTRL_INITIAL_ACCUMULATOR_VALUE = 0.1
_FTRL_L1 = 0.0
_FTRL_L2 = 0.0
_MOMENTUM = 0.9
_RMSPROP_MOMENTUM = 0.9
_RMSPROP_DECAY = 0.9


def configure_optimizer(learning_rate,
                        optimizer,
                        opt_epsilon,
                        adadelta_rho=_ADADELTA_RHO,
                        adagrad_initial_accumulator_value=_ADAGRAD_INITIAL_ACCUMULATOR_VALUE,
                        adam_beta1=_ADAM_BETA1,
                        adam_beta2=_ADAM_BETA2,
                        ftrl_learning_rate_power=_FTRL_LEARNING_RATE_POWER,
                        ftrl_initial_accumulator_value=_FTRL_INITIAL_ACCUMULATOR_VALUE,
                        ftrl_l1=_FTRL_L1,
                        ftrl_l2=_FTRL_L2,
                        momentum=_MOMENTUM,
                        rmsprop_decay=_RMSPROP_DECAY,
                        rmsprop_momentum=_RMSPROP_MOMENTUM):
    """ Configures the optimizer used for training. """

    if optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate,
            rho=adadelta_rho,
            epsilon=opt_epsilon)
    elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=adagrad_initial_accumulator_value)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=opt_epsilon)
    elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=ftrl_learning_rate_power,
            initial_accumulator_value=ftrl_initial_accumulator_value,
            l1_regularization_strength=ftrl_l1,
            l2_regularization_strength=ftrl_l2)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=momentum,
            name='Momentum')
    elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=rmsprop_decay,
            momentum=rmsprop_momentum,
            epsilon=opt_epsilon)
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [{}] was not recognized'.format(optimizer))
    return optimizer


def configure_learning_rate(global_step, batch_size, initial_learning_rate, num_replicas, num_samples_per_epoch,
                            num_epochs_per_decay, learning_rate_decay_type, learning_rate_decay_factor,
                            end_learning_rate):
    """ Configures the learning rate. """

    decay_steps = int(num_samples_per_epoch * num_epochs_per_decay / (batch_size * num_replicas))

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(initial_learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

    elif learning_rate_decay_type == 'fixed':
        return tf.constant(initial_learning_rate, name='fixed_learning_rate')

    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(initial_learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [{}] was not recognized'.format(learning_rate_decay_type))
