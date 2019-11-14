import tensorflow as tf

from src.lib.tf_commons.rnn.conv_vanilla import ConvVanilla


class Binarizer:

    def __init__(self, scope, data_format):
        self._scope = scope

        # B-Conv
        self._conv1 = ConvVanilla(32, 1, 1, data_format, tf.nn.tanh)

        self.initial_state = None

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs, is_training, reuse=None):
        with tf.variable_scope(self._scope) as scope:
            conv1 = self._conv1(inputs, scope=scope)
            t = tf.identity(conv1, name="identity")
            output = t + tf.stop_gradient(self._binarizer(conv1, is_training) - t)
            return output

    @staticmethod
    def _binarizer(inputs, is_training):
        def train_bin(inputs_):
            prob = tf.truediv(tf.add(1.0, inputs_), 2.0)
            bernoulli = tf.contrib.distributions.Bernoulli(probs=prob, dtype=tf.float32)
            return 2 * bernoulli.sample() - 1

        def valid_bin(inputs_):
            return tf.sign(inputs_)

        return tf.cond(is_training, lambda: train_bin(inputs), lambda: valid_bin(inputs))

    def trainable_variables(self, parent_scope):
        assert isinstance(parent_scope, str) or parent_scope is None
        return [v for v in tf.trainable_variables(scope=str(parent_scope) + '/' + self.scope)]
