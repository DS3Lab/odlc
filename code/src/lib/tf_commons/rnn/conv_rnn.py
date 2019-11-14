import tensorflow as tf

from src.lib.tf_commons.rnn import ConvGRU
from src.lib.tf_commons.rnn import ConvLSTM
from src.lib.tf_commons.rnn import ConvVanilla

_GRU = 'gru'
_LSTM = 'lstm'


class ConvolutionalRNN:

    def __init__(self, rnn_type, scope):
        self._scope = scope

        if rnn_type == _GRU:
            self.rnn_cell = ConvGRU
        elif rnn_type == _LSTM:
            self.rnn_cell = ConvLSTM
        else:
            self.rnn_cell = None

    @staticmethod
    def conv_layer(num_units, stride, kernel_size, data_format, activation_fn):
        return ConvVanilla(num_units=num_units,
                           stride=stride,
                           kernel_size=kernel_size,
                           data_format=data_format,
                           activation_fn=activation_fn)

    @property
    def scope(self):
        return self._scope

    def init_states(self, batch_size_tensor):
        """ to be implemented in child class """
        raise NotImplementedError

    def __call__(self, inputs, state):
        """ to be implemented in child class """
        raise NotImplementedError

    def trainable_variables(self, parent_scope):
        assert isinstance(parent_scope, str) or parent_scope is None
        return [v for v in tf.trainable_variables(scope=str(parent_scope) + '/' + self.scope)]
