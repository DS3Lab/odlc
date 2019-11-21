import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

from src.lib.tf_commons.ops import _conv


class ConvLSTM(rnn_cell_impl.RNNCell):
    """
    Convolutional LSTM network cell (ConvLSTM).

    The implementation is based on http://arxiv.org/abs/1506.04214.
    and BasicLSTMCell in TensorFlow.
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py

    """

    def __init__(self, num_units, out_height, out_width, stride, kernel_size, hidden_kernel_size, data_format,
                 forget_bias=1.0, activation=tanh):
        super(ConvLSTM, self).__init__(name='ConvLSTM')

        self._num_units = num_units
        self._out_height = out_height
        self._out_width = out_width
        self._stride = stride
        self._kernel_size = kernel_size
        self._hidden_kernel_size = hidden_kernel_size
        self._data_format = data_format
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        # if self._data_format == 'NHWC':
        #     state_shape = [batch_size, self._out_height, self._out_width, self._num_units]
        # elif self._data_format == 'NCHW':
        #     state_shape = [batch_size, self._num_units, self._out_height, self._out_width]
        # else:
        #     raise ValueError("invalid data format")
        #
        # return LSTMStateTuple(tf.zeros(state_shape), tf.zeros(state_shape))
        raise NotImplementedError

    def init_state(self, batch_size_tensor):
        if self._data_format == 'NHWC':
            state_shape = [batch_size_tensor, self._out_height, self._out_width, self._num_units]
        elif self._data_format == 'NCHW':
            state_shape = [batch_size_tensor, self._num_units, self._out_height, self._out_width]
        else:
            raise ValueError("invalid data format. Expected one of [`NHWC`, `NCHW`], got {}".format(self._data_format))

        c = tf.fill(dims=state_shape, value=0.0, name='c')
        h = tf.fill(dims=state_shape, value=0.0, name='h')

        return LSTMStateTuple(c, h)

    def __call__(self, inputs, state, scope=None):
        """Convolutional Long short-term memory cell (ConvLSTM)."""
        cell, hidden = state

        # stride _stride on inputs, stride 1 on hidden states from t-1
        with vs.variable_scope("inputs", reuse=scope.reuse) as scope_:
            new_inputs = _conv(inputs, 4 * self._num_units, self._kernel_size, self._stride, self._data_format,
                               scope=scope_)
        with vs.variable_scope("hidden", reuse=scope.reuse) as scope_:
            new_hidden = _conv(hidden, 4 * self._num_units, self._hidden_kernel_size, 1, self._data_format,
                               scope=scope_)

        new_sum = new_inputs + new_hidden

        # split gates
        input_gate, new_input, forget_gate, output_gate = array_ops.split(new_sum, 4,
                                                                          3 if self._data_format == 'NHWC' else 1)

        # computations
        forget_bias_tensor = tf.constant(self._forget_bias, dtype=tf.float32)
        new_cell = tf.add(tf.multiply(cell, sigmoid(tf.add(forget_bias_tensor, self._forget_bias))),
                          tf.multiply(sigmoid(input_gate), self._activation(new_input)))
        output = tf.multiply(self._activation(new_cell), sigmoid(output_gate))
        new_state = LSTMStateTuple(new_cell, output)

        return output, new_state
