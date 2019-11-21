import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

from src.lib.tf_commons.ops import _conv


class ConvGRU(rnn_cell_impl.RNNCell):
    """ Convolutional GRU (ConvGRU) """

    def __init__(self,
                 num_units,
                 out_height,
                 out_width,
                 stride,
                 kernel_size,
                 hidden_kernel_size,
                 data_format):
        super(ConvGRU, self).__init__(name='ConvGRU')

        self._num_units = num_units
        self._out_height = out_height
        self._out_width = out_width
        self._stride = stride
        self._kernel_size = kernel_size
        self._hidden_kernel_size = hidden_kernel_size
        self._data_format = data_format

        # constants
        self._channel_axis = 3 if data_format == 'NHWC' else 1

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size=1, dtype=None):
        # if self._data_format == 'NHWC':
        #     return tf.zeros([batch_size, self._out_height, self._out_width, self._num_units])
        # elif self._data_format == 'NCHW':
        #     return tf.zeros([batch_size, self._num_units, self._out_height, self._out_width])
        # else:
        #     raise ValueError("invalid data format")
        raise NotImplementedError

    def init_state(self, batch_size_tensor):
        if self._data_format == 'NHWC':
            return tf.fill(dims=[batch_size_tensor, self._out_height, self._out_width, self._num_units], value=0.0)
        elif self._data_format == 'NCHW':
            return tf.fill(dims=[batch_size_tensor, self._num_units, self._out_height, self._out_width], value=0.0)
        else:
            raise ValueError("invalid data format. Expected one of [`NHWC`, `NCHW`], got {}".format(self._data_format))

    def __call__(self, inputs, state, scope=None):
        """Convolutional GRU cell (ConvGRU)."""

        with vs.variable_scope('gates'):
            with vs.variable_scope('inputs') as scope_:
                inputs_conv = _conv(inputs, 2 * self._num_units, self._kernel_size, self._stride, bias=False,
                                    scope=scope_, data_format=self._data_format)
                inputs_conv_r, inputs_conv_z = array_ops.split(inputs_conv, 2, axis=self._channel_axis)
            with vs.variable_scope('hidden') as scope_:
                hidden_conv = _conv(state, 2 * self._num_units, self._hidden_kernel_size, 1, bias=False,
                                    scope=scope_, data_format=self._data_format)
                hidden_conv_r, hidden_conv_z = array_ops.split(hidden_conv, 2, axis=self._channel_axis)
            r_state, z = sigmoid(inputs_conv_r + hidden_conv_r) * state, sigmoid(inputs_conv_z + hidden_conv_z)

        with vs.variable_scope('candidate'):
            with vs.variable_scope('inputs') as scope_:
                candidate_inputs = _conv(inputs, self._num_units, self._kernel_size, self._stride, bias=False,
                                         scope=scope_, data_format=self._data_format)
            with vs.variable_scope('hidden') as scope_:
                candidate_hidden = _conv(r_state, self._num_units, self._hidden_kernel_size, 1, bias=False,
                                         scope=scope_, data_format=self._data_format)
            candidate = tanh(candidate_inputs + candidate_hidden)
            new_state = z * candidate + (1 - z) * state

        return new_state, new_state
