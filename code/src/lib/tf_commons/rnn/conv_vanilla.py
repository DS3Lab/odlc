from src.lib.tf_commons.ops import _conv


class ConvVanilla(object):

    def __init__(self,
                 num_units,
                 stride,
                 kernel_size,
                 data_format,
                 activation_fn=None):
        self._num_units = num_units
        self._stride = stride
        self._kernel_size = kernel_size
        self._data_format = data_format
        self._activation_fn = activation_fn

    def __call__(self, inputs, scope=None):
        """Convolutional Layer"""
        conv = _conv(inputs, self._num_units, self._kernel_size, stride=self._stride, data_format=self._data_format,
                     scope=scope)
        if self._activation_fn:
            conv_activation = self._activation_fn(conv)
            return conv_activation
        else:
            return conv
