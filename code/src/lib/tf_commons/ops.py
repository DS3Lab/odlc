import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest


def _conv(args, output_channels, kernel_size, stride, data_format, bias=True, bias_start=0.1, scope=None,
          place_vars_on_cpu=True):
    if args is None:
        raise ValueError("`args` must be specified")
    if nest.is_sequence(args):
        raise ValueError("`args` must be a tensor, not list or sequence")

    # check that args is 4-D tensor with order NHWC
    args_shape = args.get_shape().as_list()
    if len(args_shape) != 4:
        raise ValueError("Conv is expecting a 4-D Tensor, got shape {}".format(args_shape))
    if data_format == 'NCHW':
        input_channels = args_shape[1]
    elif data_format == 'NHWC':
        input_channels = args_shape[3]
    else:
        raise ValueError("invalid data format. Expected one of [`NHWC`, `NCHW`], got {}".format(data_format))

    # get strides
    if data_format == 'NCHW':
        strides = [1, 1, stride, stride]
    elif data_format == 'NHWC':
        strides = [1, stride, stride, 1]
    else:
        raise ValueError("invalid data format. Expected one of [`NHWC`, `NCHW`], got {}".format(data_format))

    # compute convolution
    variable_getter = _variable_on_cpu if place_vars_on_cpu else _variable
    with tf.variable_scope(scope or "Conv"):
        kernel = variable_getter("Kernel", [kernel_size, kernel_size, input_channels, output_channels],
                                 tf.contrib.layers.xavier_initializer())
        res = tf.nn.conv2d(args, kernel, strides, padding="SAME", data_format=data_format)
        if not bias:
            return res
        bias_term = variable_getter("Bias", [output_channels], initializer=init_ops.constant_initializer(bias_start))
    return tf.nn.bias_add(res, bias_term, data_format=data_format)


def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
