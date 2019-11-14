import tensorflow as tf


def conv2d(args, output_channels, kernel_size, strides, padding, activation, data_format, name, kernel_regularizer,
           bias_regularizer, trainable, filter_weights=None, biases=None):
    if not trainable and filter_weights is not None:
        kernel_initializer = tf.initializers.constant(filter_weights, verify_shape=True)
        bias_initializer = tf.initializers.constant(biases, verify_shape=True)
        kernel_regularizer = None
        bias_regularizer = None
    else:
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.initializers.constant(0.1)

    return tf.layers.conv2d(inputs=args,
                            filters=output_channels,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            data_format=_convert_data_format_name(data_format),
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            trainable=trainable,
                            name=name)


def maxpool(args, pool_size, strides, padding, data_format, name):
    return tf.layers.max_pooling2d(inputs=args,
                                   pool_size=pool_size,
                                   strides=strides,
                                   padding=padding,
                                   data_format=_convert_data_format_name(data_format),
                                   name=name)


def dense(args, units, activation, use_bias, name, trainable, kernel_regularizer, bias_regularizer, weights=None,
          biases=None):
    if not trainable and weights is not None:
        kernel_initializer = tf.initializers.constant(weights, verify_shape=True)
        bias_initializer = tf.initializers.constant(biases, verify_shape=True)
        kernel_regularizer = None
        bias_regularizer = None
    else:
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.initializers.constant(0.1)

    return tf.layers.dense(inputs=args,
                           units=units,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           trainable=trainable,
                           name=name)


def _convert_data_format_name(data_format):
    if data_format == 'NCHW':
        return 'channels_first'
    if data_format == 'NHWC':
        return 'channels_last'
    raise ValueError("invalid data format. Expected one of [`NHWC`, `NCHW`], got {}".format(data_format))
