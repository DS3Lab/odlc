import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             output_stride=16,
                             align_feature_maps=False,
                             scope=None,
                             activation_fn=tf.nn.relu):
    """Inception model from  http://arxiv.org/abs/1602.07261.
    Constructs an Inception Resnet v2 network from inputs to the given final
    endpoint. This method can construct the network up to the final inception
    block Conv2d_7b_1x1.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
      output_stride: A scalar that specifies the requested ratio of input to
        output spatial resolution. Only supports 8 and 16.
      align_feature_maps: When true, changes all the VALID paddings in the network
        to SAME padding so that the feature maps are aligned.
      scope: Optional variable_scope.
      activation_fn: Activation function for block scopes.
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.
    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
        or if the output_stride is not 8 or 16, or if the output_stride is 8 and
        we request an end point after 'PreAuxLogits'.
    """
    if output_stride != 8 and output_stride != 16:
        raise ValueError('output_stride must be 8 or 16.')

    padding = 'SAME' if align_feature_maps else 'VALID'

    end_points = {}

    def add_and_check_final(name, _net):
        end_points[name] = _net
        return name == final_endpoint

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 149 x 149 x 32
            net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                              scope='Conv2d_1a_3x3')
            if add_and_check_final('Conv2d_1a_3x3', net):
                return net, end_points

            # 147 x 147 x 32
            net = slim.conv2d(net, 32, 3, padding=padding,
                              scope='Conv2d_2a_3x3')
            if add_and_check_final('Conv2d_2a_3x3', net):
                return net, end_points
            # 147 x 147 x 64
            net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
            if add_and_check_final('Conv2d_2b_3x3', net):
                return net, end_points
            # 73 x 73 x 64
            net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                                  scope='MaxPool_3a_3x3')
            if add_and_check_final('MaxPool_3a_3x3', net):
                return net, end_points
            # 73 x 73 x 80
            net = slim.conv2d(net, 80, 1, padding=padding,
                              scope='Conv2d_3b_1x1')
            if add_and_check_final('Conv2d_3b_1x1', net):
                return net, end_points
            # 71 x 71 x 192
            net = slim.conv2d(net, 192, 3, padding=padding,
                              scope='Conv2d_4a_3x3')
            if add_and_check_final('Conv2d_4a_3x3', net):
                return net, end_points
            # 35 x 35 x 192
            net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                                  scope='MaxPool_5a_3x3')
            if add_and_check_final('MaxPool_5a_3x3', net):
                return net, end_points

            # 35 x 35 x 320
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')

                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                 scope='AvgPool_0a_3x3')
                    tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                               scope='Conv2d_0b_1x1')
                net = tf.concat(
                    [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

            if add_and_check_final('Mixed_5b', net):
                return net, end_points

            net = slim.repeat(net, 10, block35, scale=0.17,
                              activation_fn=activation_fn)

            # 17 x 17 x 1088 if output_stride == 8,
            # 33 x 33 x 1088 if output_stride == 16
            use_atrous = output_stride == 8

            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                             padding=padding,
                                             scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                stride=1 if use_atrous else 2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                                 padding=padding,
                                                 scope='MaxPool_1a_3x3')
                net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

            if add_and_check_final('Mixed_6a', net):
                return net, end_points

            with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
                net = slim.repeat(net, 20, block17, scale=0.10,
                                  activation_fn=activation_fn)
            if add_and_check_final('PreAuxLogits', net):
                return net, end_points

            if output_stride == 8:
                raise ValueError('output_stride==8 is only supported up to the '
                                 'PreAuxlogits end_point for now.')

            # 8 x 8 x 2080
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                    tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                               padding=padding,
                                               scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                padding=padding,
                                                scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_3'):
                    tower_pool = slim.max_pool2d(net, 3, stride=2,
                                                 padding=padding,
                                                 scope='MaxPool_1a_3x3')
                net = tf.concat(
                    [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

            if add_and_check_final('Mixed_7a', net):
                return net, end_points

            net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
            net = block8(net, activation_fn=None)

            # 8 x 8 x 1536
            net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
            if add_and_check_final('Conv2d_7b_1x1', net):
                return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
