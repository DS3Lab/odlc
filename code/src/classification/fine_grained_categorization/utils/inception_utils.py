import tensorflow as tf

slim = tf.contrib.slim


def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.
    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.
    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
    Returns:
      a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def trunc_normal(stddev):
    return tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.
    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.
    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.
    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256a  | Mixed_5b
    mixed_35x35x288a  | Mixed_5c
    mixed_35x35x288b  | Mixed_5d
    mixed_17x17x768a  | Mixed_6a
    mixed_17x17x768b  | Mixed_6b
    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
        'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      scope: Optional variable_scope.
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.
    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    def depth(d):
        return max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points
            # 35 x 35 x 192.

        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

        raise ValueError('Unknown final endpoint %s' % final_endpoint)
