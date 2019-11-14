import collections

import tensorflow as tf

slim = tf.contrib.slim


class NoOpScope(object):
    """No-op context manager."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
    """Generator for v1 ResNet models.
    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.
    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks.
        If 0 or None, we return the features before the logit layer.
      is_training: whether batch_norm layers are in training mode. If this is set
        to None, the callers can specify slim.batch_norm's is_training parameter
        from an outer slim.arg_scope.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          To use this parameter, the input images must be smaller than 300x300
          pixels, in which case the output logit layer does not contain spatial
          information and can be removed.
      store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is 0 or None,
        then net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes a non-zero integer, net contains the
        pre-softmax activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope(
                    [slim.batch_norm], is_training=is_training) if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_blocks_dense(net, blocks, output_stride,
                                         store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v1_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
    Returns:
      A resnet_v1 bottleneck block.
    """
    return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
    """Bottleneck residual unit variant with BN after convolutions.
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
      use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       store_non_strided_activations=False,
                       outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.
    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.
    Control of the output feature density is implemented by atrous convolution.
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the ResNet.
        For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
      outputs_collections: Collection to add the ResNet block outputs.
    Returns:
      net: Output tensor with stride equal to the specified output_stride.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
       net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)
    whereas
       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                           padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.
    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """
