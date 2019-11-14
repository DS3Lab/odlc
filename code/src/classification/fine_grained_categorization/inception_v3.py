"""
* part of this code is adapted from
    https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
"""

import tensorflow as tf

from src.classification.fine_grained_categorization import FGVCClassifier, FGVCClassifierNames
from src.data.dataloading.preprocessing import InceptionPreprocessing
from src.classification.fine_grained_categorization.utils.inception_utils import (inception_v3_base, trunc_normal,
                                                                                  reduced_kernel_size_for_small_input)

slim = tf.contrib.slim

_DROPOUT_KEEP_PROB = 0.8
_MIN_DEPTH = 16
_DEPTH_MULTIPLIER = 1.0


class InceptionV3(FGVCClassifier):
    NAME = FGVCClassifierNames.inception_v3
    SCOPE = 'InceptionV3'
    INPUT_SHAPE = (299, 299, 3)

    def __init__(self, dataset):
        super(InceptionV3, self).__init__(dataset)

    @classmethod
    def model_variables(cls):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.SCOPE)

    @staticmethod
    def arg_scope(weight_decay=0.00004, use_batch_norm=True, batch_norm_decay=0.9997, batch_norm_epsilon=0.001,
                  activation_fn=tf.nn.relu, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                  batch_norm_scale=False, *args, **kwargs):
        """ Defines the default arg scope for inception models. """
        batch_norm_params = {'decay': batch_norm_decay,  # Decay for the moving averages.
                             'epsilon': batch_norm_epsilon,  # epsilon to prevent 0s in variance.
                             'updates_collections': batch_norm_updates_collections,  # collection containing update_ops.
                             'fused': None,  # use fused batch norm if possible.
                             'scale': batch_norm_scale}
        if use_batch_norm:
            normalizer_fn = slim.batch_norm
            normalizer_params = batch_norm_params
        else:
            normalizer_fn = None
            normalizer_params = {}

        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params) as sc:
                return sc

    @classmethod
    def preprocess_image(cls, image, is_training, normalize_image=True):
        return InceptionPreprocessing.preprocess_image(image=image,
                                                       height=cls.INPUT_SHAPE[0],
                                                       width=cls.INPUT_SHAPE[1],
                                                       standardize_image_for_eval=normalize_image,
                                                       is_training=is_training)

    @staticmethod
    def standardize_tensor(input_tensor):
        return InceptionPreprocessing.standardize_image(input_tensor)

    def inference(self,
                  input_tensor,
                  is_training,
                  reuse=None,
                  arg_scope=None,
                  dropout_keep_prob=_DROPOUT_KEEP_PROB,
                  min_depth=_MIN_DEPTH,
                  depth_multiplier=_DEPTH_MULTIPLIER,
                  prediction_fn=slim.softmax,
                  spatial_squeeze=True,
                  create_aux_logits=True,
                  global_pool=False,
                  return_predictions=False, **kwargs):
        """Inception model from http://arxiv.org/abs/1512.00567. """

        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')

        def depth(d):
            return max(int(d * depth_multiplier), min_depth)

        if arg_scope is None:
            arg_scope = self.arg_scope()

        with slim.arg_scope(arg_scope):
            with tf.variable_scope(self.SCOPE, 'InceptionV3', [input_tensor], reuse=reuse) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                    net, end_points = inception_v3_base(
                        input_tensor, scope=scope, min_depth=min_depth,
                        depth_multiplier=depth_multiplier)

                    # Auxiliary Head logits
                    if create_aux_logits:
                        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                            stride=1, padding='SAME'):
                            aux_logits = end_points['Mixed_6e']
                            with tf.variable_scope('AuxLogits'):
                                aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                             scope='AvgPool_1a_5x5')
                                aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1], scope='Conv2d_1b_1x1')

                                # Shape of feature map before the final layer.
                                kernel_size = reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                                aux_logits = slim.conv2d(
                                    aux_logits, depth(768), kernel_size,
                                    weights_initializer=trunc_normal(0.01),
                                    padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                                aux_logits = slim.conv2d(
                                    aux_logits, self.num_classes, [1, 1], activation_fn=None,
                                    normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                                    scope='Conv2d_2b_1x1')
                                if spatial_squeeze:
                                    aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                                end_points['AuxLogits'] = aux_logits

                    # Final pooling and prediction
                    with tf.variable_scope('Logits'):
                        if global_pool:
                            # Global average pooling.
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
                            end_points['global_pool'] = net
                        else:
                            # Pooling with a fixed kernel size.
                            kernel_size = reduced_kernel_size_for_small_input(net, [8, 8])
                            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                                  scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                            end_points['AvgPool_1a'] = net

                        # 1 x 1 x 2048
                        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                        end_points['PreLogits'] = net
                        # 2048
                        logits = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, scope='Conv2d_1c_1x1')
                        if spatial_squeeze:
                            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                        # 1000

                    end_points['Logits'] = logits
                    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

                if return_predictions:
                    return end_points['Predictions']

                return logits, end_points
