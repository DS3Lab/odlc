"""
* part of this code is adapted from
    https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
"""

import tensorflow as tf

from src.classification.fine_grained_categorization import FGVCClassifier, FGVCClassifierNames
from src.data.dataloading.preprocessing import InceptionPreprocessing
from src.classification.fine_grained_categorization.utils.mobilenet_utils import (mobilenet_v1_base,
                                                                                  reduced_kernel_size_for_small_input)

slim = tf.contrib.slim


class MobilenetV1(FGVCClassifier):
    NAME = FGVCClassifierNames.mobilenet
    SCOPE = 'MobilenetV1'
    CLASS_OFFSET = 0
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self, dataset):
        super(MobilenetV1, self).__init__(dataset)

    @classmethod
    def model_variables(cls):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.SCOPE)

    @staticmethod
    def arg_scope(is_training=True, weight_decay=0.00004, stddev=0.09, regularize_depthwise=False,
                  batch_norm_decay=0.9997, batch_norm_epsilon=0.001,
                  batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS, normalizer_fn=slim.batch_norm, *args,
                  **kwargs):
        """ Defines the default MobilenetV1 arg scope """
        batch_norm_params = {
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'updates_collections': batch_norm_updates_collections,
        }
        if is_training is not None:
            batch_norm_params['is_training'] = is_training

        # Set weight_decay for weights in Conv and DepthSepConv layers.
        weights_init = tf.truncated_normal_initializer(stddev=stddev)
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        if regularize_depthwise:
            depthwise_regularizer = regularizer
        else:
            depthwise_regularizer = None
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_initializer=weights_init,
                            activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                    with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
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
                  dropout_keep_prob=0.999,
                  min_depth=8,
                  depth_multiplier=1.0,
                  conv_defs=None,
                  prediction_fn=tf.contrib.layers.softmax,
                  spatial_squeeze=True,
                  global_pool=False,
                  return_predictions=False, **kwargs):
        """ Mobilenet v1 model for classification """

        input_shape = input_tensor.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                             len(input_shape))

        if arg_scope is None:
            arg_scope = self.arg_scope(is_training=is_training)

        with slim.arg_scope(arg_scope):
            with tf.variable_scope(self.SCOPE, 'MobilenetV1', [input_tensor], reuse=reuse) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=is_training):
                    net, end_points = mobilenet_v1_base(input_tensor, scope=scope,
                                                        min_depth=min_depth,
                                                        depth_multiplier=depth_multiplier,
                                                        conv_defs=conv_defs)
                    with tf.variable_scope('Logits'):
                        if global_pool:
                            # Global average pooling.
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                            end_points['global_pool'] = net

                        else:
                            # Pooling with a fixed kernel size.
                            kernel_size = reduced_kernel_size_for_small_input(net, [7, 7])
                            net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                            end_points['AvgPool_1a'] = net

                        if not self.num_classes:
                            return net, end_points

                        # 1 x 1 x 1024
                        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                        logits = slim.conv2d(net, self.num_classes + self.CLASS_OFFSET, [1, 1], activation_fn=None,
                                             normalizer_fn=None, scope='Conv2d_1c_1x1')

                        if spatial_squeeze:
                            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                        logits = logits[:, self.CLASS_OFFSET:]

                    end_points['Logits'] = logits

                    if prediction_fn:
                        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

                if return_predictions:
                    return end_points['Predictions']
                else:
                    return logits, end_points
