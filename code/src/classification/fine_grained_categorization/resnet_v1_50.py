"""
* part of this code is adapted from
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py

Paper:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import tensorflow as tf

from src.classification.fine_grained_categorization import FGVCClassifier, FGVCClassifierNames
from src.classification.fine_grained_categorization.utils.resnet_utils import resnet_v1, resnet_v1_block
from src.data.dataloading.preprocessing import VGGPreprocessing

slim = tf.contrib.slim

_DROPOUT_KEEP_PROB = 0.8
_MIN_DEPTH = 16
_DEPTH_MULTIPLIER = 1.0


class ResNet50(FGVCClassifier):
    NAME = FGVCClassifierNames.resnet50
    SCOPE = 'resnet_v1_50'
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self, dataset):
        super(ResNet50, self).__init__(dataset)

    @classmethod
    def model_variables(cls):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.SCOPE)

    @staticmethod
    def arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True,
                  activation_fn=tf.nn.relu, use_batch_norm=True,
                  batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS, *args, **kwargs):
        """ Defines the default ResNet arg scope. """
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': batch_norm_updates_collections,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm if use_batch_norm else None,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                # The following implies padding='SAME' for pool1, which makes feature
                # alignment easier for dense prediction tasks. This is also used in
                # https://github.com/facebook/fb.resnet.torch. However the accompanying
                # code of 'Deep Residual Learning for Image Recognition' uses
                # padding='VALID' for pool1. You can switch to that choice by setting
                # slim.arg_scope([slim.max_pool2d], padding='VALID').
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    @classmethod
    def preprocess_image(cls, image, is_training, normalize_image=True):
        return VGGPreprocessing.preprocess_image(image=image,
                                                 output_height=cls.INPUT_SHAPE[0],
                                                 output_width=cls.INPUT_SHAPE[1],
                                                 mean_center_image=normalize_image,
                                                 is_training=is_training)

    @staticmethod
    def standardize_tensor(input_tensor):
        return VGGPreprocessing.mean_image_subtraction(input_tensor)

    def inference(self,
                  input_tensor,
                  is_training,
                  reuse=None,
                  arg_scope=None,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  store_non_strided_activations=False,
                  min_base_depth=8,
                  depth_multiplier=1,
                  return_predictions=False, **kwargs):
        """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""

        def depth_func(d):
            return max(int(d * depth_multiplier), min_base_depth)

        if arg_scope is None:
            arg_scope = self.arg_scope()

        blocks = [
            resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
                            stride=2),
            resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
                            stride=2),
            resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
                            stride=2),
            resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
                            stride=1),
        ]

        with slim.arg_scope(arg_scope):
            net, endpoints = resnet_v1(input_tensor, blocks, self.num_classes, is_training,
                                       global_pool=global_pool, output_stride=output_stride,
                                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                                       store_non_strided_activations=store_non_strided_activations,
                                       reuse=reuse, scope=self.SCOPE)

            if return_predictions:
                return endpoints['predictions']
            else:
                return net, endpoints
