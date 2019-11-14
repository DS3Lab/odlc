"""
* part of this code is adapted from
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py

Paper:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import tensorflow as tf

from src.classification.fine_grained_categorization import FGVCClassifier, FGVCClassifierNames
from src.data.dataloading.preprocessing import VGGPreprocessing

slim = tf.contrib.slim


class Vgg16(FGVCClassifier):
    NAME = FGVCClassifierNames.vgg16
    SCOPE = 'vgg_16'
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self, dataset):
        super(Vgg16, self).__init__(dataset)

    @classmethod
    def model_variables(cls):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.SCOPE)

    @staticmethod
    def arg_scope(weight_decay=0.0005, *args, **kwargs):
        """ Defines the VGG arg scope """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
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
                  dropout_keep_prob=0.5,
                  spatial_squeeze=True,
                  fc_conv_padding='VALID',
                  global_pool=False,
                  return_predictions=False, **kwargs):
        """ Oxford Net VGG 16-Layers version D Example """

        if arg_scope is None:
            arg_scope = self.arg_scope()

        with slim.arg_scope(arg_scope):
            with tf.variable_scope(self.SCOPE, 'vgg_16', [input_tensor], reuse=reuse) as sc:
                end_points_collection = sc.original_name_scope + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(input_tensor, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')

                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                        end_points['global_pool'] = net
                    if self.num_classes:
                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                        net = slim.conv2d(net, self.num_classes, [1, 1],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope='fc8')
                        if spatial_squeeze:
                            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

                        end_points[sc.name + '/fc8'] = net
                        end_points['Predictions'] = tf.nn.softmax(net, name='Predictions', axis=-1)

                if return_predictions:
                    return end_points['Predictions']
                else:
                    return net, end_points
