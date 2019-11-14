"""
* part of this code is adapted from
    https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
"""

import tensorflow as tf

from src.classification.fine_grained_categorization import FGVCClassifier, FGVCClassifierNames
from src.classification.fine_grained_categorization.utils.inception_resnet_utils import inception_resnet_v2_base
from src.data.dataloading.preprocessing import InceptionPreprocessing

slim = tf.contrib.slim


class InceptionResnetV2(FGVCClassifier):
    NAME = FGVCClassifierNames.inception_resnet_v2
    SCOPE = 'InceptionResnetV2'
    CLASS_OFFSET = 0
    INPUT_SHAPE = (299, 299, 3)

    def __init__(self, dataset):
        super(InceptionResnetV2, self).__init__(dataset)

    @classmethod
    def model_variables(cls):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=cls.SCOPE)

    @staticmethod
    def arg_scope(weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, activation_fn=tf.nn.relu,
                  batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS, batch_norm_scale=False, *args, **kwargs):
        """ Returns the scope with the default parameters for inception_resnet_v2. """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
                'updates_collections': batch_norm_updates_collections,
                'fused': None,  # Use fused batch norm if possible.
                'scale': batch_norm_scale,
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params) as scope:
                return scope

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
                  dropout_keep_prob=0.8,
                  create_aux_logits=True,
                  activation_fn=tf.nn.relu,
                  return_predictions=False, **kwargs):
        """ Creates the Inception Resnet V2 model. """

        if arg_scope is None:
            arg_scope = self.arg_scope()

        with slim.arg_scope(arg_scope):
            with tf.variable_scope(self.SCOPE, 'InceptionResnetV2', [input_tensor], reuse=reuse) as scope:
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=is_training):

                    net, end_points = inception_resnet_v2_base(input_tensor, scope=scope,
                                                               activation_fn=activation_fn)

                    if create_aux_logits and self.num_classes:
                        with tf.variable_scope('AuxLogits'):
                            aux = end_points['PreAuxLogits']
                            aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                                  scope='Conv2d_1a_3x3')
                            aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
                            aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                                              padding='VALID', scope='Conv2d_2a_5x5')
                            aux = slim.flatten(aux)
                            aux = slim.fully_connected(aux, self.num_classes + self.CLASS_OFFSET, activation_fn=None,
                                                       scope='Logits')
                            end_points['AuxLogits'] = aux

                    with tf.variable_scope('Logits'):
                        kernel_size = net.get_shape()[1:3]

                        if kernel_size.is_fully_defined():
                            net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_8x8')
                        else:
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

                        end_points['global_pool'] = net

                        if not self.num_classes:
                            return net, end_points

                        net = slim.flatten(net)
                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
                        end_points['PreLogitsFlatten'] = net
                        logits = slim.fully_connected(net, self.num_classes + self.CLASS_OFFSET, activation_fn=None,
                                                      scope='Logits')
                        logits = logits[:, self.CLASS_OFFSET:]
                        end_points['Logits'] = logits
                        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

                    if return_predictions:
                        return end_points['Predictions']

                    return logits, end_points
