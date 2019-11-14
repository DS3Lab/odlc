import numpy as np
import os

import tensorflow as tf

from src.lib.tf_commons.layers import dense, maxpool, conv2d
from src.compression.distortions.loss_networks.loss_network_base import LossNetworkBase

_N_CONVS_AND_FILTERS = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]


class VGG16(LossNetworkBase):
    SCOPE = 'vgg16'
    NAME = 'vgg16'
    R_MEAN = 123.68
    G_MEAN = 116.78
    B_MEAN = 103.94

    NUM_CLASSES = 1000

    IMAGE_SIZE = 224

    def __init__(self, data_format, weights_file):
        super(VGG16, self).__init__(data_format)

        assert all([os.path.isfile(weights_file), os.path.splitext(weights_file)[-1] == '.npy'])

        self._weights_dict = np.load(weights_file, encoding='latin1', allow_pickle=True).item()

        self._conv_kwargs = dict(kernel_size=(3, 3), strides=(1, 1), padding='SAME', data_format=data_format,
                                 activation=tf.nn.relu, kernel_regularizer=None, bias_regularizer=None, trainable=False)
        self._pool_kwargs = dict(pool_size=(2, 2), strides=(2, 2), padding='SAME', data_format=data_format)
        self._dense_kwargs = dict(units=4096, activation=tf.nn.relu, use_bias=True, trainable=False)
        self._logits_kwargs = dict(units=self.NUM_CLASSES, activation=None, use_bias=True)

    def build_convolutions(self, inputs, reuse=None):
        with tf.variable_scope(self.SCOPE, reuse=reuse):
            net = self.preprocess(inputs)
            for i, (n_convs, n_filters) in enumerate(_N_CONVS_AND_FILTERS):
                for j in range(n_convs):
                    name = 'conv{}_{}'.format(i + 1, j + 1)
                    filters, biases = self._get_filter(name), self._get_bias(name)
                    net = conv2d(net, n_filters, filter_weights=filters, biases=biases, name=name, **self._conv_kwargs)
                net = maxpool(net, name='pool{}'.format(i + 1), **self._pool_kwargs)
        return net

    def inference(self, inputs, reuse=None):
        net = self.build_convolutions(inputs, reuse)
        with tf.variable_scope(self.SCOPE, reuse=reuse):
            if self.data_format == 'NCHW':
                net = tf.transpose(net, perm=[0, 2, 3, 1])
            net = tf.layers.flatten(net, name='flatten')
            net = dense(net, name='fc6', weights=self._get_weights('fc6'), biases=self._get_bias('fc6'),
                        **self._dense_kwargs)
            net = dense(net, name='fc7', weights=self._get_weights('fc7'), biases=self._get_bias('fc7'),
                        **self._dense_kwargs)
            logits = dense(net, name='fc8', weights=self._get_weights('fc8'), biases=self._get_bias('fc8'),
                           **self._logits_kwargs)

        return logits

    def get_features(self, inputs, layer_names, prefix=None, reuse=None):
        layer_names = [layer_names] if not isinstance(layer_names, list) else layer_names
        with tf.get_default_graph().as_default() as g:
            self.build_convolutions(inputs, reuse=reuse)
            return [g.get_tensor_by_name(self._from_name_to_op(name, prefix)) for name in layer_names]

    @staticmethod
    def _from_name_to_op(name, prefix='None'):
        return {'relu1_1': prefix + 'vgg16/conv1_1/Relu:0',
                'relu1_2': prefix + 'vgg16/conv1_2/Relu:0',
                'relu2_1': prefix + 'vgg16/conv2_1/Relu:0',
                'relu2_2': prefix + 'vgg16/conv2_2/Relu:0',
                'relu3_1': prefix + 'vgg16/conv3_1/Relu:0',
                'relu3_2': prefix + 'vgg16/conv3_2/Relu:0',
                'relu3_3': prefix + 'vgg16/conv3_3/Relu:0',
                'relu4_1': prefix + 'vgg16/conv4_1/Relu:0',
                'relu4_2': prefix + 'vgg16/conv4_2/Relu:0',
                'relu4_3': prefix + 'vgg16/conv4_3/Relu:0',
                'relu5_1': prefix + 'vgg16/conv5_1/Relu:0',
                'relu5_2': prefix + 'vgg16/conv5_2/Relu:0',
                'relu5_3': prefix + 'vgg16/conv5_3/Relu:0'}[name]

    def _get_filter(self, name):
        return self._weights_dict[name][0]

    def _get_bias(self, name):
        return self._weights_dict[name][1]

    def _get_weights(self, name):
        return self._weights_dict[name][0]

    def preprocess(self, image_batch_rgb):
        with tf.name_scope('vgg_preprocess'):
            image_batch_rgb_centered = self.mean_image_subtraction(image_batch_rgb=image_batch_rgb,
                                                                   means=[self.R_MEAN, self.G_MEAN, self.B_MEAN],
                                                                   data_format=self.data_format)
            return self.rgb_to_bgr(image_batch_rgb_centered, channel_axis=3 if self.data_format == 'NHWC' else 1)

    @staticmethod
    def rgb_to_bgr(image_batch, channel_axis):
        r, g, b = tf.split(axis=channel_axis, num_or_size_splits=3, value=image_batch)
        image_batch_bgr = tf.concat(axis=channel_axis, values=[b, g, r])
        return image_batch_bgr

    @staticmethod
    def mean_image_subtraction(image_batch_rgb, means, data_format):
        """ centers a batch of images

        :param image_batch_rgb: batch of rgb-images with shape NHWC or NCHW
        :param data_format: NHWC or NCHW
        :param means: list of means in order [R, G, B]
        :return: centered rgb-image batch
        """
        if image_batch_rgb.get_shape().ndims != 4:
            raise ValueError('Input must be of rank 4')
        if data_format == 'NHWC':
            channel_axis = 3
        elif data_format == 'NCHW':
            channel_axis = 1
        else:
            raise ValueError('data_format must be one of `NHWC` or `NCHW`; got {}'.format(data_format))

        num_channels = image_batch_rgb.get_shape().as_list()[channel_axis]

        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=channel_axis, num_or_size_splits=num_channels, value=image_batch_rgb)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=channel_axis, values=channels)

    def initialize(self, **kwargs):
        sess = kwargs['sess']
        var_list = self.variables()
        assert len(var_list) > 0, 'loss_network error: no variables found in scope {}'.format(self.SCOPE)
        sess.run(tf.variables_initializer(var_list=var_list))
