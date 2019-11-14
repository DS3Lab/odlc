from abc import ABCMeta, abstractmethod
import numpy as np

import tensorflow as tf


class BaseModel(metaclass=ABCMeta):
    MODEL_SCOPE = 'rnn_compression'
    SIZE_MULTIPLE_OF = 16

    def __init__(self, image_height, image_width, num_iterations, rec_model, data_format,
                 color_means, color_vars):
        # constant vars
        self._image_height = image_height
        self._image_width = image_width
        self._num_iterations = num_iterations
        self._rec_model = rec_model
        self._data_format = data_format
        self._color_means = color_means
        self._color_vars = color_vars

    @abstractmethod
    def _initialize_model(self):
        """ override by subclasses """
        raise NotImplementedError

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """ override by subclasses """
        raise NotImplementedError

    @abstractmethod
    def _build_additive_model(self, *args, **kwargs):
        """ override by subclasses """
        raise NotImplementedError

    @abstractmethod
    def _build_oneshot_model(self, *args, **kwargs):
        """ override by subclasses """
        raise NotImplementedError

    def _get_rgb_mean(self):
        assert isinstance(self._color_means, list)
        assert len(self._color_means) == 3
        mean = np.array(self._color_means, dtype=np.float32)
        expand_axis = -1 if self._data_format == 'NCHW' else 0
        mean = np.expand_dims(np.expand_dims(mean, expand_axis), expand_axis)  # -> broadcast with NCHW / NHWC
        return mean

    def _get_rgb_var(self):
        assert isinstance(self._color_vars, list)
        assert len(self._color_vars) == 3
        var = np.array(self._color_vars, dtype=np.float32)
        expand_axis = -1 if self._data_format == 'NCHW' else 0
        mean = np.expand_dims(np.expand_dims(var, expand_axis), expand_axis)  # -> broadcast with NCHW / NHWC
        return mean

    def _normalize_np(self, inputs):
        mean, var = self._get_rgb_mean(), self._get_rgb_var()
        with tf.name_scope('normalize'):
            return (inputs - mean) / np.sqrt(var + 1e-10)

    def _normalize_tf(self, inputs):
        mean_tensor = tf.constant(self._get_rgb_mean(), dtype=tf.float32, name='rgb_means')
        var_tensor = tf.constant(self._get_rgb_var(), dtype=tf.float32, name='rgb_variances')
        if inputs.dtype != tf.float32:
            inputs = tf.to_float(inputs)
        with tf.name_scope('normalize'):
            return tf.divide(tf.subtract(inputs, mean_tensor), tf.sqrt(tf.add(var_tensor, 1e-10)))

    def _denormalize_np(self, inputs):
        mean, var = self._get_rgb_mean(), self._get_rgb_var()
        with tf.name_scope('denormalize'):
            return (inputs * np.sqrt(var + 1e-10)) + mean

    def _denormalize_tf(self, inputs):
        mean_tensor = tf.constant(self._get_rgb_mean(), dtype=tf.float32, name='rgb_means')
        var_tensor = tf.constant(self._get_rgb_var(), dtype=tf.float32, name='rgb_variances')
        with tf.name_scope('denormalize'):
            return tf.add(tf.multiply(inputs, tf.sqrt(tf.add(var_tensor, 1e-10))), mean_tensor)

    @property
    @abstractmethod
    def trainable_encoder_vars(self):
        """ override by subclasses """
        raise NotImplementedError

    @property
    @abstractmethod
    def trainable_binarizer_vars(self):
        """ override by subclasses """
        raise NotImplementedError

    @property
    @abstractmethod
    def trainable_decoder_vars(self):
        """ override by subclasses """
        raise NotImplementedError

    @property
    def model_variables(self):
        return tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.MODEL_SCOPE)

    @property
    def trainable_variables(self):
        return self.trainable_encoder_vars + self.trainable_binarizer_vars + self.trainable_decoder_vars

    @classmethod
    def pad_image_shape(cls, image_shape, extra_padding_multiples=0):
        """ calculates image shape compatible with rnn compression and optionally adds multiples of SIZE_MULTIPLES_OF"""
        assert len(image_shape) == 3

        def compute_side(s, m):
            """ computes next multiple of m if s is not multiple of m """
            if s % m == 0:
                return s
            else:
                return s + (m - s % m)

        original_height, original_width, num_channels = image_shape

        return [compute_side(original_height, cls.SIZE_MULTIPLE_OF) + extra_padding_multiples * cls.SIZE_MULTIPLE_OF,
                compute_side(original_width, cls.SIZE_MULTIPLE_OF) + extra_padding_multiples * cls.SIZE_MULTIPLE_OF,
                num_channels]
