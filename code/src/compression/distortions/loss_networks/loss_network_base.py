from abc import ABCMeta, abstractmethod

import tensorflow as tf

from src.lib.commons import AbstractAttribute


class LossNetworkBase(metaclass=ABCMeta):
    IMAGE_SIZE = AbstractAttribute('expected image size')
    SCOPE = AbstractAttribute('name of variable scope')
    NAME = AbstractAttribute('classifier name')
    NUM_CLASSES = AbstractAttribute('number of classes')

    def __init__(self, data_format):
        self.data_format = data_format

    @abstractmethod
    def build_convolutions(self, inputs, reuse=None):
        """ build graph for convolutional layers

        :param inputs: 4-D Tensor of RGB images
        :param reuse: reuse variables
        :return: output of last convolutional layer
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self, inputs, reuse=None):
        """ fully connected layers

        :param inputs: 4-D Tensor of RGB images
        :param reuse: reuse variables
        :return: logits
        """
        raise NotImplementedError

    @abstractmethod
    def get_features(self, inputs, layer_names, prefix=None, reuse=None):
        """ fetches tensors associated with layers in layer_names

        :param inputs: 4-D Tensor of RGB images
        :param layer_names: name of tensors to be fetched
        :param prefix:
        :param reuse: reuse variables
        :return: list of tensors
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image_batch_rgb):
        """ preprocesses an image

        :param image_batch_rgb: RGB image_batch of shape NHWC or NCHW
        :return: image batch preprocessed
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _from_name_to_op(name, prefix=None):
        """ convenience function define names and associated ops to be fetched by LossNetworkBase.get_features

        :param name: str readable name of op
        :param prefix: optional prefix; needed when loss network scope has parent scope
        :return: str name of op in graph
        """
        raise NotImplementedError

    @classmethod
    def variables(cls):
        return [v for v in tf.global_variables(scope=cls.SCOPE)]

    @classmethod
    def trainable_variables(cls):
        return tf.trainable_variables(scope=cls.SCOPE)

    @abstractmethod
    def initialize(self, **kwargs):
        """ implemented by subclasses"""
        raise NotImplementedError

    @staticmethod
    def parse_kwargs(key, **kwargs):
        try:
            return kwargs[key]
        except KeyError:
            return None
