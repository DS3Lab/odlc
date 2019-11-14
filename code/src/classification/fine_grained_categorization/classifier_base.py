from abc import ABCMeta

from src.lib.commons import AbstractAttribute
from src.data.datasets import Dataset


class FGVCClassifierNames:
    inception_resnet_v2 = 'inception_resnet_v2'
    inception_v3 = 'inception_v3'
    mobilenet = 'mobilenet_v1'
    resnet50 = 'resnet_v1_50'
    vgg16 = 'vgg_16'


class FGVCClassifier(metaclass=ABCMeta):
    """ base class for classifier implementation for fine grained visual categorization """
    NAME = AbstractAttribute('name of classifier')
    INPUT_SHAPE = AbstractAttribute('shape of input images - i.e. size of central crop')

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def inference(self, input_tensor, is_training, reuse, **kwargs):
        """ implemented by subclasses

        args:
          input_tensor: 4-D tensor of shape [batch_size, height, width, depth]
          is_training: python boolean
          reuse: set to True if reuse var scope
        returns:
          logits, end_points
        """
        raise NotImplementedError

    @classmethod
    def preprocess_image(cls, image, is_training, normalize_image=True):
        """ implemented by subclass

        args:
          image: 3-D Tensor of RGB images
          is_training: Boolean.
          normalize_image: Boolean. If True, the image is normalized according to the appropriate scaling method.

        returns:
          appropriately preprocessed image batch
        """
        raise NotImplementedError

    @staticmethod
    def standardize_tensor(input_tensor):
        """ implemented by subclass

        args:
          input_tensor: 4-D tensor containing images

        returns:
          standardized image batch according to the appropriate method

        """
        raise NotImplementedError

    @staticmethod
    def arg_scope(*args, **kwargs):
        """ defines arg scope for model """
        raise NotImplementedError

    @property
    def num_classes(self):
        return self._dataset.NUM_CLASSES

    # @classmethod
    # def preprocess_batch(cls, input_batch, is_training):
    #     def _preprocess(_input_image):
    #         return cls.preprocess(_input_image, is_training)
    #
    #     return tf.map_fn(_preprocess, elems=input_batch, dtype=tf.float32)
