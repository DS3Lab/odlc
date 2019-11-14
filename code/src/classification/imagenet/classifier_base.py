from abc import ABCMeta
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.contrib import graph_editor

from src.lib.commons import AbstractAttribute
from src.data.datasets import Imagenet
from src.data.dataprepare.imagenet.label_converter import LabelConverter
from dirs import IMAGENET_META_DIR
from src.lib.tf_commons.utils import get_sess_config


class ImagenetClassifierNames:
    densenet121 = 'densenet_121'
    inception_resnet_v2 = 'inception_resnet_v2'
    inception_v3 = 'inception_v3'
    mobilenet = 'mobilenet'
    resnet50 = 'resnet_50'
    vgg16 = 'vgg16'
    vgg19 = 'vgg19'
    xception = 'xception'

    allowed_classifiers = [densenet121, inception_resnet_v2, inception_v3, mobilenet, resnet50, vgg16, vgg19, xception]


class ImagenetClassifier(metaclass=ABCMeta):
    """ base class for classifier implementation """
    NAME = AbstractAttribute('name of classifier')
    MODEL_PB = AbstractAttribute('path to .pb file')
    INPUT_TENSOR_NAME_KERAS = AbstractAttribute('name of model input tensor in  graph_def')
    OUTPUT_TENSOR_NAME_KERAS = AbstractAttribute('name of model output tensor in graph_def')
    NUM_CLASSES = 1000

    # preprocessing
    DATA_FORMAT = AbstractAttribute('either `channels_last` or `channels_first`')
    INPUT_SHAPE = AbstractAttribute('shape of input images - i.e. size of central crop')

    def __init__(self):
        pass

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

    @classmethod
    def inference(cls, input_tensor, graph=None):
        """ implemented by subclasses

        args:
          input_tensor: 4-D Tensor of RGB images
          graph: graph in which model should be built

        returns:
          predictions from keras model
        """
        raise NotImplementedError

    @classmethod
    def inference0(cls, input_tensor, graph, model_input_tensor_name, model_output_tensor_name):
        """
        args:
          input_tensor: 4-D image tensor with batch_size N; inputs are assumed to be preprocessed
          graph: parent graph in which classifier is defined
          model_input_tensor_name: String. name of input tensor
          model_output_tensor_name: String. name of output tensor

        returns:
          logits, tensor of shape N x num_class
        """
        with graph.as_default():
            # read frozen graph
            with gfile.FastGFile(cls.MODEL_PB, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name=cls.NAME)

            # fetch tensors from frozen graph
            parent_name_scope = graph.get_name_scope()
            if len(parent_name_scope) > 0:
                parent_name_scope += '/'
            input_t_ph = graph.get_tensor_by_name(parent_name_scope + cls.NAME + '/' + model_input_tensor_name)
            outputs = graph.get_tensor_by_name(parent_name_scope + cls.NAME + '/' + model_output_tensor_name)

            # need to detach input_t_ph from graph and attach preprocessed_input_tensor
            graph_editor.swap_inputs(input_t_ph.op.outputs[0], [input_tensor])

        return outputs

    @staticmethod
    def permute_channels(image_batch, channel_axis, permutation):
        if image_batch.get_shape().as_list()[channel_axis] != len(permutation):
            raise ValueError('length of permutation does not equal number of color channels! {} != {}'.format(
                len(permutation), image_batch.get_shape().as_list()[channel_axis]))

        color_channels = tf.split(axis=channel_axis, num_or_size_splits=3, value=image_batch)
        image_batch_permuted = tf.concat(axis=channel_axis, values=[color_channels[i] for i in permutation])
        return image_batch_permuted

    @property
    def num_classes(self):
        return Imagenet.NUM_CLASSES

    @classmethod
    def predict(cls, image_numpy, topk=5, resize=True, normalize=True):
        """ returns topk most probable classes predicted by the classifier """
        label_converter = LabelConverter(IMAGENET_META_DIR)

        with tf.Graph().as_default() as graph:
            image_tensor = tf.convert_to_tensor(value=image_numpy, dtype=tf.uint8)

            assert len(image_tensor.get_shape().as_list()) == 3

            if not resize:
                assert image_tensor.get_shape().as_list() == list(cls.INPUT_SHAPE)
                if normalize:
                    image_preprocessed = cls.standardize_tensor(image_tensor)
                else:
                    # image is expected to be of correct shape and already normalized
                    image_preprocessed = tf.identity(image_tensor)
            else:
                image_preprocessed = cls.preprocess_image(image_tensor, is_training=False)

            image_preprocessed = tf.expand_dims(image_preprocessed, axis=0)

            class_probabilities_op = cls.inference0(image_preprocessed, graph, cls.INPUT_TENSOR_NAME_KERAS,
                                                    cls.OUTPUT_TENSOR_NAME_KERAS)

            predicted_label_op = tf.argmax(class_probabilities_op, axis=1)

        with tf.Session(graph=graph, config=get_sess_config()) as sess:
            class_probabilities, predicted_label = sess.run([class_probabilities_op, predicted_label_op])

        # topk predictions
        class_probabilities = np.squeeze(class_probabilities)
        topk_indices = class_probabilities.argsort()[-topk:][::-1]
        topk_labels_synsets_probabilities = [(label_converter.keras_index_to_name(ii),
                                              label_converter.keras_index_to_synset(ii),
                                              class_probabilities[ii]) for ii in topk_indices]

        return topk_labels_synsets_probabilities
