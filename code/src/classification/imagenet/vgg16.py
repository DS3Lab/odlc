import os

import tensorflow as tf

from dirs import KERAS_MODELS_DIR
from src.classification.imagenet import ImagenetClassifier, ImagenetClassifierNames
from src.data.dataloading.preprocessing import VGGPreprocessing


class Vgg16(ImagenetClassifier):
    NAME = ImagenetClassifierNames.vgg16
    MODEL_PB = os.path.join(KERAS_MODELS_DIR, NAME + '.pb')
    INPUT_TENSOR_NAME_KERAS = 'input_1:0'
    OUTPUT_TENSOR_NAME_KERAS = 'predictions/Softmax:0'
    NUM_CLASSES = 1000

    DATA_FORMAT = 'NHWC'
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self):
        super(Vgg16, self).__init__()

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

    @classmethod
    def inference(cls, input_tensor, graph=None):
        # convert input from RGB to BGR
        input_tensor = cls.permute_channels(input_tensor, channel_axis=3, permutation=[2, 1, 0])

        # check shapes
        input_tensor_shape = input_tensor.get_shape().as_list()[1:]
        assert input_tensor_shape == list(cls.INPUT_SHAPE)

        # fetch default graph if none
        graph = tf.get_default_graph() if graph is None else graph

        return cls.inference0(input_tensor, graph, cls.INPUT_TENSOR_NAME_KERAS, cls.OUTPUT_TENSOR_NAME_KERAS)
