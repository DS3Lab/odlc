from abc import ABCMeta

import inspect
import numpy as np
import os

import tensorflow as tf

from src.lib.logging_commons import EvalValues
from src.classification.imagenet import ImagenetClassifierNames
from src.classification.fine_grained_categorization import FGVCClassifierNames
from src.data.datasets import Imagenet, StanfordDogs, Cub200
from src.classification import classifier_factory
from src.lib.logging_commons.utils import get_logger

_FGVC_TRAIN_ID = 251


class EvalAccuracyBase(metaclass=ABCMeta):
    """ base class for accuracy evaluation """

    SIZE_MULTIPLE_OF = 16

    def __init__(self, dataset_name, records_file, logger, checkpoint_root=None):
        self._dataset_name = dataset_name
        self._checkpoint_root = checkpoint_root

        if dataset_name == Imagenet.NAME:
            self._dataset = Imagenet
        elif dataset_name == Cub200.NAME:
            self._dataset = Cub200
        elif dataset_name == StanfordDogs.NAME:
            self._dataset = StanfordDogs
        else:
            raise ValueError('unknown dataset ' + dataset_name)
        self._records_file = records_file
        self.logger = logger

    def run0(self):
        """ implemented by subclass; imagenet evaluation loop """
        raise NotImplementedError

    def run1(self):
        """ implemented by subclass; fgvc evaluation loop """
        raise NotImplementedError

    def run(self):
        if self._dataset_name == Imagenet.NAME:
            self.run0()
        else:
            self.run1()

    @property
    def imagenet_cnn_model_names(self):
        return [
            ImagenetClassifierNames.densenet121,
            ImagenetClassifierNames.inception_resnet_v2,
            ImagenetClassifierNames.inception_v3,
            ImagenetClassifierNames.mobilenet,
            ImagenetClassifierNames.resnet50,
            ImagenetClassifierNames.vgg16,
            ImagenetClassifierNames.xception
        ]

    @property
    def fgvc_cnn_model_names(self):
        return [FGVCClassifierNames.vgg16,
                FGVCClassifierNames.resnet50,
                FGVCClassifierNames.inception_resnet_v2,
                FGVCClassifierNames.inception_v3,
                FGVCClassifierNames.mobilenet]

    def get_fgvc_checkpoint(self, model_name):
        model_root = os.path.join(self._checkpoint_root, model_name + '_id{}/'.format(_FGVC_TRAIN_ID))
        model_checkpoint = tf.train.latest_checkpoint(os.path.join(model_root, 'checkpoints/'))

        if model_checkpoint is None:
            warn_str = 'checkpoint for model {} in dir {} not found!'.format(model_name, model_root)
            if os.path.exists(model_root):
                warn_str += '\n     dir contents: {}'.format(os.listdir(model_root))
            else:
                warn_str += '\n     dir {} does not exist!'.format(model_root)
            self.logger.warning(warn_str)

        else:
            self.logger.info('{}; found checkpoint: {}'.format(model_name, model_checkpoint))

        return model_checkpoint

    def _get_logger(self, dataset_name, compression_method):
        self._log_dir = os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())),
                                     'logs/{}/tmp'.format(dataset_name))
        self._log_number = np.random.choice(10000)
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        logfile = os.path.join(self._log_dir, 'eval_accuracy_{}_{}.log'.format(compression_method, self._log_number))
        return get_logger(logfile)

    def get_classifier_instance(self, classifier_name):
        return classifier_factory.get_classifier_instance(self._dataset_name, classifier_name)

    @property
    def records_file(self):
        return self._records_file

    @property
    def dataset_name(self):
        return self._dataset_name

    def log_results(self, eval_values: EvalValues):
        """ accuracies need to be logged in a uniform structure """
        for bpp, accs_dict, other_info_dict in zip(eval_values.bits_per_pixel, eval_values.accuracy_dicts,
                                                   eval_values.other_info_dicts):
            eval_str = 'EVAL: [bpp_mean={}] '.format(bpp)

            for cnn, acc_tuple in accs_dict.items():
                eval_str += '| [{}_top1acc={}, {}_top5acc={}] '.format(cnn, acc_tuple[0], cnn, acc_tuple[1])

            if other_info_dict is not None and isinstance(other_info_dict, dict):
                for key, val in other_info_dict.items():
                    eval_str += '| [{}={}] '.format(key, val)

            self.logger.info(eval_str)

    @staticmethod
    def pack_eval_values(bits_per_pixel, accuracy_dicts, other_info_dicts=None):
        return EvalValues(bits_per_pixel=bits_per_pixel,
                          accuracy_dicts=accuracy_dicts,
                          hvs_dicts=None,
                          other_info_dicts=other_info_dicts)

    @staticmethod
    def top_k_accuracy(true_labels, predicted_labels, k):
        argsorted_predictions = np.argsort(predicted_labels)[:, -k:]
        return np.any(argsorted_predictions.T == true_labels.argmax(axis=1), axis=0).mean()

    @staticmethod
    def to_categorical(y, num_classes=None, dtype='float32'):
        """ Converts a class vector (integers) to binary class matrix.

        args:
          y: class vector to be converted into a matrix (integers from 0 to num_classes).
          num_classes: total number of classes.
          dtype: The data type expected by the input, as a string (`float32`, `float64`, `int32`...)

        returns:
            A binary matrix representation of the input. The classes axis is placed last.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
