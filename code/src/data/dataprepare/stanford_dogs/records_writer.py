import numpy as np
import os
import scipy.io as scipy_io

import tensorflow as tf

from src.lib.tf_commons.utils import int64_feature, bytes_feature
from src.lib.logging_commons.utils import progress, get_logger

LABEL_OFFSET = -1


class TrainRecordsWriter:

    def __init__(self, data_dir, target_dir):
        self._data_dir = data_dir
        self._images_dir = os.path.join(data_dir, 'Images/')
        self._train_list_mat = os.path.join(data_dir, 'lists/train_list.mat')
        self._logger = get_logger(os.path.join(data_dir, 'train_data.log'))

        if target_dir is None:
            self._target_dir = data_dir
        else:
            assert os.path.isdir(target_dir)
            self._target_dir = target_dir

    def process_files(self):
        def write_example(_image_bytes, _label, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'label': int64_feature(int(_label))
            }))
            _writer.write(example.SerializeToString())

        # get image files from .mat
        train_list = scipy_io.loadmat(self._train_list_mat)
        file_list = [os.path.join(self._images_dir, fn[0][0]) for fn in train_list['file_list']]
        label_list = [label[0] for label in train_list['labels']]
        num_files = len(file_list)

        self._logger.info('data_dir={}'.format(self._data_dir))
        self._logger.info('target_dir={}'.format(self._target_dir))
        self._logger.info('train_list_mat={}'.format(self._train_list_mat))
        self._logger.info('num_files={}'.format(num_files))
        self._logger.info('unique_labels={}'.format(np.unique(label_list)))

        # create writer
        records_file = os.path.join(self._target_dir, 'train_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(records_file)

        i = 0
        for i, (image_file, label) in enumerate(zip(file_list, label_list)):
            with open(image_file, 'rb') as img_f:
                image_bytes = img_f.read()

            label += LABEL_OFFSET

            write_example(tf.compat.as_bytes(image_bytes), label, writer)
            progress(i + 1, num_files, '{}/{} training images processed...'.format(i, num_files))

        self._logger.info('num_files processed={}'.format(i + 1))


class ValRecordsWriter:

    def __init__(self, data_dir, target_dir):
        self._data_dir = data_dir
        self._images_dir = os.path.join(data_dir, 'Images/')
        self._val_list_mat = os.path.join(data_dir, 'lists/test_list.mat')
        self._logger = get_logger(os.path.join(data_dir, 'val_data.log'))

        if target_dir is None:
            self._target_dir = data_dir
        else:
            assert os.path.isdir(target_dir)
            self._target_dir = target_dir

    def process_files(self):
        def write_example(_image_bytes, _label, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'label': int64_feature(int(_label))
            }))
            _writer.write(example.SerializeToString())

        # get image files from .mat
        val_list = scipy_io.loadmat(self._val_list_mat)
        file_list = [os.path.join(self._images_dir, fn[0][0]) for fn in val_list['file_list']]
        label_list = [label[0] for label in val_list['labels']]
        num_files = len(file_list)

        self._logger.info('data_dir={}'.format(self._data_dir))
        self._logger.info('target_dir={}'.format(self._target_dir))
        self._logger.info('val_list_mat={}'.format(self._val_list_mat))
        self._logger.info('num_files={}'.format(num_files))
        self._logger.info('unique_labels={}'.format(np.unique(label_list)))

        # create writer
        records_file = os.path.join(self._target_dir, 'val_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(records_file)

        i = 0
        for i, (image_file, label) in enumerate(zip(file_list, label_list)):
            with open(image_file, 'rb') as img_f:
                image_bytes = img_f.read()

            label += LABEL_OFFSET

            write_example(tf.compat.as_bytes(image_bytes), label, writer)
            progress(i + 1, num_files, '{}/{} validation images processed...'.format(i, num_files))

        self._logger.info('num_files processed={}'.format(i + 1))
