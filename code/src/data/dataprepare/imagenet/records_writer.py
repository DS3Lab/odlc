import os

import tensorflow as tf

from dirs import IMAGENET_META_DIR
from src.lib.tf_commons.utils import int64_feature, bytes_feature
from src.lib.logging_commons.utils import progress, get_logger
from src.data.dataprepare.imagenet.label_converter import LabelConverter


class TrainRecordsWriter:

    def __init__(self, train_data_dir, target_dir=None):
        self._train_data_dir = os.path.abspath(train_data_dir)
        self._meta_dir = os.path.abspath(IMAGENET_META_DIR)
        self._target_dir = target_dir

        self._label_converter = LabelConverter(self._meta_dir)

        if not os.path.exists(self._target_dir):
            os.makedirs(self._target_dir)
        self._logger = get_logger(os.path.join(self._target_dir, 'train_data.info'))
        self._logger.info('train_data_dir: {}'.format(self._train_data_dir))
        self._logger.info('meta_dir: {}'.format(self._meta_dir))

    def process_files(self):
        def write_example(_image_bytes, _label, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'label': int64_feature(int(_label))
            }))
            _writer.write(example.SerializeToString())

        records_file = os.path.join(self._target_dir, 'train_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(records_file)
        self._logger.info('writing image data to: {}'.format(records_file))

        synsets = os.listdir(self._train_data_dir)
        num_synsets = len(synsets)
        self._logger.info('number of synsets found: {}'.format(num_synsets))
        print('')

        num_images, n = 0, 0
        for i, s in enumerate(synsets):

            keras_label = self._label_converter.synset_to_keras_index(s)
            synset_dir = os.path.join(self._train_data_dir, s)
            image_files_in_synset = [os.path.join(synset_dir, f) for f in os.listdir(synset_dir)]

            for n, image_file in enumerate(image_files_in_synset, 1):
                with open(image_file, 'rb') as img_f:
                    image_bytes = img_f.read()

                write_example(tf.compat.as_bytes(image_bytes), keras_label, writer)

            progress(i + 1, num_synsets, '{}/{} synsets processed'.format(i + 1, num_synsets))
            num_images += n

        print('')
        self._logger.info('number of train images: {}'.format(num_images))

        writer.flush()
        writer.close()


class ValRecordsWriter:

    def __init__(self, val_data_dir, target_dir=None):
        self._val_data_dir = os.path.abspath(val_data_dir)
        self._meta_dir = os.path.abspath(IMAGENET_META_DIR)
        self._target_dir = target_dir or IMAGENET_META_DIR

        self._label_converter = LabelConverter(self._meta_dir)
        self.__get_labels()

        if not os.path.exists(self._target_dir):
            os.makedirs(self._target_dir)
        self._logger = get_logger(os.path.join(self._target_dir, 'val_data.info'))
        self._logger.info('val_data_dir: {}'.format(self._val_data_dir))
        self._logger.info('meta_dir: {}'.format(self._meta_dir))

    def process_files(self):
        def write_example(_image_bytes, _label, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'label': int64_feature(int(_label))
            }))
            _writer.write(example.SerializeToString())

        fns = sorted([fn for fn in os.listdir(self._val_data_dir) if fn.endswith('.JPEG')])
        total_files = len(fns)

        records_file = os.path.join(self._target_dir, 'val_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(records_file)
        self._logger.info('writing image data to: {}'.format(records_file))

        self._logger.info('number of files: {}'.format(total_files))

        for i, (fn, label) in enumerate(zip(fns, self._labels)):
            image_file = os.path.join(self._val_data_dir, fn)

            with open(image_file, 'rb') as img_f:
                image_bytes = img_f.read()

            write_example(tf.compat.as_bytes(image_bytes), label, writer)
            progress(i, total_files, '{}/{} validation images processed...'.format(i, total_files))

        writer.flush()
        writer.close()

    def __get_labels(self):
        with open(os.path.join(self._meta_dir, 'devkit/data/ILSVRC2012_validation_ground_truth.txt'), 'r') as f:
            labels = f.read().strip().split('\n')
            labels = list(map(int, labels))
            self._labels = [self._label_converter.original_index_to_keras_index(i) for i in labels]
