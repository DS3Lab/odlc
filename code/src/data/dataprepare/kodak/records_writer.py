import os

import tensorflow as tf

from src.lib.tf_commons.utils import bytes_feature
from src.lib.logging_commons.utils import progress, get_logger


class RecordsWriter:
    BASE_FILENAME = 'kodak_data'

    def __init__(self, data_dir, target_dir=None):
        self._data_dir = os.path.abspath(data_dir)
        self._target_dir = target_dir

        if not os.path.exists(self._target_dir):
            os.makedirs(self._target_dir)
        self._logger = get_logger(os.path.join(self._target_dir, self.BASE_FILENAME + '.info'))
        self._logger.info('data_dir: {}'.format(self._data_dir))
        self._logger.info('target_dir: {}'.format(self._target_dir))
        self._logger.info('image encoding: png')
        self._logger.info('image_filename encoding: utf-8')

    def process_files(self):
        def write_example(_image_bytes, _image_file, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'image_file': bytes_feature(_image_file)
            }))
            _writer.write(example.SerializeToString())

        fns = [fn for fn in os.listdir(self._data_dir) if fn.endswith('.png')]
        total_files = len(fns)

        # log filenames
        for i, fn in enumerate(fns):
            self._logger.info('image {}: {}'.format(i, fn))

        records_file = os.path.join(self._target_dir, self.BASE_FILENAME + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(records_file)
        self._logger.info('writing image data to: {}'.format(records_file))

        self._logger.info('number of files: {}'.format(total_files))

        for i, fn in enumerate(fns):
            image_file = os.path.join(self._data_dir, fn)

            with open(image_file, 'rb') as img_f:
                image_bytes = img_f.read()

            write_example(tf.compat.as_bytes(image_bytes), bytes(image_file, encoding='utf-8'), writer)
            progress(i, total_files, '{}/{} validation images processed...'.format(i, total_files))

        writer.flush()
        writer.close()
