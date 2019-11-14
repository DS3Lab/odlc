import os
import tensorflow as tf

from src.lib.tf_commons.utils import int64_feature, bytes_feature
from src.lib.logging_commons.utils import progress, get_logger

LABEL_OFFSET = -1


class RecordsWriter:

    def __init__(self, data_dir, split, target_dir=None):
        self._data_dir = data_dir
        self._split = str(split)
        if target_dir is None:
            self._target_dir = data_dir
        else:
            assert os.path.isdir(target_dir)
            self._target_dir = target_dir

        self._images_dir = os.path.join(data_dir, 'images/')

        self._train_test_split = os.path.join(data_dir, 'train_test_split.txt')
        self._image_class_labels = os.path.join(data_dir, 'image_class_labels.txt')
        self._images_list = os.path.join(data_dir, 'images.txt')

        self._logger = get_logger(os.path.join(data_dir, '{}_data.log'.format('train' if str(split) == '1' else 'val')))

    def _get_image_ids_paths_labels_split(self, split):
        with open(self._train_test_split, 'r') as fo:
            train_test_split = [image_id_and_split.strip().split(' ') for image_id_and_split in fo.readlines()]
            train_test_split = {key: val for key, val in train_test_split}

        with open(self._image_class_labels, 'r') as fo:
            image_class_labels = [image_id_and_label.strip().split(' ') for image_id_and_label in fo.readlines()]
            image_class_labels = {key: val for key, val in image_class_labels}

        with open(self._images_list, 'r') as fo:
            image_file_names = [image_id_and_filename.strip().split(' ') for image_id_and_filename in fo.readlines()]
            image_file_names = {key: os.path.join(self._images_dir, val) for key, val in image_file_names}

        # merge all dicts
        merged_dicts = {}
        for image_id in range(1, len(image_file_names) + 1):
            image_id = str(image_id)
            if train_test_split[image_id] == split:
                merged_dicts[image_id] = {'is_training': train_test_split[image_id],
                                          'label': image_class_labels[image_id],
                                          'path': image_file_names[image_id]}

        return merged_dicts

    def process_files(self):
        def write_example(_image_bytes, _label, _writer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_bytes': bytes_feature(_image_bytes),
                'label': int64_feature(int(_label))
            }))
            _writer.write(example.SerializeToString())

        # dict with key=id, and val = {is_training, label, path}
        image_paths_labels_split = self._get_image_ids_paths_labels_split(self._split)
        num_images = len(image_paths_labels_split)

        self._logger.info('data_dir={}'.format(self._data_dir))
        self._logger.info('target_dir={}'.format(self._target_dir))
        self._logger.info('split={}'.format(self._split))
        self._logger.info('train_test_split={}'.format(self._train_test_split))
        self._logger.info('image_class_labels={}'.format(self._image_class_labels))
        self._logger.info('images_list={}'.format(self._images_list))

        # create writer
        records_file = os.path.join(self._target_dir,
                                    '{}_data.tfrecords'.format('train' if str(self._split) == '1' else 'val'))
        writer = tf.python_io.TFRecordWriter(records_file)

        # loop through files
        for i, (image_id, image_info) in enumerate(image_paths_labels_split.items(), start=1):
            image_file = image_info['path']
            image_label = int(image_info['label']) + LABEL_OFFSET
            image_split = image_info['is_training']

            assert image_split == self._split

            with open(image_file, 'rb') as img_f:
                image_bytes = img_f.read()

            write_example(image_bytes, image_label, writer)

            progress(i, num_images, '{}/{} images processed; split={}'.format(i, num_images, self._split))

        self._logger.info('num_images processed: {}'.format(num_images))
