import os
import numpy as np
import tempfile
import time

import tensorflow as tf

from src.compression.bpg import PyBPG
from src.data.dataloading.records_parsing import RecordsParser
from src.data.dataloading.preprocessing.compression_preprocessing import CompressionPreprocessing
from src.lib.tf_commons.utils import bytes_feature, int64_feature, float_feature
from src.lib.logging_commons.utils import get_logger, progress_v2

CPU_DEVICE = '/cpu:0'


class CreateRecords:

    def __init__(self, src_records_file, target_dir, quantization_level, target_height, target_width,
                 num_parallel_examples=4, records_type=RecordsParser.RECORDS_LABELLED):

        assert records_type == RecordsParser.RECORDS_LABELLED or records_type == RecordsParser.RECORDS_UNLABELLED

        # args
        self._src_records_file = src_records_file
        self._num_parallel_examples = num_parallel_examples
        self._target_dir = target_dir
        self._quantization_level = quantization_level
        self._target_height = target_height
        self._target_width = target_width
        self._records_type = records_type

        # dirs, filenames
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        basename = 'bpg_res{}_{}_'.format(str(target_height) + 'x' + str(target_width), self._quantization_level)
        basename += os.path.splitext(os.path.basename(self._src_records_file))[0]

        self._target_records_file = os.path.join(target_dir, basename + '.tfrecords')
        self._logger = get_logger(os.path.join(target_dir, basename + '.log'))

        # log specs
        self._logger.info('################# SETUP #################')
        self._logger.info('src_records_file: {}'.format(src_records_file))
        self._logger.info('num_parallel_examples: {}'.format(num_parallel_examples))
        self._logger.info('target_dir: {}'.format(target_dir))
        self._logger.info('quantization_level: {}'.format(quantization_level))
        self._logger.info('target_height: {}'.format(self._target_height))
        self._logger.info('target_width: {}'.format(self._target_width))
        self._logger.info('target_records_file: {}\n'.format(self._target_records_file))

        rand_num = np.random.choice(999999)
        self._temp_bpg_file = os.path.join(target_dir, 'tmp{}.bpg'.format(rand_num))
        self._temp_png_file = os.path.join(target_dir, 'tmp{}.png'.format(rand_num))

    def run(self):

        with tf.Graph().as_default() as graph:
            # read records
            records_reader = RecordsReader(self._src_records_file, records_type=self._records_type, batch_size=1)
            if self._records_type == RecordsParser.RECORDS_LABELLED:
                original_image_bytes, next_label = records_reader.next_element()
            else:
                original_image_bytes = records_reader.next_element()
                next_label = tf.constant(0, dtype=tf.int8)

            # decode jpeg, resize and encode as png
            image_decoded = tf.image.decode_jpeg(tf.squeeze(original_image_bytes), 3)
            min_resize_side = min(self._target_height, self._target_width)
            image_resized = CompressionPreprocessing.preprocess_image_for_eval(
                image_decoded, self._target_height, self._target_width, min_resize_side, tf.uint8)
            next_original_png_bytes = tf.image.encode_png(image_resized)

        bpp, height, width = 0.0, 0.0, 0.0
        num_images = 0

        start_time = time.time()

        with RecordsWriter(self._target_records_file) as records_writer:
            with PyBPG.remove_files_after_action([self._temp_bpg_file, self._temp_png_file]):
                with tf.Session(graph=graph) as sess:
                    try:
                        while True:
                            image_bytes, image_label = sess.run([next_original_png_bytes, next_label])

                            # write bytes to temporary file for bpgenc
                            with tempfile.NamedTemporaryFile(dir=self._target_dir) as temp_file:
                                temp_file.write(image_bytes)
                                temp_file.flush()
                                bpg_meta_data = PyBPG.encode_as_bpg(image_file=temp_file.name,
                                                                    tmp_bpg_file=self._temp_bpg_file,
                                                                    quantization_level=self._quantization_level)

                            if bpg_meta_data is None:
                                self._logger.warning(
                                    'error occurred while processing bytes starting with\n     {}'.format(
                                        image_bytes[:30]))
                                continue

                            compressed_image_png_bytes = PyBPG.decode_bpg_as_png(tmp_bpg_file=self._temp_bpg_file,
                                                                                 final_png_file=self._temp_png_file)

                            records_writer.write(feature={'image_bytes': bytes_feature(compressed_image_png_bytes),
                                                          'label': int64_feature(image_label),
                                                          'bpp': float_feature(float(bpg_meta_data.bpp)),
                                                          'height': int64_feature(int(bpg_meta_data.height)),
                                                          'width': int64_feature(int(bpg_meta_data.width))})

                            num_images += 1
                            bpp += float(bpg_meta_data.bpp)
                            height += int(bpg_meta_data.height)
                            width += int(bpg_meta_data.width)

                            images_per_second = num_images / (time.time() - start_time)
                            progress_v2(num_images, 'images processed', '{:.3f} imgs/s'.format(images_per_second))

                    except tf.errors.OutOfRangeError:
                        pass

                    except KeyboardInterrupt:
                        self._logger.warning('manual interrupt occured')

                    except Exception as e:
                        self._logger.warning('Exception occured:\n     {}'.format(e))

        # log specs
        self._logger.info('################# STATS #################')
        self._logger.info('num_images: {}'.format(num_images))
        self._logger.info('avg_bpp: {}'.format(bpp / float(num_images)))
        self._logger.info('avg_height: {}'.format(height / float(num_images)))
        self._logger.info('avg_width: {}'.format(width / float(num_images)))


class RecordsWriter:

    def __init__(self, records_path):
        self._writer = tf.python_io.TFRecordWriter(records_path)

    def __enter__(self):
        return self

    def write(self, feature):
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example.SerializeToString())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._writer.flush()
        self._writer.close()


class RecordsReader:
    JPEG_FORMAT = 'jpeg'
    PNG_FORMAT = 'png'

    def __init__(self, records_file, records_type=RecordsParser.RECORDS_LABELLED, num_parallel_batches=4, batch_size=1):
        assert records_type in [RecordsParser.RECORDS_LABELLED, RecordsParser.RECORDS_UNLABELLED]
        assert os.path.isfile(records_file)

        self._records_file = records_file
        self._records_type = records_type
        self._num_parallel_batches = num_parallel_batches
        self._batch_size = batch_size

        self._init_dataset()

    def _init_dataset(self):
        with tf.device(CPU_DEVICE):
            dataset = tf.data.TFRecordDataset(self._records_file)
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                self._read_example,
                batch_size=self._batch_size,
                num_parallel_batches=self._num_parallel_batches,
                drop_remainder=True  # -> static batch_size
            ))
            dataset = dataset.prefetch(4 * self._num_parallel_batches * self._batch_size)
            self._iterator = dataset.make_one_shot_iterator()

    def _read_example(self, example):
        features = RecordsParser.parse_example(example, self._records_type)

        outputs = []

        # decode image bytes
        image_bytes = features[RecordsParser.KW_IMAGE_BYTES]
        outputs.append(image_bytes)

        # label
        if self._records_type == RecordsParser.RECORDS_LABELLED:
            label = features[RecordsParser.KW_LABEL]
            label = tf.cast(label, tf.int32)
            label.set_shape(shape=())
            outputs.append(label)

        return outputs

    def next_element(self):
        return self._iterator.get_next()
