import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import glob
import numpy as np
import time

import tensorflow as tf

from src.compression.distortions.distortions import Distortions
from src.data.dataloading import InputPipeline
from src.data.dataloading import RecordsParser
from src.data.dataloading.preprocessing import CompressionPreprocessing
from src.data.datasets import Cub200, StanfordDogs, Imagenet, Kodak
from src.eval.hvs import EvalMSSSIMBase
from src.lib.logging_commons.utils import progress
from src.lib.tf_commons.utils import get_sess_config

CPU_DEVICE = '/cpu:0'
GPU_DEVICE = '/gpu:'


class EvalMSSSIMBpg(EvalMSSSIMBase):
    NUM_PREPROCESSING_THREADS = 16

    def __init__(self, dataset_name, bpg_records_dir, original_records, batch_size):
        logger, logfile = self._get_logger(dataset_name, 'bpg')
        super(EvalMSSSIMBpg, self).__init__(dataset_name, logger, original_records)

        self._bpg_records_files = glob.glob(os.path.join(bpg_records_dir, '*.tfrecords'))
        self._num_compression_levels = len(self._bpg_records_files)
        self._batch_size = batch_size

        if dataset_name == 'cub200':
            self._dataset = Cub200
            self._image_height = self.DEFAULT_SIZE
            self._image_width = self.DEFAULT_SIZE

        elif dataset_name == 'stanford_dogs':
            self._dataset = StanfordDogs
            self._image_height = self.DEFAULT_SIZE
            self._image_width = self.DEFAULT_SIZE

        elif dataset_name == 'imagenet':
            self._dataset = Imagenet
            self._image_height = self.DEFAULT_SIZE
            self._image_width = self.DEFAULT_SIZE

        elif dataset_name == 'kodak':
            self._dataset = Kodak
            self._image_height = Kodak.IMAGE_HEIGHT
            self._image_width = Kodak.IMAGE_WIDTH

        else:
            raise ValueError('unknown dataset {}'.format(dataset_name))

        self.logger.info('logfile={}'.format(logfile))
        self.logger.info('compression_method=bpg')
        self.logger.info('batch_size={}'.format(self._batch_size))
        self.logger.info('dataset_name={}'.format(dataset_name))
        self.logger.info('bpg_records={}'.format(self._bpg_records_files))

    def run(self):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start =====')

        with tf.Graph().as_default() as graph:

            # datafeed
            self.logger.info('* datafeed')

            ip0 = InputPipeline(records=self._records_file,
                                records_type=RecordsParser.RECORDS_UNLABELLED,
                                shuffle_buffer_size=0,
                                batch_size=self._batch_size,
                                num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                num_repeat=1,
                                preprocessing_fn=CompressionPreprocessing.preprocess_image_for_eval,
                                preprocessing_kwargs={'height': self._image_height, 'width': self._image_width,
                                                      'resize_side_min': min(self._image_height, self._image_width)},
                                drop_remainder=True,
                                compute_bpp=False,
                                shuffle=False)

            original_images = ip0.next_batch()[0]

            image_batches, bpp_op_per_compression = [], []
            for records in self._bpg_records_files:
                ip = InputPipeline(records=records,
                                   records_type=RecordsParser.RECORDS_BPP,
                                   shuffle_buffer_size=0,
                                   batch_size=self._batch_size,
                                   num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                   num_repeat=1,
                                   preprocessing_fn=CompressionPreprocessing.preprocess_image_with_identity,
                                   preprocessing_kwargs={'height': self._image_height, 'width': self._image_width,
                                                         'dtype_out': tf.uint8},
                                   drop_remainder=True,
                                   compute_bpp=False,
                                   shuffle=False)

                images, bpp = ip.next_batch()

                image_batches.append(images)
                bpp_op_per_compression.append(bpp)

            # compute distortions
            self.logger.info('* distortions')
            distortions_obj_per_compression = [Distortions(
                reconstructed_images=c_img_batch,
                original_images=original_images,
                lambda_ms_ssim=1.0,
                lambda_psnr=1.0,
                lambda_feature_loss=1.0,
                data_format=self.DATA_FORMAT,
                loss_net_kwargs=None) for c_img_batch in image_batches]

            distortions_ops_per_compression = [{'ms_ssim': d.compute_ms_ssim()} for d in
                                               distortions_obj_per_compression]

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=True), graph=graph) as sess:

            distortions_values_per_compression = [{key: list() for key in self.DISTORTION_KEYS} for _ in
                                                  range(self._num_compression_levels)]
            bpp_values_per_compression = [list() for _ in range(self._num_compression_levels)]
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, Cub200.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # compute distortions and bpp
                    batch_bpp_values_per_compression, batch_distortions_values_per_compression = sess.run(
                        [bpp_op_per_compression, distortions_ops_per_compression])

                    # collect values
                    for comp_level, (dist_comp, bpp_comp) in enumerate(
                            zip(batch_distortions_values_per_compression, batch_bpp_values_per_compression)):
                        bpp_values_per_compression[comp_level].extend(bpp_comp)
                        for key in self.DISTORTION_KEYS:
                            distortions_values_per_compression[comp_level][key].append(dist_comp[key])

                    n_images_processed += len(batch_bpp_values_per_compression[0])
                    n_images_processed_per_second.append(
                        len(batch_bpp_values_per_compression[0]) / (time.time() - batch_start_time))

                    progress(n_images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 n_images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(n_images_processed))

            except KeyboardInterrupt:
                self.logger.info(
                    'manual interrupt; processed {}/{} images'.format(n_images_processed, self._dataset.NUM_VAL))
                return [(np.nan, np.nan) for _ in range(self._num_compression_levels)]

            mean_bpp_values_per_compression = [np.mean(bpp_vals) for bpp_vals in bpp_values_per_compression]
            mean_dist_values_per_compression = [{key: np.mean(arr) for key, arr in dist_dict.items()} for dist_dict in
                                                distortions_values_per_compression]

            self._save_results(mean_bpp_values_per_compression, mean_dist_values_per_compression, 'bpg', None)
