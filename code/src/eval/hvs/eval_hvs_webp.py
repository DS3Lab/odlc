import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import numpy as np
import time

import tensorflow as tf

from src.compression.distortions.distortions import Distortions
from src.data.dataloading import InputPipeline
from src.data.dataloading import RecordsParser
from src.data.datasets import Cub200, StanfordDogs, Imagenet, Kodak
from src.compression.webp.tf_webp import TFWebp
from src.eval.hvs import EvalMSSSIMBase
from src.lib.logging_commons.utils import progress
from src.lib.tf_commons.utils import get_sess_config

CPU_DEVICE = '/cpu:0'
GPU_DEVICE = '/gpu:'


class EvalMSSSIMWebp(EvalMSSSIMBase):
    COMPRESSION_LEVELS = [1, 5, 10, 15, 20, 30, 50, 70, 95]
    NUM_PREPROCESSING_THREADS = 16

    def __init__(self, dataset_name, records_file, batch_size):
        # rnn specs
        logger, logfile = self._get_logger(dataset_name, 'webp')
        super(EvalMSSSIMWebp, self).__init__(dataset_name, logger, records_file)

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
        self.logger.info('compression_method=webp')
        self.logger.info('batch_size={}'.format(self._batch_size))
        self.logger.info('dataset_name={}'.format(dataset_name))
        self.logger.info('records_file={}'.format(records_file))

    def run(self):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start =====')

        with tf.Graph().as_default() as graph:
            # datafeed
            self.logger.info('* datafeed')

            input_pipeline = InputPipeline(records=self._records_file,
                                           records_type=RecordsParser.RECORDS_UNLABELLED,
                                           shuffle_buffer_size=0,
                                           batch_size=self._batch_size,
                                           num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                           num_repeat=1,
                                           preprocessing_fn=self._get_resize_function(self._image_height,
                                                                                      self._image_width),
                                           preprocessing_kwargs={},
                                           drop_remainder=True,
                                           compute_bpp=False,
                                           shuffle=False)

            images = input_pipeline.next_batch()[0]

            image_shape = images.get_shape().as_list()
            self.logger.info('image_shape: {}'.format(image_shape))

            # compression op
            self.logger.info('* compression')

            images_per_compression = []
            bpp_op_per_compression = []
            for j, compression_level in enumerate(self.COMPRESSION_LEVELS):
                # compress batch
                with tf.name_scope('compression_webp_{}'.format(compression_level)):
                    with tf.device(CPU_DEVICE):  # -> webp compression on cpu
                        img_batch_compressed, _bpp = TFWebp.tf_encode_decode_image_batch(
                            image_batch=tf.cast(images, tf.uint8),
                            quality=compression_level)

                    img_batch_compressed.set_shape(images.get_shape().as_list())
                    images_per_compression.append(tf.cast(img_batch_compressed, tf.float32))
                    bpp_op_per_compression.append(_bpp)

            # compute distortions
            self.logger.info('* distortions')
            distortions_obj_per_compression = [Distortions(
                reconstructed_images=c_img_batch,
                original_images=tf.cast(images, tf.float32),
                lambda_ms_ssim=1.0,
                lambda_psnr=1.0,
                lambda_feature_loss=1.0,
                data_format=self.DATA_FORMAT,
                loss_net_kwargs=None) for c_img_batch in images_per_compression]

            distortions_ops_per_compression = [{'ms_ssim': d.compute_ms_ssim()} for d in
                                               distortions_obj_per_compression]

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=True), graph=graph) as sess:

            distortions_values_per_compression = [{key: list() for key in self.DISTORTION_KEYS} for _ in
                                                  self.COMPRESSION_LEVELS]
            bpp_values_per_compression = [list() for _ in self.COMPRESSION_LEVELS]
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
                return [(np.nan, np.nan) for _ in self.COMPRESSION_LEVELS]

            mean_bpp_values_per_compression = [np.mean(bpp_vals) for bpp_vals in bpp_values_per_compression]
            mean_dist_values_per_compression = [{key: np.mean(arr) for key, arr in dist_dict.items()} for dist_dict in
                                                distortions_values_per_compression]

            self._save_results(mean_bpp_values_per_compression, mean_dist_values_per_compression, 'webp',
                               self.COMPRESSION_LEVELS)
