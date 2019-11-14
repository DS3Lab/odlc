import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import json
import numpy as np
import time

import tensorflow as tf

from src.compression.distortions.distortions import Distortions
from src.compression.rnn import RNNCompressionModel
from src.data.dataloading import InputPipeline
from src.data.dataloading import RecordsParser
from src.data.datasets import Cub200, StanfordDogs, Imagenet, Kodak
from src.eval.hvs import EvalMSSSIMBase
from src.lib.logging_commons.utils import progress
from src.lib.tf_commons.utils import get_sess_config

CPU_DEVICE = '/cpu:0'
GPU_DEVICE = '/gpu:'


class EvalMSSSIMRnn(EvalMSSSIMBase):
    NUM_PREPROCESSING_THREADS = 16

    def __init__(self, dataset_name, records_file, rnn_checkpoint_and_config_dir, batch_size):

        # rnn specs
        self._rnn_checkpoint = tf.train.latest_checkpoint(os.path.join(rnn_checkpoint_and_config_dir, 'checkpoints/'))
        model_config = self.get_configs(os.path.join(rnn_checkpoint_and_config_dir, 'config.json'), key='model')
        self._rnn_unit = model_config['rnn_unit']
        self._num_iterations = model_config['num_iterations']
        self._rec_model = model_config['rec_model']
        self._loss_name = model_config['loss_name']

        logger, logfile = self._get_logger(dataset_name, self._rnn_unit + '_' + self._loss_name)
        super(EvalMSSSIMRnn, self).__init__(dataset_name, logger, records_file)
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
        self.logger.info('compression_method=rnn')
        self.logger.info('rnn_unit={}'.format(self._rnn_unit))
        self.logger.info('num_iterations={}'.format(self._num_iterations))
        self.logger.info('batch_size={}'.format(self._batch_size))
        self.logger.info('rec_model={}'.format(self._rec_model))
        self.logger.info('loss_name={}'.format(self._loss_name))
        self.logger.info('rnn_checkpoint={}'.format(self._rnn_checkpoint))
        self.logger.info('dataset_name={}'.format(dataset_name))
        self.logger.info('records_file={}'.format(records_file))

    @staticmethod
    def get_configs(config_file, key=None):
        with open(config_file, 'r') as config_data:
            config = json.load(config_data)
        if key is None:
            return config
        else:
            return config[key]

    def _get_rnn_model(self, image_height, image_width):
        return RNNCompressionModel(rnn_type=self._rnn_unit,
                                   image_height=image_height,
                                   image_width=image_width,
                                   num_iterations=self._num_iterations,
                                   rec_model=self._rec_model,
                                   data_format=self.DATA_FORMAT)

    def run(self):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start =====')

        # datafeed
        self.logger.info('* datafeed')

        with tf.Graph().as_default() as graph:

            image_height_compression, image_width_compression, _ = RNNCompressionModel.pad_image_shape(
                image_shape=[self._image_height, self._image_width, 3])

            rnn_model = self._get_rnn_model(image_height_compression, image_width_compression)

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
            if image_height_compression != self._image_height or image_width_compression != self._image_width:
                images = tf.image.resize_image_with_crop_or_pad(images, image_height_compression,
                                                                image_width_compression)

            num_images_in_batch_op = tf.shape(images)[0]
            self.logger.info('images shape for compression: {}'.format(images.get_shape().as_list()))

            # compress images
            self.logger.info('* compression')
            images_compressed = rnn_model.build_model(images=images,
                                                      is_training=tf.cast(False, tf.bool),
                                                      reuse=tf.get_variable_scope().reuse)
            images_compressed.set_shape(
                [self._num_iterations, self._batch_size, image_height_compression, image_width_compression, 3])
            self.logger.info('compressed images shape: {}'.format(images_compressed.get_shape().as_list()))

            # compute distortions
            self.logger.info('* distortions')
            distortions_obj_per_compression = [Distortions(
                reconstructed_images=tf.image.resize_image_with_crop_or_pad(image=images_compressed[ii],
                                                                            target_width=self._image_width,
                                                                            target_height=self._image_height),
                original_images=tf.cast(images, tf.float32),
                lambda_ms_ssim=1.0,
                lambda_psnr=1.0,
                lambda_feature_loss=1.0,
                data_format=self.DATA_FORMAT,
                loss_net_kwargs=None) for ii in range(self._num_iterations)]

            distortions_ops_per_compression = [{'ms_ssim': d.compute_ms_ssim()} for d in
                                               distortions_obj_per_compression]

            # savers
            rnn_saver = tf.train.Saver(var_list=rnn_model.model_variables)

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=True), graph=graph) as sess:

            rnn_saver.restore(sess, self._rnn_checkpoint)

            distortions_values_per_compression = [{key: list() for key in self.DISTORTION_KEYS} for _ in
                                                  range(self._num_iterations)]
            bpp_values_per_compression = [list() for _ in range(self._num_iterations)]
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, Cub200.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # compute distortions and bpp
                    batch_distortions_values_per_compression, num_images_in_batch = sess.run(
                        [distortions_ops_per_compression, num_images_in_batch_op])

                    # collect values
                    for comp_level, dist_comp in enumerate(batch_distortions_values_per_compression):
                        bpp_values_per_compression[comp_level].extend([0.125 * (comp_level + 1)])
                        for key in self.DISTORTION_KEYS:
                            distortions_values_per_compression[comp_level][key].append(dist_comp[key])

                    n_images_processed += num_images_in_batch
                    n_images_processed_per_second.append(num_images_in_batch / (time.time() - batch_start_time))

                    progress(n_images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 n_images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(n_images_processed))

            except KeyboardInterrupt:
                self.logger.info(
                    'manual interrupt; processed {}/{} images'.format(n_images_processed, self._dataset.NUM_VAL))
                return

            mean_bpp_values_per_compression = [np.mean(bpp_vals) for bpp_vals in bpp_values_per_compression]
            mean_dist_values_per_compression = [{key: np.mean(arr) for key, arr in dist_dict.items()} for dist_dict in
                                                distortions_values_per_compression]

            self._save_results(mean_bpp_values_per_compression, mean_dist_values_per_compression,
                               self._rnn_unit + '_' + self._loss_name, [q + 1 for q in range(self._num_iterations)])
