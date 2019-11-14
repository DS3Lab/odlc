import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import json
import numpy as np
import time

import tensorflow as tf

from src.compression.distortions.distortions import Distortions
from src.compression.cond_prob_models import autoencoder, probclass, config_parser, logdir_helpers
from src.compression.cond_prob_models.saver import Saver
from src.compression.cond_prob_models.constants import CONFIG_BASE_AE, CONFIG_BASE_PC
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.datasets import Cub200, StanfordDogs, Imagenet, Kodak
from src.eval.hvs import EvalMSSSIMBase
from src.lib.logging_commons.utils import progress
from src.lib.tf_commons.utils import get_sess_config, nhwc_to_nchw, nchw_to_nhwc

_CPU_DEVICE = '/cpu:0'


def bitcost_to_bpp(bit_cost, input_batch):
    """
    :param bit_cost: NChw
    :param input_batch: N3HW
    :return: Chw / HW, i.e., num_bits / num_pixels
    """
    assert bit_cost.shape.ndims == input_batch.shape.ndims == 4, 'Expected NChw and N3HW, got {} and {}'.format(
        bit_cost, input_batch)
    with tf.name_scope('bitcost_to_bpp'):
        num_bits = tf.reduce_sum(bit_cost, name='num_bits')
        return num_bits / tf.to_float(num_pixels_in_input_batch(input_batch))


def num_pixels_in_input_batch(input_batch):
    assert int(input_batch.shape[1]) == 3, 'Expected N3HW, got {}'.format(input_batch)
    with tf.name_scope('num_pixels'):
        return tf.reduce_prod(tf.shape(input_batch)) / 3


class EvalHVSCPM(EvalMSSSIMBase):
    NUM_PREPROCESSING_THREADS = 16
    DATA_FORMAT = 'NHWC'
    BATCH_SIZE = 32
    DISTORTION_KEYS = ['ms_ssim', 'mse', 'psnr']

    def __init__(self, dataset_name, records_file, cpm_log_dir, job_ids, batch_size):
        logger, logfile = self._get_logger(dataset_name, 'cpm')
        super(EvalHVSCPM, self).__init__(dataset_name, logger, records_file)
        self._batch_size = batch_size

        # configs and checkpoints
        ckpt_dirs = list(logdir_helpers.iter_ckpt_dirs(cpm_log_dir, job_ids))
        self._checkpoints_and_configs = []
        self._job_ids = job_ids.split(',')
        for ckpt in ckpt_dirs:
            # build paths
            ae_config_path, pc_config_path = logdir_helpers.config_paths_from_log_dir(
                log_dir=Saver.log_dir_from_ckpt_dir(ckpt), base_dirs=[CONFIG_BASE_AE, CONFIG_BASE_PC])

            # parse configs
            ae_config, _ = config_parser.parse(ae_config_path)
            pc_config, _ = config_parser.parse(pc_config_path)

            self._checkpoints_and_configs.append(tuple([ckpt, ae_config, pc_config]))

        # Datasets
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
        self.logger.info('compression_method=cond_prob_models')
        self.logger.info('batch_size={}'.format(self._batch_size))
        self.logger.info('dataset_name={}'.format(dataset_name))
        self.logger.info('records_file={}'.format(records_file))
        self.logger.info('job_ids={}'.format(job_ids))
        self.logger.info('*** ckpts etc:\n{}'.format(self._checkpoints_and_configs))

    def run(self):
        """ imagenet evaluation loop """
        distortions_values = []
        bpp_values = []

        for cpm_ckpt, ae_config, pc_config in self._checkpoints_and_configs:
            distortions_dict, bpp_mean = self.run0(ae_config=ae_config, pc_config=pc_config, cpm_checkpoint=cpm_ckpt)

            distortions_values.append(distortions_dict)  # list of dicts
            bpp_values.append(bpp_mean)

        self._save_results(bpp_values, distortions_values, 'cpm', self._job_ids)

    @staticmethod
    def load_config(config_file, key=None):

        with open(config_file, 'r') as config_data:
            config = json.load(config_data)

        if key is None:
            return config
        else:
            return config[key]

    def run0(self, ae_config, pc_config, cpm_checkpoint):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start =====')

        with tf.Graph().as_default() as graph:
            # datafeed
            self.logger.info('* datafeed')

            input_pipeline = InputPipeline(records=self.records_file,
                                           records_type=RecordsParser.RECORDS_UNLABELLED,
                                           shuffle_buffer_size=self._batch_size * self.NUM_PREPROCESSING_THREADS,
                                           batch_size=self._batch_size,
                                           num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                           num_repeat=1,
                                           preprocessing_fn=self._get_resize_function(self._image_height,
                                                                                      self._image_width),
                                           preprocessing_kwargs={},
                                           drop_remainder=True,
                                           compute_bpp=False,
                                           shuffle=False, dtype_out=tf.float32)

            images = input_pipeline.next_batch()[0]

            # compression + inference op
            self.logger.info('* compression')
            with tf.name_scope('compression'):
                print(images.get_shape().as_list())
                images = nhwc_to_nchw(images)

                # create networks
                ae_cls = autoencoder.get_network_cls(ae_config)
                pc_cls = probclass.get_network_cls(pc_config)

                # instantiate models
                ae = ae_cls(ae_config)
                pc = pc_cls(pc_config, num_centers=ae_config.num_centers)

                enc_out_val = ae.encode(images, is_training=False)
                images_compressed = ae.decode(enc_out_val.qhard, is_training=False)

                bitcost_val = pc.bitcost(enc_out_val.qbar, enc_out_val.symbols, is_training=False,
                                         pad_value=pc.auto_pad_value(ae))
                avg_bits_per_pixel = bitcost_to_bpp(bitcost_val, images)
                images = nchw_to_nhwc(images)
                images_compressed = nchw_to_nhwc(images_compressed)

            # compute distortions
            self.logger.info('* distortions')
            with tf.name_scope('distortions'):
                distortions_obj = Distortions(
                    reconstructed_images=images_compressed,
                    original_images=tf.cast(images, tf.float32),
                    lambda_ms_ssim=1.0,
                    lambda_psnr=1.0,
                    lambda_feature_loss=1.0,
                    data_format=self.DATA_FORMAT,
                    loss_net_kwargs=None)

                distortions_ops = {'ms_ssim': distortions_obj.compute_ms_ssim(),
                                   'mse': distortions_obj.compute_mse(),
                                   'psnr': distortions_obj.compute_psnr()}

            # cpm saver
            cpm_saver = Saver(cpm_checkpoint, var_list=Saver.get_var_list_of_ckpt_dir(cpm_checkpoint))
            ckpt_itr, cpm_ckpt_path = Saver.all_ckpts_with_iterations(cpm_checkpoint)[-1]
            self.logger.info('ckpt_itr={}'.format(ckpt_itr))
            self.logger.info('ckpt_path={}'.format(cpm_ckpt_path))

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=False), graph=graph) as sess:

            cpm_saver.restore_ckpt(sess, cpm_ckpt_path)

            distortions_values = {key: list() for key in self.DISTORTION_KEYS}
            bpp_values = []
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, self._dataset.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # compute distortions and bpp
                    batch_bpp_mean_values, batch_distortions_values = sess.run([avg_bits_per_pixel, distortions_ops])

                    # collect values
                    bpp_values.append(batch_bpp_mean_values)
                    for key in self.DISTORTION_KEYS:
                        distortions_values[key].append(batch_distortions_values[key])

                    n_images_processed += self._batch_size
                    n_images_processed_per_second.append(self._batch_size / (time.time() - batch_start_time))

                    progress(n_images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 n_images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(n_images_processed))

            except KeyboardInterrupt:
                self.logger.info(
                    'manual interrupt; processed {}/{} images'.format(n_images_processed, self._dataset.NUM_VAL))

                mean_bpp_values = np.mean(bpp_values)
                mean_dist_values = {key: np.mean(arr) for key, arr in distortions_values.items()}

                print('*** intermediate results:')
                print('bits per pixel: {}'.format(mean_bpp_values))
                for key in self.DISTORTION_KEYS:
                    print('{}: {}'.format(key, mean_dist_values[key]))

                return {key: np.nan for key in self.DISTORTION_KEYS}, np.nan

        mean_bpp_values = np.mean(bpp_values)
        mean_dist_values = {key: np.mean(arr) for key, arr in distortions_values.items()}

        return mean_dist_values, mean_bpp_values
