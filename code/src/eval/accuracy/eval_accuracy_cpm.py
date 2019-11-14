import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa

from collections import deque
import json
import numpy as np
import time

import tensorflow as tf

from src.classification.imagenet import ImagenetClassifier
from src.classification.fine_grained_categorization import FGVCClassifier
from src.compression.cond_prob_models import autoencoder, probclass, config_parser, logdir_helpers
from src.compression.cond_prob_models.saver import Saver
from src.compression.cond_prob_models.constants import CONFIG_BASE_AE, CONFIG_BASE_PC
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.dataloading.preprocessing import CompressionPreprocessing
from src.data.datasets import Imagenet
from src.eval.accuracy import EvalAccuracyBase
from src.lib.logging_commons.utils import progress, write_to_csv
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


class EvalAccuracyCPM(EvalAccuracyBase):
    NUM_PREPROCESSING_THREADS = 16
    DATA_FORMAT = 'NHWC'
    BATCH_SIZE = 32

    def __init__(self, dataset_name, records_file, cpm_log_dir, job_ids, fgvc_checkpoint_root=None):
        logger = self._get_logger(dataset_name, 'cpm')
        super(EvalAccuracyCPM, self).__init__(dataset_name, records_file, logger, fgvc_checkpoint_root)

        # configs and checkpoints
        ckpt_dirs = list(logdir_helpers.iter_ckpt_dirs(cpm_log_dir, job_ids))
        self._checkpoints_and_configs = []
        for ckpt in ckpt_dirs:
            # build paths
            ae_config_path, pc_config_path = logdir_helpers.config_paths_from_log_dir(
                log_dir=Saver.log_dir_from_ckpt_dir(ckpt), base_dirs=[CONFIG_BASE_AE, CONFIG_BASE_PC])

            # parse configs
            ae_config, _ = config_parser.parse(ae_config_path)
            pc_config, _ = config_parser.parse(pc_config_path)

            self._checkpoints_and_configs.append(tuple([ckpt, ae_config, pc_config]))

        self.logger.info('compression_method=cond_prob_models')
        self.logger.info('fgvc_checkpoint_root={}'.format(fgvc_checkpoint_root))
        self.logger.info('job_ids={}'.format(job_ids))
        self.logger.info('batch_size={}'.format(self.BATCH_SIZE))
        self.logger.info('*** ckpts etc:\n{}'.format(self._checkpoints_and_configs))

    def run0(self):
        """ imagenet evaluation loop """

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.imagenet_cnn_model_names]))

        for cnn_model_name in self.imagenet_cnn_model_names:

            accuracies = []
            bpp_values = []

            for cpm_ckpt, ae_config, pc_config in self._checkpoints_and_configs:
                accuracy, bpp_value = self.eval_classifier_model(ae_config=ae_config,
                                                                 pc_config=pc_config,
                                                                 cpm_checkpoint=cpm_ckpt,
                                                                 cnn_model=self.get_classifier_instance(cnn_model_name))

                accuracies.append(accuracy)  # list of tuples (top1, top5)
                bpp_values.append(bpp_value)

            self._save_results(bpp_values, accuracies, cnn_model_name)

    def run1(self):
        """ FGVC evaluation loop """

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.fgvc_cnn_model_names]))

        for cnn_model_name in self.fgvc_cnn_model_names:
            fgvc_checkpoint = self.get_fgvc_checkpoint(cnn_model_name)

            accuracies = []
            bpp_values = []

            if fgvc_checkpoint is None:
                continue

            for cpm_ckpt, ae_config, pc_config in self._checkpoints_and_configs:
                accuracy, bpp_value = self.eval_classifier_model(ae_config=ae_config,
                                                                 pc_config=pc_config,
                                                                 cpm_checkpoint=cpm_ckpt,
                                                                 cnn_model=self.get_classifier_instance(cnn_model_name),
                                                                 fgvc_ckpt_path=fgvc_checkpoint)

                accuracies.append(accuracy)  # list of tuples (top1, top5)
                bpp_values.append(bpp_value)

            self._save_results(bpp_values, accuracies, cnn_model_name)

    def _save_results(self, bpp, accs, cnn_model_name):
        # write results to csv
        csv_file = os.path.join(self._log_dir, 'cpm_{}_accuracy_{}.csv'.format(cnn_model_name, self._log_number))
        write_to_csv(csv_file, [(b, *a) for b, a in zip(bpp, accs)])

        # log results
        accuracy_dicts = [{cnn_model_name: acc} for acc in accs]
        self.log_results(self.pack_eval_values(bits_per_pixel=bpp, accuracy_dicts=accuracy_dicts))

    @staticmethod
    def load_config(config_file, key=None):

        with open(config_file, 'r') as config_data:
            config = json.load(config_data)

        if key is None:
            return config
        else:
            return config[key]

    def eval_classifier_model(self, ae_config, pc_config, cpm_checkpoint,
                              cnn_model: any([ImagenetClassifier, FGVCClassifier]), fgvc_ckpt_path=None):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start: {} ====='.format(cnn_model.NAME))

        # assertions
        assert self._dataset.NUM_CLASSES == cnn_model.num_classes, 'incostent number of classes ({} != {})'.format(
            self._dataset.NUM_CLASSES, cnn_model.num_classes)

        # image shapes
        image_shape_classification = cnn_model.INPUT_SHAPE
        image_shape_compression = CompressionPreprocessing.pad_image_shape(image_shape=image_shape_classification,
                                                                           size_multiple_of=self.SIZE_MULTIPLE_OF,
                                                                           extra_padding_multiples=2)

        # log image sizes
        self.logger.info('image_shape_classification={}'.format(image_shape_classification))
        self.logger.info('image_shape_compression={}'.format(image_shape_compression))

        with tf.Graph().as_default() as graph:
            # datafeed
            self.logger.info('* datafeed')

            input_pipeline = InputPipeline(records=self.records_file,
                                           records_type=RecordsParser.RECORDS_LABELLED,
                                           shuffle_buffer_size=self.BATCH_SIZE * self.NUM_PREPROCESSING_THREADS,
                                           batch_size=self.BATCH_SIZE,
                                           num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                           num_repeat=1,
                                           preprocessing_fn=CompressionPreprocessing.preprocess_image,
                                           preprocessing_kwargs={'height': image_shape_compression[0],
                                                                 'width': image_shape_compression[1],
                                                                 'resize_side_min': min(image_shape_compression[:2]),
                                                                 'is_training': False,
                                                                 'dtype_out': tf.uint8},
                                           drop_remainder=False,
                                           compute_bpp=False,
                                           shuffle=False, dtype_out=tf.float32)

            images, labels = input_pipeline.next_batch()

            # compression + inference op
            self.logger.info('* compression')
            with tf.name_scope('compression'):

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
                images_compressed = nchw_to_nhwc(images_compressed)

                # inference kwargs
            self.logger.info('* inference')
            if self._dataset_name == Imagenet.NAME:
                def inference_kwargs(**kwargs):
                    return dict(graph=kwargs['graph'])
            else:
                def inference_kwargs(**kwargs):
                    return dict(arg_scope=cnn_model.arg_scope(weight_decay=float(0)),
                                is_training=False,
                                return_predictions=True,
                                reuse=None)

            with tf.name_scope('inference_rnn'):

                # take central crop of images in batch
                images_compressed = tf.image.resize_image_with_crop_or_pad(
                    image=images_compressed,
                    target_height=image_shape_classification[0],
                    target_width=image_shape_classification[1])

                # standardize appropriately
                images_compressed = cnn_model.standardize_tensor(
                    images_compressed)

                # predict
                predictions = cnn_model.inference(images_compressed, **inference_kwargs(graph=graph))

                # aggregate
                self.logger.info('predictions_shape: {}'.format(predictions.get_shape().as_list()))

            # restorers
            if self._dataset_name == Imagenet.NAME:
                classifier_saver = None
            else:
                classifier_saver = tf.train.Saver(var_list=cnn_model.model_variables())

            # cpm saver
            cpm_saver = Saver(cpm_checkpoint, var_list=Saver.get_var_list_of_ckpt_dir(cpm_checkpoint))
            ckpt_itr, cpm_ckpt_path = Saver.all_ckpts_with_iterations(cpm_checkpoint)[-1]
            self.logger.info('ckpt_itr={}'.format(ckpt_itr))
            self.logger.info('ckpt_path={}'.format(cpm_ckpt_path))

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=False), graph=graph) as sess:

            cpm_saver.restore_ckpt(sess, cpm_ckpt_path)

            if classifier_saver is not None:
                classifier_saver.restore(sess, fgvc_ckpt_path)

            labels_values = []
            predictions_values = []
            bpp_values = []
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, self._dataset.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # run inference
                    batch_predictions_values, batch_label_values, batch_avg_bpp_values = sess.run(
                        [predictions, labels, avg_bits_per_pixel])

                    # collect predictions
                    predictions_values.append(batch_predictions_values)

                    # collect labels and bpp
                    labels_values.append(self.to_categorical(batch_label_values, Imagenet.NUM_CLASSES))
                    bpp_values.append(batch_avg_bpp_values)

                    n_images_processed += len(batch_label_values)
                    n_images_processed_per_second.append(len(batch_label_values) / (time.time() - batch_start_time))

                    progress(n_images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 n_images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(n_images_processed))

            except KeyboardInterrupt:
                self.logger.info(
                    'manual interrupt; processed {}/{} images'.format(n_images_processed, self._dataset.NUM_VAL))

                labels_values = np.concatenate(labels_values, axis=0)
                predictions_values = np.concatenate(predictions_values, axis=0)
                bpp_values_mean = np.mean(bpp_values)

                accuracies = (self.top_k_accuracy(labels_values, predictions_values, 1),
                              self.top_k_accuracy(labels_values, predictions_values, 5))

                print('*** intermediate results:')
                print('bits per pixel: {}'.format(bpp_values_mean))
                print('Top-1 Accuracy: {}'.format(accuracies[0]))
                print('Top-5 Accuracy: {}'.format(accuracies[1]))

                return (np.nan, np.nan), np.nan

        labels_values = np.concatenate(labels_values, axis=0)
        predictions_values = np.concatenate(predictions_values, axis=0)
        bpp_values_mean = np.mean(bpp_values)

        accuracies = (self.top_k_accuracy(labels_values, predictions_values, 1),
                      self.top_k_accuracy(labels_values, predictions_values, 5))

        return accuracies, bpp_values_mean
