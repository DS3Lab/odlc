import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import json
import numpy as np
import time

import tensorflow as tf

from src.classification.imagenet import ImagenetClassifier
from src.classification.fine_grained_categorization import FGVCClassifier
from src.compression.rnn import RNNCompressionModel
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.dataloading.preprocessing import CompressionPreprocessing
from src.data.datasets import Imagenet
from src.eval.accuracy import EvalAccuracyBase
from src.lib.logging_commons.utils import progress, write_to_csv
from src.lib.tf_commons.utils import get_sess_config

_CPU_DEVICE = '/cpu:0'


class EvalAccuracyRnn(EvalAccuracyBase):
    NUM_PREPROCESSING_THREADS = 16
    DATA_FORMAT = 'NHWC'
    BATCH_SIZE = 8

    def __init__(self, dataset_name, records_file, rnn_checkpoint_and_config_dir, checkpoint_root=None):
        # rnn specs
        self._rnn_checkpoint = tf.train.latest_checkpoint(os.path.join(rnn_checkpoint_and_config_dir, 'checkpoints/'))
        model_config = self.load_config(os.path.join(rnn_checkpoint_and_config_dir, 'config.json'), key='model')
        self._rnn_unit = model_config['rnn_unit']
        self._num_iterations = model_config['num_iterations']
        self._rec_model = model_config['rec_model']
        self._loss_name = model_config['loss_name']

        logger = self._get_logger(dataset_name, self._rnn_unit + '_' + self._loss_name)
        super(EvalAccuracyRnn, self).__init__(dataset_name, records_file, logger, checkpoint_root)

        self.logger.info('compression_method=rnn')
        self.logger.info('rnn_unit={}'.format(self._rnn_unit))
        self.logger.info('num_iterations={}'.format(self._num_iterations))
        self.logger.info('batch_size={}'.format(self.BATCH_SIZE))
        self.logger.info('rec_model={}'.format(self._rec_model))
        self.logger.info('loss_name={}'.format(self._loss_name))
        self.logger.info('rnn_checkpoint={}'.format(self._rnn_checkpoint))

    def run0(self):
        """ imagenet evaluation loop """
        bpp = [0.125 * (i + 1) for i in range(self._num_iterations)]  # this is fixed for rnn compression

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.imagenet_cnn_model_names]))

        for cnn_model_name in self.imagenet_cnn_model_names:
            accs = self.eval_classifier_model(self.get_classifier_instance(cnn_model_name))
            self._save_results(bpp, accs, cnn_model_name)

    def run1(self):
        """ FGVC evaluation loop """
        bpp = [0.125 * (i + 1) for i in range(self._num_iterations)]  # this is fixed for rnn compression

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.fgvc_cnn_model_names]))

        for cnn_model_name in self.fgvc_cnn_model_names:
            model_checkpoint = self.get_fgvc_checkpoint(cnn_model_name)

            if model_checkpoint is None:
                continue

            accs = self.eval_classifier_model(cnn_model=self.get_classifier_instance(cnn_model_name),
                                              ckpt_path=model_checkpoint)
            self._save_results(bpp, accs, cnn_model_name)

    def _save_results(self, bpp, accs, cnn_model_name):
        # write results to csv
        csv_file = os.path.join(self._log_dir, '{}_{}_{}_accuracy_{}.csv'.format(self._rnn_unit,
                                                                                 self._loss_name,
                                                                                 cnn_model_name,
                                                                                 self._log_number))
        write_to_csv(csv_file, [(b, *a) for b, a in zip(bpp, accs)])

        # log results
        accuracy_dicts = [{cnn_model_name: accs[i]} for i in range(self._num_iterations)]
        self.log_results(self.pack_eval_values(bits_per_pixel=bpp, accuracy_dicts=accuracy_dicts))

    @staticmethod
    def load_config(config_file, key=None):

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

    def eval_classifier_model(self, cnn_model: any([ImagenetClassifier, FGVCClassifier]), ckpt_path=None):

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

            rnn_model = self._get_rnn_model(image_shape_compression[0], image_shape_compression[1])

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
                                           shuffle=False)

            images, labels = input_pipeline.next_batch()

            # compression + inference op
            self.logger.info('* compression')
            with tf.name_scope('rnn_compression'):
                image_batch_compressed = rnn_model.build_model(images=images,
                                                               is_training=tf.cast(False, tf.bool),
                                                               reuse=tf.get_variable_scope().reuse)

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
                                reuse=True if kwargs['j'] > 0 else False)

            predictions_per_compression = []

            with tf.name_scope('inference_rnn'):

                for rnn_iteration in range(self._num_iterations):
                    with tf.name_scope('iteration_{}'.format(rnn_iteration)):
                        image_batch_compressed_iteration = tf.cast(image_batch_compressed[rnn_iteration], tf.float32)

                        # take central crop of images in batch
                        image_batch_compressed_iteration = tf.image.resize_image_with_crop_or_pad(
                            image=image_batch_compressed_iteration,
                            target_height=image_shape_classification[0],
                            target_width=image_shape_classification[1])

                        # standardize appropriately
                        image_batch_compressed_iteration = cnn_model.standardize_tensor(
                            image_batch_compressed_iteration)

                        # predict
                        preds = cnn_model.inference(image_batch_compressed_iteration,
                                                    **inference_kwargs(graph=graph, j=rnn_iteration))
                        predictions_per_compression.append(preds)

                # aggregate
                predictions_per_compression_op = tf.stack(predictions_per_compression, axis=0)
                self.logger.info('predictions_shape: {}'.format(predictions_per_compression_op.get_shape().as_list()))

            # restorers
            if self._dataset_name == Imagenet.NAME:
                classifier_saver = None
            else:
                classifier_saver = tf.train.Saver(var_list=cnn_model.model_variables())

            # rnn saver
            saver = tf.train.Saver(var_list=rnn_model.model_variables)

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=False), graph=graph) as sess:

            if classifier_saver is not None:
                classifier_saver.restore(sess, ckpt_path)

            saver.restore(sess, self._rnn_checkpoint)

            labels_values = []
            predictions_all_iters_values = [list() for _ in range(self._num_iterations)]
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, self._dataset.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # run inference
                    batch_predictions_all_iters_values, batch_label_values = sess.run(
                        [predictions_per_compression_op, labels])

                    # collect predictions
                    for rnn_itr, preds_itr in enumerate(batch_predictions_all_iters_values):
                        predictions_all_iters_values[rnn_itr].append(preds_itr)

                    # collect labels and bpp
                    labels_values.append(self.to_categorical(batch_label_values, Imagenet.NUM_CLASSES))

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
                    'manual interrupt; processed {}/{} images'.format(n_images_processed, Imagenet.NUM_VAL))
                return [(np.nan, np.nan) for _ in range(self._num_iterations)], [np.nan for _ in
                                                                                 range(self._num_iterations)]

        labels_values = np.concatenate(labels_values, axis=0)
        predictions_all_iters_values = [np.concatenate(preds_iter_values, axis=0) for preds_iter_values in
                                        predictions_all_iters_values]

        accuracies = [(self.top_k_accuracy(labels_values, preds_iter_values, 1),
                       self.top_k_accuracy(labels_values, preds_iter_values, 5))
                      for preds_iter_values in predictions_all_iters_values]

        return accuracies
