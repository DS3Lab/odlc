import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

from collections import deque
import glob
import numpy as np
import time

import tensorflow as tf

from src.classification.imagenet import ImagenetClassifier
from src.classification.fine_grained_categorization import FGVCClassifier
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.dataloading.preprocessing import CompressionPreprocessing
from src.data.datasets import Imagenet
from src.eval.accuracy import EvalAccuracyBase
from src.lib.logging_commons.utils import progress, write_to_csv
from src.lib.tf_commons.utils import get_sess_config


class EvalAccuracyBpg(EvalAccuracyBase):
    NUM_PREPROCESSING_THREADS = 16
    BATCH_SIZE = 16
    DATA_FORMAT = 'NHWC'

    def __init__(self, dataset_name, bpg_records_dir256, bpg_records_dir336, checkpoint_root=None):
        logger = self._get_logger(dataset_name, 'bpg')
        super(EvalAccuracyBpg, self).__init__(dataset_name, None, logger, checkpoint_root)

        # records with bpg decoded image files and bpp values per image
        self._bpg_records_files256 = glob.glob(os.path.join(bpg_records_dir256, '*.tfrecords'))
        self._bpg_records_files336 = glob.glob(os.path.join(bpg_records_dir336, '*.tfrecords'))

        assert len(self._bpg_records_files336) == len(self._bpg_records_files256)
        self._num_records = len(self._bpg_records_files256)

        self.logger.info('compression_method=bpg')
        self.logger.info('batch_size={}'.format(self.BATCH_SIZE))

    def run0(self):
        """ imagenet evaluation loop """
        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.imagenet_cnn_model_names]))

        for cnn_model_name in self.imagenet_cnn_model_names:
            accs, bpp = self.eval_classifier_model(self.get_classifier_instance(cnn_model_name))
            self._save_results(bpp, accs, cnn_model_name)

    def run1(self):
        """ FGVC evaluation loop """
        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.fgvc_cnn_model_names]))

        for cnn_model_name in self.fgvc_cnn_model_names:
            model_checkpoint = self.get_fgvc_checkpoint(cnn_model_name)

            if model_checkpoint is None:
                continue

            accs, bpp = self.eval_classifier_model(cnn_model=self.get_classifier_instance(cnn_model_name),
                                                   ckpt_path=model_checkpoint)
            self._save_results(bpp, accs, cnn_model_name)

    def _save_results(self, bpp, accs, cnn_model_name):
        # write results to csv
        csv_file = os.path.join(self._log_dir, 'bpg_{}_accuracy_{}.csv'.format(cnn_model_name, self._log_number))
        write_to_csv(csv_file, [(b, *a) for b, a in zip(bpp, accs)])

        # log results
        accuracy_dicts = [{cnn_model_name: accs[i]} for i in range(self._num_records)]
        self.log_results(self.pack_eval_values(bits_per_pixel=bpp, accuracy_dicts=accuracy_dicts))

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

        # records files depending on inference resolution
        if image_shape_classification[0] < 256:
            bpg_records_files = list(self._bpg_records_files256)
        else:
            bpg_records_files = list(self._bpg_records_files336)

        self.logger.info('bpg_records_files: {}'.format(bpg_records_files))

        with tf.Graph().as_default() as graph:

            # datafeed
            self.logger.info('* datafeed')
            image_batches, labels_batches, bpp_batches = [], [], []
            for records in bpg_records_files:
                ip = InputPipeline(records=records,
                                   records_type=RecordsParser.RECORDS_LABELLED_BPP,
                                   shuffle_buffer_size=1,
                                   batch_size=self.BATCH_SIZE,
                                   num_preprocessing_threads=self.NUM_PREPROCESSING_THREADS,
                                   num_repeat=1,
                                   preprocessing_fn=CompressionPreprocessing.preprocess_image_with_identity,
                                   preprocessing_kwargs={'height': image_shape_compression[0],
                                                         'width': image_shape_compression[1],
                                                         'dtype_out': tf.uint8},
                                   drop_remainder=True,
                                   compute_bpp=False,
                                   shuffle=False,
                                   dtype_out=tf.uint8)

                images, labels, bpp = ip.next_batch()

                image_batches.append(images)
                labels_batches.append(labels)
                bpp_batches.append(bpp)

            # compression + inference op
            self.logger.info('* inference')

            predictions_per_compression = []

            # inference kwargs
            if self._dataset_name == Imagenet.NAME:
                def inference_kwargs(**kwargs):
                    return dict(graph=kwargs['graph'])
            else:
                def inference_kwargs(**kwargs):
                    return dict(arg_scope=cnn_model.arg_scope(weight_decay=float(0)),
                                is_training=False,
                                return_predictions=True,
                                reuse=True if kwargs['j'] > 0 else False)

            for j, image_batch_compressed in enumerate(image_batches):
                with tf.name_scope('inference_bpg{}'.format(j)):
                    # crop center
                    image_batch_compressed = tf.image.resize_image_with_crop_or_pad(
                        image_batch_compressed, image_shape_classification[0], image_shape_classification[1])

                    # standardize appropriately
                    image_batch_for_inference = cnn_model.standardize_tensor(image_batch_compressed)

                    # predict
                    preds = cnn_model.inference(image_batch_for_inference, **inference_kwargs(graph=graph, j=j))
                    predictions_per_compression.append(preds)

            # aggregate
            predictions_per_compression_op = tf.stack(predictions_per_compression, axis=0)
            self.logger.info('predictions_shape: {}'.format(predictions_per_compression_op.get_shape().as_list()))

            # restore
            if self._dataset_name == Imagenet.NAME:
                classifier_saver = None
            else:
                classifier_saver = tf.train.Saver(var_list=cnn_model.model_variables())

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=False), graph=graph) as sess:

            if classifier_saver is not None:
                classifier_saver.restore(sess, ckpt_path)

            labels_all_comp_values = [list() for _ in range(self._num_records)]
            predictions_all_comp_values = [list() for _ in range(self._num_records)]
            bpp_all_comp_values = [list() for _ in range(self._num_records)]
            n_images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(n_images_processed, self._dataset.NUM_VAL,
                     '{}/{} images processed'.format(n_images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # run inference
                    (batch_predictions_all_comp_values, batch_label_all_comp_values,
                     batch_bpp_all_comp_values) = sess.run(
                        [predictions_per_compression_op, labels_batches, bpp_batches])

                    # collect predictions
                    for comp_level, (preds_comp, bpp_comp, labels_comp) in enumerate(
                            zip(batch_predictions_all_comp_values,
                                batch_bpp_all_comp_values,
                                batch_label_all_comp_values)):
                        predictions_all_comp_values[comp_level].append(preds_comp)
                        bpp_all_comp_values[comp_level].append(bpp_comp)
                        labels_all_comp_values[comp_level].append(
                            self.to_categorical(labels_comp, cnn_model.num_classes))

                    n_images_processed += len(batch_label_all_comp_values[0])
                    n_images_processed_per_second.append(
                        len(batch_label_all_comp_values[0]) / (time.time() - batch_start_time))

                    progress(n_images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 n_images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(n_images_processed))

            except KeyboardInterrupt:
                self.logger.info('manual interrupt; processed {}/{} images'.format(
                    n_images_processed, Imagenet.NUM_VAL))
                return [(np.nan, np.nan) for _ in range(self._num_records)], [np.nan for _ in range(self._num_records)]

        labels_all_comp_values = [np.concatenate(labels_comp_values, axis=0) for labels_comp_values in
                                  labels_all_comp_values]
        bpp_all_comp_values = [np.mean(np.concatenate(bpp_values, 0)) for bpp_values in bpp_all_comp_values]
        predictions_all_comp_values = [np.concatenate(preds_comp_values, axis=0) for preds_comp_values in
                                       predictions_all_comp_values]

        accuracies = [(self.top_k_accuracy(labels_values, preds_comp_values, 1),
                       self.top_k_accuracy(labels_values, preds_comp_values, 5))
                      for preds_comp_values, labels_values in zip(predictions_all_comp_values, labels_all_comp_values)]

        return accuracies, bpp_all_comp_values
