import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # noqa

from collections import deque
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

_FGVC_TRAIN_ID = 251


class EvalAccuracyRaw(EvalAccuracyBase):
    NUM_PREPROCESSING_THREADS = 16
    DATA_FORMAT = 'NHWC'
    BATCH_SIZE = 64

    def __init__(self, dataset_name, records_file, checkpoint_root=None):
        logger = self._get_logger(dataset_name, 'original')
        super(EvalAccuracyRaw, self).__init__(dataset_name, records_file, logger, checkpoint_root)

        self.logger.info('compression_method: None')
        self.logger.info('batch_size={}'.format(self.BATCH_SIZE))

    def run0(self):
        """ imagenet evaluation loop """
        acc_dict = {}
        bits_per_pixel = []

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.imagenet_cnn_model_names]))

        for cnn_model_name in self.imagenet_cnn_model_names:
            bpp, accs = self.eval_classifier_model(self.get_classifier_instance(cnn_model_name))
            acc_dict[cnn_model_name] = accs
            bits_per_pixel.append(bpp)

            # write results to csv
            csv_file = os.path.join(self._log_dir,
                                    '{}_accuracy_original_{}.csv'.format(cnn_model_name, self._log_number))
            write_to_csv(csv_file, [(bpp, *accs)])

            # log results
            accuracy_dicts = [{cnn_model_name: accs}]
            self.log_results(self.pack_eval_values(bits_per_pixel=[bpp], accuracy_dicts=accuracy_dicts))

    def run1(self):
        """ FGVC evaluation loop """
        acc_dict = {}
        bits_per_pixel = []

        self.logger.info('classifiers for evaluation: {}'.format([m for m in self.fgvc_cnn_model_names]))

        for cnn_model_name in self.fgvc_cnn_model_names:
            model_checkpoint = self.get_fgvc_checkpoint(cnn_model_name)

            if model_checkpoint is None:
                continue

            bpp, accs = self.eval_classifier_model(cnn_model=self.get_classifier_instance(cnn_model_name),
                                                   ckpt_path=model_checkpoint)
            acc_dict[cnn_model_name] = accs
            bits_per_pixel.append(bpp)

            # write results to csv
            csv_file = os.path.join(self._log_dir,
                                    '{}_accuracy_original_{}.csv'.format(cnn_model_name, self._log_number))
            write_to_csv(csv_file, [(bpp, *accs)])

            # log results
            accuracy_dicts = [{cnn_model_name: accs}]
            self.log_results(self.pack_eval_values(bits_per_pixel=[bpp], accuracy_dicts=accuracy_dicts))

    def eval_classifier_model(self, cnn_model: any([ImagenetClassifier, FGVCClassifier]), ckpt_path=None):

        tf.reset_default_graph()

        print('')
        self.logger.info('===== building graph start: {} ====='.format(cnn_model.NAME))

        # assertions
        assert self._dataset.NUM_CLASSES == cnn_model.num_classes, 'inconsistent number of classes ({} != {})'.format(
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
                                           compute_bpp=True,
                                           shuffle=False)

            images, labels, bpp = input_pipeline.next_batch()

            # inference op
            self.logger.info('* inference')

            # inference kwargs
            if self._dataset_name == Imagenet.NAME:
                def inference_kwargs(**kwargs):
                    return dict(graph=kwargs['graph'])
            else:
                def inference_kwargs(**kwargs):
                    return dict(arg_scope=cnn_model.arg_scope(weight_decay=float(0)),
                                is_training=False,
                                return_predictions=True)

            # take central crop of images
            images = tf.image.resize_image_with_crop_or_pad(image=images,
                                                            target_height=image_shape_classification[0],
                                                            target_width=image_shape_classification[1])

            # standardize appropriately
            images_for_inference = cnn_model.standardize_tensor(images)

            # predict
            predictions_op = cnn_model.inference(images_for_inference, **inference_kwargs(graph=graph))

            # restore
            if self._dataset_name == Imagenet.NAME:
                classifier_saver = None
            else:
                classifier_saver = tf.train.Saver(var_list=cnn_model.model_variables())

        graph.finalize()

        with tf.Session(config=get_sess_config(allow_growth=False), graph=graph) as sess:

            if classifier_saver is not None:
                classifier_saver.restore(sess, ckpt_path)

            labels_values, bpp_values, predictions_values = [], [], []
            images_processed = 0
            n_images_processed_per_second = deque(10 * [0.0], 10)
            progress(images_processed, self._dataset.NUM_VAL,
                     '{}/{} images processed'.format(images_processed, self._dataset.NUM_VAL))

            try:
                while True:
                    batch_start_time = time.time()

                    # run inference
                    batch_predictions_values, batch_label_values, batch_bpp_values = sess.run(
                        [predictions_op, labels, bpp])

                    # collect predictions
                    predictions_values.append(batch_predictions_values)

                    # collect labels and bpp
                    labels_values.append(self.to_categorical(batch_label_values, self._dataset.NUM_CLASSES))
                    bpp_values.append(batch_bpp_values)

                    images_processed += len(batch_label_values)
                    n_images_processed_per_second.append(self.BATCH_SIZE / (time.time() - batch_start_time))

                    progress(images_processed, self._dataset.NUM_VAL,
                             status='{}/{} images processed ({} img/s)'.format(
                                 images_processed, self._dataset.NUM_VAL,
                                 np.mean([t for t in n_images_processed_per_second])))

            except tf.errors.OutOfRangeError:
                self.logger.info('reached end of dataset; processed {} images'.format(images_processed))

            except KeyboardInterrupt:
                self.logger.info(
                    'manual interrupt; processed {}/{} images'.format(images_processed, self._dataset.NUM_VAL))
                return np.nan, (np.nan, np.nan)

        labels_values = np.concatenate(labels_values, axis=0)
        bpp_mean = np.mean(np.concatenate(bpp_values, axis=0))
        predictions_values = np.concatenate(predictions_values, axis=0)

        accuracies = (self.top_k_accuracy(labels_values, predictions_values, 1),
                      self.top_k_accuracy(labels_values, predictions_values, 5))

        return bpp_mean, accuracies
