import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

import inspect
from collections import deque
import json
import numpy as np
import os
from shutil import copyfile
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dirs import FGC_CLASSIFICATION_DIR
from src.classification import classifier_factory
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.datasets import Cub200, StanfordDogs, Dataset
from src.lib.tf_commons.train_helpers import configure_learning_rate, configure_optimizer
from src.lib.tf_commons.utils import get_sess_config
from src.lib.logging_commons.utils import seconds_to_minutes_seconds, get_logger, log_configs

_CPU_DEVICE = '/cpu:0'


def get_dataset(ds_name) -> Dataset:
    if ds_name == Cub200.NAME:
        return Cub200()
    if ds_name == StanfordDogs.NAME:
        return StanfordDogs()
    else:
        raise ValueError('unknown dataset {}'.format(ds_name))


class TrainModel:
    ALLOWED_MODELS = ['inception_v3', 'mobilenet_v1', 'resnet_v1_50', 'vgg_16', 'inception_resnet_v2']
    ALLOWED_DATASETS = [Cub200.NAME, StanfordDogs.NAME]

    def __init__(self, model_name, dataset_name, train_records, val_records, config_path, config_optimizer,
                 config_learning_rate, config_data, config_transfer_learning, init_checkpoint_path, job_id, eval_epochs,
                 checkpoint_epochs):

        # assertions
        assert os.path.isfile(train_records), 'train_records not found'
        assert os.path.isfile(val_records), 'val_records not found'

        assert model_name in self.ALLOWED_MODELS, 'unknown model'
        assert dataset_name in self.ALLOWED_DATASETS, 'unknown dataset'

        # general args
        self._model_name = model_name
        self._dataset = get_dataset(dataset_name)
        self._config_path = config_path
        self._init_checkpoint_path = init_checkpoint_path
        self._eval_epochs = eval_epochs
        self._checkpoint_epochs = checkpoint_epochs

        # optimizer
        self._optimizer_name = config_optimizer['name']
        self._opt_epsilon = config_optimizer['opt_epsilon']
        self._weight_decay = config_optimizer['weight_decay']

        # learning rate
        self._initial_learning_rate = config_learning_rate['initial_learning_rate']
        self._learning_rate_decay_type = config_learning_rate['learning_rate_decay_type']
        self._learning_rate_decay_factor = config_learning_rate['learning_rate_decay_factor']
        self._num_epochs_per_decay = config_learning_rate['num_epochs_per_decay']
        self._end_learning_rate = config_learning_rate['end_learning_rate']

        # data
        self._batch_size = config_data['batch_size']
        self._num_epochs = config_data['num_epochs']
        self._shuffle_buffer_size = config_data['shuffle_buffer_size']
        self._num_preprocessing_threads = config_data['num_preprocessing_threads']
        self._train_records = train_records
        self._val_records = val_records

        # transfer_learning
        self._trainable_scopes = config_transfer_learning['trainable_scopes']
        self._checkpoint_exclude_scopes = config_transfer_learning['checkpoint_exclude_scopes']

        # timing
        self._epoch_times = deque(10 * [0.0], 10)

        # setup logging, directories
        self._job_id = job_id
        self._setup_dirs_and_logging()

        # log params
        log_configs(self._logger, [config_optimizer, config_transfer_learning, config_data, config_learning_rate])
        self._logger.info('model_name: {}'.format(model_name))
        self._logger.info('train_records: {}'.format(train_records))
        self._logger.info('val_records: {}'.format(val_records))
        self._logger.info('job_id: {}'.format(job_id))

        # build graph
        self._build_graph()

    def _init_model(self):
        self._classifier = classifier_factory.get_fgvc_classifier(self._dataset.NAME, self._model_name)

    def _build_input_pipelines(self):
        self._train_input_pipeline = InputPipeline(records=self._train_records,
                                                   records_type=RecordsParser.RECORDS_LABELLED,
                                                   shuffle_buffer_size=self._shuffle_buffer_size,
                                                   batch_size=self._batch_size,
                                                   num_preprocessing_threads=self._num_preprocessing_threads,
                                                   num_repeat=1,
                                                   preprocessing_fn=self._classifier.preprocess_image,
                                                   preprocessing_kwargs={'is_training': True},
                                                   drop_remainder=False,
                                                   iterator_type=InputPipeline.INITIALIZABLE_ITERATOR)

        self._val_input_pipeline = InputPipeline(records=self._val_records,
                                                 records_type=RecordsParser.RECORDS_LABELLED,
                                                 shuffle_buffer_size=self._shuffle_buffer_size,
                                                 batch_size=self._batch_size,
                                                 num_preprocessing_threads=self._num_preprocessing_threads,
                                                 num_repeat=-1,
                                                 preprocessing_fn=self._classifier.preprocess_image,
                                                 preprocessing_kwargs={'is_training': False},
                                                 drop_remainder=False,
                                                 iterator_type=InputPipeline.ONESHOT_ITERATOR)

    def _compute_loss(self, logits, labels, end_points):
        # cross entropy loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        # aux loss
        if 'AuxLogits' in end_points:
            aux_loss = 0.4 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                            logits=end_points['AuxLogits'])
            aux_loss = tf.reduce_mean(aux_loss, name='aux_loss')
        else:
            aux_loss = tf.cast(0.0, tf.float32)

        # regularization loss
        if self._trainable_scopes is None:
            regularization_terms = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        else:
            regularization_terms = []
            for sc in self._trainable_scopes:
                regularization_terms.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                                              scope=tf.get_default_graph().get_name_scope() + '/' + sc))

        regularization_loss = tf.reduce_sum(regularization_terms, name='regularization_loss')

        return cross_entropy_loss, aux_loss, regularization_loss

    def _get_trainable_variables(self, verbose=False):
        if self._trainable_scopes is None:
            return tf.trainable_variables()

        assert isinstance(self._trainable_scopes, list)

        variables_to_train = []
        for scope in self._trainable_scopes:
            scope_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(scope_variables)

        if verbose:
            self._logger.info('====== trainable variables:')
            self._logger.info('trainable_scopes: {}'.format(self._trainable_scopes))
            self._logger.info('num_trainable_variables: {}'.format(len(variables_to_train)))
            for v in variables_to_train:
                self._logger.info('name: {}, shape: {}, dtype: {}'.format(v.name, v.get_shape().as_list(), v.dtype))
            self._logger.info('===========================')

        return variables_to_train

    def _build_graph(self):
        with tf.Graph().as_default() as graph:
            self._init_model()
            self._build_input_pipelines()
            self._build_graph0()

        graph.finalize()
        self._graph = graph

    def _build_graph0(self):

        self._global_step = tf.train.get_or_create_global_step(tf.get_default_graph())

        # ========= train
        with tf.name_scope('train'):
            train_images, train_labels = self._train_input_pipeline.next_batch()
            train_images.set_shape([None, *self._classifier.INPUT_SHAPE])

            # ========= inference
            arg_scope = self._classifier.arg_scope(weight_decay=self._weight_decay, is_training=True)
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                train_logits, train_end_points = self._classifier.inference(
                    input_tensor=train_images, is_training=True, reuse=None, arg_scope=arg_scope)

            # log shapes
            self._logger.info('images_shape: {}'.format(train_images.get_shape().as_list()))
            self._logger.info('logits_shape: {}'.format(train_logits.get_shape().as_list()))

            trainable_variables = self._get_trainable_variables(verbose=True)

            # ========= compute losses
            self._cross_entropy_loss, self._aux_loss, self._regularization_loss = self._compute_loss(
                train_logits, train_labels, train_end_points)
            self._train_loss = self._regularization_loss + self._cross_entropy_loss + self._aux_loss

            # ========= accuracy
            train_predictions = tf.argmax(tf.nn.softmax(train_logits), axis=1, name='train_predictions')
            self._train_accuracy, self._train_accuracy_update = tf.metrics.accuracy(train_labels, train_predictions,
                                                                                    name='train_accuracy')

            # ========= configure optimizer
            learning_rate = configure_learning_rate(self._global_step, self._batch_size,
                                                    self._initial_learning_rate,
                                                    1, self._dataset.NUM_TRAIN, self._num_epochs_per_decay,
                                                    self._learning_rate_decay_type,
                                                    self._learning_rate_decay_factor, self._end_learning_rate)

            self._optimizer = configure_optimizer(learning_rate, self._optimizer_name, self._opt_epsilon)

            # ========= optimization
            if self._trainable_scopes is None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            else:
                update_ops = []
                for sc in self._trainable_scopes:
                    update_ops.extend(
                        tf.get_collection(key=tf.GraphKeys.UPDATE_OPS,
                                          scope=tf.get_default_graph().get_name_scope() + '/' + sc))

            self._logger.info('update_ops: {}'.format(update_ops))

            with tf.control_dependencies(update_ops):
                self._train_op = self._optimizer.minimize(self._train_loss, self._global_step, trainable_variables)

            # ========= summaries
            self._train_summaries = self._collect_summaries('train', self._train_loss, self._cross_entropy_loss,
                                                            self._aux_loss, self._regularization_loss,
                                                            self._train_accuracy, learning_rate, trainable_variables,
                                                            train_end_points)

        # ========= validation
        with tf.name_scope('val'):
            val_images, val_labels = self._val_input_pipeline.next_batch()

            # ========= inference
            arg_scope = self._classifier.arg_scope(self._weight_decay)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                val_logits, val_end_points = self._classifier.inference(
                    input_tensor=val_images, is_training=False, reuse=True, arg_scope=arg_scope)

            # ========= compute losses
            self._val_cross_entropy_loss, self._val_aux_loss, self._val_regularization_loss = self._compute_loss(
                val_logits, val_labels, val_end_points)
            self._val_loss = self._val_regularization_loss + self._val_cross_entropy_loss + self._val_aux_loss

            # ========= accuracy
            val_predictions = tf.argmax(tf.nn.softmax(val_logits), axis=1)
            self._val_accuracy, self._val_accuracy_update = tf.metrics.accuracy(val_labels, val_predictions,
                                                                                name='validation_accuracy')

            # ========= summaries
            self._val_summaries = self._collect_summaries('val', self._val_loss, self._val_cross_entropy_loss,
                                                          self._val_aux_loss, self._val_regularization_loss,
                                                          self._val_accuracy)

        # ========= saving and restoring
        self._optimizer.get_name()
        vars_to_save = [v for v in tf.global_variables() if 'Adam' not in v.op.name]
        vars_to_save.extend([v for v in tf.local_variables() if 'Adam' not in v.op.name])
        self._logger.info(('vars_to_save: {}'.format(vars_to_save)))
        self._model_saver = tf.train.Saver(var_list=vars_to_save,
                                           save_relative_paths=True,
                                           max_to_keep=2)
        self._init_fn = self._get_init_fn(self._classifier.model_variables(),
                                          self._init_checkpoint_path,
                                          self._checkpoint_exclude_scopes,
                                          verbose=True)

    def train(self):
        train_writer = tf.summary.FileWriter(os.path.join(self._tensorboard_dir, 'train/'), self._graph)
        val_writer = tf.summary.FileWriter(os.path.join(self._tensorboard_dir, 'val/'), self._graph)

        with tf.Session(graph=self._graph, config=get_sess_config(allow_growth=True)) as sess:

            # initialize model and train input pipeline
            self._init_fn(sess)
            self._train_input_pipeline.initialize(sess)

            step = sess.run(self._global_step)

            for epoch in range(1, self._num_epochs + 1):

                epoch_start_time = time.time()

                try:
                    while True:
                        _, step = sess.run([self._train_op, self._global_step])

                except tf.errors.OutOfRangeError:
                    pass

                except KeyboardInterrupt:
                    self._logger.info('manual interrupt')
                    self._clean_up(sess, step, [train_writer, val_writer])
                    return

                self._epoch_times.append(time.time() - epoch_start_time)
                self._train_input_pipeline.initialize(sess)

                if epoch % self._eval_epochs == 0 or epoch == 1:
                    self._eval_procedure(sess, step, epoch, train_writer, val_writer)

                if epoch % self._checkpoint_epochs == 0:
                    self._model_saver.save(sess, os.path.join(self._checkpoint_dir, 'model.ckpt'), global_step=step,
                                           write_meta_graph=False)
                    self._logger.info('[{} steps]saved model.'.format(step))

            self._clean_up(sess, step, [train_writer, val_writer])

    def _get_init_fn(self, model_variables, checkpoint_path, checkpoint_exclude_scopes, ignore_missing_vars=False,
                     verbose=False):
        """ returns fetches, feed_dict for restoring model """

        # ========== continue with training
        latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
        if latest_checkpoint is not None:
            self._logger.info('continue training from {}'.format(latest_checkpoint))

            def _init_fn(sess):
                self._model_saver.restore(sess, latest_checkpoint)

            return _init_fn

        # ========== start training from scratch if no checkpoint is provided
        if checkpoint_path is None:
            self._logger.info('no init_checkpoint_path provided; training network from scratch.')

            def _init_fn(sess):
                sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()]))

            return _init_fn

        # ========== fine tune trainable variables
        exclusions = [v.op.name for v in self._optimizer.variables()]
        if checkpoint_exclude_scopes:
            assert isinstance(checkpoint_exclude_scopes, list)
            exclusions.extend([scope.strip() for scope in checkpoint_exclude_scopes])

        vars_to_restore_from_checkpoint = []
        for var in model_variables:
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                vars_to_restore_from_checkpoint.append(var)

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

        vars_to_init_from_scratch = [v for v in [*tf.global_variables(), *tf.local_variables()] if
                                     v not in vars_to_restore_from_checkpoint]

        if verbose:
            self._logger.info('====== variables to be initialized from checkpoint:')
            self._logger.info('num: {}'.format(len(vars_to_restore_from_checkpoint)))
            for v in vars_to_restore_from_checkpoint:
                self._logger.info('name: {}, shape: {}, dtype: {}'.format(v.op.name, v.get_shape().as_list(), v.dtype))
            self._logger.info('===========================\n')

            self._logger.info('====== variables to be initialized from scratch:')
            self._logger.info('num: {}'.format(len(vars_to_init_from_scratch)))
            for v in vars_to_init_from_scratch:
                self._logger.info('name: {}, shape: {}, dtype: {}'.format(v.op.name, v.get_shape().as_list(), v.dtype))
            self._logger.info('===========================\n')

        # randomly initialize vars to train
        init_fetches = [tf.variables_initializer(var_list=vars_to_init_from_scratch)]

        # restore rest of vars from pretrained model

        restore_op, restore_dict = slim.assign_from_checkpoint(checkpoint_path,
                                                               var_list=vars_to_restore_from_checkpoint,
                                                               ignore_missing_vars=ignore_missing_vars)
        init_fetches.append(restore_op)

        def _init_fn(sess):
            sess.run(init_fetches, feed_dict=restore_dict)

        return _init_fn

    def _clean_up(self, sess, step, summary_writers=None):
        print('cleanup...')

        # stop writers
        summary_writers = [summary_writers] if not isinstance(summary_writers, list) else summary_writers
        for writer in summary_writers:
            if writer is None:
                continue
            else:
                writer.flush()
                writer.close()

        # save model
        self._model_saver.save(sess, os.path.join(self._checkpoint_dir, 'model.ckpt'), step, write_meta_graph=False)
        self._logger.info('[{} steps]saved model.'.format(step))

    @staticmethod
    def _collect_summaries(split, total_loss, cross_entropy_loss, aux_loss, regularization_loss, accuracy,
                           learning_rate=None, variables=None, end_points=None):
        summaries = list()
        summaries.append(tf.summary.scalar(split + '/total_loss', total_loss))
        summaries.append(tf.summary.scalar(split + '/accuracy', accuracy))
        summaries.append(tf.summary.scalar(split + '/cross_entropy_loss', cross_entropy_loss))
        summaries.append(tf.summary.scalar(split + '/aux_loss', aux_loss))
        summaries.append(tf.summary.scalar(split + '/regularization_loss', regularization_loss))

        if split == 'val':
            return tf.summary.merge(summaries, name=split + '_summaries')

        summaries.append(tf.summary.scalar(split + '/learning_rate', learning_rate))
        for v in variables:
            summaries.append(tf.summary.histogram(split + '/variables/{}'.format(v.op.name), v))
        for ep in end_points:
            a = end_points[ep]
            summaries.append(tf.summary.histogram(split + '/activations/{}'.format(ep), a))

        return tf.summary.merge(summaries, name=split + '_summaries')

    def _eval_procedure(self, sess, step, epoch, train_summary_writer, val_summary_writer):

        # train stats
        (train_summaries, train_total_loss, train_regularization_loss, train_cross_entropy_loss, train_aux_loss,
         train_accuracy, _) = sess.run(
            [self._train_summaries, self._train_loss, self._regularization_loss, self._cross_entropy_loss,
             self._aux_loss, self._train_accuracy, self._train_accuracy_update])

        train_summary_writer.add_summary(train_summaries, epoch)
        train_summary_writer.flush()

        # val stats
        (val_summaries, val_total_loss, val_cross_entropy_loss, val_aux_loss, val_accuracy,
         _) = sess.run([self._val_summaries, self._val_loss, self._val_cross_entropy_loss,
                        self._val_aux_loss, self._val_accuracy, self._val_accuracy_update])

        val_summary_writer.add_summary(val_summaries, epoch)
        val_summary_writer.flush()

        # compute average epoch time
        avg_epoch_time = np.mean([t for t in self._epoch_times if t > 0])
        evg_epoch_time_hms = seconds_to_minutes_seconds(avg_epoch_time)

        # print stats
        self._logger.info(self._eval_str(epoch=epoch, num_epochs=self._num_epochs, step=step,
                                         avg_epoch_time=evg_epoch_time_hms,
                                         total_loss=train_total_loss, val_total_loss=val_total_loss,
                                         regularization_loss=train_regularization_loss,
                                         cross_entropy_loss=train_cross_entropy_loss,
                                         val_cross_entropy_loss=val_cross_entropy_loss,
                                         aux_loss=train_aux_loss, val_aux_loss=val_aux_loss,
                                         accuracy=100.0 * train_accuracy, val_accuracy=100.0 * val_accuracy))

    @staticmethod
    def _eval_str(epoch, num_epochs, step, avg_epoch_time, total_loss, val_total_loss, regularization_loss,
                  cross_entropy_loss, val_cross_entropy_loss, aux_loss, val_aux_loss, accuracy, val_accuracy):

        eval_str = "[{epoch}/{num_epochs} epochs ({avg_epoch_time} / epoch)] | "
        eval_str += "total_loss: {total_loss:.4f} ({val_total_loss:.4f})| "
        eval_str += "regularization_loss: {regularization_loss:.6f} (-) | "
        eval_str += "cross_entropy_loss: {cross_entropy_loss:.4f} ({val_cross_entropy_loss:.4f}) | "
        eval_str += "aux_loss: {aux_loss:.4f} ({val_aux_loss:.4f}) | "
        eval_str += "accuracy: {accuracy:.2f}% ({val_accuracy:.2f}%)"

        return eval_str.format(epoch=epoch, num_epochs=num_epochs, step=step, avg_epoch_time=avg_epoch_time,
                               total_loss=total_loss, val_total_loss=val_total_loss,
                               regularization_loss=regularization_loss,
                               cross_entropy_loss=cross_entropy_loss, val_cross_entropy_loss=val_cross_entropy_loss,
                               aux_loss=aux_loss, val_aux_loss=val_aux_loss,
                               accuracy=accuracy, val_accuracy=val_accuracy)

    @property
    def graph(self):
        return self._graph

    def _setup_dirs_and_logging(self):
        self._module_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        base_dir = os.path.join(self._module_dir, 'checkpoints/{}_id{}/'.format(self._model_name, self._job_id))

        self._checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        self._tensorboard_dir = os.path.join(base_dir, 'tensorboard')
        self._log_dir = os.path.join(base_dir, 'logs')

        log_strings = []
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            log_strings.append('created dir {}'.format(base_dir))

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
            log_strings.append('created dir {}'.format(self._checkpoint_dir))

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
            log_strings.append('created dir {}'.format(self._log_dir))

        if not os.path.exists(self._tensorboard_dir):
            os.makedirs(self._tensorboard_dir)
            log_strings.append('created dir {}'.format(self._tensorboard_dir))

        # copy config file to base dir
        config_dest_file = os.path.join(base_dir, 'config.json')
        copyfile(self._config_path, config_dest_file)

        # logging
        logfile = os.path.join(self._log_dir, 'train_{}_id{}.log'.format(self._model_name, self._job_id))
        self._logger = get_logger(logfile)
        for s in log_strings:
            self._logger.info(s)
        self._logger.info('logfile={}'.format(logfile))


def _load_config(config_path):
    assert os.path.isfile(config_path)
    with open(config_path, 'r') as config_data:
        config = json.load(config_data)
    return config


def _get_config_path(dataset_name):
    return os.path.join(FGC_CLASSIFICATION_DIR, 'training/configs/{}.json'.format(dataset_name))


def train(model_name, dataset_name, train_records, val_records, init_checkpoint_path, job_id, eval_epchs,
          checkpoint_epochs):
    config_path = _get_config_path(dataset_name)
    config = _load_config(config_path)[model_name]
    job_id = np.random.choice(99999) if job_id is None else job_id

    train_model = TrainModel(model_name=model_name,
                             dataset_name=dataset_name,
                             train_records=train_records,
                             val_records=val_records,
                             config_path=config_path,
                             config_optimizer=config['optimizer'],
                             config_learning_rate=config['learning_rate'],
                             config_data=config['data'],
                             config_transfer_learning=config['transfer_learning'],
                             init_checkpoint_path=init_checkpoint_path,
                             job_id=job_id,
                             eval_epochs=eval_epchs,
                             checkpoint_epochs=checkpoint_epochs)
    train_model.train()
