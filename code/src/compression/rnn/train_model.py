import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # noqa

import cv2
import inspect
import json
import numpy as np
from shutil import copyfile
import time

import tensorflow.contrib.slim as slim
import tensorflow as tf

from src.compression.distortions import Distortions
from src.data.dataloading import InputPipeline, RecordsParser
from src.data.dataloading.preprocessing import CompressionPreprocessing
from src.data.datasets import Dataset
from src.compression.rnn import RNNCompressionModel
from src.lib.logging_commons.utils import get_logger, log_configs, seconds_to_hours_minutes
from src.lib.tf_commons.utils import close_summary_writers, get_available_gpus
from src.lib.np_commons import utils as np_utils

# reproducibility
tf.set_random_seed(648)


class TrainModel:
    """ Training procedure for RNN compression """
    _RESIZE_SIDE_MIN = 256
    _RESIZE_SIDE_MAX = 256

    def __init__(self, dataset: Dataset, config_name, config_file, job_id, train_records, val_records, alpha,
                 model_config, train_config, general_config, feature_loss_kwargs=None):

        # gpu devices
        self._available_gpus = get_available_gpus()
        self._num_gpus = len(self._available_gpus)

        if len(self._available_gpus) == 0:
            self._devices = ['/cpu:0']  # we only have cpu available
        else:
            self._devices = list(self._available_gpus)

        self._dataset = dataset
        self._job_id = job_id
        self._alpha = float(alpha)
        self._config_name = config_name
        self._config_file = config_file
        self._rnn_unit = model_config['rnn_unit']
        self._rec_model = model_config['rec_model']
        self._feature_loss_kwargs = feature_loss_kwargs
        self._feature_loss_components = general_config['feature_loss_components']
        self._data_format = 'NHWC'
        self._setup_dirs_and_logging()

        # parse configs
        self._num_iterations = model_config['num_iterations']
        self._batch_size = train_config['batch_size']
        self._shuffle_buffer_size = train_config['shuffle_buffer_size']
        self._num_epochs = train_config['num_epochs']
        self._learning_rate = train_config['learning_rate']
        self._crop_size = general_config['crop_size']
        self._eval_steps = general_config['eval_steps']
        self._num_preprocessing_threads = general_config['num_preprocessing_threads']
        self._lambda_ms_ssim = general_config['lambda_ms_ssim']
        self._lambda_psnr = general_config['lambda_psnr']
        self._lambda_feature_loss = general_config['lambda_feature_loss']

        # assertions
        assert self._rec_model in [0, 1]
        assert self._crop_size % 16 == 0, 'crop size needs to be mutliple of 16; got {}'.format(self._crop_size)
        if self._alpha > 0:
            assert isinstance(feature_loss_kwargs,
                              dict), 'feature_loss_kwargs=None; disallowed if alpha>0'

        # records files
        self._train_records = train_records
        self._val_records = val_records

        # log args
        log_configs(self._logger, [model_config, train_config, general_config])
        self._logger.info('alpha={}'.format(self._alpha))
        self._logger.info('crop_size={}'.format(self._crop_size))
        self._logger.info('data_format={}'.format(self._data_format))
        self._logger.info('feature_loss_kwargs={}'.format(feature_loss_kwargs))
        self._logger.info('job_id={}'.format(self._job_id))
        self._logger.info('training_data={}'.format(os.path.abspath(self._train_records)))
        self._logger.info('testing_data={}'.format(os.path.abspath(self._val_records)))
        self._logger.info('available_gpus={}'.format(self._available_gpus))
        self._logger.info('devices={}'.format(self._devices))

        # tensorflow session
        self.sess = None

        # initialize
        self._init_input_pipelines()
        self._init_compression_model()
        self._build_graph()

    def _init_input_pipelines(self):
        self._train_input_pipeline = InputPipeline(records=self._train_records,
                                                   records_type=RecordsParser.RECORDS_UNLABELLED,
                                                   preprocessing_fn=CompressionPreprocessing.preprocess_image,
                                                   shuffle_buffer_size=self._shuffle_buffer_size,
                                                   batch_size=self._batch_size,
                                                   num_preprocessing_threads=self._num_preprocessing_threads,
                                                   num_repeat=self._num_epochs,
                                                   drop_remainder=True,
                                                   preprocessing_kwargs={'height': self._crop_size,
                                                                         'width': self._crop_size,
                                                                         'resize_side_min': self._RESIZE_SIDE_MIN,
                                                                         'resize_side_max': self._RESIZE_SIDE_MAX,
                                                                         'is_training': True,
                                                                         'dtype_out': tf.uint8})
        self._val_input_pipeline = InputPipeline(records=self._val_records,
                                                 records_type=RecordsParser.RECORDS_UNLABELLED,
                                                 preprocessing_fn=CompressionPreprocessing.preprocess_image,
                                                 shuffle_buffer_size=self._shuffle_buffer_size,
                                                 batch_size=self._batch_size,
                                                 num_preprocessing_threads=self._num_preprocessing_threads,
                                                 num_repeat=-1,
                                                 drop_remainder=True,
                                                 preprocessing_kwargs={'height': self._crop_size,
                                                                       'width': self._crop_size,
                                                                       'resize_side_min': self._RESIZE_SIDE_MIN,
                                                                       'resize_side_max': self._RESIZE_SIDE_MAX,
                                                                       'is_training': False,
                                                                       'dtype_out': tf.uint8})

    def _init_compression_model(self):
        self._compression_model = RNNCompressionModel(rnn_type=self._rnn_unit,
                                                      image_height=self._crop_size,
                                                      image_width=self._crop_size,
                                                      num_iterations=self._num_iterations,
                                                      rec_model=self._rec_model,
                                                      data_format=self._data_format)

    def _log_model_vars(self):
        trainable_vars = self._compression_model.trainable_variables
        vars_str = '\n' + '=' * 20 + ' trainable model variables ({}) '.format(len(trainable_vars)) + '=' * 20 + '\n'
        for v in trainable_vars:
            vars_str += 'name={}, shape={}\n'.format(v.name, v.shape)
        self._logger.info(vars_str + '=' * 50)

    def _collect_summaries(self, split, total_loss, ms_ssim):
        summaries = [tf.summary.scalar(split + '/learning_rate', self._learning_rate)] if split == 'train' else list()
        summaries.append(tf.summary.scalar(split + '/total_loss', total_loss))
        summaries.append(tf.summary.scalar(split + '/ms_ssim', ms_ssim))

        return tf.summary.merge(summaries)

    def _compute_tower_loss(self, original_images, reuse, is_training):

        reconstructions = self._compression_model.build_model(original_images,
                                                              is_training=tf.cast(is_training, tf.bool),
                                                              reuse=reuse)

        distortions_per_iteration = [Distortions(
            reconstructed_images=reconstructions[i],
            original_images=original_images,
            lambda_ms_ssim=self._lambda_ms_ssim,
            lambda_psnr=self._lambda_psnr,
            lambda_feature_loss=self._lambda_feature_loss,
            data_format=self._data_format,
            loss_net_kwargs=self._feature_loss_kwargs,
            alpha=self._alpha,
            reuse=reuse if i == 0 else True,
            feature_loss_components=self._feature_loss_components) for i in range(self._num_iterations)]

        if self._alpha > 0 and not reuse:
            self._logger.info(
                'feature_loss_components: {}'.format(distortions_per_iteration[0].feature_loss_components))

        ms_ssim_mean = tf.reduce_mean([d.compute_ms_ssim() for d in distortions_per_iteration])

        loss_per_iteration = [d.distortion_loss for d in distortions_per_iteration]
        total_loss = tf.reduce_mean(loss_per_iteration)

        return reconstructions, total_loss, ms_ssim_mean

    def _compute_gradients(self, total_loss, optimizer):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._compression_model.MODEL_SCOPE)
        with tf.control_dependencies(update_ops):
            grads_and_vars = optimizer.compute_gradients(total_loss, self._compression_model.trainable_variables)
        return grads_and_vars

    @staticmethod
    def _average_gradients(tower_grads):
        average_grads = []

        for grads_and_vars in zip(*tower_grads):

            # collect gradients
            grads = []
            for g, _ in grads_and_vars:
                expanded_g = tf.expand_dims(g, 0)  # 0-th dimension represents tower
                grads.append(expanded_g)  # append on tower dimension

            # avg over the tower dimension
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # vars are redundant since they are shared among towers -> return first tower's pointer to the variable
            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    @staticmethod
    def _average_towers(tower_tensors):
        expanded_tower_tensors = []
        for tensor in tower_tensors:
            expanded_tensor = tf.expand_dims(tensor, 0)
            expanded_tower_tensors.append(expanded_tensor)

        expanded_tower_tensors = tf.concat(axis=0, values=expanded_tower_tensors)
        average_towers = tf.reduce_mean(expanded_tower_tensors, 0)

        return average_towers

    def _build_graph(self):

        self.global_step = tf.train.get_or_create_global_step(tf.get_default_graph())

        optimizer = tf.train.AdamOptimizer(self._learning_rate, name='Adam')

        with tf.device('/cpu:0'):  # place everything on cpu per default, use gpus for running model

            # ========= train
            with tf.name_scope('train'):
                # data pipeline
                self._train_images = self._train_input_pipeline.next_batch()[0]
                batch_queue = slim.prefetch_queue.prefetch_queue(tensors=[self._train_images],
                                                                 capacity=2 * max(self._num_gpus, 1),
                                                                 num_threads=self._num_preprocessing_threads)

                reuse_variable_scope = False
                tower_grads, tower_losses, tower_ms_ssims = [], [], []
                for i, gpu_device in enumerate(self._devices):
                    with tf.device(gpu_device):
                        with tf.name_scope('tower_{}'.format(i)):
                            # load batch for current gpu device
                            image_batch = batch_queue.dequeue()

                            # compute tower loss -> this function builds the model and shares vars across towers
                            _, loss, ms_ssim = self._compute_tower_loss(original_images=image_batch,
                                                                        reuse=reuse_variable_scope,
                                                                        is_training=True)
                            tower_losses.append(loss)
                            tower_ms_ssims.append(ms_ssim)

                            # reuse variables for the next tower
                            reuse_variable_scope = True

                            # compute gradients
                            grads = self._compute_gradients(loss, optimizer)
                            tower_grads.append(grads)

                # average gradients (on cpu)
                grads = self._average_gradients(tower_grads)
                self._train_op = optimizer.apply_gradients(grads, global_step=self.global_step)

                self._train_loss = self._average_towers(tower_losses)
                self._train_ms_ssim = self._average_towers(tower_ms_ssims)

                # collect summaries
                self._train_summaries = self._collect_summaries('train', self._train_loss, self._train_ms_ssim)

                # log model vars
                self._log_model_vars()

            # ========= val -> we only evaluate one batch on a single gpu
            with tf.name_scope('val'):
                # data pipeline
                self._val_images = self._val_input_pipeline.next_batch()[0]

                with tf.device(self._devices[0]):
                    self._val_reconstructions, self._val_loss, self._val_ms_ssim = self._compute_tower_loss(
                        self._val_images, reuse=True, is_training=tf.cast(False, tf.bool))

                # collect summaries
                self._val_summaries = self._collect_summaries('val', self._val_loss, self._val_ms_ssim)

        self._graph = tf.get_default_graph()

    def train(self):
        # summary writers
        train_writer = tf.summary.FileWriter(os.path.join(self._tensorboard_dir, 'train/'), graph=self._graph)
        val_writer = tf.summary.FileWriter(os.path.join(self._tensorboard_dir, 'val/'), graph=self._graph)

        self._logger.info("start training for {} epochs".format(self._num_epochs))

        # string for evaluation
        eval_str = "[{epoch}/{num_epochs} epochs][{step}/{num_steps} epoch steps][~{rem_time} remaining] | "
        eval_str += "total_loss: {train_total_loss:.2f} ({val_total_loss:.2f}) | "
        eval_str += "mean ms-ssim: {train_msssim:.3f} ({val_msssim:.3f})"

        num_examples_per_step = self._batch_size * max(self._num_gpus, 1)

        step = self.sess.run(self.global_step)

        steps_per_epoch = self._dataset.NUM_TRAIN // num_examples_per_step
        eval_steps_per_epoch = steps_per_epoch // (self._eval_steps + 1)
        train_steps_per_epoch = steps_per_epoch - eval_steps_per_epoch
        total_training_steps = self._dataset.NUM_TRAIN * self._num_epochs // num_examples_per_step

        self._logger.info('num_examples_per_step={}'.format(num_examples_per_step))
        self._logger.info('steps_per_epoch={}'.format(steps_per_epoch))
        self._logger.info('eval_steps_per_epoch={}'.format(eval_steps_per_epoch))
        self._logger.info('train_steps_per_epoch={}'.format(train_steps_per_epoch))
        self._logger.info('total_training_steps={}'.format(total_training_steps))

        # timing
        train_start_time = time.time()

        # training loop
        try:
            while True:
                # train
                _, step = self.sess.run([self._train_op, self.global_step])

                if step % self._eval_steps == 0:
                    epoch = step * num_examples_per_step // self._dataset.NUM_TRAIN

                    # train stats
                    train_summary, train_total_loss, train_ms_ssim = self.sess.run(
                        [self._train_summaries, self._train_loss, self._train_ms_ssim])

                    train_writer.add_summary(train_summary, step)
                    train_writer.flush()

                    # validation stats
                    val_summary, val_total_loss, val_ms_ssim = self.sess.run(
                        [self._val_summaries, self._val_loss, self._val_ms_ssim])

                    val_writer.add_summary(val_summary, step)
                    val_writer.flush()

                    # compute remaining training time
                    seconds_since_start = time.time() - train_start_time
                    total_training_seconds = seconds_since_start / (step + 1) * total_training_steps
                    seconds_remaining = total_training_seconds - seconds_since_start

                    # log stats
                    self._logger.info(eval_str.format(
                        epoch=epoch + 1, num_epochs=self._num_epochs,
                        step=step % train_steps_per_epoch,
                        rem_time=seconds_to_hours_minutes(seconds_remaining),
                        num_steps=train_steps_per_epoch,
                        train_total_loss=train_total_loss,
                        val_total_loss=val_total_loss,
                        train_msssim=np.mean(train_ms_ssim),
                        val_msssim=np.mean(val_ms_ssim)))

                    # plot some samples
                    self._plot_samples(step)

        except tf.errors.OutOfRangeError:
            self._logger.info('reached end of training.')

        except KeyboardInterrupt:
            self._logger.info("Training terminated after {} steps due to KeyboardInterrupt.".format(step))
            close_summary_writers(val_writer, train_writer)
            return

        # close summary writers
        close_summary_writers(val_writer, train_writer)

    def _plot_samples(self, step):
        # load some samples and reconstructions
        originals, reconstructions = self.sess.run([self._val_images, self._val_reconstructions])

        # post process
        if self._data_format == 'NCHW':
            reconstructions = np_utils.clip_to_rgb(np.transpose(reconstructions, axes=[0, 1, 3, 4, 2]))
            originals = np_utils.nchw_to_nhwc(originals)
        else:
            reconstructions = np_utils.clip_to_rgb(reconstructions)

        # concatenate images
        originals = np.concatenate([orig for orig in originals], axis=0)
        reconstructions = np.concatenate([x for x in np.concatenate([rec for rec in reconstructions], axis=2)], axis=0)
        image_mat = np.concatenate([originals, reconstructions], axis=1)
        image_mat = image_mat[:self._crop_size, ...]
        image_mat = cv2.cvtColor(image_mat, cv2.COLOR_RGB2BGR)
        image_mat = cv2.resize(image_mat, (0, 0), fx=0.75, fy=0.75)

        # save as image file
        save_as = os.path.join(self._image_dir, 'sample_reconstructions_step_{}.png'.format(step))
        cv2.imwrite(save_as, np.uint8(image_mat))

    def set_session(self, sess):
        self.sess = sess

    def _setup_dirs_and_logging(self):
        self._module_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        base_dir = os.path.join(self._module_dir, 'checkpoints/{}/'.format(self._dataset.NAME),
                                "{}{}_alpha={}_id{}".format(self._rnn_unit, self._rec_model, self._alpha, self._job_id))

        self._checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        self._tensorboard_dir = os.path.join(base_dir, 'tensorboard')
        self._image_dir = os.path.join(base_dir, 'images')
        self._log_dir = os.path.join(base_dir, 'logs')

        log_strings = []
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            log_strings.append('created dir {}'.format(base_dir))

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
            log_strings.append('created dir {}'.format(self._checkpoint_dir))

        if not os.path.exists(self._image_dir):
            os.makedirs(self._image_dir)
            log_strings.append('created dir {}'.format(self._image_dir))

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
            log_strings.append('created dir {}'.format(self._log_dir))

        if not os.path.exists(self._tensorboard_dir):
            os.makedirs(self._tensorboard_dir)
            log_strings.append('created dir {}'.format(self._tensorboard_dir))

        # copy config file to base dir
        config_dest_file = os.path.join(base_dir, 'config.json')
        copyfile(self._config_file, config_dest_file)

        # logging
        logfile = os.path.join(self._log_dir,
                               '{}{}_train_id{}_[{}].log'.format(self._rnn_unit, self._rec_model, self._job_id,
                                                                 int(time.time())))

        self._logger = get_logger(logfile)
        for s in log_strings:
            self._logger.info(s)
        self._logger.info('logfile={}'.format(logfile))

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def logger(self):
        return self._logger


def _load_config(config_path):
    with open(config_path, 'r') as config_data:
        config = json.load(config_data)
    return config


def _get_dataset(dataset_name, _train_recs, _val_recs):
    if dataset_name == 'imagenet':
        from src.data.datasets import Imagenet
        return Imagenet()

    raise ValueError('unknown dataset `{}`'.format(dataset_name))


def train(dataset_name, config_path, job_id, train_records, val_records, alpha, feature_loss_kwargs):
    # load configs
    config = _load_config(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    job_id = np.random.choice(99999) if not job_id else job_id

    # init model
    train_model = TrainModel(dataset=_get_dataset(dataset_name, train_records, val_records),
                             config_name=config_name,
                             config_file=config_path,
                             job_id=job_id,
                             train_records=train_records,
                             val_records=val_records,
                             alpha=alpha,
                             model_config=config['model'],
                             train_config=config['train'],
                             general_config=config['general'],
                             feature_loss_kwargs=feature_loss_kwargs)

    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=2, save_relative_paths=True))

    # configure session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    # monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=train_model.checkpoint_dir,
            scaffold=scaffold,
            hooks=None,
            config=sess_config,
            save_summaries_steps=None,
            save_summaries_secs=None,
            save_checkpoint_secs=None,
            log_step_count_steps=0,  # disable logging of steps/s to avoid TF warning in validation sets
            save_checkpoint_steps=config['general']['save_checkpoint_steps']) as sess:
        train_model.set_session(sess)
        train_model.train()

    train_model.logger.info('Training finished.')
