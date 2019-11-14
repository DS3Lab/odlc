import os
import tensorflow as tf

from src.data.dataloading.records_parsing import RecordsParser

_ONESHOT_ITERATOR = 'one-shot'
_INITIALIZABLE_ITERATOR = 'initializable'


class InputPipeline:
    ONESHOT_ITERATOR = _ONESHOT_ITERATOR
    INITIALIZABLE_ITERATOR = _INITIALIZABLE_ITERATOR

    def __init__(self,
                 records,
                 records_type,
                 shuffle_buffer_size,
                 batch_size,
                 num_preprocessing_threads,
                 num_repeat,
                 preprocessing_fn,
                 preprocessing_kwargs=None,
                 drop_remainder=True,
                 compute_bpp=False,
                 iterator_type=_ONESHOT_ITERATOR,
                 datafeed_device='/cpu:0',
                 dtype_out=tf.float32,
                 shuffle=True):

        # assertions
        assert records_type in RecordsParser.ALLOWED_RECORDS_TYPES, 'records type must be on of ' + str(
            RecordsParser.ALLOWED_RECORDS_TYPES)
        assert os.path.isfile(records), 'records not found!'

        self._records = records
        self._records_type = records_type
        self._shuffle_buffer_size = shuffle_buffer_size
        self._datafeed_device = datafeed_device
        self._preprocessing_kwargs = preprocessing_kwargs or {}
        self._dtype_out = dtype_out
        self._preprocess_image_fn = preprocessing_fn

        self._compute_bpp = compute_bpp
        self._batch_size = batch_size
        self._num_repeat = num_repeat
        self._drop_remainder = drop_remainder
        self._num_preprocessing_threads = num_preprocessing_threads

        self._iterator_type = iterator_type
        self._init_iterator(shuffle=shuffle)

    def next_batch(self):
        return self.iterator.get_next()

    def initialize(self, sess):
        if self._iterator_type == self.INITIALIZABLE_ITERATOR:
            sess.run(self.iterator.initializer)

    def _init_iterator(self, shuffle):
        dataset = self._get_dataset(shuffle)
        with tf.name_scope('dataset_iterator'):
            if self._iterator_type == self.ONESHOT_ITERATOR:
                self.iterator = dataset.make_one_shot_iterator()
            elif self._iterator_type == self.INITIALIZABLE_ITERATOR:
                self.iterator = dataset.make_initializable_iterator()
            else:
                raise ValueError('unknown iterator type {}'.format(self._iterator_type))

    def _get_dataset(self, shuffle):
        with tf.name_scope('dataset'):
            with tf.device(self._datafeed_device):
                dataset = tf.data.TFRecordDataset(self._records)
                if shuffle:
                    dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)
                dataset = dataset.apply(tf.contrib.data.map_and_batch(
                    self._preprocess_example,
                    batch_size=self._batch_size,
                    num_parallel_batches=self._num_preprocessing_threads,
                    drop_remainder=self._drop_remainder
                ))
                dataset = dataset.repeat(self._num_repeat)
                dataset = dataset.prefetch(4 * self._batch_size)
        return dataset

    def _preprocess_example(self, example):
        """ outputs are always ordered according to

            image, label, bpp

        """

        features = RecordsParser.parse_example(example, self._records_type)
        outputs = []

        # process image bytes
        image_bytes = features[RecordsParser.KW_IMAGE_BYTES]
        image_preprocessed, image_shape = self._process_image_bytes(image_bytes)
        outputs.append(image_preprocessed)

        # parse label
        if RecordsParser.KW_LABEL in features:
            outputs.append(self._process_label(features[RecordsParser.KW_LABEL]))

        # parse bpp
        if RecordsParser.KW_BPP in features and not self._compute_bpp:
            outputs.append(self._process_bpp(features[RecordsParser.KW_BPP]))

        # compute bpp from bytes
        elif self._compute_bpp:
            outputs.append(self._compute_bpp_from_bytes(image_bytes, image_shape))

        return outputs

    def _process_image_bytes(self, image_bytes):
        image_decoded = tf.image.decode_jpeg(image_bytes, channels=3)
        image_shape = tf.shape(image_decoded)
        image_decoded.set_shape([None, None, 3])
        image_preprocessed = self._preprocess_image_fn(image_decoded, **self._preprocessing_kwargs)
        image_preprocessed = tf.cast(image_preprocessed, self._dtype_out)
        return image_preprocessed, image_shape

    @staticmethod
    def _process_label(label):
        label = tf.cast(label, tf.int32)
        label.set_shape(shape=())
        return label

    @staticmethod
    def _process_bpp(bpp):
        bpp = tf.cast(bpp, tf.float32)
        bpp.set_shape(shape=())
        return bpp

    @staticmethod
    def _compute_bpp_from_bytes(image_bytes, image_shape):
        num_bytes = tf.cast(tf.size(tf.string_split([image_bytes], "")), tf.float32)
        num_pixels = tf.cast(tf.multiply(image_shape[0], image_shape[1]), tf.float32)
        bpp = tf.divide(8 * (num_bytes - 2), num_pixels)
        return bpp
