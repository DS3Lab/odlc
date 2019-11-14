import tensorflow as tf

from src.compression.rnn import BaseModel
from src.compression.rnn.modules import Encoder, Binarizer, Decoder

_RNN_GRU = 'gru'
_RNN_LSTM = 'lstm'

_R_MEAN = 121.85369873
_G_MEAN = 113.58860779
_B_MEAN = 100.63715363

_R_VAR = 4746.37695312
_G_VAR = 4454.13964844
_B_VAR = 4812.234375


class RNNCompressionModel(BaseModel):

    def __init__(self, rnn_type, image_height, image_width, num_iterations, rec_model, data_format,
                 color_means=None, color_vars=None):

        if color_means is None:
            color_means = [_R_MEAN, _G_MEAN, _B_MEAN]

        if color_vars is None:
            color_vars = [_R_VAR, _G_VAR, _B_VAR]

        assert rnn_type in [_RNN_GRU, _RNN_LSTM]
        self._rnn_type = rnn_type

        super(RNNCompressionModel, self).__init__(
            image_height=image_height,
            image_width=image_width,
            num_iterations=num_iterations,
            rec_model=rec_model,
            data_format=data_format,
            color_means=color_means,
            color_vars=color_vars)
        self._initialize_model()

    def _initialize_model(self):
        # initialize model
        self._encoder = Encoder(self._rnn_type, self._image_height, self._image_width, self._data_format)
        self._binarizer = Binarizer(self._rnn_type.upper() + 'Binarizer', self._data_format)
        self._decoder = Decoder(self._rnn_type, self._image_height, self._image_width, self._data_format)

    def build_model(self, images, is_training, reuse, **kwargs):

        if self._rec_model == 0:
            return self._build_additive_model(images, is_training, reuse=reuse)

        elif self._rec_model == 1:
            return self._build_oneshot_model(images, is_training, reuse=reuse)

        raise ValueError("model {} unknown;".format(self._rec_model))

    def _build_additive_model(self, images, is_training, reuse):

        # unroll rnn
        with tf.variable_scope(self.MODEL_SCOPE, reuse=reuse) as scope:

            residuals = self._normalize_tf(images)
            outputs = []

            # initialize states
            self._encoder.init_states(tf.shape(images)[0])
            self._decoder.init_states(tf.shape(images)[0])

            decoded_images = tf.zeros_like(residuals, dtype=tf.float32)
            encoder_state = self._encoder.initial_state
            decoder_state = self._decoder.initial_state

            for i in range(self._num_iterations):
                # propagate variables through time
                if i > 0:
                    scope.reuse_variables()

                # encode, binarize, decode
                encoded_images, encoder_state = self._encoder(residuals, encoder_state)
                binary_code = self._binarizer(encoded_images, is_training=is_training)
                decoded_residuals, decoder_state = self._decoder(binary_code, decoder_state)

                # add up residual
                decoded_images += decoded_residuals

                # denormalize
                outputs.append(self._denormalize_tf(decoded_images))

                # update residuals
                residuals = residuals - decoded_residuals

            return tf.stack(outputs, 0)

    def _build_oneshot_model(self, images, is_training, reuse):

        # unroll rnn
        with tf.variable_scope(self.MODEL_SCOPE, reuse=reuse) as scope:

            residuals = self._normalize_tf(images)
            outputs = []

            # initialize states
            self._encoder.init_states(tf.shape(images)[0])
            self._decoder.init_states(tf.shape(images)[0])

            encoder_state = self._encoder.initial_state
            decoder_state = self._decoder.initial_state

            for i in range(self._num_iterations):
                # propagate variables through time
                if i > 0:
                    scope.reuse_variables()

                # encode, binarize, decode
                encoded_images, encoder_state = self._encoder(residuals, encoder_state)
                binary_code = self._binarizer(encoded_images, is_training=is_training)
                decoded_images, decoder_state = self._decoder(binary_code, decoder_state)

                # denormalize
                outputs.append(self._denormalize_tf(decoded_images))

                # update residuals
                residuals = residuals - decoded_images

            return tf.stack(outputs, 0)

    @property
    def trainable_encoder_vars(self):
        return self._encoder.trainable_variables(parent_scope=self.MODEL_SCOPE)

    @property
    def trainable_binarizer_vars(self):
        return self._binarizer.trainable_variables(parent_scope=self.MODEL_SCOPE)

    @property
    def trainable_decoder_vars(self):
        return self._decoder.trainable_variables(parent_scope=self.MODEL_SCOPE)
