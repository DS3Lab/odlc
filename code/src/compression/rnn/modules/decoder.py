import tensorflow as tf

from src.lib.tf_commons.rnn.conv_rnn import ConvolutionalRNN


class Decoder(ConvolutionalRNN):

    def __init__(self, rnn_type, image_height, image_width, data_format):
        assert rnn_type in ['gru', 'lstm']
        super(Decoder, self).__init__(rnn_type, scope=rnn_type.upper() + 'Decoder')

        self._scope = rnn_type.upper() + 'Decoder'

        self._data_format = data_format
        # D-Conv#1
        self._conv1 = self.conv_layer(512, 1, 1, data_format, None)
        # D-RNN#1
        self._rnn1 = self.rnn_cell(512, image_height // 16, image_width // 16, 1, 2, 1, data_format)
        # D-RNN#2
        self._rnn2 = self.rnn_cell(512, image_height // 8, image_width // 8, 1, 3, 1, data_format)
        # D-RNN#3
        self._rnn3 = self.rnn_cell(256, image_height // 4, image_width // 4, 1, 3, 3, data_format)
        # D-RNN#4
        self._rnn4 = self.rnn_cell(128, image_height // 2, image_width // 2, 1, 3, 3, data_format)
        # D-Conv#2
        self._conv2 = self.conv_layer(3, 1, 1, data_format, None)

        self.initial_state = None

    def init_states(self, batch_size_tensor):
        self.initial_state = (self._rnn1.init_state(batch_size_tensor),
                              self._rnn2.init_state(batch_size_tensor),
                              self._rnn3.init_state(batch_size_tensor),
                              self._rnn4.init_state(batch_size_tensor))

    def __call__(self, inputs, state):
        assert self.initial_state is not None, 'initial_state not initialized!'

        with tf.variable_scope(self._scope):
            with tf.variable_scope("conv1") as scope1:
                conv1 = self._conv1(inputs, scope=scope1)

            with tf.variable_scope("rnn1") as scope2:
                hidden_1, state_1 = self._rnn1(conv1, state[0], scope=scope2)
                hidden_1_upsampled = tf.depth_to_space(hidden_1, 2, data_format=self._data_format)

            with tf.variable_scope("rnn2") as scope3:
                hidden_2, state_2 = self._rnn2(hidden_1_upsampled, state[1], scope=scope3)
                hidden_2_upsampled = tf.depth_to_space(hidden_2, 2, data_format=self._data_format)

            with tf.variable_scope("rnn3") as scope4:
                hidden_3, state_3 = self._rnn3(hidden_2_upsampled, state[2], scope=scope4)
                hidden_3_upsampled = tf.depth_to_space(hidden_3, 2, data_format=self._data_format)

            with tf.variable_scope("rnn4") as scope5:
                hidden_4, state_4 = self._rnn4(hidden_3_upsampled, state[3], scope=scope5)
                hidden_4_upsampled = tf.depth_to_space(hidden_4, 2, data_format=self._data_format)

            with tf.variable_scope("conv2") as scope6:
                conv2 = self._conv2(hidden_4_upsampled, scope=scope6)

            return conv2, (state_1, state_2, state_3, state_4)
