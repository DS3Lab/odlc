import tensorflow as tf

from src.lib.tf_commons.rnn.conv_rnn import ConvolutionalRNN


class Encoder(ConvolutionalRNN):

    def __init__(self, rnn_type, image_height, image_width, data_format):
        assert rnn_type in ['gru', 'lstm']
        super(Encoder, self).__init__(rnn_type, scope=rnn_type.upper() + 'Encoder')

        # E-Conv
        self._conv1 = self.conv_layer(64, 2, 3, data_format, None)
        # E-RNN#1
        self._rnn1 = self.rnn_cell(256, image_height // 4, image_width // 4, 2, 3, 1, data_format)
        # E-RNN#2
        self._rnn2 = self.rnn_cell(512, image_height // 8, image_width // 8, 2, 3, 1, data_format)
        # E-RNN#3
        self._rnn3 = self.rnn_cell(512, image_height // 16, image_width // 16, 2, 3, 1, data_format)

        self.initial_state = None

    def init_states(self, batch_size_tensor):
        self.initial_state = (self._rnn1.init_state(batch_size_tensor),
                              self._rnn2.init_state(batch_size_tensor),
                              self._rnn3.init_state(batch_size_tensor))

    def __call__(self, inputs, state):
        assert self.initial_state is not None, 'initial_state not initialized!'

        with tf.variable_scope(self._scope):
            with tf.variable_scope("conv1") as scope1:
                conv1 = self._conv1(inputs, scope=scope1)
            with tf.variable_scope("rnn1") as scope2:
                hidden_1, state_1 = self._rnn1(conv1, state[0], scope=scope2)
            with tf.variable_scope("rnn2") as scope3:
                hidden_2, state_2 = self._rnn2(hidden_1, state[1], scope=scope3)
            with tf.variable_scope("rnn3") as scope4:
                hidden_3, state_3 = self._rnn3(hidden_2, state[2], scope=scope4)

        return hidden_3, (state_1, state_2, state_3)
