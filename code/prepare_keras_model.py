import argparse
import os

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from src.classification.imagenet import (DenseNet121,
                                         InceptionResnetV2,
                                         InceptionV3,
                                         MobileNet,
                                         ResNet50,
                                         Vgg16,
                                         Vgg19,
                                         Xception)

from src.lib.logging_commons.utils import get_logger
from dirs import TEMP_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='name of model to be converted')
options = parser.parse_args()


def main(_opts):
    model_converter = ModelConverter(frozen_graphs_dir=TEMP_DIR)

    model_converter.convert(_opts.model)


class ModelConverter:
    DENSENET_121 = DenseNet121.NAME
    INCEPTION_RESNET_V2 = InceptionResnetV2.NAME
    INCEPTION_V3 = InceptionV3.NAME
    MOBILENET = MobileNet.NAME
    RESNET_50 = ResNet50.NAME
    VGG_16 = Vgg16.NAME
    VGG_19 = Vgg19.NAME
    XCEPTION = Xception.NAME

    _ALLOWED_MODELS = [DENSENET_121, INCEPTION_RESNET_V2, INCEPTION_V3, MOBILENET, RESNET_50, VGG_16, VGG_19, XCEPTION]

    def __init__(self, frozen_graphs_dir, weights='imagenet', include_top=True, num_classes=1000):
        self._frozen_graphs_dir = frozen_graphs_dir
        self._weights = weights
        self._include_top = include_top
        self._num_classes = num_classes
        self._logger = None

    def convert(self, keras_model_name):

        assert keras_model_name in self._ALLOWED_MODELS, 'unknown model {}; must be one of {}'.format(
            keras_model_name, self._ALLOWED_MODELS)

        save_model_as = os.path.join(self._frozen_graphs_dir, keras_model_name + '.pb')
        if os.path.isfile(save_model_as):
            print('model {} already exists as frozen .pb file: {}'.format(keras_model_name, save_model_as))
            return

        self._logger = get_logger(os.path.join(self._frozen_graphs_dir, keras_model_name + '.log'))

        self._logger.info('conversion start')

        model = self._load_keras_model(keras_model_name, self._weights, self._include_top, self._num_classes)
        model_output_node_names = [out.op.name for out in model.outputs]
        model_inputs_node_names = [inp.op.name for inp in model.inputs]

        # get keras session and freeze graph
        keras_session = tf.keras.backend.get_session()
        frozen_graph = self.freeze_graph(session=keras_session,
                                         output_node_names=model_output_node_names,
                                         input_node_names=model_inputs_node_names)

        # save graph
        tf.train.write_graph(frozen_graph, self._frozen_graphs_dir, keras_model_name + '.pb', as_text=False)
        self._logger.info('writing frozen graph to {}'.format(save_model_as))

    def freeze_graph(self, session, output_node_names, input_node_names, clear_devices=True):
        assert self._logger is not None
        graph = session.graph
        with graph.as_default():
            freeze_var_names = [v.op.name for v in tf.global_variables()]

            self._logger.info('\n============== freeze_var_names ==============')
            for node_name in freeze_var_names:
                self._logger.info(node_name)
            self._logger.info('==============\n')

            output_node_names = output_node_names or []
            output_node_names += [v.op.name for v in tf.global_variables()]

            self._logger.info('\n============== output_node_names ==============')
            for node_name in output_node_names:
                self._logger.info(node_name)
            self._logger.info('==============\n')

            self._logger.info('\n============== input_node_names ==============')
            for node_name in input_node_names:
                self._logger.info(node_name)
            self._logger.info('==============\n')

            input_graph_def = graph.as_graph_def()

            self._logger.info('clear_devices={}'.format(clear_devices))
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ''

            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_node_names, freeze_var_names)

        return frozen_graph

    @classmethod
    def _load_keras_model(cls, model_name, weights, include_top, num_classes):
        if model_name not in cls._ALLOWED_MODELS:
            raise ValueError('unknown model {}; must be one of {}'.format(model_name, cls._ALLOWED_MODELS))

        model_kwargs = dict(include_top=include_top, weights=weights, classes=num_classes)

        # see issue https://github.com/keras-team/keras/issues/12547
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)  # we need the graph in inference mode

        if model_name == cls.DENSENET_121:
            return tf.keras.applications.DenseNet121(input_shape=DenseNet121.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.INCEPTION_RESNET_V2:
            return tf.keras.applications.InceptionResNetV2(input_shape=InceptionResnetV2.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.INCEPTION_V3:
            return tf.keras.applications.InceptionV3(input_shape=InceptionV3.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.MOBILENET:
            return tf.keras.applications.MobileNet(input_shape=MobileNet.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.RESNET_50:
            return tf.keras.applications.ResNet50(input_shape=ResNet50.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.VGG_16:
            return tf.keras.applications.VGG16(input_shape=Vgg16.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.VGG_19:
            return tf.keras.applications.VGG19(input_shape=Vgg19.INPUT_SHAPE, **model_kwargs)

        if model_name == cls.XCEPTION:
            return tf.keras.applications.Xception(input_shape=Xception.INPUT_SHAPE, **model_kwargs)


if __name__ == '__main__':
    main(options)
