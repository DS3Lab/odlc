import inspect
import os

ROOT_DIR = os.path.dirname(inspect.getabsfile(inspect.currentframe()))
SRC_DIR = os.path.join(ROOT_DIR, 'src/')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources/')

# special dirs
TRAINED_MODELS_DIR = os.path.join(RESOURCES_DIR, 'trained_models/')
KERAS_MODELS_DIR = os.path.join(RESOURCES_DIR, 'keras_models/')
IMAGENET_META_DIR = os.path.join(RESOURCES_DIR, 'imagenet/meta/')
TEMP_DIR = os.path.join(RESOURCES_DIR, 'tmp/')

# RNN dirs and patterns
RNN_CKPT_PATTERN = 'gru0_interpolation_{alpha}_loss_id1'

# FGC Classification dirs
FGC_CLASSIFICATION_DIR = os.path.join(SRC_DIR, 'classification/fine_grained_categorization/')
