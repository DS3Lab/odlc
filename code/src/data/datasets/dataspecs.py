from abc import ABCMeta

from src.lib.commons import AbstractAttribute


class Dataset(metaclass=ABCMeta):
    NUM_TRAIN = AbstractAttribute('number of training samples')
    NUM_VAL = AbstractAttribute('number of validation samples')
    NUM_CLASSES = AbstractAttribute('number of distinct classes')
    NAME = AbstractAttribute('name of dataset')


class Imagenet(Dataset):
    NUM_TRAIN = 1281167
    NUM_VAL = 50000
    NUM_CLASSES = 1000
    NAME = 'imagenet'


class Cub200(Dataset):
    NUM_TRAIN = 5994
    NUM_VAL = 5794
    NUM_CLASSES = 200
    NAME = 'cub200'


class StanfordDogs(Dataset):
    NUM_TRAIN = 12000
    NUM_VAL = 8580
    NUM_CLASSES = 120
    NAME = 'stanford_dogs'


class Kodak(Dataset):
    NUM_TRAIN = 0
    NUM_VAL = 24
    NUM_CLASSES = 0
    NAME = 'kodak'
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 768
