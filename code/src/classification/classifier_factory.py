def get_classifier_instance(dataset_name, classifier_name):
    """ Returns an instance of a CNN classifier

    args:
      dataset_name: String.
      classifier_name: String.

    returns:
      instance of classifier with name classifer_name on corresponding dataset

    raises:
      ValueError if dataset is not known
    """
    if dataset_name == 'imagenet':
        return get_imagenet_classifier(classifier_name)

    else:
        return get_fgvc_classifier(dataset_name, classifier_name)


def get_imagenet_classifier(classifier_name):
    """ Returns an instance of a CNN classifier on imagenet

    args:
      classifier_name: String.

    returns:
      instance of classifier with name classifer_name
    """
    from src.classification.imagenet import ImagenetClassifierNames

    if classifier_name == ImagenetClassifierNames.densenet121:
        from src.classification.imagenet import DenseNet121
        return DenseNet121()

    if classifier_name == ImagenetClassifierNames.inception_resnet_v2:
        from src.classification.imagenet import InceptionResnetV2
        return InceptionResnetV2()

    if classifier_name == ImagenetClassifierNames.inception_v3:
        from src.classification.imagenet import InceptionV3
        return InceptionV3()

    if classifier_name == ImagenetClassifierNames.mobilenet:
        from src.classification.imagenet import MobileNet
        return MobileNet()

    if classifier_name == ImagenetClassifierNames.resnet50:
        from src.classification.imagenet import ResNet50
        return ResNet50()

    if classifier_name == ImagenetClassifierNames.vgg16:
        from src.classification.imagenet import Vgg16
        return Vgg16()

    if classifier_name == ImagenetClassifierNames.vgg19:
        from src.classification.imagenet import Vgg19
        return Vgg19()

    if classifier_name == ImagenetClassifierNames.xception:
        from src.classification.imagenet import Xception
        return Xception()

    raise ValueError('invalid classifier `{}` for `imagenet`'.format(classifier_name))


def get_fgvc_classifier(dataset_name, classifier_name):
    """ Returns an instance of a CNN classifier on CUB-200-2011

    args:
      dataset_name: String.
      classifier_name: String.

    returns:
      instance of classifier with name classifer_name
    """
    from src.classification.fine_grained_categorization import FGVCClassifierNames
    from src.data.datasets import Cub200, StanfordDogs

    if dataset_name == Cub200.NAME:
        dataset = Cub200()
    elif dataset_name == StanfordDogs.NAME:
        dataset = StanfordDogs()
    else:
        raise ValueError('invalid dataset `{}`'.format(dataset_name))

    if classifier_name == FGVCClassifierNames.inception_resnet_v2:
        from src.classification.fine_grained_categorization import InceptionResnetV2
        return InceptionResnetV2(dataset)

    if classifier_name == FGVCClassifierNames.inception_v3:
        from src.classification.fine_grained_categorization import InceptionV3
        return InceptionV3(dataset)

    if classifier_name == FGVCClassifierNames.mobilenet:
        from src.classification.fine_grained_categorization import MobilenetV1
        return MobilenetV1(dataset)

    if classifier_name == FGVCClassifierNames.resnet50:
        from src.classification.fine_grained_categorization import ResNet50
        return ResNet50(dataset)

    if classifier_name == FGVCClassifierNames.vgg16:
        from src.classification.fine_grained_categorization import Vgg16
        return Vgg16(dataset)

    raise ValueError('invalid classifier `{}` for `{}`'.format(classifier_name, dataset_name))
