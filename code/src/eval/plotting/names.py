from src.classification.imagenet import ImagenetClassifierNames
from src.lib.logging_commons.logs_parser import LogsParser

CLASSIFIER_NAMES = {
    ImagenetClassifierNames.densenet121: 'DenseNet-121',
    ImagenetClassifierNames.inception_resnet_v2: 'Inception-ResNet-V2',
    ImagenetClassifierNames.inception_v3: 'Inception-V3',
    ImagenetClassifierNames.mobilenet: 'MobileNet',
    ImagenetClassifierNames.resnet50: 'ResNet-50',
    ImagenetClassifierNames.vgg16: 'VGG-16',
    ImagenetClassifierNames.xception: 'Xception',

    # stanford_dogs.InceptionResnetV2: 'Inception-ResNet-V2',
    # stanford_dogs.InceptionV3.NAME: 'Inception-V3',
    # stanford_dogs.Vgg16.NAME: 'VGG-16',
    # stanford_dogs.MobilenetV1.NAME: 'MobileNet',
    # stanford_dogs.ResNetV1_50.NAME: 'ResNet-50',
    #
    # cub200.InceptionResnetV2: 'Inception-ResNet-V2',
    # cub200.InceptionV3.NAME: 'Inception-V3',
    # cub200.Vgg16.NAME: 'VGG-16',
    # cub200.MobilenetV1.NAME: 'MobileNet',
    # cub200.ResNetV1_50.NAME: 'ResNet-50'
}

COMPRESSION_NAMES = {
    'cpm': 'Mentzer et al.',
    'rnn': r'RNN-$\alpha$',
    'gru0': 'RNN-C (Ours)',
    'gru025': r'Ours w/ $\alpha$=0.25',
    'gru05': r'Ours w/ $\alpha$=0.5',
    'gru075': r'Ours w/ $\alpha$=0.75',
    'gru1': 'RNN-H (Ours)',
    'bpg': 'BPG',
    'webp': 'WebP',
    'jpeg': 'JPEG (4:2:0)',
    'gru1all': r'$\mathcal{I}=\mathcal{I}_0$',
    'gru1top': r'$\mathcal{I}=\{\phi_{5.1}\}$',
    'gru1bottom': r'$\mathcal{I}=\{\phi_{1.1}\}$',
    'gru1topbottom': r'$\mathcal{I}=\{\phi_{1.1},\,\phi_{5.1}\}$',
}

COMPRESSION_LABELS = {
    'gru0': r'$\alpha=1$',
    'gru025': r'$\alpha=0.75$',
    'gru05': r'$\alpha=0.5$',
    'gru075': r'$\alpha=0.25$',
    'gru1': r'$\alpha=0$',
    'bpg': 'BPG',
    'webp': 'WebP',
    'jpeg': 'JPEG (4:2:0)'
}

HVS_METRIC_NAMES = {
    LogsParser.PSNR_KW: 'PSNR',
    LogsParser.MSSSIM_KW: 'MS-SSIM',
    LogsParser.MSE_KW: 'MSE',
    LogsParser.L1_KW: r'L_1-distance'
}

DATASET_NAMES = {
    'imagenet': 'ILSVRC-2012',
    'cub_200': 'CUB-200-2011',
    'stanford_dogs': 'Stanford Dogs',
    'kodak': 'Kodak'
}