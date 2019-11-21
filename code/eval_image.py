import argparse
import os

###################
# constants
CLASSIFIERS = ['densenet_121', 'inception_resnet_v2', 'inception_v3', 'mobilenet', 'resnet_50', 'vgg16', 'xception']
JPEG = 'jpeg'
WEBP = 'webp'
BPG = 'bpg'
RNN = 'rnn'

parser = argparse.ArgumentParser()
# general
parser.add_argument('--image', required=True, help='path to image file; jpeg or png')
parser.add_argument('--save', action='store_true')
parser.add_argument('--save_dir', default=None, type=str, help='dir where outputs will be saved')
parser.add_argument('--show', action='store_true')

# compression
parser.add_argument('--compression', default='rnn', type=str, choices=[JPEG, WEBP, BPG, RNN])
parser.add_argument('--quality', '-q', default=0, type=int, help='compression quality; varies between algorithms')
parser.add_argument('--alpha', default=0.0, type=float, help='trade off parameter in RNN compression')

# classification
parser.add_argument('--classifier', default=None, type=str, help='classifier for inference', choices=CLASSIFIERS)
parser.add_argument('--topk', default=5, type=int, help='predict top k labels in classification', choices=[1, 5])

options = parser.parse_args()


def main(_opts):
    # load image
    image_numpy = load_image(_opts.image, color_mode='RGB')

    # instantiate imagenet classifier
    if _opts.classifier is not None:
        classifier = get_imagenet_classifier(_opts.classifier)
    else:
        classifier = None

    # determine shape
    inference_shape, compression_shape = compute_shapes(image_numpy.shape, classifier)

    # compress image
    image_resized, image_compressed, bits_per_pixel, ms_ssim = compress_image(
        image_numpy, _opts.quality, compression_shape, _opts.compression, _opts.alpha)

    # classify compressed image
    predictions_compressed = classify_image(image_compressed, classifier, _opts.topk, True, inference_shape)

    # classify uncompressed image
    predictions_uncompressed = classify_image(image_resized, classifier, _opts.topk, True, inference_shape)

    # write output text
    output_text = '\n============== compression, classification results ==============\n'
    output_text += '* compression method: {}\n'.format(str(_opts.compression).upper())
    output_text += '* quality parameter: {}\n'.format(_opts.quality)
    output_text += '* bitrate: {:.3f} bpp\n'.format(bits_per_pixel)
    output_text += '* MS-SSIM: {:.3f} bpp\n'.format(ms_ssim)

    if classifier is not None:
        output_text += '* classifier: {}\n'.format(classifier.NAME)
        output_text += '\n*** predicted label(s) on compressed image ***\n'
        output_text += tabulate(predictions_compressed, headers=['label', 'synset', 'p(y|x)'], tablefmt='presto',
                                floatfmt=".4f")

        output_text += '\n\n*** predicted label(s) on uncompressed image ***\n'
        output_text += tabulate(predictions_uncompressed, headers=['label', 'synset', 'p(y|x)'], tablefmt='presto',
                                floatfmt=".4f")
        output_text += '\n\n'

    output_text += '================================================================='

    print(output_text)

    if _opts.show:
        show_images([image_resized, image_compressed], str(_opts.compression).upper(), bits_per_pixel)

    if _opts.save:
        save_results(image_resized, image_compressed, _opts.compression, output_text, _opts.alpha, _opts.quality,
                     _opts.save_dir)


def show_images(list_of_images, compression_method, bpp):
    images_numpy = np.concatenate(list_of_images, axis=1)
    images_numpy = cv2.cvtColor(images_numpy, cv2.COLOR_RGB2BGR)
    cv2.imshow('{} compression - original vs. compressed ({:.3f} bpp)'.format(compression_method, bpp), images_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(image_original, image_compressed, compression_method, output_text, rnn_alpha, quality, save_dir=None):
    compression_name = compression_method

    if compression_method == RNN:
        compression_name += '_alpha={}'.format(rnn_alpha)

    if save_dir is None:
        save_dir = os.path.join(TEMP_DIR, str(np.random.choice(999999)) + '/')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save uncompressed image
    save_as = os.path.join(save_dir, 'original.png')
    image_original = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_as, image_original)
    print('* saved uncompressed image as \n      {}'.format(save_as))

    # save compressed image
    save_as = os.path.join(save_dir, '{}_q={}.png'.format(compression_name, quality))
    image_compressed = cv2.cvtColor(image_compressed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_as, image_compressed)
    print('* saved compressed image as \n      {}'.format(save_as))

    # save text
    save_as = os.path.join(save_dir, 'output_{}_q={}.txt'.format(compression_name, quality))
    with open(save_as, 'w') as fo:
        fo.write(output_text)
    print('* saved output text as \n      {}'.format(save_as))


def compute_shapes(image_shape, classifier):
    if classifier is None:
        compression_shape = CompressionPreprocessing.pad_image_shape(
            image_shape=image_shape, size_multiple_of=RNNCompressionModel.SIZE_MULTIPLE_OF)

        # we take square image with smalles_side sides -> so we don't get error with crop size smaller than image size
        smallest_side = min(compression_shape[:2])
        compression_shape = [smallest_side, smallest_side, 3]

        return None, compression_shape

    else:
        inference_shape = classifier.INPUT_SHAPE
        compression_shape = CompressionPreprocessing.pad_image_shape(
            image_shape=inference_shape, size_multiple_of=RNNCompressionModel.SIZE_MULTIPLE_OF,
            extra_padding_multiples=2)

        return inference_shape, compression_shape


def resize(image_tensor, resize_shape):
    # resize op: bilinear interpolation and crop center
    return CompressionPreprocessing.preprocess_image(
        image_tensor, height=resize_shape[0], width=resize_shape[1], is_training=False,
        resize_side_min=min(resize_shape[:2]), dtype_out=tf.uint8)


def compress_image(image_numpy, quality, resize_shape, compression_method, rnn_alpha=None):
    # resize image
    image = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
    image_resized = resize(image, resize_shape)

    print('* image shape for compression: {}'.format(image_resized.get_shape().as_list()))

    # build graph
    image_compressed, bits_per_pixel, saver = build_fetches(image_resized, compression_method, quality, rnn_alpha)

    # compute ms-ssim
    ms_ssim = multiscale_ssim(tf.cast(tf.expand_dims(image_compressed, axis=0), tf.float32),
                              tf.cast(tf.expand_dims(image_resized, axis=0), tf.float32))

    # execute
    with tf.Session() as sess:
        if saver is not None:
            saver['saver'].restore(sess, saver['ckpt'])

        image_resized_val, image_compressed_val, bits_per_pixel_val, ms_ssim_val = sess.run(
            [image_resized, image_compressed, bits_per_pixel, ms_ssim])

    return image_resized_val, image_compressed_val, bits_per_pixel_val, ms_ssim_val


def classify_image(image_numpy, classifier, topk, central_crop, target_shape=None):
    if classifier is None:
        return None

    if image_numpy.ndim != 3:
        raise ValueError('wrong number of dims; expected ndim=3, got {}'.format(image_numpy.ndim))

    assert image_numpy.dtype == np.uint8

    if central_crop:
        assert target_shape is not None and len(target_shape) == 3

        target_height = target_shape[0]
        target_width = target_shape[1]

        image_shape = image_numpy.shape
        height = image_shape[0]
        width = image_shape[1]

        left = (width - target_width) // 2
        top = (height - target_height) // 2

        image_numpy = image_numpy[left:(left + target_width), top:(top + target_height), :]

    print('* image shape for inference: {}'.format(image_numpy.shape))

    return classifier.predict(image_numpy, topk, resize=False, normalize=True)


def build_fetches(image_resized, compression_method, quality, rnn_alpha):
    if str(compression_method).lower() == RNN:
        return rnn_fetches(image_resized, quality, rnn_alpha)

    if str(compression_method).lower() == JPEG:
        return jpeg_fetches(image_resized, quality)

    if str(compression_method).lower() == WEBP:
        return webp_fetches(image_resized, quality)

    if str(compression_method).lower() == BPG:
        return bpg_fetches(image_resized, quality)


def get_rnn_ckpt_dir(a):
    from dirs import TRAINED_MODELS_DIR, RNN_CKPT_PATTERN
    if str(a).split('.')[-1] == '0':
        a = int(a)
    a = str(a).replace('.', '')
    return os.path.join(TRAINED_MODELS_DIR, RNN_CKPT_PATTERN.format(alpha=a))


def rnn_fetches(image_tensor, quality, rnn_alpha):
    import json
    from src.compression.rnn import RNNCompressionModel

    image_shape = image_tensor.get_shape().as_list()

    rnn_ckpt_dir = get_rnn_ckpt_dir(rnn_alpha)

    # load rnn specs
    rnn_checkpoint = tf.train.latest_checkpoint(os.path.join(rnn_ckpt_dir, 'checkpoints/'))
    config_file = os.path.join(rnn_ckpt_dir, 'config.json')

    if rnn_checkpoint is None:
        raise ValueError('rnn checkpoint in {} does not exist!'.format(rnn_ckpt_dir))

    with open(config_file, 'r') as config_data:
        config = json.load(config_data)['model']

    rnn_unit = config['rnn_unit']
    num_iterations = config['num_iterations']
    rec_model = config['rec_model']

    if quality > num_iterations or quality == 0:
        raise ValueError('quality parameter for rnn compression must be int in [{}, .., {}]'.format(1, num_iterations))

    # compress
    rnn_model = RNNCompressionModel(rnn_type=rnn_unit,
                                    image_height=image_shape[0],
                                    image_width=image_shape[1],
                                    num_iterations=quality,
                                    rec_model=rec_model, data_format='NHWC')

    image_compressed = rnn_model.build_model(images=tf.expand_dims(image_tensor, axis=0),
                                             is_training=tf.cast(False, tf.bool),
                                             reuse=None)[-1]

    # set shape, clip, types
    image_compressed.set_shape([1, *image_shape])
    image_compressed = tf.squeeze(image_compressed)
    image_compressed = tf.cast(tf.clip_by_value(image_compressed, 0, 255), tf.uint8)

    # restore
    rnn_saver = tf.train.Saver(var_list=rnn_model.model_variables)

    return image_compressed, tf.convert_to_tensor(0.125 * quality, tf.float32), {'saver': rnn_saver,
                                                                                 'ckpt': rnn_checkpoint}


def jpeg_fetches(image_tensor, quality):
    from src.compression.jpeg.tf_jpeg import TFJpeg

    # compress
    image_compressed, bits_per_pixel = TFJpeg.encode_decode_bpp(
        image_tensor=image_tensor, quality=quality, image_shape=image_tensor.get_shape().as_list())

    return image_compressed, bits_per_pixel, None


def webp_fetches(image_tensor, quality):
    from src.compression.webp.tf_webp import TFWebp

    # compress
    image_compressed, bits_per_pixel = TFWebp.tf_encode_decode_image(image=image_tensor, quality=quality)
    image_compressed.set_shape(image_tensor.get_shape().as_list())

    return image_compressed, bits_per_pixel, None


def bpg_fetches(image_tensor, quality):
    import tempfile

    from src.compression.bpg.py_bpg import PyBPG
    from dirs import TEMP_DIR

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    image_png_bytes = tf.image.encode_png(image_tensor)  # encode as png

    with tf.Session() as sess:
        image_png_bytes_val = sess.run(image_png_bytes)

    # tmp files
    tmp_bpg_file = os.path.join(TEMP_DIR, str(np.random.choice(99999)) + PyBPG.TMP_BPG_EXT)
    tmp_bpg_as_png_file = os.path.join(TEMP_DIR, str(np.random.choice(99999)) + PyBPG.BPG_AS_PNG_EXT)

    with PyBPG.remove_files_after_action([tmp_bpg_file, tmp_bpg_as_png_file]):
        with tempfile.NamedTemporaryFile(dir=TEMP_DIR) as temp_file:
            temp_file.write(image_png_bytes_val)
            temp_file.flush()
            bpg_meta_data = PyBPG.encode_as_bpg(image_file=temp_file.name,
                                                tmp_bpg_file=tmp_bpg_file,
                                                quantization_level=quality)

            if bpg_meta_data is None:
                raise ValueError('image compression with bpg failed!')

            # decode and laod temp png file
            _ = PyBPG.decode_bpg_as_png(tmp_bpg_file=tmp_bpg_file, final_png_file=tmp_bpg_as_png_file)
            image_compressed = load_image(tmp_bpg_as_png_file, color_mode='RGB')

    return tf.convert_to_tensor(image_compressed, tf.uint8), tf.convert_to_tensor(bpg_meta_data.bpp, tf.float32), None


def load_image(image_file, color_mode='RGB'):
    image_numpy = cv2.imread(image_file)

    if color_mode == 'RGB':
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)

    return image_numpy


def check_options(_opts):
    # image file
    if not os.path.isfile(_opts.image):
        raise FileNotFoundError('--image {} does not exist!'.format(_opts.image))

    file_ext = str(os.path.splitext(_opts.image)[-1]).lower()

    if file_ext not in ['.jpeg', '.jpg', '.png']:
        raise ValueError('unknown image format! got {}, must be one of `.jpeg`, `.jpg`, `.png`'.format(file_ext))

    # check if classifier is valid or None
    if _opts.classifier not in CLASSIFIERS and _opts.classifier is not None:
        raise ValueError('invalid classifier! got {}, must be one of {}'.format(
            _opts.classifier, CLASSIFIERS))

    # compression method
    if str(_opts.compression).lower() not in [JPEG, WEBP, BPG, RNN]:
        raise ValueError('--compression must be one of {}; got {}'.format([JPEG, WEBP, BPG, RNN], _opts.compression))

    # quality parameter
    if str(_opts.compression).lower() in [JPEG, WEBP]:
        if not 1 <= _opts.quality <= 100:
            raise ValueError(
                'invalid quality parameter! must be int in [1, 100] for {} compression; got {}'.format(
                    _opts.compression, _opts.quality))

    if _opts.compression == BPG:
        if not 1 <= _opts.quality <= 51:
            raise ValueError(
                'invalid quality parameter! must be int in [1, 51] for {} compression; got {}'.format(
                    _opts.compression, _opts.quality))

    if _opts.compression == RNN:
        if not 1 <= _opts.quality <= 8:
            raise ValueError(
                'invalid quality parameter! must be int in [1, 8] for {} compression; got {}'.format(
                    _opts.compression, _opts.quality))


if __name__ == '__main__':
    # make sure args are fine
    check_options(options)

    # imports
    import cv2
    import numpy as np
    from tabulate import tabulate
    import tensorflow as tf

    from dirs import TEMP_DIR
    from src.classification.classifier_factory import get_imagenet_classifier
    from src.data.dataloading.preprocessing import CompressionPreprocessing
    from src.compression.rnn.model_impl import RNNCompressionModel
    from src.compression.distortions.ms_ssim_tf import multiscale_ssim

    # compression, inference, display
    main(options)
