# ============================================================================================================
#
# Parts of this code are adapted from:
#       https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
#
# ============================================================================================================
""" Provides utilities to preprocess images for DenseNet. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_R_MEAN = 0.485
_G_MEAN = 0.456
_B_MEAN = 0.406

_R_STD = 0.229
_G_STD = 0.224
_B_STD = 0.225

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


class DenseNetPreprocessing:

    @classmethod
    def preprocess_image(cls, image, output_height, output_width, normalize_image=True, is_training=False,
                         resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX):
        """Preprocesses an image for training or evaluation.

        Args:
          image: A `Tensor` representing an image of arbitrary size; RGB values in 0, ..., 255.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          normalize_image: If True, per channel image normalization according to Imagenet data.
          is_training: `True` if we're preprocessing the image for training and `False` otherwise.
          resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. If
            `is_training` is `False`, then this value is used for rescaling.
          resize_side_max: The upper bound for the smallest side of the image for aspect-preserving resizing. If
            `is_training` is `False`, this value is ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max].

        Returns:
          A preprocessed image.
        """

        if is_training:
            return cls.preprocess_for_train(image, output_height, output_width, resize_side_min, resize_side_max,
                                            normalize_image)
        else:
            return cls.preprocess_for_eval(image, output_height, output_width, resize_side_min, normalize_image)

    @classmethod
    def preprocess_for_eval(cls, image, output_height, output_width, resize_side, normalize_image=True):
        """Preprocesses the given image for evaluation.

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          resize_side: The smallest side of the image for aspect-preserving resizing.
          normalize_image: If True, per channel image normalization according to Imagenet data.

        Returns:
          A preprocessed image.
        """

        image = cls.aspect_preserving_resize(image, resize_side)
        image = cls.central_crop([image], output_height, output_width)[0]
        image.set_shape([output_height, output_width, 3])
        if normalize_image:
            return cls.normalize_image(image, [_R_MEAN, _G_MEAN, _B_MEAN], [_R_STD, _G_STD, _B_STD])
        else:
            return image

    @classmethod
    def preprocess_for_train(cls, image, output_height, output_width, resize_side_min=_RESIZE_SIDE_MIN,
                             resize_side_max=_RESIZE_SIDE_MAX, normalize_image=True):
        """Preprocesses the given image for training.

        Note that the actual resizing scale is sampled from [`resize_size_min`, `resize_size_max`].

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          normalize_image: If True, per channel image normalization according to Imagenet data.
          resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing.
          resize_side_max: The upper bound for the smallest side of the image for aspect-preserving resizing.

        Returns:
          A preprocessed image.
        """

        resize_side = tf.random_uniform(
            [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

        image = cls.aspect_preserving_resize(image, resize_side)
        image = cls._random_crop([image], output_height, output_width)[0]
        image.set_shape([output_height, output_width, 3])
        image = tf.image.random_flip_left_right(image)

        if normalize_image:
            image = tf.to_float(image)
            return cls.normalize_image(image, [_R_MEAN, _G_MEAN, _B_MEAN], [_R_STD, _G_STD, _B_STD])
        else:
            return image

    @staticmethod
    def _smallest_size_at_least(height, width, smallest_side):
        """Computes new shape with the smallest side equal to `smallest_side`.

        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.

        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          smallest_side: A python integer or scalar `Tensor` indicating the size of the smallest side after resize.

        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: and int32 scalar tensor indicating the new width.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(tf.rint(height * scale))
        new_width = tf.to_int32(tf.rint(width * scale))
        return new_height, new_width

    @classmethod
    def aspect_preserving_resize(cls, image, smallest_side):
        """Resize images preserving the original aspect ratio.

        Args:
          image: A 3-D image `Tensor`.
          smallest_side: A python integer or scalar `Tensor` indicating the size of the smallest side after resize.

        Returns:
          resized_image: A 3-D tensor containing the resized image.
        """
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        new_height, new_width = cls._smallest_size_at_least(height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    @classmethod
    def central_crop(cls, image_list, crop_height, crop_width):
        """Performs central crops of the given image list.
        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the height of the image following the crop.
          crop_width: the width of the image following the crop.
        Returns:
          the list of cropped images.
        """

        outputs = []

        for image in image_list:
            image_height = tf.shape(image)[0]
            image_width = tf.shape(image)[1]

            offset_height = (image_height - crop_height) / 2
            offset_width = (image_width - crop_width) / 2

            outputs.append(cls._crop(image, offset_height, offset_width, crop_height, crop_width))

        return outputs

    @classmethod
    def _random_crop(cls, image_list, crop_height, crop_width):
        """Crops the given list of images.

        The function applies the same crop to each image in the list. This can be
        effectively applied when there are multiple image inputs of the same
        dimension such as:
          image, depths, normals = _random_crop([image, depths, normals], 120, 150)

        Args:
          image_list: a list of image tensors of the same dimension but possibly varying channel.
          crop_height: the new height.
          crop_width: the new width.

        Returns:
          the image_list with cropped images.

        Raises:
          ValueError: if there are multiple image inputs provided with different size
            or the images are smaller than the crop dimensions.
        """
        if not image_list:
            raise ValueError('Empty image_list.')

        # Compute the rank assertions.
        rank_assertions = []
        for i in range(len(image_list)):
            image_rank = tf.rank(image_list[i])
            rank_assert = tf.Assert(
                tf.equal(image_rank, 3),
                ['Wrong rank for tensor  %s [expected] [actual]',
                 image_list[i].name, 3, image_rank])
            rank_assertions.append(rank_assert)

        with tf.control_dependencies([rank_assertions[0]]):
            image_shape = tf.shape(image_list[0])
        image_height = image_shape[0]
        image_width = image_shape[1]
        crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height),
                                                    tf.greater_equal(image_width, crop_width)),
                                     ['Crop size greater than the image size.'])

        asserts = [rank_assertions[0], crop_size_assert]

        for i in range(1, len(image_list)):
            image = image_list[i]
            asserts.append(rank_assertions[i])
            with tf.control_dependencies([rank_assertions[i]]):
                shape = tf.shape(image)
            height = shape[0]
            width = shape[1]

            height_assert = tf.Assert(
                tf.equal(height, image_height),
                ['Wrong height for tensor %s [expected][actual]',
                 image.name, height, image_height])

            width_assert = tf.Assert(tf.equal(width, image_width),
                                     ['Wrong width for tensor %s [expected][actual]',
                                      image.name, width, image_width])

            asserts.extend([height_assert, width_assert])

        # Create a random bounding box.
        #
        # Use tf.random_uniform and not numpy.random.rand as doing the former would
        # generate random numbers at graph eval time, unlike the latter which
        # generates random numbers at graph definition time.
        with tf.control_dependencies(asserts):
            max_offset_height = tf.reshape(image_height - crop_height + 1, [])

        with tf.control_dependencies(asserts):
            max_offset_width = tf.reshape(image_width - crop_width + 1, [])

        offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
        offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

        return [cls._crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]

    @staticmethod
    def _crop(image, offset_height, offset_width, crop_height, crop_width):
        """Crops the given image using the provided offsets and sizes.

        Note that the method doesn't assume we know the input image size but it does
        assume we know the input image rank.

        Args:
          image: an image of shape [height, width, channels].
          offset_height: a scalar tensor indicating the height offset.
          offset_width: a scalar tensor indicating the width offset.
          crop_height: the height of the cropped image.
          crop_width: the width of the cropped image.

        Returns:
          the cropped (and resized) image.

        Raises:
          InvalidArgumentError: if the rank is not 3 or if the image dimensions are
            less than the crop size.
        """
        original_shape = tf.shape(image)

        rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 3),
            ['Rank of image must be equal to 3.'])
        with tf.control_dependencies([rank_assertion]):
            cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

        size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(original_shape[0], crop_height),
                tf.greater_equal(original_shape[1], crop_width)),
            ['Crop size greater than the image size.'])

        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

        # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
        # define the crop size.
        with tf.control_dependencies([size_assertion]):
            image = tf.slice(image, offsets, cropped_shape)
        return tf.reshape(image, cropped_shape)

    @staticmethod
    def normalize_image(image, means=None, stds=None):
        """ normalizes an image x according to (x - mean) / std.

        For example:
          means = [0.485, 0.456, 0.406]
          stds = [0.229, 0.224, 0.225]
          image = _normalize_image(image, means)
        Note that the rank of `image` must be known.

        Args:
          image: a tensor of size [height, width, C] or [batch_size, height, width, C].
          means: a C-vector of values to subtract from each channel.
          stds: a C-vector of values by which each mean centered channel is divided.

        Returns:
          the normalized image.

        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other than three or if the number of
            channels in `image` doesn't match the number of values in `means`.
        """

        # convert to float and scale to [0, 1]
        image = tf.to_float(image)
        image = tf.div(image, 255.0)

        ndims = image.get_shape().ndims

        if ndims != 3 and ndims != 4:
            raise ValueError('Input must be of size {}'.format(
                '[height, width, C>0]' if ndims == 3 else '[batch_size, height, width, C>0]'))

        num_channels = image.get_shape().as_list()[-1]

        if means is None:
            means = [_R_MEAN, _G_MEAN, _B_MEAN]

        if stds is None:
            stds = [_R_STD, _G_STD, _B_STD]

        if len(means) != num_channels or len(stds) != num_channels:
            raise ValueError('len(means) and len(stds) must match the number of channels')

        channels = tf.split(axis=-1, num_or_size_splits=num_channels, value=image)

        for i in range(num_channels):
            channels[i] -= means[i]
            channels[i] /= stds[i]

        return tf.concat(axis=-1, values=channels)

    # @staticmethod
    # def standardize_image_batch(image_batch, means, stds):
    #     """ normalizes an image_batch x according to (x - mean) / std.
    #
    #     For example:
    #       means = [0.485, 0.456, 0.406]
    #       stds = [0.229, 0.224, 0.225]
    #       image = _normalize_image(image, means)
    #     Note that the rank of `image` must be known.
    #
    #     Args:
    #       image_batch: a tensor of size [N, height, width, C].
    #       means: a C-vector of values to subtract from each channel.
    #       stds: a C-vector of values by which each mean centered channel is divided.
    #
    #     Returns:
    #       the normalized image batch.
    #
    #     Raises:
    #       ValueError: If the rank of `image_batch` is unknown, if `image_batch` has a rank other than four or if the
    #       number of channels in `image` doesn't match the number of values in `means`.
    #     """
    #
    #     # convert to float and scale to [0, 1]
    #     image = tf.to_float(image_batch)
    #     image = tf.div(image, 255.0)
    #
    #     if image.get_shape().ndims != 4:
    #         raise ValueError('Input must be of size [batch_size, height, width, C>0]')
    #     num_channels = image.get_shape().as_list()[-1]
    #     if len(means) != num_channels or len(stds) != num_channels:
    #         raise ValueError('len(means) and len(stds) must match the number of channels')
    #
    #     channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
    #     for i in range(num_channels):
    #         channels[i] -= means[i]
    #         channels[i] /= stds[i]
    #     return tf.concat(axis=3, values=channels)
