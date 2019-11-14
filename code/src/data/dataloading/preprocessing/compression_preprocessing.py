""" Provides utilities to preprocess images for subsequent compression. """

import tensorflow as tf

from src.data.dataloading.preprocessing.vgg_preprocessing import VGGPreprocessing

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 256


class CompressionPreprocessing:

    @classmethod
    def preprocess_image(cls, image, height, width, is_training, resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX, dtype_out=tf.uint8):
        """ Preprocesses an image for training or evaluation

        if is_training:
            - sample resize_side uniformly from [resize_side_min, resize_side_max]
            - aspect-preserving resize, so that smaller side equals `resize_side`,
            - random crop of size (`height`, `width`)
            - random flip left, right
        else:
            - aspect-preserving resize, so that smaller side equals `resize_side_min`,
            - central crop of size (`height`, `width`)


        -> Note that here, we only resize the images - no normalizing

        Args:
          image: A `Tensor` representing an image of arbitrary size
          height: Integer
          width: Integer
          is_training: Boolean. `True` if we're preprocessing the image for training and `False` otherwise
          resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing
          resize_side_max: The upper bound for the smallest side of the image for aspect-preserving resizing. Only used
            when is_training is set to True
          dtype_out: output type

        returns:
          resized image of type dtype_out
        """
        if is_training:
            return cls.preprocess_image_for_train(image=image,
                                                  height=height,
                                                  width=width,
                                                  resize_side_min=resize_side_min,
                                                  resize_side_max=resize_side_max,
                                                  dtype_out=dtype_out)
        else:
            return cls.preprocess_image_for_eval(image=image,
                                                 height=height,
                                                 width=width,
                                                 resize_side_min=resize_side_min,
                                                 dtype_out=dtype_out)

    @staticmethod
    def preprocess_image_for_train(image, height, width, resize_side_min, resize_side_max, dtype_out):
        """ Preprocesses an image for training.

        Here we only resize the images. Normalizing is done directly in the RNN implementation.

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          height: Integer
          width: Integer
          resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. Only used
            when preprocessing_mode is set to `vgg`.
          resize_side_max: The upper bound for the smallest side of the image for aspect-preserving resizing. Resize
            side is sampled from [resize_size_min, resize_size_max].
          dtype_out: output type.

        returns:
          Appropriately resized image.

        """
        resized_image = VGGPreprocessing.preprocess_for_train(image=image,
                                                              output_height=height,
                                                              output_width=width,
                                                              resize_side_min=resize_side_min,
                                                              resize_side_max=resize_side_max,
                                                              mean_center_image=False)
        return tf.cast(resized_image, dtype=dtype_out)

    @staticmethod
    def preprocess_image_with_identity(image, height, width, dtype_out):
        image.set_shape([height, width, 3])
        image = tf.cast(image, dtype=dtype_out)
        return image

    @staticmethod
    def preprocess_image_for_eval(image, height, width, resize_side_min, dtype_out):
        """ Preprocesses an image for evaluation.

        Here we only resize the images. Normalizing is done directly in the RNN implementation.

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          height: Integer
          width: Integer
          resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. Only used
            when preprocessing_mode is set to `vgg`.
          dtype_out: output type.

        returns:
          Appropriately resized image.

        """

        resized_image = VGGPreprocessing.preprocess_for_eval(image=image,
                                                             output_height=height,
                                                             output_width=width,
                                                             resize_side=resize_side_min,
                                                             mean_center_image=False)
        return tf.cast(resized_image, dtype=dtype_out)

    @staticmethod
    def pad_image_shape(image_shape, size_multiple_of, extra_padding_multiples=0):
        """ for each spatial dimensions computes the next bigger or equal multiple of `size_multiple_of` and adds
          an additional extra_padding_multiples x size_multiple of

        args:
          image_shape: rank-1 tuple of ints (height, width, channels)
          size_multiple_of: integer
          extra_padding_multiples: number of additional multiples of `size_multiple_of`

        returns:
          rank-1 tuple of ints (new_height, new_width, channels)
        """
        assert len(image_shape) == 3

        def compute_side(s, m):
            """ computes int that is the next >= multiple of m """
            if s % m == 0:
                return s
            else:
                return s + (m - s % m)

        original_height, original_width, num_channels = image_shape

        return [compute_side(original_height, size_multiple_of) + extra_padding_multiples * size_multiple_of,
                compute_side(original_width, size_multiple_of) + extra_padding_multiples * size_multiple_of,
                num_channels]
