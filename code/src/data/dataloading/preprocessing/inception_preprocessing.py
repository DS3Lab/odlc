# ============================================================================================================
#
# This code is adapted from:
#       https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
#
# ============================================================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


class InceptionPreprocessing:

    @classmethod
    def preprocess_image(cls, image, height, width, standardize_image_for_eval=True, is_training=False, bbox=None,
                         fast_mode=True, crop_image=True):
        """ Preprocess an image for training or evaluation.

        Args:
          image: 3-D Tensor of image.
          height: integer
          width: integer
          standardize_image_for_eval: Optional Boolean. If True, image is converted to dtype tf.float32 and standardized
            to [-1,1]. If False, no scaling is applied and image is returned with type tf.uint8 and range [0, 255]. Only
            takes effect if is_training is set to False.
          is_training: Boolean. If true it would transforms image for training, otherwise for evaluation.
          bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords] where each coordinate is [0, 1) and
            the coordinates are arranged as [ymin, xmin, ymax, xmax].
          fast_mode: Optional boolean, if True avoids slower transformations in training mode.
          crop_image: Whether to enable cropping of images during preprocessing for both training and evaluation.

        Returns:
          3-D float Tensor containing an appropriately preprocessed image.

        """
        if is_training:
            return cls.preprocess_for_train(image=image,
                                            height=height,
                                            width=width,
                                            bbox=bbox,
                                            fast_mode=fast_mode,
                                            random_crop=crop_image)
        else:
            return cls.preprocess_for_eval(image=image,
                                           height=height,
                                           width=width,
                                           standardize_image=standardize_image_for_eval,
                                           central_crop=crop_image)

    @classmethod
    def preprocess_for_eval(cls, image, height, width, standardize_image=True, central_fraction=0.875, scope=None,
                            central_crop=True):
        """ Prepare one image for evaluation.

        If height and width are specified it would output an image with that size by
        applying resize_bilinear.

        If central_fraction is specified it would crop the central fraction of the
        input image.

        If standardize_image is set to True we only resize the image without any scaling.

        Args:
          image: 3-D Tensor of image with dtype tf.uint8
          height: integer
          width: integer
          standardize_image: Optional Boolean. If True, image is converted to dtype tf.float32 and standardized to
            [-1,1]. If False, no scaling is applied and image is returned with type tf.uint8 and range [0, 255].
          central_fraction: Optional Float, fraction of the image to crop.
          scope: Optional scope for name_scope.
          central_crop: Enable central cropping of images during preprocessing for evaluation.

        Returns:
          3-D float Tensor of prepared image with type tf.uint8 or tf.float32.
        """

        assert image.dtype == tf.uint8

        with tf.name_scope(scope, 'eval_image', [image, height, width]):

            if standardize_image:
                image = cls.standardize_image(image)

            if central_crop and central_fraction:
                # Crop the central region of the image with an area containing 87.5% of the original image.
                image = tf.image.central_crop(image, central_fraction=central_fraction)

            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
                image = tf.squeeze(image, [0])

            return image

    @staticmethod
    def standardize_image(image):
        """
        args:
          image: image tensor (or batch of images) either of type tf.uint8 with values expected in [0, 256) or
            tf.float32 with values expected in [0, 1). For further details see the documentation of
            tf.image.convert_image_dtype

        returns:
          standardized image tensor
        """
        if image.dtype == tf.uint8:
            # convert image to tf.float32 -> maps to [0, 1]
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            # standardize to [-1,1]
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

        else:
            # expect image to be float with values in [0, 255]
            image = tf.subtract(tf.div(image, 127.5), 1.0)

        return image

    @classmethod
    def preprocess_for_train(cls, image, height, width, bbox, fast_mode=True, scope=None,
                             random_crop=True):
        """ Distort one image for training a network.

        Distorting images provides a useful technique for augmenting the data set during training in order to make the
        network invariant to aspects of the image that do not effect the label.

        Args:
          image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
            [0, 1], otherwise it would converted to tf.float32 assuming that the range
            is [0, MAX], where MAX is largest positive representable number for
            int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
          height: integer
          width: integer
          bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax].
          fast_mode: Optional boolean, if True avoids slower transformations (i.e.
            bi-cubic resizing, random_hue or random_contrast).
          scope: Optional scope for name_scope.
          random_crop: Enable random cropping of images during preprocessing for
            training.

        Returns:
          3-D float Tensor of distorted image used for training with range [-1, 1].
        """

        with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):

            if bbox is None:
                bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                                   dtype=tf.float32,
                                   shape=[1, 1, 4])
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            if not random_crop:
                distorted_image = image

            else:
                distorted_image, distorted_bbox = cls.distorted_bounding_box_crop(image, bbox)
                # # Restore the shape since the dynamic slice based upon the bbox_size loses
                # # the third dimension.
                # distorted_image.set_shape([None, None, 3])
                # image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distorted_bbox)

            # This resizing operation may distort the images because the aspect
            # ratio is not respected. We select a resize method in a round robin
            # fashion based on the thread number.
            # Note that ResizeMethod contains 4 enumerated resizing methods.
            # We select only 1 case for fast_mode bilinear.
            num_resize_cases = 1 if fast_mode else 4
            distorted_image = cls.apply_with_random_selector(
                distorted_image,
                lambda x, method: tf.image.resize_images(x, [height, width], method),
                num_cases=num_resize_cases)

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Randomly distort the colors. There are 1 or 4 ways to do it.
            num_distort_cases = 1 if fast_mode else 4
            distorted_image = cls.apply_with_random_selector(
                distorted_image,
                lambda x, ordering: cls.distort_color(x, ordering, fast_mode),
                num_cases=num_distort_cases)

            distorted_image = tf.subtract(distorted_image, 0.5)
            distorted_image = tf.multiply(distorted_image, 2.0)

            return distorted_image

    @staticmethod
    def apply_with_random_selector(x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].

        Args:
          x: input Tensor.
          func: Python function to apply.
          num_cases: Python int32, number of cases to sample sel from.

        Returns:
          The result of func(x, sel), where func receives the value of the selector as a python integer, but sel is
          sampled dynamically.
        """
        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

    @staticmethod
    def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
        """Distort the color of a Tensor image.

        Each color distortion is non-commutative and thus ordering of the color ops matters. Ideally we would randomly
        permute the ordering of the color ops. Rather then adding that level of complication, we select a distinct
        ordering of color ops for each preprocessing thread.

        Args:
          image: 3-D Tensor containing single image in [0, 1].
          color_ordering: Python int, a type of distortion (valid values: 0-3).
          fast_mode: Avoids slower ops (random_hue and random_contrast)
          scope: Optional scope for name_scope.

        Returns:
          3-D Tensor color-distorted image on range [0, 1]

        Raises:
          ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                else:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                elif color_ordering == 1:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                elif color_ordering == 2:
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                elif color_ordering == 3:
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                else:
                    raise ValueError('color_ordering must be in [0, 3]')

            # The random_* ops do not necessarily clamp.
            return tf.clip_by_value(image, 0.0, 1.0)

    @staticmethod
    def distorted_bounding_box_crop(image,
                                    bbox,
                                    min_object_covered=0.1,
                                    aspect_ratio_range=(0.75, 1.33),
                                    area_range=(0.05, 1.0),
                                    max_attempts=100,
                                    scope=None):
        """Generates cropped_image using a one of the bboxes randomly distorted.

        See `tf.image.sample_distorted_bounding_box` for more documentation.

        Args:
          image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
          bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
          min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
          aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
          area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
          max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
          scope: Optional scope for name_scope.

        Returns:
          A tuple, a 3-D Tensor cropped_image and the distorted bbox
        """
        with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
            # Each bounding box has shape [1, num_boxes, box coords] and
            # the coordinates are ordered [ymin, xmin, ymax, xmax].

            # A large fraction of image datasets contain a human-annotated bounding
            # box delineating the region of the image containing the object of interest.
            # We choose to create a new bounding box for the object which is a randomly
            # distorted version of the human-annotated bounding box that obeys an
            # allowed range of aspect ratios, sizes and overlap with the human-annotated
            # bounding box. If no box is supplied, then we assume the bounding box is
            # the entire image.
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            cropped_image = tf.slice(image, bbox_begin, bbox_size)
            return cropped_image, distort_bbox
