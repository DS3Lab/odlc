# """ Provides utilities to preprocess images for RNN compression. """
#
# from src.data.dataloading.preprocessing.vgg_preprocessing import VGGPreprocessing
# from src.data.dataloading.preprocessing.inception_preprocessing import InceptionPreprocessing
#
# _VGG_PREPROCESSING = 'vgg'
# _INCEPTION_PREPROCESSING = 'inception'
# _RESIZE_SIDE_MIN = 256
#
#
# class RNNPreprocessing:
#
#     @classmethod
#     def preprocess_image(cls, image, height, width, is_training=False, resize_side_min=_RESIZE_SIDE_MIN,
#                          preprocessing_mode=_VGG_PREPROCESSING):
#         """ Preprocesses an image for training or evaluation
#
#         Note that here, we only resize the images. Normalizing is done directly in the RNN implementation.
#
#         Args:
#           image: A `Tensor` representing an image of arbitrary size.
#           height: Integer
#           width: Integer
#           is_training: Boolean. `True` if we're preprocessing the image for training and `False` otherwise.
#           resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. If
#             `is_training` is `False`, then this value is used for rescaling. Only used when preprocessing_mode is set to
#              `vgg`.
#           preprocessing_mode: String. Must be one of `vgg` or `inception`, indicating the preprocessing method.
#
#         returns:
#           Appropriately resized image.
#         """
#         if is_training:
#             return cls.preprocess_image_for_train(image=image,
#                                                   height=height,
#                                                   width=width,
#                                                   resize_side_min=resize_side_min,
#                                                   resize_side_max=resize_side_min,
#                                                   preprocessing_mode=preprocessing_mode)
#         else:
#             return cls.preprocess_image_for_eval(image=image,
#                                                  height=height,
#                                                  width=width,
#                                                  resize_side_min=resize_side_min,
#                                                  preprocessing_mode=preprocessing_mode)
#
#     @staticmethod
#     def preprocess_image_for_train(image, height, width, resize_side_min, resize_side_max, preprocessing_mode):
#         """ Preprocesses an image for training.
#
#         Here we only resize the images. Normalizing is done directly in the RNN implementation.
#
#         Args:
#           image: A `Tensor` representing an image of arbitrary size.
#           height: Integer
#           width: Integer
#           resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. Only used
#             when preprocessing_mode is set to `vgg`.
#           resize_side_max: The upper bound for the smallest side of the image for aspect-preserving resizing. Resize
#             side is sampled from [resize_size_min, resize_size_max].
#           preprocessing_mode: String. Must be one of `vgg` or `inception`, indicating the preprocessing method.
#
#         returns:
#           Appropriately resized image.
#
#         raises:
#           NotImplementedError if preprocessing_mode is set to `inception`.
#           ValueError if preprocessing mode is different from `vgg` or `inception`.
#         """
#         if preprocessing_mode == _VGG_PREPROCESSING:
#             return VGGPreprocessing.preprocess_for_train(
#                 image, height, width, resize_side_min, resize_side_max, mean_center_image=False)
#         elif preprocessing_mode == _INCEPTION_PREPROCESSING:
#             raise NotImplementedError('preprocessing mode `{}` not implemented'.format(preprocessing_mode))
#         else:
#             raise ValueError('unknown preprocessing mode `{}`'.format(preprocessing_mode))
#
#     @staticmethod
#     def preprocess_image_for_eval(image, height, width, resize_side_min, preprocessing_mode):
#         """ Preprocesses an image for evaluation.
#
#         Here we only resize the images. Normalizing is done directly in the RNN implementation.
#
#         Args:
#           image: A `Tensor` representing an image of arbitrary size.
#           height: Integer
#           width: Integer
#           resize_side_min: The lower bound for the smallest side of the image for aspect-preserving resizing. Only used
#             when preprocessing_mode is set to `vgg`.
#           preprocessing_mode: String. Must be one of `vgg` or `inception`, indicating the preprocessing method.
#
#         returns:
#           Appropriately resized image.
#
#         raises:
#           ValueError if preprocessing mode is different from `vgg` or `inception`.
#         """
#         if preprocessing_mode == _VGG_PREPROCESSING:
#             return VGGPreprocessing.preprocess_for_eval(image=image,
#                                                         output_height=height,
#                                                         output_width=width,
#                                                         resize_side=resize_side_min,
#                                                         mean_center_image=False)
#
#         elif preprocessing_mode == _INCEPTION_PREPROCESSING:
#             return InceptionPreprocessing.preprocess_for_eval(image=image,
#                                                               height=height,
#                                                               width=width,
#                                                               standardize_image=False)
#
#         else:
#             raise ValueError('unknown preprocessing mode `{}`'.format(preprocessing_mode))
