# import tensorflow as tf
#
#
# def resize_input(x, data_format, mode):
#     if mode == 'tf':
#         # corresponds to inception preprocessing; scale to [-1, 1]
#         return x
#
#     if mode == 'torch':
#         # corresponds to densenet preprocessing; scale to [0, 1] and then normalize channel wise wrt imagenet dataset
#         return x
#     else:
#         # corresponds to vgg preprocessing; zero center wrt imagenet dataset without scaling
#         return x
#
#
# def scale_input(x, data_format, mode):
#     """Preprocesses a tensor encoding a batch of images.
#
#     :param x: Input tensor, 3D or 4D.
#     :param data_format: Data format of the image tensor.
#     :param mode: One of "caffe", "tf" or "torch".
#             - caffe: will convert the images from RGB to BGR,
#                 then will zero-center each color channel with
#                 respect to the ImageNet dataset,
#                 without scaling.
#             - tf: will scale pixels between -1 and 1,
#                 sample-wise.
#             - torch: will scale pixels between 0 and 1 and then
#                 will normalize each channel with respect to the
#                 ImageNet dataset.
#
#     :returns Preprocessed tensor.
#     """
#
#     assert str(data_format).startswith('N'), 'data_format must be either `NCHW` or `NHWC`'
#
#     def rgb_to_bgr(image_batch, channel_axis):
#         r, g, b = tf.split(axis=channel_axis, num_or_size_splits=3, value=image_batch)
#         image_batch_bgr = tf.concat(axis=channel_axis, values=[b,g,r])
#         return image_batch_bgr
#
#     if mode == 'tf':
#         return tf.subtract(tf.div(x, 127.5), 1.0)
#
#     if mode == 'torch':
#         # densenet preprocessing
#         x = tf.div(x, 255.)
#         mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32, shape=(3,))
#         std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32, shape=(3,))
#
#     else:
#         # caffe models
#         if data_format == 'NCHW':
#             # RGB -> BGR
#             if len(x.get_shape().as_list()) == 3:
#                 x = rgb_to_bgr(x, channel_axis=0)
#             else:
#                 x = rgb_to_bgr(x, channel_axis=1)
#         else:
#             # 'RGB'->'BGR'
#             x = rgb_to_bgr(x, channel_axis=-1)
#
#         mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=(3,))
#         std = None
#
#     # Zero-center by mean pixel
#     x = tf.nn.bias_add(x, -mean, data_format=data_format)
#
#     if std is not None:
#         x = tf.div(x, std)
#
#     return x
