from functools import partial
import tensorflow as tf


class TFJpeg:
    """ class with basic methods for jpeg compression """
    _HEADER_BYTES_TOTAL = 2

    @classmethod
    def encode_decode_bpp(cls, image_tensor, quality, image_shape, chroma_downsampling=True):
        image_encoded = tf.image.encode_jpeg(image_tensor, quality=quality, optimize_size=True,
                                             chroma_downsampling=chroma_downsampling)
        num_bytes = tf.cast(tf.size(tf.string_split([image_encoded], "")), tf.float32)
        num_pixels = tf.cast(tf.multiply(tf.shape(image_tensor)[0], tf.shape(image_tensor)[1]), tf.float32)

        image_bpp = tf.divide(8 * (num_bytes - cls._HEADER_BYTES_TOTAL), num_pixels)

        image_decoded = tf.cast(tf.image.decode_jpeg(image_encoded, channels=3), tf.uint8)
        image_decoded.set_shape(image_shape)

        return image_decoded, image_bpp

    @classmethod
    def encode_decode_image_batch(cls, image_batch, quality, image_shape):
        return tf.map_fn(fn=partial(cls.encode_decode_bpp, quality=quality, image_shape=image_shape),
                         elems=image_batch,
                         dtype=(tf.uint8, tf.float32))
