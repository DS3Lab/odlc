import cv2
import numpy as np
import tensorflow as tf


class TFWebp:
    """ class with methods for webp compression """

    _EXT = '.webp'
    _HEADER_BYTES_TOTAL = 12

    @classmethod
    def encode(cls, image, quality, rgb=True):
        """ encodes an RGB image

        :param image: 3-D numpy array with shape (height, width, channels) RGB format
        :param quality: integer ≤ 100
        :param rgb: boolean; if True, color channels expected in RGB order
        :return: webp compressed bytes array
        """

        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return bytes(cv2.imencode(cls._EXT, image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])[1])

    @staticmethod
    def decode(image_bytes, rgb=True):
        """ decodes a webp encoded image from bytes

        :param image_bytes: webp bytes
        :param rgb: boolean; if True, returns image with color channels in RGB ordering
        :return: decoded numpy in RGB order
        """
        image_numpy_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if rgb:
            return cv2.cvtColor(image_numpy_bgr, cv2.COLOR_BGR2RGB)
        else:
            return image_numpy_bgr

    @classmethod
    def encode_decode_image(cls, image, quality, rgb=True):
        """ encodes and decodes an image

        :param image: 3-D numpy array with shape (height, width, channels)
        :param quality: integer ≤ 100
        :param rgb: boolean; if True, color channels are expected and returned in RGB ordering
        :return: tuple containing decoded image as numpy array and bits per pixel (excluding header)
        """
        img_height, img_width, _ = np.shape(image)
        image_bytes = cls.encode(image, int(quality), rgb=rgb)
        image_bpp = np.float32((len(image_bytes) - cls._HEADER_BYTES_TOTAL) * 8.0 / (img_height * img_width))
        image_numpy_compressed = np.array(cls.decode(image_bytes, rgb=rgb), dtype=np.uint8)
        return image_numpy_compressed, image_bpp

    @classmethod
    def encode_decode_image_batch(cls, image_batch, quality, rgb=True):
        """ compresses a batch of images

        :param image_batch: 4-D numpy array with shape (batch_size, height, width, channels)
        :param quality: integer ≤ 100
        :param rgb: boolean; if True, color channels are expected and returned in RGB ordering
        :return: tuple containing decoded image batch and bits per pixel (excluding header)
        """
        image_batch_compressed = []
        image_batch_bpp = []
        for image in image_batch:
            image_compressed, image_bpp = cls.encode_decode_image(image, quality, rgb=rgb)
            image_batch_compressed.append(np.uint8(image_compressed))
            image_batch_bpp.append(np.float32(image_bpp))
        return np.stack(image_batch_compressed, axis=0), np.stack(image_batch_bpp, axis=0)

    @classmethod
    def tf_encode_decode_image_batch(cls, image_batch, quality, rgb=True):
        """

        :param image_batch: 4-D tensor with shape (batch_size, height, width, channels)
        :param quality: integer ≤ 100
        :param rgb: boolean; if True, color channels are expected and returned in RGB ordering
        :return: tensorflow op to compress a batch of image tensors + bits per pixel (excluding header)
        """
        return tf.py_func(cls.encode_decode_image_batch, inp=[image_batch, quality, rgb], Tout=[tf.uint8, tf.float32])

    @classmethod
    def tf_encode_decode_image(cls, image, quality, rgb=True):
        """

        :param image: 3-D tensor with shape (height, width, channels)
        :param quality: integer ≤ 100
        :param rgb: boolean; if True, color channels are expected and returned in RGB ordering
        :return: tensorflow op to compress a batch of image tensors + bits per pixel (excluding header)
        """
        return tf.py_func(cls.encode_decode_image, inp=[image, quality, rgb], Tout=[tf.uint8, tf.float32])
