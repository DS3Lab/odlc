import tensorflow as tf

from src.compression.distortions.ms_ssim_tf import multiscale_ssim
from src.compression.distortions.loss_networks.vgg16 import VGG16


class Distortions:

    def __init__(self, reconstructed_images, original_images, lambda_ms_ssim, lambda_psnr, lambda_feature_loss,
                 data_format, loss_net_kwargs, alpha=None, reuse=None, feature_loss_components=None):

        self._lambda_ms_ssim = lambda_ms_ssim
        self._lambda_psnr = lambda_psnr
        self._lambda_feature_loss = lambda_feature_loss
        self._data_format = data_format
        self._reuse = reuse
        self._alpha = alpha  # interpolation loss
        self._feature_loss_components = feature_loss_components

        # init loss network if needed
        if self._alpha > 0 and self._alpha is not None:
            assert self._alpha <= 1.0
            assert 'weights_file' in loss_net_kwargs, 'missing weights file for feature reconstruction loss!'

            self.vgg16 = VGG16(data_format, loss_net_kwargs['weights_file'])

        # image tensors
        self._reconstructed_images = reconstructed_images
        self._original_images = original_images

    @property
    def distortion_loss(self):
        with tf.name_scope('distortion_loss'):
            return self._get_distortion_loss()

    @property
    def feature_loss_components(self):
        return self._feature_loss_components

    def _get_distortion_loss(self):

        assert self._alpha is not None

        if self._alpha == 0.0:
            # human friendly loss
            return self._lambda_ms_ssim * (1 - self.compute_ms_ssim())

        elif self._alpha == 1.0:
            # classification friendly loss
            return self._lambda_feature_loss * self.compute_feature_reconstruction_loss()

        else:
            # interpolation loss
            ms_ssim_loss = self._lambda_ms_ssim * (1 - self.compute_ms_ssim())
            feature_reconstruction_loss = self._lambda_feature_loss * self.compute_feature_reconstruction_loss()
            return self._alpha * ms_ssim_loss + (1 - self._alpha) * feature_reconstruction_loss

    def compute_ms_ssim(self):
        with tf.name_scope('ms_ssim'):
            return multiscale_ssim(self._reconstructed_images, self._original_images, data_format=self._data_format,
                                   name='MS-SSIM')

    def compute_mse(self):
        with tf.name_scope('mse_mean'):
            return tf.reduce_mean(Distortions.compute_mse_per_image(self._reconstructed_images, self._original_images))

    def compute_l1(self):
        with tf.name_scope('l1_mean'):
            return tf.reduce_mean(
                Distortions.compute_l1_per_image(self._reconstructed_images, self._original_images))

    def compute_feature_reconstruction_loss(self):
        assert self.vgg16 is not None
        with tf.name_scope('feature_reconstruction_loss'):
            with tf.name_scope('original_features'):
                scope_prefix = tf.get_default_graph().get_name_scope() + '/'
                features_orig = self.vgg16.get_features(self._original_images, self._feature_loss_components,
                                                        prefix=scope_prefix, reuse=self._reuse)
            with tf.name_scope('reconstructed_features'):
                scope_prefix = tf.get_default_graph().get_name_scope() + '/'
                features_rec = self.vgg16.get_features(self._reconstructed_images, self._feature_loss_components,
                                                       prefix=scope_prefix, reuse=True)

            features_mse = [tf.reduce_mean(tf.squared_difference(x, y)) for x, y in zip(features_orig, features_rec)]
            return tf.reduce_mean(features_mse, name='mse_features')

    def compute_psnr(self):
        with tf.name_scope('psnr_mean'):
            return tf.reduce_mean(Distortions.compute_psnr_per_image(self._reconstructed_images, self._original_images))

    @staticmethod
    def compute_mse_per_image(x, y):
        with tf.name_scope('mse'):
            squared_error = tf.square(x - y)
            squared_error_float = tf.to_float(squared_error)
            mse_per_image = tf.reduce_mean(squared_error_float, axis=[1, 2, 3])
            return mse_per_image

    @staticmethod
    def compute_l1_per_image(x, y):
        with tf.name_scope('l1'):
            abs_diff = tf.abs(tf.subtract(x, y), name='absolut_difference')
            abs_diff_float = tf.to_float(abs_diff)
            abs_diff_per_image = tf.reduce_mean(abs_diff_float, axis=[1, 2, 3])
            return abs_diff_per_image

    @staticmethod
    def compute_psnr_per_image(x, y):
        with tf.name_scope('psnr'):
            mse_per_image = Distortions.compute_mse_per_image(x, y)
            psnr_per_image = 10 * _log10(255.0 * 255.0 / mse_per_image)
            return psnr_per_image


def _log10(_x):
    return tf.log(_x) / tf.log(tf.constant(10, dtype=_x.dtype))
