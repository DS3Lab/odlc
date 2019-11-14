from abc import ABCMeta
import inspect
import os
import numpy as np

from src.data.dataloading.preprocessing import VGGPreprocessing
from src.lib.logging_commons import EvalValues
from src.lib.logging_commons.utils import get_logger, write_to_csv


class EvalMSSSIMBase(metaclass=ABCMeta):
    DATA_FORMAT = 'NHWC'
    DEFAULT_SIZE = 256
    DISTORTION_KEYS = ['ms_ssim']

    def __init__(self, dataset_name, logger, records_file):
        self._dataset_name = dataset_name
        self._records_file = records_file
        self.logger = logger

    def run(self):
        """ implemented by subclass """
        raise NotImplementedError

    @property
    def records_file(self):
        return self._records_file

    @property
    def dataset_name(self):
        return self._dataset_name

    def _get_logger(self, dataset_name, compression_method):
        self._log_dir = os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())),
                                     'logs/{}/tmp'.format(dataset_name))
        self._log_number = np.random.choice(10000)
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        logfile = os.path.join(self._log_dir, 'eval_hvs_{}_{}.log'.format(compression_method, self._log_number))
        return get_logger(logfile), logfile

    def log_results(self, eval_values: EvalValues):
        """ ms-ssim values need to be logged in a uniform structure """
        for bpp, hvs_dict, other_info_dict in zip(eval_values.bits_per_pixel, eval_values.hvs_dicts,
                                                  eval_values.other_info_dicts):
            eval_str = 'EVAL: [bpp_mean={}] | '.format(bpp)

            for metric, value in hvs_dict.items():
                eval_str += ' [{}={}] |'.format(metric, value)

            if other_info_dict is not None and isinstance(other_info_dict, dict):
                for key, val in other_info_dict.items():
                    eval_str += ' [{}={}] |'.format(key, val)

            self.logger.info(eval_str)

    def _save_results(self, bpp, distortion_dicts, compression_method, compression_levels):
        # write results to csv
        csv_file = os.path.join(self._log_dir, '{}_hvs_{}.csv'.format(compression_method, self._log_number))

        csv_rows = [('bpp', *self.DISTORTION_KEYS)]
        csv_rows.extend(
            [(b, *[dist_dict[k] for k in self.DISTORTION_KEYS]) for b, dist_dict in zip(bpp, distortion_dicts)])

        write_to_csv(csv_file, csv_rows)

        # log results
        other_info_dicts = None
        if compression_levels is not None:
            other_info_dicts = [{'quality': q} for q in compression_levels]

        self.log_results(
            self.pack_eval_values(bits_per_pixel=bpp, hvs_dicts=distortion_dicts, other_info_dicts=other_info_dicts))

    @staticmethod
    def pack_eval_values(bits_per_pixel, hvs_dicts, other_info_dicts=None):
        return EvalValues(bits_per_pixel=bits_per_pixel,
                          accuracy_dicts=None,
                          hvs_dicts=hvs_dicts,
                          other_info_dicts=other_info_dicts)

    def _get_resize_function(self, image_height, image_width):
        def resize(img):
            return self._resize_for_compression(img, image_height, image_width)

        return resize

    def _resize_for_compression(self, _input_image, height, width):
        """ resize image as in VGG preprocessing

        args:
          _input_image: 3-D tensor

        returns:
          resized image
        """
        if self._dataset_name == 'kodak':
            _input_image.set_shape([height, width, 3])
            return _input_image
        else:
            return VGGPreprocessing.preprocess_image(image=_input_image,
                                                     output_height=height,
                                                     output_width=width,
                                                     mean_center_image=False)
