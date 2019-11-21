from contextlib import contextmanager
import os
import subprocess

from src.compression.bpg.bpg_utils import get_bpg_meta_data


class PyBPG:
    BPGENC = os.environ.get('BPGENC', 'bpgenc')
    BPGDEC = os.environ.get('BPGDEC', 'bpgdec')

    # extensions
    TMP_BPG_EXT = '_tmp_bpg.bpg'
    TMP_JPEG_AS_PNG_EXT = '_tmp_jpeg_as_png.png'
    BPG_AS_PNG_EXT = '_bpg_as_png.png'

    @classmethod
    def encode_as_bpg(cls, image_file, tmp_bpg_file, quantization_level, chroma_fmt=444, verbose=False):
        num_failures = 0
        while True:
            return_code = subprocess.call(
                [cls.BPGENC, '-q', str(quantization_level), '-o', tmp_bpg_file, '-f', str(chroma_fmt), image_file])
            if return_code == 0 or num_failures > 9:
                break
            else:
                num_failures += 1
                print('--bpgenc failure; received code {}. Trying again ({}/10)'.format(return_code, num_failures))

        if verbose:
            print('--bpgenc return_code: {}; image_file={}'.format(return_code, image_file))

        # return obj containing metadata about imagefile
        try:
            return get_bpg_meta_data(tmp_bpg_file)

        except AssertionError as e:
            print('failed to compress {}\n      got {}'.format(image_file, e))
            return None

    @classmethod
    def decode_bpg_as_png(cls, tmp_bpg_file, final_png_file):
        subprocess.call([cls.BPGDEC, '-o', final_png_file, tmp_bpg_file])

        with open(final_png_file, 'rb') as img_f:
            image_bytes = img_f.read()

        return image_bytes

    @staticmethod
    @contextmanager
    def remove_files_after_action(list_of_files_to_remove):
        yield list_of_files_to_remove
        for f in list_of_files_to_remove:
            os.remove(f)
