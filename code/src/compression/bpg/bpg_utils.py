""" functions in this module are adapted from https://github.com/fab-jul/imgcomp-cvpr"""
from contextlib import contextmanager
import json
import os


_BPG_MAGIC_NUMBER = bytearray.fromhex('425047fb')


def get_bpg_meta_data(bpg_file):
    with open(bpg_file, 'rb') as f:
        magic_number = f.read(4)
        assert magic_number == _BPG_MAGIC_NUMBER, 'not a BPG file; got magic number {}, expected {}'.format(
            magic_number, _BPG_MAGIC_NUMBER)
        _ = f.read(2)  # contains header info
        width = read_ue7(f)
        height = read_ue7(f)
        picture_data_length = read_ue7(f)
        num_bytes_for_picture = number_of_bytes_until_eof(f) if picture_data_length == 0 else picture_data_length

    return Meta(height, width, num_bytes_for_picture)


class Meta:
    def __init__(self, height, width, num_bytes):
        self.height = height
        self.width = width
        self.num_bytes = num_bytes
        self.bpp = 8.0 * num_bytes / float(height * width)


@contextmanager
def remove_after_context(f):
    yield f
    os.remove(f)


def number_of_bytes_until_eof(f):
    return sum(1 for _ in byte_generator(f))


def byte_generator(f):
    while True:
        byte = f.read(1)
        if byte == b"":
            break
        yield byte


def read_ue7(f):
    """
    ue7 means it's a bunch of bytes all starting with a 1 until one byte starts
    with 0. from all those bytes you take all bits except the first one and
    merge them. E.G.

    some ue7-encoded number:      10001001 01000010
    take all bits except first ->  0001001  1000010
    merge ->                            10011000010 = 1218
    """

    bits = 0
    first_bit_mask = 1 << 7
    value_holding_bits_mask = int(7 * '1', 2)
    for byte in byte_generator(f):
        byte_as_int = byte[0]
        more_bits_are_coming = byte_as_int & first_bit_mask
        bits_from_this_byte = byte_as_int & value_holding_bits_mask
        bits = (bits << 7) | bits_from_this_byte
        if not more_bits_are_coming:
            return bits


def meta_from_local_to_newroot(meta_file, new_root='/mnt/ds3lab-scratch/webermau/'):
    """

    args:
        meta_file: filepath to json containing dict {<LOCAL_PATH>/*bpg_as_png.png:
            {bpg_file: <LOCAL_PATH>/*tmp_bpg.bpg, data}}
        new_root: new root dir

    returns:
        dict meta file where <LOCAL_PATH> is replaced with <new_root>
    """
    old_root = "/Volumes/HD1/image-compression-for-classification/"
    with open(meta_file, 'r') as meta_in:
        old_meta_data = json.load(meta_in)

    old_meta_fn, ext = os.path.splitext(os.path.basename(meta_file))
    new_meta_file = os.path.join(os.path.dirname(meta_file), old_meta_fn + '_new' + ext)

    # replace
    new_meta_data = {}
    for k, v in old_meta_data.items():
        k = k.replace(old_root, new_root)
        v['bpg_file'] = v['bpg_file'].replace(old_root, new_root)
        new_meta_data[k] = v

    print('saving meta to {}'.format(new_meta_file))
    with open(new_meta_file, 'w') as meta_out:
        meta_out.write(json.dumps(new_meta_data, sort_keys=True))
