import tensorflow as tf
from tensorflow.python.client import device_lib


def get_sess_config(allow_growth=True, allow_soft_placement=True):
    sess_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement)
    sess_config.gpu_options.allow_growth = allow_growth
    return sess_config


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def nchw_to_nhwc(t):
    return tf.transpose(t, perm=[0, 2, 3, 1], name='to_NHWC')


def nhwc_to_nchw(t):
    return tf.transpose(t, perm=[0, 3, 1, 2], name='to_NCHW')


def clip_to_rgb_range(_t):
    return tf.clip_by_value(_t, 0, 255, 'clip')


def close_summary_writers(*writers):
    for w in writers:
        w.flush()
        w.close()


def convert_data_format_naming(data_format):
    if data_format == 'NHWC':
        return 'channels_last'
    elif data_format == 'NCHW':
        return 'channels_first'
    elif data_format == 'channels_last':
        return 'NHWC'
    elif data_format == 'channels_first':
        return 'NCHW'
    else:
        raise ValueError('unknown data_format: {}'.format(data_format))

