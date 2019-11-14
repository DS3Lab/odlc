import numpy as np


def clip_to_rgb(inputs):
    return np.clip(inputs, a_min=0, a_max=255)


def nchw_to_nhwc(inputs):
    return np.transpose(inputs, axes=[0, 2, 3, 1])


def nhwc_to_nchw(inputs):
    return np.transpose(inputs, axes=[0, 3, 1, 2])
