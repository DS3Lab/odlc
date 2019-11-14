from tensorflow import parse_single_example, FixedLenFeature
from tensorflow.python.framework import dtypes


class RecordsParser:
    # features kewywords
    KW_IMAGE_BYTES = 'image_bytes'
    KW_LABEL = 'label'
    KW_BPP = 'bpp'

    # records types
    RECORDS_LABELLED = 'labelled'
    RECORDS_UNLABELLED = 'unlabelled'
    RECORDS_BPP = 'bpp'
    RECORDS_LABELLED_BPP = 'labels_and_bpp'

    ALLOWED_RECORDS_TYPES = [RECORDS_LABELLED, RECORDS_UNLABELLED, RECORDS_BPP, RECORDS_LABELLED_BPP]

    @classmethod
    def parse_example(cls, example, records_type):
        mode_map = {cls.RECORDS_LABELLED: cls.parse_example_with_label,
                    cls.RECORDS_UNLABELLED: cls.parse_example_unlabelled,
                    cls.RECORDS_BPP: cls.parse_example_with_bpp,
                    cls.RECORDS_LABELLED_BPP: cls.parse_example_with_label_bpp}

        return mode_map[records_type](example)

    @classmethod
    def parse_example_unlabelled(cls, example):
        return parse_single_example(example, features={
            cls.KW_IMAGE_BYTES: FixedLenFeature([], dtypes.string)})

    @classmethod
    def parse_example_with_label(cls, example):
        return parse_single_example(example, features={
            cls.KW_IMAGE_BYTES: FixedLenFeature([], dtypes.string),
            cls.KW_LABEL: FixedLenFeature([], dtypes.int64)})

    @classmethod
    def parse_example_with_bpp(cls, example):
        return parse_single_example(example, features={
            cls.KW_IMAGE_BYTES: FixedLenFeature([], dtypes.string),
            cls.KW_BPP: FixedLenFeature([], dtypes.float32)})

    @classmethod
    def parse_example_with_label_bpp(cls, example):
        return parse_single_example(example, features={
            cls.KW_IMAGE_BYTES: FixedLenFeature([], dtypes.string),
            cls.KW_LABEL: FixedLenFeature([], dtypes.int64),
            cls.KW_BPP: FixedLenFeature([], dtypes.float32)
        })
