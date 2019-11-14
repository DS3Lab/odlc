class EvalValues:
    def __init__(self, bits_per_pixel, accuracy_dicts=None, hvs_dicts=None, other_info_dicts=None):
        self.bits_per_pixel = bits_per_pixel
        self.__num_compression_levels = len(bits_per_pixel)

        self.hvs_dicts = self.__check_and_transform_list(hvs_dicts)
        self.accuracy_dicts = self.__check_and_transform_list(accuracy_dicts)
        self.other_info_dicts = self.__check_and_transform_list(other_info_dicts)

    def __check_and_transform_list(self, ls):
        if ls is None:
            return [None for _ in range(self.__num_compression_levels)]
        else:
            if isinstance(ls, list) and len(ls) == self.__num_compression_levels:
                return ls
            else:
                return [None for _ in range(self.__num_compression_levels)]
