import os
import scipy.io as scipy_io


class LabelConverter:
    """ conversion between keras labels, original labels, synsets, class names"""

    def __init__(self, meta_dir):
        self._meta_dir = os.path.abspath(meta_dir)
        self.__init_converters()

    def keras_index_to_name(self, idx):
        return self._keras_idx_to_name[idx]

    def keras_index_to_synset(self, idx):
        return self._keras_idx_to_synset[idx]

    def synset_to_keras_index(self, synset):
        return self._synset_to_keras_idx[synset]

    def synset_to_name(self, synset):
        return self._synset_to_name[synset]

    def original_index_to_keras_index(self, idx):
        return self._synset_to_keras_idx[self._original_idx_to_synset[idx]]

    def original_index_to_synset(self, idx):
        return self._original_idx_to_synset[idx]

    def __init_converters(self):
        # label processing
        meta = scipy_io.loadmat(os.path.join(self._meta_dir, 'devkit/data/meta.mat'))
        self._original_idx_to_synset = {}
        self._synset_to_name = {}
        self._synset_to_keras_idx = {}
        self._keras_idx_to_name = {}

        for i in range(1000):
            ilsvrc2012_id = int(meta['synsets'][i, 0][0][0][0])
            synset = meta['synsets'][i, 0][1][0]
            name = meta['synsets'][i, 0][2][0]
            self._original_idx_to_synset[ilsvrc2012_id] = synset
            self._synset_to_name[synset] = name

        with open(os.path.join(self._meta_dir, 'synset_words.txt'), 'r') as f:
            idx = 0
            for line in f:
                parts = line.strip().split(" ")
                self._synset_to_keras_idx[parts[0]] = idx
                self._keras_idx_to_name[idx] = " ".join(parts[1:])
                idx += 1

        self._keras_idx_to_synset = {v: k for k, v in self._synset_to_keras_idx.items()}
