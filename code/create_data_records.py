import argparse

_DATASETS = ['imagenet', 'cub200', 'stanford_dogs']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str, choices=_DATASETS)
parser.add_argument('--split', required=True, type=str, choices=['train', 'val'])
parser.add_argument('--data_dir', required=True, type=str, help='dir where data is located')
parser.add_argument('--target_dir', required=True, type=str, help='dir where records files are saved to')

options = parser.parse_args()


def main(_opts):
    if _opts.dataset == 'imagenet':
        if _opts.split == 'val':
            from src.data.dataprepare.imagenet.records_writer import ValRecordsWriter

            records_writer = ValRecordsWriter(_opts.data_dir, _opts.target_dir)
            records_writer.process_files()

            return

        if _opts.split == 'train':
            from src.data.dataprepare.imagenet.records_writer import TrainRecordsWriter

            records_writer = TrainRecordsWriter(_opts.data_dir, _opts.target_dir)
            records_writer.process_files()
            return

    if _opts.dataset == 'stanford_dogs':
        if _opts.split == 'val':
            from src.data.dataprepare.stanford_dogs.records_writer import ValRecordsWriter

            records_writer = ValRecordsWriter(_opts.data_dir, _opts.target_dir)
            records_writer.process_files()

            return

        if _opts.split == 'train':
            from src.data.dataprepare.stanford_dogs.records_writer import TrainRecordsWriter

            records_writer = TrainRecordsWriter(_opts.data_dir, _opts.target_dir)
            records_writer.process_files()
            return

    if _opts.dataset == 'cub200':
        assert _opts.data_dir is not None
        from src.data.dataprepare.cub200.records_writer import RecordsWriter

        records_writer = RecordsWriter(_opts.data_dir, 1 if _opts.split == 'train' else 0, _opts.target_dir)
        records_writer.process_files()
        return


if __name__ == '__main__':
    main(options)
