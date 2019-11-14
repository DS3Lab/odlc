import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['imagenet', 'stanford_dogs', 'cub200', 'kodak'])
parser.add_argument('--src_records', required=True, help='source records that need to be compressed')
parser.add_argument('--target_dir', required=True, help='dir where records files are saved to')
parser.add_argument('--target_height', required=True, type=int, help='target height to which image should be resized')
parser.add_argument('--target_width', required=True, type=int, help='target width to which image should be resized')
parser.add_argument('--quantization', '-q', required=True, help='bpg quantization level')
parser.add_argument('--num_cores', '-nc', default=4, type=int, required=False, help='num cpus available')

options = parser.parse_args()


def main(_opts):
    from src.data.datasets import Kodak
    from src.data.dataloading.records_parsing import RecordsParser
    from src.compression.bpg.create_records import CreateRecords

    if _opts.dataset == Kodak.NAME:
        records_type = RecordsParser.RECORDS_UNLABELLED
    else:
        records_type = RecordsParser.RECORDS_LABELLED

    records_writer = CreateRecords(_opts.src_records, _opts.target_dir, _opts.quantization, _opts.target_height,
                                   _opts.target_width, _opts.num_cores, records_type)
    records_writer.run()


if __name__ == '__main__':
    main(options)
