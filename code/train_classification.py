"""
usage:

python3 train_classifier.py --dataset <ds_name> --train_records path/to/train/records \
    --val_records path/to/val/records --job_id 251 --config path/to/config --eval_epochs 1 --checkpoint_epochs 10 \
    --classifier <model_name> --pretrained_model path/to/pretrained/model

"""

import argparse

_CLASSIFIERS = ['inception_v3', 'mobilenet_v1', 'resnet_v1_50', 'vgg_16']
_DATASETS = ['stanford_dogs', 'cub200']


parser = argparse.ArgumentParser()
parser.add_argument('--classifier', required=True, choices=_CLASSIFIERS)
parser.add_argument('--dataset', required=False, choices=_DATASETS)
parser.add_argument('--train_records', required=True, help='records file with training data')
parser.add_argument('--val_records', required=False, help='records file with validation data')
parser.add_argument('--job_id', required=False, help='id for training run')
parser.add_argument('--pretrained_model', required=False, help='path to a pretrained model')
parser.add_argument('--eval_epochs', required=False, default=1, type=int, help='eval every x epochs')
parser.add_argument('--checkpoint_epochs', required=False, default=10, type=int, help='save model every x epochs')

options = parser.parse_args()


def main(_opts):
    assert _opts.pretrained_model is not None

    from src.classification.fine_grained_categorization.training.train_model import train

    train(model_name=_opts.classifier,
          dataset_name=_opts.dataset,
          train_records=_opts.train_records,
          val_records=_opts.val_records,
          init_checkpoint_path=_opts.pretrained_model,
          job_id=_opts.job_id,
          eval_epchs=_opts.eval_epochs,
          checkpoint_epochs=_opts.checkpoint_epochs)


if __name__ == '__main__':
    main(options)
