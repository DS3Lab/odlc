import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_records', required=True, help='records file with training data')
parser.add_argument('--val_records', required=True, help='records file with validation data')
parser.add_argument('--job_id', required=False, help='id for training run')
parser.add_argument('--alpha', required=True, help='param controlling trade off')
parser.add_argument('--config', required=True, help='config.json file')
parser.add_argument('--vgg_weights', required=False, help='path to .npy file with vgg weights for feature reconst loss')
options = parser.parse_args()


def main(_opts):
    from src.compression.rnn.train_model import train

    if _opts.vgg_weights is not None:
        feature_loss_kwargs = {'weights_file': _opts.vgg_weights}
    else:
        feature_loss_kwargs = None

    train('imagenet', _opts.config, _opts.job_id, _opts.train_records, _opts.val_records, alpha=_opts.alpha,
          feature_loss_kwargs=feature_loss_kwargs)


if __name__ == '__main__':
    main(options)
