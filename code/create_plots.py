import argparse

COMPR_KEYS = ['jpeg', 'bpg', 'webp', 'gru0', 'gru025', 'gru05', 'gru075', 'gru1', 'cpm']
_COMPR_KEYS_DEFAULTS = ['gru0', 'gru1', 'cpm', 'bpg', 'webp', 'jpeg']
ALLOWED_CLASSIFIERS = ['all', 'densenet_121', 'inception_resnet_v2', 'inception_v3', 'mobilenet', 'resnet_50', 'vgg16',
                       'vgg19', 'xception', 'resnet_v1_50', 'vgg_16', 'mobilenet_v1']

parser = argparse.ArgumentParser()

# general args
parser.add_argument('--dataset', required=False)
parser.add_argument('--type', required=True, choices=['accuracy', 'hvs', 'tradeoff'])
parser.add_argument('--show', required=False, action='store_true', help='')
parser.add_argument('--save', required=False, action='store_true', help='')

# accuarcy args
parser.add_argument('--acc_csv_dir', help='dir contatining accuracy csv files')
parser.add_argument('--classifiers', nargs='+', default=['inception_resnet_v2'], choices=ALLOWED_CLASSIFIERS)
parser.add_argument('--compr_keys', required=False, nargs='+', default=_COMPR_KEYS_DEFAULTS,
                    help='list of compression keys', choices=COMPR_KEYS)
parser.add_argument('--topk', required=False, type=int, default=1, help='topk accuracy', choices=[1, 5])
parser.add_argument('--bpp_min', required=False, type=float, default=0.1)

# hvs args
parser.add_argument('--hvs_csv_dir', required=False, help='dir contatining hvs logs')

# tradeoff args
parser.add_argument('--bitrate', required=False, type=float, help='')
parser.add_argument('--baselines', required=False, nargs='+', help='')

# plot style args
parser.add_argument('--title', required=False, action='store_true')
parser.add_argument('--figsize', required=False, nargs='+', type=float, default=None)
parser.add_argument('--ymin', required=False, type=float, default=None)

options = parser.parse_args()

_ACCURACY_DEFAULTS = {
    'imagenet': {'xmin': 0.09375, 'xmax': None, 'ymin': 0, 'ymax': 82, 'figsize': (3.7, 4.1), 'logscale': True},
    'cub200': {'xmin': 0.09375, 'xmax': None, 'ymin': 0, 'ymax': 82, 'figsize': (3.7, 4.1), 'logscale': True},
    'stanford_dogs': {'xmin': 0.09375, 'xmax': None, 'ymin': 0, 'ymax': 95, 'figsize': (3.7, 4.1), 'logscale': True}
}

_HVS_DEFAULTS = {
    'kodak': {'xmin': 0.03125, 'xmax': 1.0625, 'ymin': 0.76, 'ymax': 1.005, 'figsize': (3.9, 4.2)},
    'imagenet': {'xmin': 0.03125, 'xmax': 1.0625, 'ymin': 0.76, 'ymax': 1.005, 'figsize': (3.9, 4.2)},
    'cub200': {'xmin': 0.03125, 'xmax': 1.0625, 'ymin': 0.76, 'ymax': 1.005, 'figsize': (3.9, 4.2)},
    'stanford_dogs': {'xmin': 0.03125, 'xmax': 1.0625, 'ymin': 0.76, 'ymax': 1.005, 'figsize': (3.9, 4.2)},
}

_TRADEOFF_DEFAULTS = {'ymin': 0.63, 'ymax': 1.02, 'figsize': (5, 2.8), 'baselines': ['cpm', 'bpg', 'webp']}

_FONT_NAME = 'times_roman'
_PERC_METRIC = 'ms_ssim'


def main(_opts):
    if _opts.type == 'accuracy':
        assert _opts.dataset is not None
        assert _opts.acc_csv_dir is not None

        from src.eval.plotting.accuracy.create import AccuracyPlots

        accuracy_plotter = AccuracyPlots(_opts.acc_csv_dir, _opts.dataset)

        if _opts.classifiers[0] == 'all':
            classifiers = ALLOWED_CLASSIFIERS

        else:
            classifiers = _opts.classifiers

        if _opts.figsize is not None:
            _ACCURACY_DEFAULTS[_opts.dataset]['figsize'] = _opts.figsize

        if _opts.ymin is not None:
            _ACCURACY_DEFAULTS[_opts.dataset]['ymin'] = _opts.ymin

        if _opts.compr_keys is not None:
            compression_keys = [str(x) for x in _opts.compr_keys]
        else:
            compression_keys = _COMPR_KEYS_DEFAULTS

        for classifier in classifiers:
            accuracy_plotter.make_plot(classifier,
                                       show=_opts.show,
                                       save=_opts.save,
                                       compression_keys=compression_keys,
                                       topk=_opts.topk,
                                       make_title=_opts.title,
                                       font_name=_FONT_NAME,
                                       bpp_min=_opts.bpp_min,
                                       **_ACCURACY_DEFAULTS[_opts.dataset])

        return

    if _opts.type == 'hvs':
        assert _opts.dataset is not None
        assert _opts.hvs_csv_dir is not None

        from src.eval.plotting.hvs.create import HVSPlots

        hvs_plotter = HVSPlots(_opts.hvs_csv_dir, _opts.dataset)

        if _opts.compr_keys is not None:
            compression_keys = [str(x) for x in _opts.compr_keys]
        else:
            compression_keys = _COMPR_KEYS_DEFAULTS

        if _opts.figsize is not None:
            _HVS_DEFAULTS[_opts.dataset]['figsize'] = _opts.figsize

        hvs_plotter.make_plot(perception_metric=_PERC_METRIC,
                              show=_opts.show,
                              save=_opts.save,
                              compression_keys=compression_keys,
                              font_name=_FONT_NAME,
                              bpp_min=_opts.bpp_min,
                              **_HVS_DEFAULTS[_opts.dataset])

        return

    if _opts.type == 'tradeoff':
        from src.eval.plotting.tradeoff.create import TradeOffPlots

        plotter = TradeOffPlots(_opts.acc_csv_dir, _opts.hvs_csv_dir, _opts.dataset)

        for c in _opts.classifiers:
            plotter.make_plot(classifier=c,
                              show=_opts.show,
                              save=_opts.save,
                              target_bitrate=_opts.bitrate,
                              hvs_metric=_PERC_METRIC,
                              topk=_opts.topk,
                              font_name=_FONT_NAME,
                              **_TRADEOFF_DEFAULTS)

        return


if __name__ == '__main__':
    main(options)
