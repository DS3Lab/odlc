import inspect
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, LogLocator, FormatStrFormatter, NullFormatter

import json
import matplotlib.pyplot as plt
import numpy as np
import os

from src.lib.logging_commons.logs_parser import LogsParser
from src.lib.logging_commons.utils import read_csv
from src.eval.plotting.names import COMPRESSION_NAMES, CLASSIFIER_NAMES, DATASET_NAMES
from src.eval.plotting.fonts import get_font, FONTSIZES
from src.eval.plotting.colors import COLORMAP

CSV_FILE_BASE = '{}_{}_accuracy.csv'
FACECOLOR = (1, 1, 1)


class AccuracyPlots:

    def __init__(self, csv_dir, dataset):

        assert os.path.exists(csv_dir), 'csv_dir not found!'

        self._module_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self._csv_dir = csv_dir
        self._dataset = dataset
        self._plots_save_dir = os.path.join(self._module_dir, 'plots/', dataset + '/')

    def load_config(self, compression_key, classifier):
        with open(os.path.join(self._module_dir, 'configs.json'), 'r') as fo:
            configs = json.load(fo)
        cfg = configs[compression_key]
        return {**cfg, 'csv_file': os.path.join(self._csv_dir, CSV_FILE_BASE.format(compression_key, classifier))}

    @staticmethod
    def keys_to_color(keys):
        return {key: COLORMAP[key] for key in keys}

    def make_plot(self, classifier, show, save, compression_keys, topk=1, make_title=False, xmin=0.032, xmax=None,
                  ymin=0, ymax=100, figsize=(6, 3.75), logscale=True, font_name='times_roman', legend_loc='lower right',
                  bpp_min=None):

        if compression_keys is None:
            raise ValueError('compression_keys must not be None')

        if not isinstance(compression_keys, list):
            compression_keys = list([compression_keys])

        if int(topk) != 1 and int(topk) != 5:
            raise ValueError('topk must either be `1` or `5`')

        topkacc_kw = LogsParser.TOP1ACC_KW if int(topk) == 1 else LogsParser.TOP5ACC_KW
        topkacc_idx = 1 if int(topk) == 1 else 2

        configs = {ckey: self.load_config(ckey, classifier) for ckey in compression_keys}
        csv_original = os.path.join(self._csv_dir, CSV_FILE_BASE.format('original', classifier))
        corrupted_logfiles = []

        if os.path.isfile(csv_original):
            acc_data_lossless = np.array(read_csv(csv_original), dtype=np.float32)
        else:
            print('WARNING! {} not found.'.format(csv_original))
            corrupted_logfiles.append('original')
            acc_data_lossless = None

        # ========= parse csv files
        parsed_data = {}
        for key in compression_keys:
            cfg = configs[key]
            csv_file = cfg['csv_file']

            if not os.path.isfile(csv_file):
                print('WARNING! {} not found.'.format(csv_file))
                corrupted_logfiles.append(key)
                continue

            acc_data = np.array(read_csv(csv_file), dtype=np.float32)

            # sort data
            acc_data = acc_data[acc_data[:, 0].argsort()]

            if bpp_min is not None:
                include_idx = np.where(acc_data[:, 0] >= float(bpp_min))[0]
                acc_data = acc_data[include_idx, :]

            parsed_data[key] = acc_data[:, 0], 100.0 * acc_data[:, topkacc_idx]

        if len(parsed_data) == 0:
            print('parsed_data empty')
            return

        compression_keys = list([k for k in compression_keys if k not in corrupted_logfiles])
        configs = {k: configs[k] for k in compression_keys}
        compression_colors = self.keys_to_color([k for k in compression_keys])

        # ========= make plot
        # determine plot boundaries
        all_bpp_data = np.unique(np.concatenate([data[0] for data in parsed_data.values()]))
        if xmax is None:
            xmax = all_bpp_data.max() + 2.0
        x_lim = xmin, xmax
        y_lim = ymin, ymax

        # setup fig
        fig = plt.figure(figsize=figsize)

        # plot data
        ax = plt.gca()
        plot_func = ax.semilogx if logscale else ax.plot
        for key in compression_keys:
            bpp_array, acc_array = parsed_data[key]
            cfg = configs[key]

            plot_func(bpp_array,
                      acc_array,
                      lw=cfg['lw'],
                      color=compression_colors[key],
                      label=COMPRESSION_NAMES[key],
                      marker=cfg['marker'],
                      markersize=3 * cfg['lw'],
                      linestyle=cfg['linestyle'])

        if acc_data_lossless is not None:
            ax.axhline(100.0 * acc_data_lossless[0, topkacc_idx], xmin=0, xmax=24, color='dimgrey', lw=0.75,
                       linestyle='--')
            plot_func(acc_data_lossless[:, 0], 100.0 * acc_data_lossless[:, topkacc_idx], marker='^', markersize=6,
                      color='red')

        # format plot
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        ax.set_xlabel('bpp', fontproperties=get_font(font_name, FONTSIZES.Large))
        ax.set_ylabel('Validation Accuracy (%)', fontproperties=get_font(font_name, FONTSIZES.Large))
        ax.grid(True, color=(0.91, 0.91, 0.91), linewidth=0.5)

        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1d'))

        if logscale:
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(1.2, 1.4, 1.6, 1.8)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_locator(LogLocator(base=2))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(0.125))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.125))

        ax.tick_params(which='minor', width=0.4)

        ax.set_facecolor(FACECOLOR)
        for label in ax.get_xticklabels():
            label.set_fontproperties(get_font(font_name, FONTSIZES.large))
        for label in ax.get_yticklabels():
            label.set_fontproperties(get_font(font_name, FONTSIZES.large))
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('black')

        # legend
        legend_labels = [COMPRESSION_NAMES[k] for k in compression_keys]
        legend_labels += ['Original']
        legend = plt.legend(**self._get_legend_kwargs(configs=[(k, configs[k]) for k in compression_keys],
                                                      labels=legend_labels,
                                                      legend_loc=legend_loc,
                                                      font_name=font_name,
                                                      compression_colors=compression_colors))
        ax.add_artist(legend)

        # title
        if make_title:
            plt.suptitle(
                t='Validation Accuracy on {}, Top-{}, %'.format(DATASET_NAMES[self._dataset],
                                                                1 if topkacc_kw == LogsParser.TOP1ACC_KW else 5),
                fontproperties=get_font(font_name, FONTSIZES.Large))
            plt.title(CLASSIFIER_NAMES[classifier], fontproperties=get_font(font_name, FONTSIZES.large))
            plt.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.86)

        else:
            fig.tight_layout()

        if show:
            plt.show()

        if save:
            if not os.path.exists(self._plots_save_dir):
                os.makedirs(self._plots_save_dir)

            save_as = '{}_accuracy_{}_{}.png'.format(self._dataset, classifier, topkacc_kw)
            fig.savefig(os.path.join(self._plots_save_dir, save_as), dpi=200)
            print('plot saved as {}'.format(os.path.join(self._plots_save_dir, save_as)))

        plt.close(fig)

    @staticmethod
    def _get_legend_kwargs(configs, labels, legend_loc, font_name, compression_colors):

        def get_line(_cfg=None, key=None, color=None, linestyle=None, marker=None, **kwargs):
            if _cfg is not None:
                marker = _cfg['marker']
                color = compression_colors[key]
                linestyle = _cfg['linestyle']

            return Line2D([0], [0], color=color, lw=1.2, linestyle=linestyle, marker=marker, **kwargs)

        # legend
        custom_lines_compressors = [get_line(cfg, k) for k, cfg in configs]
        marker_kwargs = dict(markerfacecolor='red', markeredgecolor='red')
        custom_lines_compressors += [get_line(color='dimgrey', linestyle='--', marker='^', **marker_kwargs)]

        return dict(fancybox=False,
                    framealpha=0.4,
                    handles=custom_lines_compressors,
                    labels=labels,
                    loc=legend_loc,
                    prop=get_font(font_name, FONTSIZES.big),
                    ncol=1)
