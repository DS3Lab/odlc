import inspect
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import json
import matplotlib.pyplot as plt
import numpy as np
import os

from src.eval.plotting.names import COMPRESSION_NAMES, CLASSIFIER_NAMES
from src.eval.plotting.fonts import get_font, FONTSIZES
from src.lib.logging_commons.utils import read_csv

_CSV_FILE_BASE_ACCURACY = '{}_{}_accuracy.csv'
_CSV_FILE_BASE_HVS = '{}_hvs.csv'
_FACECOLOR = (1, 1, 1)
_GRID_COLOR = (0.91, 0.91, 0.91)

_LINEWIDTH = 2.5

_BASELINE_IDX_METHOD = 0
_BASELINE_IDX_BPP = 1
_BASELINE_IDX_VAL = 2

_RNN_IDX_ALPHA = 0
_RNN_IDX_VAL = 1


def convert_linestyle(ls):
    if ls == 'denselydasheddotdotted':
        return tuple((0, (3, 1, 1, 1, 1, 1)))
    return ls


class TradeOffPlots:
    ALPHA_VALUES = [0, 0.25, 0.5, 0.75, 1]

    def __init__(self, acc_csv_dir, hvs_csv_dir, dataset):

        assert os.path.exists(acc_csv_dir), 'acc_csv_dir not found!'
        assert os.path.exists(hvs_csv_dir), 'hvs_csv_dir not found!'

        self._module_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self._acc_csv_dir = acc_csv_dir
        self._hvs_csv_dir = hvs_csv_dir
        self._dataset = dataset
        self._plots_save_dir = os.path.join(self._module_dir, 'plots/', dataset + '/')
        self._configs = self._load_config()

    def _load_config(self):
        with open(os.path.join(self._module_dir, 'configs.json'), 'r') as fo:
            configs = json.load(fo)
        return configs

    def make_plot(self, classifier, show, save, target_bitrate, hvs_metric, ymin, ymax, figsize, baselines=None,
                  topk=1, font_name='times_roman'):

        # hvs_color = 'darkturquoise'
        # hvs_color = 'palevioletred'
        hvs_color = 'grey'
        acc_color = 'midnightblue'

        if baselines is None:
            baselines = []

        # ========= get data
        # get rnn compression data
        rnn_accuracy_data = self.get_accuracy_data_rnn(target_bitrate, classifier, topk, True)
        rnn_hvs_data = self.get_hvs_data_rnn(target_bitrate, hvs_metric)

        baselines_accuracy_data = self.get_accuracy_data_baselines(target_bitrate, classifier, topk, baselines, True,
                                                                   True)
        baselines_hvs_data = self.get_hvs_data_baselines(target_bitrate, hvs_metric, baselines, True)

        # ========= make plot
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        # baseline data
        for baseline_hvs, key in zip(baselines_hvs_data, baselines):
            ax1.axhline(float(baseline_hvs[_BASELINE_IDX_VAL]), color=hvs_color,
                        linestyle=convert_linestyle(self._configs[key]['linestyle']), label=COMPRESSION_NAMES[key],
                        lw=self._configs[key]['lw'])

        for baseline_acc, key in zip(baselines_accuracy_data, baselines):
            ax1.axhline(float(baseline_acc[_BASELINE_IDX_VAL]), color=acc_color,
                        linestyle=convert_linestyle(convert_linestyle(self._configs[key]['linestyle'])),
                        label=COMPRESSION_NAMES[key], lw=self._configs[key]['lw'])

        # == fill in data RNN compression
        # left y-axis (MS-SSIM)
        ax1.set_xlabel(r'$\alpha$', fontproperties=get_font(font_name, FONTSIZES.Large))
        ax1.set_ylabel('MS-SSIM', fontproperties=get_font(font_name, FONTSIZES.Large), labelpad=10, color=hvs_color)
        ax1.plot(rnn_hvs_data[:, _RNN_IDX_ALPHA], rnn_hvs_data[:, _RNN_IDX_VAL], color=hvs_color,
                 lw=self._configs['rnn']['lw'], linestyle=self._configs['rnn']['linestyle'],
                 markersize=3 * self._configs['rnn']['lw'], marker=self._configs['rnn']['marker'])
        ax1.tick_params(axis='y', labelcolor=hvs_color)
        ax1.tick_params(axis='y', which='minor', width=0.7, colors=hvs_color)
        ax1.tick_params(axis='y', which='major', colors=hvs_color)

        # right y-axis (Accuracy)
        ax2.set_ylabel('Preserved Val. Accuracy', fontproperties=get_font(font_name, FONTSIZES.Large), labelpad=10,
                       color=acc_color)
        ax2.plot(rnn_accuracy_data[:, _RNN_IDX_ALPHA], rnn_accuracy_data[:, _RNN_IDX_VAL], color=acc_color,
                 lw=self._configs['rnn']['lw'], linestyle=self._configs['rnn']['linestyle'],
                 markersize=3 * self._configs['rnn']['lw'], marker=self._configs['rnn']['marker'])
        ax2.tick_params(axis='y', labelcolor=acc_color)
        ax2.tick_params(axis='y', which='minor', width=0.7, colors=acc_color)
        ax2.tick_params(axis='y', which='major', colors=acc_color)

        # == format
        # axis limits
        ax1.set_xlim((-0.05, 1.05))
        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))

        # ticks
        ax1.yaxis.set_major_locator(MultipleLocator(0.05))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax1.set_xticks(np.arange(0, 1.25, 0.25))
        ax2.yaxis.set_major_locator(MultipleLocator(0.05))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax2.set_xticks(np.arange(0, 1.25, 0.25))

        # fontprops
        for labelx in ax1.get_xticklabels():
            labelx.set_fontproperties(get_font(font_name, FONTSIZES.large))

        for labely1 in ax1.get_yticklabels():
            labely1.set_fontproperties(get_font(font_name, FONTSIZES.large))
            labely1.set_color(hvs_color)

        for labely2 in ax2.get_yticklabels():
            labely2.set_fontproperties(get_font(font_name, FONTSIZES.large))
            labely2.set_color(acc_color)

        # grid, facecolor
        ax1.grid(True, color=_GRID_COLOR, linewidth=0.5)
        ax1.set_facecolor(_FACECOLOR)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(acc_color)
        ax2.spines['left'].set_visible(True)
        ax2.spines['left'].set_color(hvs_color)

        # legend for compression method
        legend1 = ax1.legend(**self._get_compression_legend_kwargs(configs=self._configs,
                                                                   legend_loc='lower center',
                                                                   font_name=font_name,
                                                                   compression_keys=['rnn', *baselines]))
        ax1.add_artist(legend1)
        fig.tight_layout()

        if show:
            plt.show()

        if save:
            if not os.path.exists(self._plots_save_dir):
                os.makedirs(self._plots_save_dir)

            save_as = '{}_tradeoff_{}_{}_{}bpp.png'.format(self._dataset, hvs_metric, classifier, target_bitrate)
            fig.savefig(os.path.join(self._plots_save_dir, save_as), dpi=200)
            print('plot saved as {}'.format(os.path.join(self._plots_save_dir, save_as)))

        plt.close(fig)

    @classmethod
    def _get_compression_legend_kwargs(cls, configs, legend_loc, font_name, compression_keys):
        custom_lines = list([])

        for key in compression_keys:
            custom_lines.append(cls.get_line({**configs[key], 'marker': 'None', 'color': 'black', 'lw': 1.2}))

        return dict(fancybox=False,
                    framealpha=0.4,
                    handles=custom_lines,
                    labels=[COMPRESSION_NAMES[k] for k in compression_keys],
                    loc=legend_loc,
                    bbox_to_anchor=(0.0, 0.02, 1., .102),
                    prop=get_font(font_name, FONTSIZES.medium),
                    ncol=2)

    def get_accuracy_data_baselines(self, target_bitrate, classifier, topk, baselines, normalize=False,
                                    first_bigger=False):
        """ parse csv files for accuracy values

        args:
          target_bitrate: float.
          classifier: string.
          topk: int. Either 1 or 5

        returns:
          2-dim numpy array of accuracy values for different values of alpha, floats
        """
        csv_files = []
        for b in baselines:
            fn = _CSV_FILE_BASE_ACCURACY.format(b, classifier)
            fp = os.path.join(self._acc_csv_dir, fn)
            csv_files.append(fp)

        topk_idx = 1 if topk == 1 else 2

        original_accuracy = 1.0
        if normalize:
            original_accuracy_file = os.path.join(self._acc_csv_dir,
                                                  _CSV_FILE_BASE_ACCURACY.format('original', classifier))
            original_accuracy = float(self.parse_accuracy_csv(original_accuracy_file, 10.0, False)[topk_idx])

        data_mat = []
        for k, f in zip(baselines, csv_files):

            if not os.path.isfile(f):
                print('::WARNING:: file not found {}'.format(f))
                continue

            data_row = self.parse_accuracy_csv(f, target_bitrate, first_bigger)
            data_mat.append([k, float(data_row[0]), float(data_row[topk_idx]) / original_accuracy])

        return data_mat

    def get_hvs_data_baselines(self, target_bitrate, hvs_metric, baselines, first_bigger=False):
        """ parse csv files for hvs values

        args:
          target_bitrate: float.
          hvs_metric: string.

        returns:
          2-dim numpy array of accuracy values for different values of alpha, floats
        """
        csv_files = []
        for b in baselines:
            fn = _CSV_FILE_BASE_HVS.format(b)
            fp = os.path.join(self._hvs_csv_dir, fn)
            csv_files.append(fp)

        data_mat = []
        for k, f in zip(baselines, csv_files):

            if not os.path.isfile(f):
                print('::WARNING:: file not found {}'.format(f))
                continue

            data = self.parse_hvs_csv(f, target_bitrate, hvs_metric, first_bigger)
            data_mat.append([k, *data])

        return data_mat

    def get_accuracy_data_rnn(self, target_bitrate, classifier, topk, normalize=False):
        """ parse csv files for accuracy values

        args:
          target_bitrate: float.
          classifier: string.
          topk: int. Either 1 or 5

        returns:
          2-dim numpy array of accuracy values for different values of alpha, floats
        """
        csv_files = []
        for a in self.ALPHA_VALUES:
            fn = _CSV_FILE_BASE_ACCURACY.format('gru{}'.format(str(a).replace('.', '')), classifier)
            fp = os.path.join(self._acc_csv_dir, fn)
            csv_files.append(fp)

        topk_idx = 1 if topk == 1 else 2

        original_accuracy = 1.0
        if normalize:
            original_accuracy_file = os.path.join(self._acc_csv_dir,
                                                  _CSV_FILE_BASE_ACCURACY.format('original', classifier))
            original_accuracy = float(self.parse_accuracy_csv(original_accuracy_file, 10.0, False)[topk_idx])

        data_mat = []
        for a, f in zip(self.ALPHA_VALUES, csv_files):

            if not os.path.isfile(f):
                print('::WARNING:: file not found {}'.format(f))
                continue

            data_row = self.parse_accuracy_csv(f, target_bitrate, False)
            data_mat.append([1 - a, float(data_row[topk_idx]) / original_accuracy])

        return np.array(data_mat, np.float32)

    def get_hvs_data_rnn(self, target_bitrate, hvs_metric):
        """ parse csv files for hvs values """
        csv_files = []
        for a in self.ALPHA_VALUES:
            fn = _CSV_FILE_BASE_HVS.format('gru{}'.format(str(a).replace('.', '')))
            fp = os.path.join(self._hvs_csv_dir, fn)
            csv_files.append(fp)

        data_mat = []
        for a, f in zip(self.ALPHA_VALUES, csv_files):

            if not os.path.isfile(f):
                print('::WARNING:: file not found {}'.format(f))
                continue

            _, data = self.parse_hvs_csv(f, target_bitrate, hvs_metric, first_bigger=False)
            data_mat.append([1 - a, data])

        return np.array(data_mat, np.float32)

    @staticmethod
    def parse_accuracy_csv(csv_file, target_bitrate, first_bigger):
        data_rows = read_csv(csv_file)
        diff = 999
        accuracy_data = None

        if first_bigger:
            print('\n---> ', csv_file)
            data_rows = np.array(data_rows, dtype=np.float32)
            data_rows.sort(axis=0)
            for row in data_rows:
                print('* Accuracy', row)
                if row[0] >= target_bitrate:
                    return list(row)

        for row in data_rows:
            if abs(float(row[0]) - target_bitrate) <= diff:
                accuracy_data = list(row)
                diff = abs(float(row[0]) - target_bitrate)

        return accuracy_data

    @staticmethod
    def parse_hvs_csv(csv_file, target_bitrate, hvs_metric, first_bigger):
        data_rows = read_csv(csv_file)
        header = data_rows[0]
        col_idx = header.index(hvs_metric)

        hvs_data = None
        diff = 999

        if first_bigger:
            print('\n---> ', csv_file)
            data_rows = np.array(data_rows[1:], dtype=np.float32)
            data_rows.sort(axis=0)
            for row in data_rows:
                print('* HVS', row, row.shape)
                if row[0] >= target_bitrate:
                    return row[0], row[col_idx]

        for row in data_rows[1:]:
            new_diff = abs(float(row[0]) - target_bitrate)
            if new_diff <= diff:
                hvs_data = row[0], row[col_idx]
                diff = float(new_diff)

        return hvs_data

    @staticmethod
    def get_line(config):
        return Line2D([0], [0], color=config['color'], lw=config['lw'],
                      linestyle=convert_linestyle(config['linestyle']), marker=config['marker'])

# @classmethod
# def _get_axes_legends_kwargs(cls, legend_loc, font_name):
#     lines = list([cls.get_line({'color': 'black', 'marker': None, 'linestyle': '--', 'lw': 1.5})])
#     lines.append(cls.get_line({'color': 'black', 'marker': None, 'linestyle': '-', 'lw': 1.5}))
#
#     return dict(fancybox=False,
#                 framealpha=0.4,
#                 handles=lines,
#                 labels=['MS-SSIM', 'Val. Accuracy'],
#                 loc=legend_loc,
#                 prop=get_font(font_name, FONTSIZES.big),
#                 bbox_to_anchor=(0., 0.15, 1., .15),
#                 borderaxespad=0.,
#                 # mode="expand",
#                 ncol=2)

# @classmethod
# def _get_compression_legend_kwargs(cls, configs, legend_loc, font_name, compression_keys):
#     custom_lines = list([])
#
#     for key in compression_keys:
#         custom_lines.append(cls.get_line({**configs[key], 'color': COLORMAP[key], 'linestyle': 'None', 'lw': 1.2}))
#
#     return dict(fancybox=False,
#                 framealpha=0.4,
#                 handles=custom_lines,
#                 labels=[COMPRESSION_NAMES[k] for k in compression_keys],
#                 loc=legend_loc,
#                 bbox_to_anchor=(0., 0.0, 1., .102),
#                 prop=get_font(font_name, FONTSIZES.big),
#                 ncol=3)

# def make_plot0(self, classifier, show, save, target_bitrate, hvs_metric, ymin, ymax, figsize, baselines=None,
#                topk=1, font_name='times_roman'):
#
#     if baselines is None:
#         baselines = []
#
#     # ========= get data
#     # get rnn compression data
#     rnn_accuracy_data = self.get_accuracy_data_rnn(target_bitrate, classifier, topk, True)
#     rnn_hvs_data = self.get_hvs_data_rnn(target_bitrate, hvs_metric)
#
#     baselines_accuracy_data = self.get_accuracy_data_baselines(target_bitrate, classifier, topk, baselines, True,
#                                                                True)
#     baselines_hvs_data = self.get_hvs_data_baselines(target_bitrate, hvs_metric, baselines, True)
#
#     # ========= make plot
#     fig, ax1 = plt.subplots(figsize=figsize)
#
#     # == fill in data RNN compression
#     # left y-axis (MS-SSIM)
#     ax1.set_xlabel(r'$\alpha$', fontproperties=get_font(font_name, FONTSIZES.Large))
#     ax1.set_ylabel('MS-SSIM', fontproperties=get_font(font_name, FONTSIZES.Large), labelpad=10)
#     ax1.plot(rnn_hvs_data[:, _RNN_IDX_ALPHA], rnn_hvs_data[:, _RNN_IDX_VAL], marker=self._configs['rnn']['marker'],
#              color=COLORMAP['rnn'], lw=self._configs['rnn']['lw'], linestyle='--',
#              markersize=3 * self._configs['rnn']['lw'])
#     ax1.tick_params(axis='y', labelcolor='black')
#     ax1.tick_params(which='minor', width=0.7)
#
#     # right y-axis (Accuracy)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Preserved Val. Accuracy\n{}'.format(CLASSIFIER_NAMES[classifier]),
#                    fontproperties=get_font(font_name, FONTSIZES.Large),
#                    labelpad=10)
#     ax2.plot(rnn_accuracy_data[:, _RNN_IDX_ALPHA], rnn_accuracy_data[:, _RNN_IDX_VAL],
#              marker=self._configs['rnn']['marker'], color=COLORMAP['rnn'], lw=self._configs['rnn']['lw'],
#              markersize=3 * self._configs['rnn']['lw'])
#     ax2.tick_params(axis='y', labelcolor='black')
#     ax2.tick_params(which='minor', width=0.7)
#
#     # baseline data
#     for baseline_hvs, key in zip(baselines_hvs_data, baselines):
#         ax1.plot(-0.15, float(baseline_hvs[_BASELINE_IDX_VAL]), color=COLORMAP[key], linestyle='--',
#                  label=COMPRESSION_NAMES[key], marker=self._configs[key]['marker'], lw=self._configs[key]['lw'])
#
#         ax1.hlines(float(baseline_hvs[_BASELINE_IDX_VAL]), xmin=-0.5, xmax=-0.15, color=COLORMAP[key],
#                    linestyle='--', label=COMPRESSION_NAMES[key], lw=self._configs[key]['lw'])
#
#     for baseline_acc, key in zip(baselines_accuracy_data, baselines):
#         ax2.plot(1.15, float(baseline_acc[_BASELINE_IDX_VAL]), color=COLORMAP[key], linestyle='-',
#                  label=COMPRESSION_NAMES[key], marker=self._configs[key]['marker'], lw=self._configs[key]['lw'])
#
#         ax2.hlines(float(baseline_acc[_BASELINE_IDX_VAL]), xmin=1.15, xmax=1.5, color=COLORMAP[key],
#                    linestyle='-', label=COMPRESSION_NAMES[key], lw=self._configs[key]['lw'])
#
#     # == format
#     # axis limits
#     ax1.set_xlim((-0.3, 1.3))
#     ax1.set_ylim((ymin, ymax))
#     ax2.set_ylim((ymin, ymax))
#
#     # ticks
#     ax1.yaxis.set_major_locator(MultipleLocator(0.05))
#     ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
#     ax1.set_xticks(np.arange(0, 1.25, 0.25))
#     ax2.yaxis.set_major_locator(MultipleLocator(0.05))
#     ax2.yaxis.set_minor_locator(MultipleLocator(0.025))
#     ax2.set_xticks(np.arange(0, 1.25, 0.25))
#
#     # fontprops
#     for labelx in ax1.get_xticklabels():
#         labelx.set_fontproperties(get_font(font_name, FONTSIZES.large))
#
#     for labely1 in ax1.get_yticklabels():
#         labely1.set_fontproperties(get_font(font_name, FONTSIZES.large))
#
#     for labely2 in ax2.get_yticklabels():
#         labely2.set_fontproperties(get_font(font_name, FONTSIZES.large))
#
#     # grid, facecolor, spines
#     ax1.grid(True, color=_GRID_COLOR, linewidth=0.5)
#     ax1.set_facecolor(_FACECOLOR)
#
#     # legend to identify y-axes
#     legend2 = ax1.legend(**self._get_axes_legends_kwargs(legend_loc='lower center', font_name=font_name), )
#     ax1.add_artist(legend2)
#
#     # legend for compression method
#     legend1 = ax1.legend(**self._get_compression_legend_kwargs(configs=self._configs,
#                                                                legend_loc='lower center',
#                                                                font_name=font_name,
#                                                                compression_keys=['rnn', *baselines]))
#     ax1.add_artist(legend1)
#
#     fig.tight_layout()
#
#     if show:
#         plt.show()
#
#     if save:
#         if not os.path.exists(self._plots_save_dir):
#             os.makedirs(self._plots_save_dir)
#
#         save_as = '{}_tradeoff_{}_{}_{}bpp.png'.format(self._dataset, hvs_metric, classifier, target_bitrate)
#         fig.savefig(os.path.join(self._plots_save_dir, save_as), dpi=200)
#         print('plot saved as {}'.format(os.path.join(self._plots_save_dir, save_as)))
#
#     plt.close(fig)
