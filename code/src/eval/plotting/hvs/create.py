import inspect
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import json
import matplotlib.pyplot as plt
import numpy as np
import os

from src.lib.logging_commons.logs_parser import LogsParser
from src.lib.logging_commons.utils import read_csv
from src.eval.plotting.names import COMPRESSION_NAMES, HVS_METRIC_NAMES
from src.eval.plotting.fonts import get_font, FONTSIZES
from src.eval.plotting.colors import COLORMAP

LOGFILE_BASE = 'eval_hvs_{}.log'
CSV_FILE_BASE = '{}_hvs.csv'
FACECOLOR = (1, 1, 1)

TICK_MULTIPLES = {LogsParser.MSSSIM_KW: 0.01,
                  LogsParser.PSNR_KW: 2,
                  LogsParser.L1_KW: 1,
                  LogsParser.MSE_KW: 1}


class HVSPlots:

    def __init__(self, csv_dir, dataset):

        assert os.path.exists(csv_dir), 'log_dir not found!'

        self._module_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self._csv_dir = csv_dir
        self._dataset = dataset
        self._plots_save_dir = os.path.join(self._module_dir, 'plots/', dataset + '/')

    def load_config(self, compression_key):
        with open(os.path.join(self._module_dir, 'configs.json'), 'r') as fo:
            configs = json.load(fo)
        cfg = configs[compression_key]
        return {**cfg, 'csv_file': os.path.join(self._csv_dir, CSV_FILE_BASE.format(compression_key))}

    @staticmethod
    def keys_to_color(keys):
        return {key: COLORMAP[key] for key in keys}

    def make_plot(self, perception_metric, show, save, compression_keys, xmin=0.0, xmax=1.0, ymin=0,
                  ymax=1, figsize=(6, 3.75), font_name='times_roman', legend_loc='lower right', bpp_min=None):

        assert perception_metric in [LogsParser.L1_KW, LogsParser.MSE_KW, LogsParser.MSSSIM_KW, LogsParser.PSNR_KW]

        if compression_keys is None:
            raise ValueError('compression_keys must not be None')

        if not isinstance(compression_keys, list):
            compression_keys = list([compression_keys])

        configs = {ckey: self.load_config(ckey) for ckey in compression_keys}
        corrupted_logfiles = []

        # ========= parse csv files
        parsed_data = {}
        for key in compression_keys:
            cfg = configs[key]
            csv_file = cfg['csv_file']

            if not os.path.isfile(csv_file):
                print('WARNING! {} not found.'.format(csv_file))
                corrupted_logfiles.append(key)
                continue

            # parse csv
            hvs_data = read_csv(csv_file)
            idx = hvs_data[0].index(perception_metric)
            hvs_data = np.array(hvs_data[1:], dtype=np.float32)
            hvs_data = hvs_data[:, [0, idx]]

            # sort data
            hvs_data.sort(axis=0)

            if bpp_min is not None:
                include_idx = np.where(hvs_data[:, 0] >= float(bpp_min))[0]
                hvs_data = hvs_data[include_idx, :]

            parsed_data[key] = hvs_data[:, 0], hvs_data[:, 1]

        if len(parsed_data) == 0:
            print('parsed data empty')
            return

        compression_keys = list([k for k in compression_keys if k not in corrupted_logfiles])
        configs = {k: configs[k] for k in compression_keys}
        compression_colors = self.keys_to_color([k for k in compression_keys])

        # ========= make plot
        # determine plot boundaries
        x_lim = xmin, xmax
        y_lim = ymin, ymax

        # setup fig
        fig = plt.figure(figsize=figsize)

        # plot data
        ax = plt.gca()
        for key in compression_keys:
            bpp_array, hvs_array = parsed_data[key]
            cfg = configs[key]

            ax.plot(bpp_array, hvs_array, lw=cfg['lw'], color=compression_colors[key], label=COMPRESSION_NAMES[key],
                    marker=cfg['marker'], markersize=3 * cfg['lw'], linestyle=cfg['linestyle'])

        # format plot
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        ax.set_xlabel('bpp', fontproperties=get_font(font_name, FONTSIZES.Large))
        ax.set_ylabel(HVS_METRIC_NAMES[perception_metric], fontproperties=get_font(font_name, FONTSIZES.Large))
        ax.grid(True, color=(0.91, 0.91, 0.91), linewidth=0.5)

        tick_multiples = TICK_MULTIPLES[perception_metric]
        ax.yaxis.set_minor_locator(MultipleLocator(tick_multiples))
        ax.yaxis.set_major_locator(MultipleLocator(2 * tick_multiples))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if tick_multiples < 1 else '%d'))

        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.125))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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
        legend = plt.legend(**self._get_legend_kwargs(configs=[(k, configs[k]) for k in compression_keys],
                                                      labels=legend_labels,
                                                      legend_loc=legend_loc,
                                                      font_name=font_name,
                                                      compression_colors=compression_colors))
        ax.add_artist(legend)

        plt.tight_layout()

        if show:
            plt.show()

        if save:
            if not os.path.exists(self._plots_save_dir):
                os.makedirs(self._plots_save_dir)

            save_as = 'hvs_{}.png'.format(perception_metric)
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
                    prop=get_font(font_name, FONTSIZES.large),
                    ncol=1)
