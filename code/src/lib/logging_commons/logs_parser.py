import numpy as np
import re


class LogsParser:
    # keywords
    BPP_KW = 'bpp_mean'

    TOP1ACC_KW = 'top1acc'
    TOP5ACC_KW = 'top5acc'

    MSSSIM_KW = 'ms_ssim'
    PSNR_KW = 'psnr'
    MSE_KW = 'mse'
    L1_KW = 'l1'

    EVAL_LINE_KW = 'EVAL:'

    @staticmethod
    def get_regex_for_acc(classifier, acc_kw):
        """ expects logs to contain data in the format eg `vgg16_top5acc=0.9132` """
        return re.compile('{}_{}='.format(classifier, acc_kw) + '(?P<{}>\d+\.\d+)'.format(acc_kw))

    @staticmethod
    def get_regex_for_hvs(hvs_metric):
        """ expects logs to contain data in the format eg `ms_ssim=0.956` """
        return re.compile('{}='.format(hvs_metric) + '(?P<{}>\d+\.\d+)'.format(hvs_metric))

    @classmethod
    def get_regex_for_bpp(cls):
        """ expects logs to contain data in the format eg `bpp_mean=0.192` """
        return re.compile('{}=(?P<bpp_mean>\d+\.\d+)'.format(cls.BPP_KW))

    @classmethod
    def parse_accuracy_file(cls, logfile, classifier, topkacc_kw, case_sensitive=True):
        assert topkacc_kw == cls.TOP1ACC_KW or topkacc_kw == cls.TOP5ACC_KW
        parsed_acc, parsed_bpp = [], []

        # manage case sensitivity
        eval_line_kw = cls.EVAL_LINE_KW if case_sensitive else cls.EVAL_LINE_KW.lower()
        classifier = classifier if case_sensitive else classifier.lower()

        # read file, parse values
        with open(logfile, 'r') as f:
            for line in f:
                line = line if case_sensitive else line.lower()
                if eval_line_kw in line:
                    parsed_bpp.append(float(cls.get_regex_for_bpp().search(line).group(cls.BPP_KW)))
                    parsed_acc.append(
                        float(cls.get_regex_for_acc(classifier, topkacc_kw).search(line).group(topkacc_kw)))

        # convert to numpy and sort
        indices_sorted = np.argsort(np.array(parsed_bpp))
        parsed_bpp = np.array(parsed_bpp)[indices_sorted]
        parsed_acc = np.array(parsed_acc)[indices_sorted]

        return {cls.BPP_KW: parsed_bpp, topkacc_kw: parsed_acc}

    @classmethod
    def parse_hvs_file(cls, logfile, case_sensitive=True):
        parsed_bpp, parsed_msssim, parsed_psnr, parsed_mse, parsed_l1 = [], [], [], [], []

        # manage case sensitivity
        eval_line_kw = cls.EVAL_LINE_KW if case_sensitive else cls.EVAL_LINE_KW.lower()

        # read file, parse values
        with open(logfile, 'r') as f:
            for line in f:
                line = line if case_sensitive else line.lower()
                if eval_line_kw in line:
                    parsed_bpp.append(float(cls.get_regex_for_bpp().search(line).group(cls.BPP_KW)))
                    parsed_msssim.append(
                        float(cls.get_regex_for_hvs(cls.MSSSIM_KW).search(line).group(cls.MSSSIM_KW)))
                    parsed_psnr.append(
                        float(cls.get_regex_for_hvs(cls.PSNR_KW).search(line).group(cls.PSNR_KW)))
                    parsed_mse.append(
                        float(cls.get_regex_for_hvs(cls.MSE_KW).search(line).group(cls.MSE_KW)))
                    # parsed_l1.append(
                    #     float(cls.get_regex_for_hvs(cls.L1_KW).search(line).group(cls.L1_KW)))

        indices_sorted = np.argsort(np.array(parsed_bpp))
        parsed_bpp = np.array(parsed_bpp)[indices_sorted]
        parsed_msssim = np.array(parsed_msssim)[indices_sorted]
        parsed_psnr = np.array(parsed_psnr)[indices_sorted]
        parsed_mse = np.array(parsed_mse)[indices_sorted]
        # parsed_l1 = np.array(parsed_l1)[indices_sorted]

        return {cls.BPP_KW: parsed_bpp, cls.MSSSIM_KW: parsed_msssim, cls.PSNR_KW: parsed_psnr, cls.MSE_KW: parsed_mse,
                cls.L1_KW: parsed_l1}
