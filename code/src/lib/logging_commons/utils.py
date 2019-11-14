import csv
import logging
import math
import sys


def get_logger(logfile, logging_mode='w'):
    formatter = logging.Formatter(fmt='%(levelname)s::%(name)s:: %(asctime)s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(logfile, mode=logging_mode)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def write_to_csv(csv_file, data_rows):
    with open(csv_file, mode='w') as csv_file_obj:
        csv_writer = csv.writer(csv_file_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data_rows:
            csv_writer.writerow(row)


def read_csv(csv_file, delimiter=','):
    with open(csv_file, mode='r') as csv_file_obj:
        csv_reader = csv.reader(csv_file_obj, delimiter=delimiter)

        data = []
        for row in csv_reader:
            data.append(row)

        return data


def log_flags_and_configs(logger, flags, config_list):
    main_dict = {}
    for config in config_list:
        for key, val in config.items():
            if key not in main_dict.keys():
                main_dict[key] = val
            else:
                logger.error('key, val pair duplicate found in configs: key={}, val={}'.format(key, val))
                main_dict[key + '_' + str(2)] = val

    for key, val in flags.flag_values_dict().items():
        main_dict[key] = val

    # log everything in alphabetical order
    for key in sorted(main_dict.keys(), key=lambda x: x.lower()):
        logger.info("param {key}={val}".format(key=key, val=main_dict[key]))


def log_configs(logger, config_list):
    main_dict = {}
    for config in config_list:
        for key, val in config.items():
            if key not in main_dict.keys():
                main_dict[key] = val
            else:
                logger.error('key, val pair duplicate found in configs: key={}, val={}'.format(key, val))
                main_dict[key + '_' + str(2)] = val

    # log everything in alphabetical order
    for key in sorted(main_dict.keys(), key=lambda x: x.lower()):
        logger.info("param {key}={val}".format(key=key, val=main_dict[key]))


def log_flags(logger, flags):
    main_dict = {}
    for key, val in flags.flag_values_dict().items():
        main_dict[key] = val

    # log everything in alphabetical order
    for key in sorted(main_dict.keys(), key=lambda x: x.lower()):
        logger.info("param {key}={val}".format(key=key, val=main_dict[key]))


def progress(count, total, status=''):
    """progress bar"""
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s | %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def progress_v2(count, msg='', status=''):
    sys.stdout.write('[%s %s] | %s\r' % (count, msg, status))
    sys.stdout.flush()


def progress_v3(count, total, status=''):
    sys.stdout.write('[%s/%s] | %s\r' % (count, total, status))
    sys.stdout.flush()


def seconds_to_hours_minutes(seconds):
    hours = (seconds - seconds % 3600) // 3600
    minutes = ((seconds - hours * 3600) - (seconds - hours * 3600) % 60) // 60
    return '{}h{}m'.format(int(hours if not math.isnan(hours) else 0), int(minutes if not math.isnan(minutes) else 0))


def seconds_to_minutes_seconds(seconds):
    minutes = seconds // 60
    rem_seconds = seconds - minutes * 60
    return '{}m{}s'.format(int(minutes), int(rem_seconds))
