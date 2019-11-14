import os
import matplotlib.font_manager as font_manager
from dirs import RESOURCES_DIR


def get_font(font_name, font_size):
    fonts_files = {'din_pro': 'fonts/DIN pro.tff',
                   'times_roman': 'fonts/times-ro.ttf'}
    return font_manager.FontProperties(fname=os.path.join(RESOURCES_DIR, fonts_files[font_name]),
                                       size=font_size)


class FONTSIZES:
    tiny = 6
    small = 8
    medium = 10
    big = 12
    large = 14
    Large = 16
    LARGE = 18
