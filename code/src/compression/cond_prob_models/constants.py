import inspect
import os

CONFIG_BASE_AE = os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())), 'ae_configs/')
CONFIG_BASE_PC = os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())), 'pc_configs/')
