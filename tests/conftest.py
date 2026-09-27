import logging
import warnings

logging.getLogger("pandapower").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
