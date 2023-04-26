import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys

import coloredlogs


def prepare_logger(log_dir: str = None, name: str = None):
    datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
    if name is not None:
        log_path = os.path.join(log_dir, datetime_str + '_' + name)
    else:
        log_path = os.path.join(log_dir, datetime_str)
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))
    return logger, log_path
