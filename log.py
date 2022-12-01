import logging
import sys
import os

def config_logging(dir_name):
    log_file = os.path.join(dir_name, 'log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding="utf8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(module)s.%(lineno)d %(message)s',datefmt="%Y/%m/%d %H:%M:%S"))
    # file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt="%Y/%m/%d %H:%M:%S"))
    # console_handler.setLevel(console_level)

    logging.basicConfig(level=logging.NOTSET,handlers=[file_handler, console_handler],)