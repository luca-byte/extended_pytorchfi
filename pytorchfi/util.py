import random
import logging

def random_value(min_val: int = -1, max_val: int = 1):
    return random.uniform(min_val, max_val)


LOGGING_FORMAT = '%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s'

logging.basicConfig(
    format=LOGGING_FORMAT,
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)

def_logger_pfi=logging.getLogger()