import logging

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

def setup_logging(log_level=LOG_LEVEL):
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
