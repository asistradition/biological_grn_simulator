import logging
import sys

logger = logging.Logger('BiologicalGRNSimulator')
logger_handler = logging.StreamHandler(sys.stderr)
logger_handler.setFormatter(
    logging.Formatter('%(asctime)-15s %(message)s')
)
logger.addHandler(logger_handler)
