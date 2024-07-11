import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

__all__ = ['setup_logger']

logger_initialized = []


def setup_logger(name="augment", output='p2g_logs/p2gnet.log'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] P2GNET %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fh = TimedRotatingFileHandler(filename=filename, when='midnight', interval=1, backupCount=7)
        fh.suffix = "%Y-%m-%d.log"
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger
