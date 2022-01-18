import logging
import logging.handlers
import os
import sys


def init_logging(level='INFO', file=True, stdout=True):
    """
    Initialize logging

    :param str level: logging level
    :param bool file: enable file logging
    :param bool stdout: enable system out logging
    :return: logging logger
    :rtype: logging.Logger
    """
    # logging format
    elements = [
        '%(asctime)s',
        '%(levelname)-8s',
        '%(name)-8s',
        '[%(filename)s:%(lineno)s]',
        '%(message)s',
    ]
    fmt = ' '.join(elements)

    root = logging.getLogger(__name__)
    formatter = logging.Formatter(fmt.replace('\n', ' ').strip())

    # get logging level
    level = logging.getLevelName(level)

    if isinstance(level, str) and level.startswith('Level'):
        level = logging.INFO

    root.setLevel(level)
    root.propagate = False

    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)

    if file:
        file_path = os.environ.get('LOGFILE', '/tmp/identity-detector.log')
        file_handler = logging.handlers.WatchedFileHandler(file_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root
