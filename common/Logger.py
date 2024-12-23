import logging
import os
import colorlog


class Logger:
    """
    default logger in ReIDFaaS
    Args:
        name(str) : Logger name, default is 'ReIDFaaS'
    """

    def __init__(self, name='ReIDFaaS', logLevel='INFO'):
        self.logger = logging.getLogger(name)

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] %(filename)s(%(lineno)d)'
            ' [%(levelname)s]%(reset)s - %(message)s', )

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(level=logLevel)
        self.logger.propagate = False


LOGGER = Logger(logLevel=os.getenv("LOG_LEVEL", "INFO")).logger
