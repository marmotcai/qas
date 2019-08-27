import logging

from utils import tools as my_tools

################################################################################

class logger():
    def __init__(self, logfilename, appname = "", level = logging.INFO):
        my_tools.mkdir(logfilename)

        if len(appname) <= 0: appname = __name__
        self.logger = logging.getLogger(appname)
        self.logger.setLevel(level = logging.INFO)
        self.handler = logging.FileHandler(logfilename)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def debug(self, log):
        self.logger.debug(log)

    def info(self, log):
        self.logger.info(log)

    def warning(self, log):
        self.logger.warning(log)

    def error(self, log):
        self.logger.error(log)

    def critical(self, log):
        self.logger.critical(log)
