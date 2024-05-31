## Reused code from:
## Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., and Yan, S.
## Better diffusion models further improve adversarial training, June 2023
## Code available at https://github.com/wzekai99/DM-Improves-AT

import logging


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        self.logger = logging.getLogger()
        self.path = path
        self.setup_file_logger()
        print('Logging to file: ', self.path)

    def setup_file_logger(self):
        hdlr = logging.FileHandler(self.path, 'a')
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        print(message)
        self.logger.info(message)
