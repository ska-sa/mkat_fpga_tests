import logging
import os
import pwd
import sys
from getpass import getuser as getusername

try:
    get_username = getusername()
except OSError:
    get_username = pwd.getpwuid(os.getuid()).pw_name


class LoggingClass:

    @property
    def logger(self):
        log_format = (
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(pathname)s : '
            '%(lineno)d - %(message)s')
        _file_name = os.path.basename(sys.argv[0])
        name = '.'.join([_file_name, self.__class__.__name__])
        logger = logging.getLogger(name)
        if not len(logger.handlers):
            Formatter = logging.Formatter(log_format)
            Handler = logging.FileHandler("/tmp/test_ran_by_%s.log" % (get_username))
            Handler.setLevel(logging.DEBUG)
            Handler.setFormatter(Formatter)
            logger.addHandler(Handler)
        return logger
