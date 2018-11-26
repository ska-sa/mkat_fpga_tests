import logging
import os
import pwd
import sys
from getpass import getuser as getusername

try:
    get_username = getusername()
except OSError:
    get_username = pwd.getpwuid(os.getuid()).pw_name

# Set same logging level as per nosetests
LOGGING_LEVEL = ''.join(
    [i.split('=')[-1] for i in sys.argv if i.startswith('--logging-level')])
if not LOGGING_LEVEL:
    LOGGING_LEVEL = "INFO"


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
            try:
                Handler.setLevel(LOGGING_LEVEL)
            except ValueError:
                LOGGING_LEVEL = "INFO"
                Handler.setLevel(LOGGING_LEVEL)
            logger.setLevel(LOGGING_LEVEL)
            Handler.setFormatter(Formatter)
            logger.addHandler(Handler)
        return logger
