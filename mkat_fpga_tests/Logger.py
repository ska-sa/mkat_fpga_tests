import logging
import os
import pwd
import sys
import tempfile
from getpass import getuser as getusername

try:
    GET_USERNAME = getusername()
except OSError:
    GET_USERNAME = pwd.getpwuid(os.getuid()).pw_name

# Set same logging level as per nosetests
LOGGING_LEVEL = ''.join(
    [i.split('=')[-1] for i in sys.argv if i.startswith('--logging-level')])
if not LOGGING_LEVEL:
    LOGGING_LEVEL = "INFO"


class LoggingClass:

    @property
    def logger(self):
        """
        Logger
        """
        log_format = (
            '%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(pathname)s : '
            '%(lineno)d - %(message)s')
        _file_name = os.path.basename(sys.argv[0])
        name = '.'.join([_file_name, self.__class__.__name__])
        logger = logging.getLogger(name)
        if not len(logger.handlers):
            formatter = logging.Formatter(log_format)
            handler = logging.FileHandler(
                os.path.join(
                    tempfile.gettempdir(),
                    "test_ran_by_%s.log" % GET_USERNAME
                    )
                )
            try:
                handler.setLevel(getattr(logging, LOGGING_LEVEL.upper()))
                logger.setLevel(getattr(logging, LOGGING_LEVEL.upper()))
            except ValueError:
                pass
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
