import os
import logging
import subprocess

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as test_config

from corr2 import fxcorrelator
from corr2 import utils

LOGGER = logging.getLogger(__name__)


class CorrelatorFixture(object):

    def __init__(self, config_filename=None):
        if config_filename is None:
            config_filename = os.environ['CORR2INI']
        self.config_filename = config_filename
        os.environ['CORR2INI'] = config_filename
        self._correlator = None
        """Assume correlator is already running if this flag is True.
        IOW, don't do start_correlator() if set."""

    @property
    def correlator(self):
        if self._correlator is not None:
            return self._correlator
        else:
            if int(test_config.get('start_correlator', False)):
                # Is it not easier to just call a self._correlator method?
                self.start_correlator()

            # get config file from /etc/corr/{array-name}-{instrument-name}, e.g.
            # /etc/corr/array0-c8n856M4k
            self._correlator = fxcorrelator.FxCorrelator(
                'test correlator', config_source=self.config_filename)
            self.correlator.initialise(program=False)
            return self._correlator

    def start_stop_data(self, start_or_stop, engine_class):
        assert start_or_stop in ('start', 'stop')
        assert engine_class in ('xengine', 'fengine')
        # kcpcmd -s localhost:{array-port} capture-destination c856M4k 10.100.201.1
        #subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-s' ,
            #'localhost:{}'.format(array_port) ,'capture-destination' ,
                #'{}'.format(mode), '{}'.format(destination)])

        # kcpcmd -s localhost:{array-port} capture-start c856M4k
        # or
        # kcpcmd -s localhost:{array-port} capture-stop c856M4k
        #subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-s' ,
            #'localhost:{}'.format(array_port) ,'capture-{}'.format(start_or_stop) ,
                #'{}'.format('c856M4k')]

        #subprocess.check_call([
            #'corr2_start_stop_tx.py', '--{}'.format(start_or_stop),
            #'--class', engine_class])


    def start_x_data(self):
        # On array interf
        self.start_stop_data('start', 'xengine')

    def stop_x_data(self):
        self.start_stop_data('stop', 'xengine')

    def start_correlator(self, retries=5, loglevel='INFO')
