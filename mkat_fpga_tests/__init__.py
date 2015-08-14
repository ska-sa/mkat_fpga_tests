import os
import logging
import subprocess
import time

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as test_config

from corr2 import fxcorrelator
from corr2 import utils



LOGGER = logging.getLogger(__name__)


class CorrelatorFixture(object):

    def __init__(self, config_filename=None):
        try:
            subprocess.check_call(['corr2_dsim_control.py',
            '--program', '--start'])
            LOGGER.info('D-Eng Started succesfully')
        except subprocess.CalledProcessError:
            LOGGER.warn('Failed to start D-Eng')

        if config_filename is None:
            try:
                config_filename = utils.parse_ini_file('/etc/corr/array0-c8n856M4k')
            except IOError:
                LOGGER.info ("ERROR Config File Does Not Exist.")
                config_filename = os.environ['CORR2INI']
            self.config_filename = config_filename

        self._correlator = None

        """Assume correlator is already running if this flag is True.
        IOW, don't do start_correlator() if set."""

    @property
    def correlator(self):
        if self._correlator is not None:
            LOGGER.info('Correlator started succesfully')
            return self._correlator

        else:
            if int(test_config.get('start_correlator', False)):
                # Is it not easier to just call a self._correlator method?
                print "-"*100
                time.sleep(5)
                self.start_correlator()

            self._correlator = fxcorrelator.FxCorrelator(
                'test correlator', config_source='/etc/corr/array0-c8n856M4k')
            self.correlator.initialise(program=False)
            LOGGER.info('Correlator started succesfully')
            return self._correlator

    def start_stop_data(self, start_or_stop, modes):
        self.modes = modes
        LOGGER.info('Correlator starting to capture data.')
        assert start_or_stop in ('start', 'stop')
        assert self.modes in ('c856M4k', 'c856M32k')

        destination = self.correlator.configd['xengine']['output_destination_ip']
        destination_port = self.correlator.configd['xengine']['output_destination_port']

        subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-destination' ,
                '{}'.format(self.modes), '{}:{}'.format(destination, destination_port)])

        subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-t','30', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-{}'.format(start_or_stop) ,
                '{}'.format(self.modes)])

    def start_x_data(self):
        # On array interf
        self.start_stop_data('start','c856M4k')

    def stop_x_data(self):
        self.start_stop_data('stop', 'c856M4k')

    def start_correlator(self, retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        import IPython;IPython.embed()
        array_no = 0
        host_port = int(self.config_filename['FxCorrelator']['katcp_port'])
        multicast_ip = self.config_filename['fengine']['source_mcast_ips']
        self.instrument_name = "c8n856M4k"

        # Clear out any arrays, if they exist
        subprocess.check_call(['/usr/local/bin/kcpcmd', 'array-halt', 'array0'])
        while retries and not success:
            try:
                subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30', '-s', 'localhost:7147',
                    'array-assign', 'array0'] + multicast_ip.split(','))

                self.katcp_port = int(subprocess.Popen("/usr/local/bin/kcpcmd -s localhost:{0} array-list array0\
                | grep array{1} | cut -f3 -d ' '".format(host_port,array_no)
                    , shell=True, stdout=subprocess.PIPE).stdout.read())

                LOGGER.info ("Starting Correlator.")
                success = 0 == subprocess.check_call(['/usr/local/bin/kcpcmd','-t','500','-s',
                    'localhost:{}'.format(self.katcp_port), 'instrument-activate',
                        'c8n856M4k'])
                retries -= 1
                if success == True:
                    LOGGER.info('Correlator started succesfully')
                else:
                    LOGGER.warn('Failed to start correlator, {} attempts left.\
                        \nRestarting Correlator.'
                            .format(retries))

            except Exception:
            #except subprocess.CalledProcessError:
                subprocess.check_call(['/usr/local/bin/kcpcmd', 'array-halt', 'array0'])
                retries -= 1
                LOGGER.info ('\nFailed to start correlator, {} attempts left.\n'
                            .format(retries))
                time.sleep(5)
            print "-"*50

        if not success:
            raise RuntimeError('Could not successfully start correlator within {} retries'
                               .format(retries_requested))

    def issue_metadata(self):
        subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-meta', self.modes])
correlator_fixture = CorrelatorFixture()
