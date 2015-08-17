import os
import logging
import subprocess
import time

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as test_config
from corr2.dsimhost_fpga import FpgaDsimHost

from corr2 import fxcorrelator
from corr2 import utils

LOGGER = logging.getLogger(__name__)

class CorrelatorFixture(object):

    def __init__(self, test_config_filename=None):

        if test_config_filename is None:
            test_config_filename = os.environ['CORR2TESTINI']
            self.dsim_conf = utils.parse_ini_file(
                test_config_filename)['dsimengine']

        self._correlator = None
        self._dhost = None

    @property
    def dhost(self):
        if self._dhost is not None:
            return self._dhost
        else:
            dig_host = self.dsim_conf['host']
            self._dhost = FpgaDsimHost(dig_host, config=dsim_conf)
            # Check if D-eng is running else start it.
            if self._dhost.is_running():
                LOGGER.info('D-Eng is running')
            else:
                # Programming and starting D-Eng
                self._dhost.initialise()
                self._dhost.enable_data_output(enabled=True)
                self._dhost.registers.control.write(gbe_txen=True)
                if self._dhost.is_running():
                    LOGGER.info('D-Eng Started succesfully')
            return self._dhost

    @property
    def correlator(self):
        LOGGER.debug('it is here')
        if self._correlator is not None:
            LOGGER.info('Correlator started succesfully')
            return self._correlator
        else:
            if int(test_config.get('start_correlator', False)):
                self.start_correlator()

            # We assume either start_correlator() above has been called, or
            # the c8n856M4k instrument was started on array0 before running the
            # test.

            # TODO: hard-coded config location
            self.config_filename = '/etc/corr/array0-c8n856M4k'
            #self.corr_conf = utils.parse_ini_file(self.config_filename)
            self._correlator = fxcorrelator.FxCorrelator(
                'test correlator', config_source=self.config_filename)
            self.correlator.initialise(program=False)
            LOGGER.info('Correlator started succesfully')
            return self._correlator

    def start_stop_data(self, start_or_stop, modes):
        self.modes = modes
        LOGGER.info('Correlator starting to capture data.')
        assert start_or_stop in ('start', 'stop')

        destination = self.correlator.configd['xengine']['output_destination_ip']
        destination_port = self.correlator.configd['xengine']['output_destination_port']
        self.katcp_port = int(subprocess.Popen("/usr/local/bin/kcpcmd \
                    -s localhost:7147 array-list array0\
                        | grep array0 | cut -f3 -d ' '"
                            , shell=True, stdout=subprocess.PIPE).
                                    stdout.read())

        subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-destination' ,
                '{}'.format(self.modes), '{}:{}'.format(destination,
                    destination_port)])

        subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-t','30', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-{}'
                .format(start_or_stop), '{}'.format(self.modes)])

    def start_x_data(self):
        # On array interf
        self.start_stop_data('start','c856M4k')

    def stop_x_data(self):
        self.start_stop_data('stop', 'c856M4k')

    def start_correlator(self, retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        array_no = 0
        import IPython;IPython.embed()
        host_port = self.config_filename['FxCorrelator']['katcp_port']
        multicast_ip = self.config_filename['fengine']['source_mcast_ips']
        try:
            # Clear out any arrays, if exist
            subprocess.check_call(['/usr/local/bin/kcpcmd', '-s', 'localhost',
                'array-halt', 'array0'])
        except:
            LOGGER.info ("Already cleared array")

        while retries and not success:
            try:
                subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30',
                    '-s', 'localhost:7147', 'array-assign', 'array0']
                        + multicast_ip.split(','))

                self.katcp_port = int(subprocess.Popen("/usr/local/bin/kcpcmd \
                    -s localhost:{0} array-list array0\
                        | grep array{1} | cut -f3 -d ' '"
                            .format(host_port,array_no)
                                , shell=True, stdout=subprocess.PIPE).
                                    stdout.read())

                LOGGER.info ("Starting Correlator.")
                success = 0 == subprocess.check_call(['/usr/local/bin/kcpcmd',
                    '-t','500','-s', 'localhost:{}'.format(self.katcp_port),
                        'instrument-activate', 'c8n856M4k'])
                retries -= 1
                if success == True:
                    LOGGER.info('Correlator started succesfully')
                else:
                    LOGGER.warn('Failed to start correlator, {} attempts left.\
                        \nRestarting Correlator.'
                            .format(retries))

            except Exception:
                subprocess.check_call(['/usr/local/bin/kcpcmd', '-s',
                'localhost', 'array-halt', 'array0'])
                retries -= 1
                LOGGER.warn ('\nFailed to start correlator, {} attempts left.\n'
                            .format(retries))
        if not success:
            raise RuntimeError('Could not successfully start correlator within {}\
                retries'.format(retries_requested))

    def issue_metadata(self):
        subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '100', '-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-meta', self.modes])
correlator_fixture = CorrelatorFixture()
