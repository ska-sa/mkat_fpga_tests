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

        if config_filename is None:
            try:
                config_filename = utils.parse_ini_file('/etc/corr/array0-c8n856M4k')#,['dsimengine'])
            except IOError:
                print "ERROR Config File Does Not Exist.\
                \nReading for CORR_TEMPLATE"
                config_filename = os.environ['CORR2INI']
#                config_filename = utils.parse_ini_file(os.environ['CORR2INI'])

            self.config_filename = config_filename

        self._correlator = None

        """Assume correlator is already running if this flag is True.
        IOW, don't do start_correlator() if set."""

    @property
    def correlator(self):
        if self._correlator is not None:
            print "What are you doing here."
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
            #
            return self._correlator

    def start_stop_data(self, start_or_stop, modes):
        print '>'*30
        time.sleep(30)
        #import IPython;IPython.embed()
        assert start_or_stop in ('start', 'stop')
        assert modes in ('c856M4k', 'c856M32k')

        destination = self.config_filename['xengine']['output_destination_ip']

        # kcpcmd -s localhost:{array-port} capture-destination c856M4k 10.100.201.1
        subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-s' ,
            'localhost:{}'.format(self.katcp_port) ,'capture-destination' ,
                '{}'.format(self.instrument_name), '{}'.format(destination)])


        subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-s' ,
            'localhost:{}'.format(array_port) ,'capture-{}'.format(start_or_stop) ,
                '{}'.format(mode)])

        #subprocess.check_call([
        #    'corr2_start_stop_tx.py', '--{}'.format(start_or_stop),
        #    '--class', engine_class])


    def start_x_data(self):
        # On array interf
        self.start_stop_data('start','c856M4k')

    def stop_x_data(self):
        self.start_stop_data('stop', 'c856M4k')

    def start_correlator(self, retries=10, loglevel='INFO'):
        success = False
        start_dsim = False
        retries_requested = retries

        array_no = 0
        host_port = int(self.config_filename['FxCorrelator']['katcp_port'])
        multicast_ip = self.config_filename['fengine']['source_mcast_ips']
        self.instrument_name = "c8n856M4k"

        #start_dsim = 0 == subprocess.check_call(['corr2_dsim_control.py',
            #'--program', '--start'])
        #if start_dsim:
            #LOGGER.info('D-Eng Started succesfully')
        #else:
            #LOGGER.warn('Failed to start D-Eng')
        # Clear out any arrays, if they exist
        subprocess.check_call(['kcpcmd', 'array-halt', 'array0'])
        while retries and not success:
            try:
                subprocess.check_call(['kcpcmd', '-t', '30', '-s', 'localhost:7147',
                    'array-assign', 'array0'] + multicast_ip.split(','))

                self.katcp_port = int(subprocess.Popen("kcpcmd -s localhost:{0} array-list array0\
                | grep array{1} | cut -f3 -d ' '".format(host_port,array_no)
                    , shell=True, stdout=subprocess.PIPE).stdout.read())

                time.sleep(5)
                print "\nStarting Correlator.\n"
                success = 0 == subprocess.check_call(['kcpcmd','-t','500','-s',
                    'localhost:{}'.format(self.katcp_port), 'instrument-activate',
                        'c8n856M4k'])
                retries -= 1
                if success:
                    LOGGER.info('Correlator started succesfully')
                else:
                    LOGGER.warn('Failed to start correlator, {} attempts left.\
                        \nRestarting Correlator.'
                            .format(retries))

            except Exception:
                subprocess.check_call(['kcpcmd', 'array-halt', 'array0'])
                retries -= 1
                print ('\nFailed to start correlator, {} attempts left\n'
                            .format(retries))
                time.sleep(5)

            print "-"*50
            time.sleep(5)

        if not success:
            raise RuntimeError('Could not successfully start correlator within {} retries'
                               .format(retries_requested))

    def issue_metadata(self):
        subprocess.check_call('corr2_issue_spead_metadata.py')

correlator_fixture = CorrelatorFixture()
