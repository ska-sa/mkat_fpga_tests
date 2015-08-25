import os
import logging
import subprocess
import time

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as test_config
from corr2.dsimhost_fpga import FpgaDsimHost
from katcp import resource_client
from katcp import ioloop_manager

from corr2 import fxcorrelator
from corr2 import utils

from katcp import resource_client
from katcp import ioloop_manager

LOGGER = logging.getLogger(__name__)

class CorrelatorFixture(object):

    def __init__(self, test_config_filename=None):

        if test_config_filename is None:
            test_config_filename = os.environ['CORR2TESTINI']
            self.corr_conf = utils.parse_ini_file(
                test_config_filename)
            self.dsim_conf = self.corr_conf['dsimengine']

        self._correlator = None
        self._dhost = None
        """
        self.iomanager = ioloop_manager.IOLoopManager()
        self.iowrapper = resource_client.IOLoopThreadWrapper(self.iomanager.get_ioloop())
        self.iomanager.start()
        self.resource_client = resource_client.KATCPClientResource(
                dict(name='localhost', address=('localhost', '7147'),
                    controlled=True))
        self.resource_client.set_ioloop(self.iomanager.get_ioloop())
        self.rct = resource_client.ThreadSafeKATCPClientResourceWrapper(
            self.resource_client, self.iowrapper)
        self.rct.start()
        self.rct.until_synced()
        """

        self.io_manager = ioloop_manager.IOLoopManager()
        self.io_wrapper = resource_client.IOLoopThreadWrapper(
            self.io_manager.get_ioloop())
        self.io_manager.start()
        self.rc = resource_client.KATCPClientResource(dict(name='localhost',
            address=('localhost', '7147'), controlled=True))
        self.rc.set_ioloop(self.io_manager.get_ioloop())
        self.rct = resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc, self.io_wrapper)
        self.rct.start()
        self.rct.until_synced()

    @property
    def dhost(self):
        if self._dhost is not None:
            return self._dhost
        else:
            dig_host = self.dsim_conf['host']
            self._dhost = FpgaDsimHost(dig_host, config=self.dsim_conf)
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
            self._correlator = fxcorrelator.FxCorrelator(
                'test correlator', config_source=self.config_filename)
            self.correlator.initialise(program=False)
            LOGGER.info('Correlator started succesfully')
            return self._correlator

#    def start_stop_data(self, start_or_stop, modes):
    def start_stop_data(self, modes):
        self.modes = modes
        LOGGER.info('Correlator starting to capture data.')
        #assert start_or_stop in ('start', 'stop')

        destination = self.correlator.configd['xengine']['output_destination_ip']
        destination_port = self.correlator.configd['xengine']['output_destination_port']
        """
        self.katcp_port = int(subprocess.Popen("/usr/local/bin/kcpcmd \
                    -s localhost:7147 array-list array0\
                        | grep array0 | cut -f3 -d ' '"
                            , shell=True, stdout=subprocess.PIPE).
                                    stdout.read())
        """
        self.katcp_array_port = int(
                        self.rct.req.array_list()[1][0].arguments[1])

        #subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30', '-s' ,
            #'localhost:{}'.format(self.katcp_port) ,'capture-destination' ,
                #'{}'.format(self.modes), '{}:{}'.format(destination,
                    #destination_port)])
        self.katcp_rct.req.capture_destination(self.modes, destination, destination_port)
        #subprocess.check_call(['/usr/local/bin/kcpcmd' ,'-t','30', '-s' ,
            #'localhost:{}'.format(self.katcp_port) ,'capture-{}'
                #.format(start_or_stop), '{}'.format(self.modes)])

        #katcp_rct.req.capture_start(self.modes)
    def start_x_data(self):
        # On array interf
        #self.start_stop_data('start','c856M4k')
        katcp_rct.req.capture_start(self.modes)

    def stop_x_data(self):
        #self.start_stop_data('stop', 'c856M4k')
        katcp_rct.req.capture_stop(self.modes)

    def start_correlator(self, retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        #array_no = 0
        self.dhost
        host_port = self.corr_conf['test_confs']['katcp_port']
        multicast_ip = self.corr_conf['test_confs']['source_mcast_ips']
        instrument = 'c8n856M4k'

        # Clear out any arrays, if exist
        #subprocess.check_call(['/usr/local/bin/kcpcmd', '-s', 'localhost',
            #'array-halt', 'array0'])

        array_list_status, array_list_messages = self.rct.req.array_list()
        array_number = array_list_messages[0].arguments[0]
        try:
            if bool(array_list_messages) is False:
                self.rct.req.array_halt(array_number)
        except:
            LOGGER.info ("Already cleared array")
        finally:
            while retries and not success:
                try:
                    """
                    subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '30',
                        '-s', 'localhost:7147', 'array-assign', 'array0']
                            + multicast_ip.split(','))
                    """
                    self.rct.req.array_assign('array0',
                        *multicast_ip.split(','))
                    self.katcp_array_port = int(
                        self.rct.req.array_list()[1][0].arguments[1])
                    """
                    self.katcp_port = int(subprocess.Popen("/usr/local/bin/kcpcmd \
                        -s localhost:{0} array-list array0\
                            | grep array{1} | cut -f3 -d ' '"
                                .format(host_port,array_no)
                                    , shell=True, stdout=subprocess.PIPE).
                                        stdout.read())
                    """
                    # make a plan for this to work
                    self.katcp_rc = resource_client.KATCPClientResource(
                        dict(name='localhost', address=(
                            'localhost', '{}'.format(self.katcp_array_port)),
                            controlled=True))
                    self.katcp_rc.set_ioloop(self.io_manager.get_ioloop())
                    self.katcp_rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(
                        self.katcp_rc, self.io_wrapper))
                    self.katcp_rct.start()
                    self.katcp_rct.until_synced()


                    LOGGER.info ("Starting Correlator.")
                    success = 0 == (self.katcp_rct.req.instrument_activate(
                        instrument, timeout=500))

                    #success = 0 == subprocess.check_call(['/usr/local/bin/kcpcmd',
                        #'-t','500','-s', 'localhost:{}'.format(self.katcp_port),
                            #'instrument-activate', 'c8n856M4k'])

                    retries -= 1
                    if success == True:
                        LOGGER.info('Correlator started succesfully')
                    else:
                        LOGGER.warn('Failed to start correlator, {} attempts left.\
                            \nRestarting Correlator.'
                                .format(retries))
                except Exception:
                    #subprocess.check_call(['/usr/local/bin/kcpcmd', '-s',
                    #'localhost', 'array-halt', 'array0'])
                    self.rct.req.array_halt(array_number)
                    retries -= 1
                    LOGGER.warn ('\nFailed to start correlator, {} attempts left.\n'
                                .format(retries))
            if not success:
                raise RuntimeError('Could not successfully start correlator within {}\
                    retries'.format(retries_requested))

    def issue_metadata(self):
        #subprocess.check_call(['/usr/local/bin/kcpcmd', '-t', '100', '-s' ,
            #'localhost:{}'.format(self.katcp_port) ,'capture-meta', self.modes])
        self.katcp_rct.req.capture_meta(self.modes)
correlator_fixture = CorrelatorFixture()
