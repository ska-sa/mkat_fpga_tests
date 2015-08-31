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

        self.io_manager = ioloop_manager.IOLoopManager()
        self.io_wrapper = resource_client.IOLoopThreadWrapper(
            self.io_manager.get_ioloop())
        self.io_manager.start()
        self.rc = resource_client.KATCPClientResource(dict(name='localhost',
            address=('localhost', '7147'), controlled=True))
        self.rc.set_ioloop(self.io_manager.get_ioloop())
        self.rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc,
            self.io_wrapper))
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
        else: # include a check, if correlator is runninge else start it.
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


    def start_stop_data(self, modes):
        LOGGER.info('Correlator starting to capture data.')
        destination = self.correlator.configd['xengine']['output_destination_ip']
        destination_port = (self.correlator.configd['xengine']
            ['output_destination_port'])
        self.katcp_array_port = int(
                        self.rct.req.array_list()[1][0].arguments[1])
        self.katcp_rct.req.capture_destination(self.modes, destination,
            destination_port)

    def start_x_data(self):
        LOGGER.info ('Start X data capture')
        self.output_product = (self.correlator.configd['xengine']
            ['output_products'][0])
        self.katcp_rct.req.capture_start(self.output_product)

    def stop_x_data(self):
        LOGGER.info ('Stop X data capture')
        self.katcp_rct.req.capture_stop(self.output_product)

    def start_correlator(self, retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        self.dhost # starting d-engine
        host_port = self.corr_conf['test_confs']['katcp_port']
        multicast_ip = self.corr_conf['test_confs']['source_mcast_ips']
        instrument = 'c8n856M4k'
        array_list_status, array_list_messages = self.rct.req.array_list()

        try:
            if array_list_messages:
                self.array_number = array_list_messages[0].arguments[0]
                self.rct.req.array_halt(self.array_number)
        except:
            LOGGER.info ("Already cleared array")

        finally:
            while retries and not success:
                try:
                    self.rct.req.array_assign('array0',
                        *multicast_ip.split(','))
                    self.katcp_array_port = int(
                        self.rct.req.array_list()[1][0].arguments[1])

                    self.katcp_rc = resource_client.KATCPClientResource(
                        dict(name='localhost', address=(
                            'localhost', '{}'.format(self.katcp_array_port)),
                            controlled=True))
                    self.katcp_rc.set_ioloop(self.io_manager.get_ioloop())
                    self.katcp_rct = (
                    resource_client.ThreadSafeKATCPClientResourceWrapper(
                        self.katcp_rc, self.io_wrapper))
                    self.katcp_rct.start()
                    self.katcp_rct.until_synced()

                    LOGGER.info ("Starting Correlator.")
                    reply, informs = self.katcp_rct.req.instrument_activate(
                        instrument, timeout=500)
                    success = reply.reply_ok()
                    retries -= 1

                    if success == True:
                        LOGGER.info('Correlator started succesfully')
                    else:
                        LOGGER.warn('Failed to start correlator, {} attempts left.'
                            '\nRestarting Correlator.\nReply:{}, Informs: {}'
                                .format(retries, reply, informs))
                        self.rct.req.array_halt(self.array_number)

                except Exception:
                    self.rct.req.array_halt(self.array_number)
                    retries -= 1
                    LOGGER.warn ('\nFailed to start correlator,'
                        '{} attempts left.\n'.format(retries))
            if not success:
                raise RuntimeError('Could not successfully start correlator'
                'within {} retries'.format(retries_requested))

    def issue_metadata(self):
        self.katcp_rct.req.capture_meta(self.output_product)
correlator_fixture = CorrelatorFixture()
