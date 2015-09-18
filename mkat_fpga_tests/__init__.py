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

cleanups = []
"""Callables that will be called in reverse order at package teardown

Stored as a tuples of (callable, args, kwargs)
"""

def add_cleanup(fn, *args, **kwargs):
    cleanups.append((fn, args, kwargs))

def teardown_package():
    while cleanups:
        fn, args, kwargs = cleanups.pop()
        try:
            fn(*args, **kwargs)
        except Exception:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):

    def __init__(self, test_config_filename=None):
        if test_config_filename is None:
            test_config_filename = os.environ.get(
                'CORR2TESTINI',
                './mkat_fpga_tests/config_templates/test_conf.ini')
            self.corr_conf = utils.parse_ini_file(
                test_config_filename)
            self.dsim_conf = self.corr_conf['dsimengine']
        # Assume the correlator is already started if start_correlator is False
        self._correlator_started = not int(
            test_config.get('start_correlator', False))
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None

        self.io_manager = ioloop_manager.IOLoopManager()
        self.io_wrapper = resource_client.IOLoopThreadWrapper(
            self.io_manager.get_ioloop())
        add_cleanup(self.io_manager.stop)
        self.io_manager.start()
        self.rc = resource_client.KATCPClientResource(dict(name='localhost',
            address=('localhost', '7147'), controlled=True))
        self.rc.set_ioloop(self.io_manager.get_ioloop())
        self.rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc,
            self.io_wrapper))
        self.rct.start()
        add_cleanup(self.rct.stop)
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
            LOGGER.info('Using cached correlator instance')
            return self._correlator
        else: # include a check, if correlator is running else start it.
            if not self._correlator_started:
                self.start_correlator()

            # We assume either start_correlator() above has been called, or
            # the c8n856M4k instrument was started on array0 before running the
            # test.

            # TODO: hard-coded config location
            self.config_filename = '/etc/corr/array0-c8n856M4k'
            LOGGER.info('Making new correlator instance')
            self._correlator = fxcorrelator.FxCorrelator(
                'test correlator', config_source=self.config_filename)
            self.correlator.initialise(program=False)
            return self._correlator

    def halt_array(self):
        if not self._correlator:
            raise RuntimeError('Array not yet initialised')

        self.katcp_rct.req.halt()
        self._correlator_started = False
        self.katcp_rct.stop()

        self._katcp_rct = None
        # TODO: MM(2015-09-11) Proper teardown of corr object(katcp connections etc.)
        # Must still be implemented.
        self._correlator = None

    @property
    def katcp_rct(self):
        if self._katcp_rct is None:
            try:
                self.katcp_array_port = int(
                    self.rct.req.array_list()[1][0].arguments[1])
            except IndexError:
                LOGGER.error('Failed to assign katcp array port number.')
                raise RuntimeError('Failed to assign katcp array port number.')

            katcp_rc = resource_client.KATCPClientResource(
                dict(name='localhost', address=(
                    'localhost', '{}'.format(self.katcp_array_port)),
                    controlled=True))
            katcp_rc.set_ioloop(self.io_manager.get_ioloop())
            self._katcp_rct = (
            resource_client.ThreadSafeKATCPClientResourceWrapper(
                katcp_rc, self.io_wrapper))
            self._katcp_rct.start()
            self._katcp_rct.until_synced()
        return self._katcp_rct

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
        # starting d-engine before correlator
        self.dhost
        host_port = self.corr_conf['test_confs']['katcp_port']
        multicast_ip = self.corr_conf['test_confs']['source_mcast_ips']
        instrument = 'c8n856M4k'
        array_list_status, array_list_messages = self.rct.req.array_list()

        try:
            if array_list_messages:
                self.array_number = array_list_messages[0].arguments[0]
                self.rct.req.array_halt(self.array_number)
        except IndexError:
            LOGGER.error("Unable to halt array due to empty array number")
            raise RuntimeError("Unable to halt array due to empty array number")

        while retries and not success:
            try:
                self.rct.req.array_assign('array0',
                    *multicast_ip.split(','))

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
                try:
                    self.rct.req.array_halt(self.array_number)
                except IndexError:
                    LOGGER.error("Unable to halt array due to empty array number")
                    raise RuntimeError("Unable to halt array due to empty array"
                        "number")

                self.katcp_rct.stop()
                retries -= 1
                LOGGER.warn ('\nFailed to start correlator,'
                    '{} attempts left.\n'.format(retries))
        if success:
            self._correlator_started = True
        else:
            self._correlator_started = False
            raise RuntimeError('Could not successfully start correlator'
            'within {} retries'.format(retries_requested))

    def issue_metadata(self):
        self.katcp_rct.req.capture_meta(self.output_product)

correlator_fixture = CorrelatorFixture()
