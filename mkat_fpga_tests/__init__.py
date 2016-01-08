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
            self.test_conf = utils.parse_ini_file(
                test_config_filename)
            self.dsim_conf = self.test_conf['dsimengine']
        # Assume the correlator is already started if start_correlator is False
        self._correlator_started = not int(
            test_config.get('start_correlator', False))

        # TODO: hard-coded Array number
        # We assume either start_correlator() above has been called, or the instrument
        # was started with the name contained in self.array_name before running the
        # test.
        self.array_name = 'array0'
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None
        self._rct = None

    @property
    def rct(self):
        if self._rct is not None:
            return self._rct
        else:
            self.io_manager = ioloop_manager.IOLoopManager()
            self.io_wrapper = resource_client.IOLoopThreadWrapper(
                self.io_manager.get_ioloop())
            add_cleanup(self.io_manager.stop)
            self.io_wrapper.default_timeout = 10
            self.io_manager.start()
            self.rc = resource_client.KATCPClientResource(dict(name='localhost',
                address=('localhost', '7147'), controlled=True))
            self.rc.set_ioloop(self.io_manager.get_ioloop())
            self._rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc,
                self.io_wrapper))
            self._rct.start()
            add_cleanup(self._rct.stop)
            self._rct.until_synced()
        return self._rct

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

            # We assume either start_correlator() above has been called, or the instrument
            # was started with the name contained in self.array_name before running the
            # test.

            # TODO: hard-coded config location
            self.config_filename = '/etc/corr/{}-{}'.format(
                self.array_name, self.instrument)
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

    def ensure_instrument(self, instrument, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required

        """
        if not self.check_instrument(instrument):
            self.start_correlator(instrument, **kwargs)

    def check_instrument(self, instrument):
        """Return true if named instrument is enabled on correlator array

        Uses the correlator array KATCP interface to check if the requested instrument is
        active

        """
        # Get a list of products associated with instrument
        try:
            reply = self.katcp_rct.req.instrument_list(instrument)
        except RuntimeError:
            # This probably means that no array has been defined yet and therefore the
            # katcp_rct client cannot be created. IOW, the desired instrument would not be
            # available
            return False
        if not reply.succeeded:
            raise RuntimeError('Array request failed: {}'.format(reply))
         # instrument_products = set(reply.informs[0].arguments[1:])
        instrument_products = reply.informs[0].arguments[1:]

        # Get list of available data products and check that the products belonging to the
        # requested instrument is available
        reply = self.katcp_rct.req.capture_list()
        if not reply.succeeded:
            raise RuntimeError('Array request failed: {}'.format(reply))
        # products = set(i.arguments[0] for i in reply.informs)
        products = reply.informs[0].arguments[0][1:-1]

        # instrument_products_present = products.intersection(
        #     instrument_products) == instrument_products
        instrument_products_present = products == instrument_products[0]
        if instrument_products_present:
            self.instrument = instrument
        return instrument_products_present

    def start_correlator(self, instrument='c8n856M4k',
                         retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        self.instrument = instrument
        self._correlator = None # Invalidate cached correlator instance
        # starting d-engine before correlator
        self.dhost
        host_port = self.test_conf['test_confs']['katcp_port']
        multicast_ip = self.test_conf['test_confs']['source_mcast_ips']
        array_list_status, array_list_messages = self.rct.req.array_list(
            self.array_name)
        if array_list_messages:
            reply = self.rct.req.array_halt(self.array_name)
            if not reply.succeeded:
                raise RuntimeError("Unable to halt array {}: {}"
                                   .format(self.array_name, reply))

        while retries and not success:
            try:
                self.rct.req.array_assign(self.array_name,
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
                    self.rct.req.array_halt(self.array_name)

            except Exception:
                try:
                    self.rct.req.array_halt(self.array_name)
                except IndexError:
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
