import os
import logging
import subprocess
import time

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as test_config

from corr2.dsimhost_fpga import FpgaDsimHost
from casperfpga import utils as fpgautils
from casperfpga import katcp_fpga

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
            test_config_filename = os.environ.get('CORR2TESTINI',
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
                LOGGER.info('D-Eng is already running.')
            else:
                # Programming and starting D-Eng
                self._dhost.initialise()
                self._dhost.enable_data_output(enabled=True)
                self._dhost.registers.control.write(gbe_txen=True)
                if self._dhost.is_running():
                    LOGGER.info('D-Eng started succesfully')
            return self._dhost

    @property
    def correlator(self):
        if self._correlator is not None:
            LOGGER.info('Using cached correlator instance')
            return self._correlator
        else: # include a check, if correlator is running else start it.
            if not self._correlator_started:
                LOGGER.info('Correlator not running, now starting.')
                self.start_correlator()

            # We assume either start_correlator() above has been called, or the
            # instrument was started with the name contained in self.array_name
            # before running the test.

            # TODO: hard-coded config location
            self.config_filename = '/etc/corr/{}-{}'.format(
            self.array_name, self.instrument)
            if os.path.exists(self.config_filename):
                LOGGER.info('Making new correlator instance')
                self._correlator = fxcorrelator.FxCorrelator(
                    'test correlator', config_source=self.config_filename)
                self.correlator.initialise(program=False)
                return self._correlator
            else:
                self.start_correlator()

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
                LOGGER.error('Failed to assign katcp array port number')
                return False

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
        """
        Enable/Start output product capture
        """
        LOGGER.info ('Start X data capture')
        try:
            self.output_product = (self.correlator.configd['xengine']
                ['output_products'][0])
        except IndexError:
            LOGGER.error('CORR2INI files does not contain Xengine output products')
            raise RuntimeError('CORR2INI files does not contain Xengine output products')

        try:
            reply = self.katcp_rct.req.capture_start(self.output_product)
        except Exception as errmsg:
            LOGGER.error('Failed to capture start: {}'.format(errmsg))
            return False
        else:
            if not reply.succeeded:
                return False
            else:
                return True

    def stop_x_data(self):
        """
        Disable/Stop output product capture
        """
        LOGGER.info ('Stop X data capture')
        try:
            reply = self.katcp_rct.req.capture_stop(self.output_product)
        except Exception as errmsg:
            LOGGER.error('Failed to capture stop: {}'.format(errmsg))
            return False
        else:
            if not reply.succeeded:
                return False
            else:
                return True

    def deprogram_fpgas(self):
        """
        Deprogram CASPER devices listed on dnsmasq leases
        """
        HOSTCLASS = katcp_fpga.KatcpFpga
        try:
            _running_instrument = self.katcp_rct.sensor.instrument_state.get_value()
            config_file = '/etc/corr/{}-{}'.format(self.array_name, _running_instrument)
            if os.path.exists(config_file):
                fhosts = utils.parse_hosts(config_file, section='fengine')
                xhosts = utils.parse_hosts(config_file, section='xengine')
                hosts = fhosts + xhosts
            else:
                raise Exception
        except Exception as errmsg:
            LOGGER.error('Sensor request failed, off to plan B - dnsmasq')
            hosts = []
            masq_path = '/var/lib/misc/dnsmasq.leases'
            if os.path.isfile(masq_path):
                with open(masq_path) as masqfile:
                    for line in masqfile:
                        if line.find('roach') > 0:
                            roachname = line[line.find('roach'):line.find(
                                        ' ', line.find('roach') + 1)].strip()

                            result = (subprocess.call(['ping', '-c', '1', roachname],
                                      stdout = subprocess.PIPE, stderr = subprocess.PIPE))
                            if result == 0:
                                hosts.append(roachname)
        try:
            if not len(hosts) == 0:
                connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(
                                  HOSTCLASS, set(hosts))
                deprogrammed_fpgas = fpgautils.threaded_fpga_function(
                                    connected_fpgas, 10, 'deprogram')
                LOGGER.info('FPGAs in dnsmasq all deprogrammed')
                return True
            else:
                LOGGER.error('Failed to deprogram FPGAs no hosts available: {}'.format(
                            errmsg))
                raise RuntimeError('No hosts available.')
                return False
        except Exception as errmsg:
            LOGGER.error('Failed to deprogram FPGAs: {}'.format(errmsg))
            return False

    def ensure_instrument(self, instrument, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required

        """
        if not self.check_instrument(instrument):
            LOGGER.info('Correlator not running requested instrument, will restart.')
            self.deprogram_fpgas()
            self.instrument = instrument
            self.start_correlator(self.instrument, **kwargs)

    def check_instrument(self, instrument):
        """Return true if named instrument is enabled on correlator array

        Uses the correlator array KATCP interface to check if the requested
        instrument is active

        """
        # Get a list of instruments associated with instrument
        try:
            _rsync = self.katcp_rct.start()
        except AttributeError:

            LOGGER.error('katcp rct has no attribute.')
            return False

        except RuntimeError:
            # This probably means that no array has been defined yet and therefore the
            # katcp_rct client cannot be created. IOW, the desired instrument would
            # not be available
            LOGGER.error('Could not resynchronise katcp connection.')
            return False

        else:
            if self.katcp_rct.state == 'synced':
                reply = self.katcp_rct.req.instrument_list(instrument)
                if not reply.succeeded:
                    return False
                    LOGGER.error('Array request failed: {}'.format(reply))
                    raise RuntimeError('Array request failed: {}'.format(reply))
            else:
                return False
                LOGGER.error('Could not resynchronise katcp connection.')
                raise RuntimeError('Could not resynchronise katcp connection.')

            instruments_available = [instrument_avail.arguments[0]
                                     for instrument_avail in reply.informs]
            # Test to see if requested instrument is available on the instrument list
            if instrument not in instruments_available:
                return False
                LOGGER.error('Instrument: {} is not in instrument list: {}'.format(
                                    instrument, instruments_available))
                raise RuntimeError('Instrument: {} is not in instrument list: {}'.format(
                                    instrument, instruments_available))

            # Get currently running instrument listed on the sensor(s)
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
            if not reply.istatus:
                return False
                raise RuntimeError('Sensor request failed: {}'.format(reply))
            running_intrument = reply.value

            instrument_present = instrument == running_intrument
            if instrument_present:
                self.instrument = instrument
            return instrument_present

    def start_correlator(self, instrument='bc8n856M4k', retries=30, loglevel='INFO'):
        success = False
        retries_requested = retries
        self.instrument = instrument
        self._correlator = None # Invalidate cached correlator instance
        LOGGER.info('Confirm DEngine is running before starting correlator')
        if not self.dhost.is_running():
            raise RuntimeError('DEngine: {} not running.'.format(self.dhost.host))
        _d = self.test_conf
        host_port = _d['test_confs']['katcp_port']
        multicast_ip = _d['test_confs']['source_mcast_ips']
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

                LOGGER.info ("Starting Correlator. Try: {}".format(retries))
                reply, informs = self.katcp_rct.req.instrument_activate(
                    self.instrument, timeout=500)

                success = reply.reply_ok()
                retries -= 1

                if success == True:
                    LOGGER.info('Correlator started succesfully')
                else:
                    LOGGER.warn('Failed to start correlator, {} attempts left.'
                        '\nRestarting Correlator.\nReply:{}, Informs: {}'
                            .format(retries, reply, informs))
                    self.deprogram_fpgas()
                    self.rct.req.array_halt(self.array_name)
                    self.katcp_rct.stop()
                    self._katcp_rct = None
                    self.katcp_array_port = None

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
            try:
                self._correlator_started = False
                self.katcp_rct.stop()
                self._katcp_rct = None
                self._correlator = None
            except:
                raise RuntimeError('Could not successfully start correlator '
                'within {} retries'.format(retries_requested))

    def issue_metadata(self):
        self.katcp_rct.req.capture_meta(self.output_product)

correlator_fixture = CorrelatorFixture()
