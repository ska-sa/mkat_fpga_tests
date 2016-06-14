import os
import sys
import logging
import subprocess
import time
import corr2

# Config using nose-testconfig plugin, set variables with --tc on nose command line
from testconfig import config as nose_test_config

from corr2.dsimhost_fpga import FpgaDsimHost
from casperfpga import utils as fpgautils
from casperfpga import katcp_fpga

from katcp import resource_client
from katcp import ioloop_manager

from katcp import KatcpClientError
from katcp import KatcpDeviceError
from katcp import KatcpSyntaxError

from corr2 import fxcorrelator

LOGGER = logging.getLogger(__name__)

cleanups = []
"""Callables that will be called in reverse order at package teardown

Stored as a tuples of (callable, args, kwargs)
"""

def add_cleanup(fn, *args, **kwargs):
    LOGGER.info('Function Tear Down')
    cleanups.append((fn, args, kwargs))

def teardown_package():
    while cleanups:
        fn, args, kwargs = cleanups.pop()
        try:
            fn(*args, **kwargs)
        except:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):

    def __init__(self, array=None, instrument=None, resource_clt=None):

        # TODO: hard-coded Array number
        # We assume either start_correlator() above has been called, or the instrument
        # was started with the name contained in self.array_name before running the
        # test.
        self.array_name = array
        self.instrument = instrument
        self.resource_clt = resource_clt
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None
        self._rct = None
        self.katcp_array_port = None
        # Assume the correlator is already started if start_correlator is False
        self._correlator_started = not int(
        nose_test_config.get('start_correlator', False))
        self.test_conf = self.test_config_file()

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
            self.rc = resource_client.KATCPClientResource(
                dict(name='{}'.format(self.resource_clt),
                     address=('{}'.format(self.resource_clt),
                             '7147'),
                     controlled=True))
            self.rc.set_ioloop(self.io_manager.get_ioloop())
            self._rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc,
                self.io_wrapper))
            self._rct.start()
            add_cleanup(self._rct.stop)
            self._rct.until_synced(timeout=60)
        return self._rct

    @property
    def dhost(self):
        if self._dhost is not None:
            return self._dhost
        else:
            self.config_filename = '/etc/corr/{}-{}'.format(
                self.array_name, self.instrument)
            if os.path.exists(self.config_filename):
                self.dsim_conf = corr2.utils.parse_ini_file(self.config_filename)['dsimengine']
                dig_host = self.dsim_conf['host']
            else:
                LOGGER.error('Could not retrieve information from config file, resorting to test_conf.ini')
                self.dsim_conf = self.test_conf['dsimengine']
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
                try:
                    self._correlator = fxcorrelator.FxCorrelator(
                    'test correlator', config_source=self.config_filename)
                    self.correlator.initialise(program=False)
                    return self._correlator
                except:
                    self.start_correlator(self.instrument, **kwargs)
            else:
                LOGGER.info('Starting correlator with default instrument')
                self.start_correlator()

    def halt_array(self):
        """
        Halting of primary and secondary katcp arrays and ensure that the correlator
        object is teared-down
        """
        #if not self._correlator:
            #raise RuntimeError('Array not yet initialised')
        LOGGER.info('Halting primary array.')
        self.katcp_rct.stop()
        reply, informs = self.rct.req.array_halt(self.array_name)
        self.rct.stop()
        self._rct = None
        self._katcp_rct = None
        # TODO: MM(2015-09-11) Proper teardown of corr object(katcp connections etc.)
        # Must still be implemented.
        self._correlator_started = False
        self._correlator = None
        LOGGER.info('Array {} halted and teared-down'.format(self.array_name))

    @property
    def katcp_rct(self):
        multicast_ip = self.get_multicast_ips(self.instrument)
        if self._katcp_rct is None:
            reply, informs = self.rct.req.array_list(self.array_name)
            # If no sub-array present create one, but this could cause problems
            # if more than one sub-array is present. Update this to check for
            # required sub-array.
            if reply.reply_ok():
                self.katcp_array_port = int(informs[0].arguments[1])
            else:
                LOGGER.error('Array has not been assigned yet, will try to assign.')
                try:
                    reply, informs = self.rct.req.array_assign(self.array_name,
                        *multicast_ip)
                except ValueError:
                    LOGGER.exception('')
                else:
                    if len(reply.arguments) == 2:
                        try:
                            self.katcp_array_port = int(reply.arguments[-1])
                            LOGGER.info('Array {} assigned successfully'\
                                        ''.format(self.katcp_array_port))
                        except ValueError:
                            LOGGER.exception('Array assign failed: {}'.format(reply))
                            #self.rct.req.array_halt(self.array_name)
                            #self.rct.stop()
                            #self.rct.start()
                            #self.rct.until_synced(timeout=60)
                            #reply, informs = self.rct.req.array_assign(self.array_name,
                                #*multicast_ip)

            katcp_rc = resource_client.KATCPClientResource(
                        dict(name='{}'.format(self.resource_clt),
                             address=('{}'.format(self.resource_clt),
                                     '{}'.format(self.katcp_array_port)),
                             controlled=True))
            katcp_rc.set_ioloop(self.io_manager.get_ioloop())
            self._katcp_rct = (
                              resource_client.ThreadSafeKATCPClientResourceWrapper(
                              katcp_rc, self.io_wrapper))
            self._katcp_rct.start()
            self._katcp_rct.until_synced(timeout=60)
            add_cleanup(self._katcp_rct.stop)
        else:
            self._katcp_rct.start()
            self._katcp_rct.until_synced(timeout=60)
        return self._katcp_rct

    def start_x_data(self):
        """
        Enable/Start output product capture
        """
        LOGGER.info ('Start X data capture')
        try:
            reply = self.katcp_rct.req.capture_list()
        except IndexError:
            LOGGER.error('Config file does not contain Xengine output products.')
            return False
        else:
            if reply.succeeded:
                self.output_product = reply.informs[0].arguments[0]
            else:
                self.output_product = self.correlator.configd['xengine']['output_products'][0]

        try:
            reply = self.katcp_rct.req.capture_start(self.output_product)
        except:
            errmsg = 'Failed to capture start'
            LOGGER.exception(errmsg)
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
        except:
            LOGGER.exception('Failed to capture stop')
            return False
        else:
            if not reply.succeeded:
                return False
            else:
                return True

    def deprogram_fpgas(self, instrument):
        """
        Deprogram CASPER devices listed on config file or dnsmasq leases
        """
        hostclass = katcp_fpga.KatcpFpga
        self.instrument = instrument
        try:
            try:
                LOGGER.info('Retrieving running instrument from sensors')
                _running_instrument = self.katcp_rct.sensor.instrument_state.get_value()
            except:
                if self.instrument is not None:
                    _running_instrument = self.instrument

            if len(_running_instrument) > 4:
                config_file = '/etc/corr/{}-{}'.format(self.array_name, _running_instrument)
            else:
                config_file = '/etc/corr/{}-{}'.format(self.array_name, self.instrument)

            if os.path.exists(config_file):
                LOGGER.info('Retrieving running hosts from running config')
                fhosts = corr2.utils.parse_hosts(config_file, section='fengine')
                xhosts = corr2.utils.parse_hosts(config_file, section='xengine')
                hosts = fhosts + xhosts
            else:
                raise Exception
        except:
            LOGGER.error('Could not get instrument from sensors and config file does not exist'
                        ', Resorting to plan B - retreiving roach list from CORR2INI'
                        ', In order to deprogram')
            corr2ini_link = os.environ.get('CORR2INI')
            if corr2ini_link is not None:
                fhosts = corr2.utils.parse_hosts(corr2ini_link, section='fengine')
                xhosts = corr2.utils.parse_hosts(corr2ini_link, section='xengine')
                hosts = fhosts + xhosts
            else:
                LOGGER.error('Failed to retrieve hosts from CORR2INI')
                return False

        if not len(hosts) == 0:
            try:
                try:
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(
                          hostclass, list(set(hosts)))
                except:
                    raise Exception
                try:
                    hosts = [host.host for host in connected_fpgas if host.ping() == True]
                except:
                    raise Exception
                try:
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(
                                      hostclass, hosts)
                except:
                    raise Exception
                try:
                    deprogrammed_fpgas = fpgautils.threaded_fpga_function(
                                        connected_fpgas, 10, 'deprogram')
                except:
                    raise Exception
                LOGGER.info('FPGAs in dnsmasq all deprogrammed')
                return True
            except (katcp_fpga.KatcpRequestFail, KatcpClientError, KatcpDeviceError, KatcpSyntaxError):
                errmsg = 'Failed to connect to roaches, reboot devices to fix.'
                LOGGER.exception(errmsg)
                sys.exit(errmsg)
            except Exception:
                errmsg = 'Failed to connect to roaches, reboot devices to fix.'
                LOGGER.exception(errmsg)
                sys.exit(errmsg)
        else:
            LOGGER.error('Failed to deprogram FPGAs no hosts available')
            return False

    def get_running_intrument(self):
        """
        Returns currently running instrument listed on the sensor(s)
        """
        try:
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
        except AttributeError:
            LOGGER.error('KATCP Request does not contain attributes')
            return False

        if not reply.istatus:
            LOGGER.error('Sensor request failed: {}'.format(reply))
            return False
        running_intrument = reply.value
        return running_intrument

    def ensure_instrument(self, instrument, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required

        """
        if self.check_instrument(instrument) == True:
            return True
        else:
            LOGGER.info('Correlator not running requested instrument, will restart.')
            deprogram_status = self.deprogram_fpgas(instrument)
            if not deprogram_status:
                LOGGER.info('Could not deprogram the hosts')
            self.instrument = instrument
            corr_success = self.start_correlator(self.instrument, **kwargs)
            if corr_success == True:
                return True
            else:
                return False

    def check_instrument(self, instrument):
        """Return true if named instrument is enabled on correlator array

        Uses the correlator array KATCP interface to check if the requested
        instrument is active

        """
        try:
            assert isinstance(self.katcp_rct,
                        resource_client.ThreadSafeKATCPClientResourceWrapper)
            self.katcp_rct.start()
            self.katcp_rct.until_synced()
        except:
            # This probably means that no array has been defined yet and therefore the
            # katcp_rct client cannot be created. IOW, the desired instrument would
            # not be available
            LOGGER.error('katcp rct has no attribute or no correlator instance is running.')
            return False
        else:
            if self.katcp_rct.state == 'synced':
                reply = self.katcp_rct.req.instrument_list(instrument)
                if not reply.succeeded:
                    LOGGER.error('Array request failed: {}'.format(reply))
                    return False
            else:
                LOGGER.error('Could not resynchronise katcp connection.')
                return False

            instruments_available = [instrument_avail.arguments[0]
                                     for instrument_avail in reply.informs]
            # Test to see if requested instrument is available on the instrument list
            if instrument not in instruments_available:
                LOGGER.error('Instrument: {} is not in instrument list: {}'.format(
                                    instrument, instruments_available))
                return False

            # Get currently running instrument listed on the sensor(s)
            try:
                reply = self.katcp_rct.sensor.instrument_state.get_reading()
            except AttributeError:
                LOGGER.error('Instrument state could not be retrieved from the sensors')
                return False
            else:
                if not reply.istatus:
                    LOGGER.error('Sensor request failed: {}'.format(reply))
                    return False
                running_intrument = reply.value

                instrument_present = instrument == running_intrument
                if instrument_present:
                    self.instrument = instrument
                    LOGGER.info('Confirm that the named instrument {} is enabled on '
                                'correlator array.'.format(self.instrument))
                return instrument_present

    def test_config_file(self):
        """
        Configuration file containing information such as dsim, pdu and dataswitch ip's
        return: Dict
        """
        path, _None = os.path.split(__file__)
        path, _None = os.path.split(path)
        conf_path = '/config/test_conf.ini'
        config_file = path + conf_path
        if os.path.exists(config_file):
            try:
                test_conf = corr2.utils.parse_ini_file(config_file)
                return test_conf
            except IOError:
                errmsg = 'Failed to read test config file, Test will exit'
                LOGGER.error(errmsg)
                sys.exit(errmsg)

    def get_multicast_ips(self, instrument):
        """
        Retrieves multicast ips from test configuration file and calculates the number
        of inputs depending on which instrument is being initialised
        Param: String
            Instrument
        Return: List
            list of multicast ip addresses"""
        if instrument is None:
            return False
        self.test_conf = self.test_config_file()
        multicast_ip_inp = self.test_conf['inst_param']['source_mcast_ips'].split(',')
        if self.instrument.startswith('bc') or self.instrument.startswith('c'):
            if self.instrument[0] == 'b':
                multicast_ip = multicast_ip_inp * (int(self.instrument[2]) / 2)
            else:
                multicast_ip = multicast_ip_inp * (int(self.instrument[1]) / 2)

        return multicast_ip

    def start_correlator(self, instrument=None, retries=10, loglevel='INFO'):
        LOGGER.info('Will now try to start the correlator')
        success = False
        retries_requested = retries
        if instrument is not None:
            self.instrument = instrument
        self._correlator = None # Invalidate cached correlator instance
        LOGGER.info('Confirm DEngine is running before starting correlator')
        if not self.dhost.is_running():
            raise RuntimeError('DEngine: {} not running.'.format(self.dhost.host))
        multicast_ip = self.get_multicast_ips(self.instrument)
        self.rct.start()
        self.rct.until_synced(timeout=60)
        reply, informs = self.rct.req.array_list(self.array_name)

        if not reply.reply_ok():
            LOGGER.error('Failed to halt down the array in primary interface: reply: {}'.format(reply))
            return False
        else:
            try:
                informs = informs[0]
                self.katcp_array_port = int([i for i in informs.arguments if len(i) == 5][0])
            except ValueError:
                LOGGER.exception('Failed to assign katcp port: Reply: {}'.format(reply))
                reply = self.rct.req.array_halt(self.array_name)
                if not reply.succeeded:
                    LOGGER.error('Unable to halt array {}: {}'.format(self.array_name, reply))
                    return False

        while retries and not success:
            self.katcp_array_port = None
            try:
                if self.katcp_array_port is None:
                    LOGGER.info('Assigning array port number')
                    self.rct.start()
                    self.rct.until_synced(timeout=10)
                    reply, informs = self.rct.req.array_assign(self.array_name,
                        *multicast_ip)
                    if reply.reply_ok():
                        self.katcp_array_port = int(reply.arguments[-1])
                    else:
                        self.katcp_array_port = None
                """
                try:
                    #self.katcp_array_port = int(reply.arguments[-1])
                    LOGGER.info('Array port assigned: {}'.format(self.katcp_array_port))
                except ValueError:
                    LOGGER.fatal('Failed to assign array port number on {}'.format(self.array_name))
                else:
                    if not reply.reply_ok():
                        LOGGER.fatal('Failed to assign array port number on {}'.format(self.array_name))
                        return False
                """
                instrument_param = [int(i) for i in self.test_conf['inst_param']['instrument_param']
                                    if i != ',']
                LOGGER.info ("Starting {} with {} parameters. Try #{}".format(
                    self.instrument, instrument_param, retries))
                reply = self.katcp_rct.req.instrument_activate(
                    self.instrument, *instrument_param, timeout=500)
                success = reply.succeeded
                retries -= 1

                if success == True:
                    LOGGER.info('Instrument {} started succesfully'.format(self.instrument))
                else:
                    LOGGER.warn('Failed to start correlator, {} attempts left. '
                        'Restarting Correlator. Reply:{}'
                            .format(retries, reply))
                    self.halt_array()
                    success = False
                    LOGGER.debug('Katcp teardown and restarting correlator.')

            except Exception:
                try:
                    self.rct.req.array_halt(self.array_name)
                except IndexError:
                    LOGGER.error("Unable to halt array: Empty Array number")
                try:
                    assert isinstance(self.katcp_rct,
                        resource_client.ThreadSafeKATCPClientResourceWrapper)
                except AssertionError:
                    return False
                else:
                    try:
                        self.katcp_rct.stop()
                    except AttributeError:
                        LOGGER.error('KATCP request does not contain attributes')
                        return False
                    else:
                        retries -= 1
                        LOGGER.warn ('Failed to start correlator,'
                            '{} attempts left.\n'.format(retries))
            if retries < 0:
                success = False
                return False

        if success:
            self._correlator_started = True
            return True
        else:
            try:
                self.halt_array()
                self._correlator_started = False
                self.katcp_rct.stop()
                self.rct.stop()
                self._katcp_rct = None
                self._correlator = None
            except:
                self.deprogram_fpgas(self.instrument)
                LOGGER.critical('Could not successfully start correlator '
                                 'within {} retries'.format(retries_requested))
                return False
            return False

    def issue_metadata(self):
        """Issue Spead metadata"""
        try:
            self.katcp_rct.req.capture_meta(self.output_product)
            LOGGER.info('New metadata issued')
            return True
        except:
            LOGGER.error('Failed to issue new metadata')
            return False

correlator_fixture = CorrelatorFixture()
