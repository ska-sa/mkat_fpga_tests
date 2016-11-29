import logging
import os
import glob
import sys
import socket
import struct

from inspect import currentframe, getframeinfo

import corr2
from casperfpga import katcp_fpga
from casperfpga import tengbe
from casperfpga import utils as fpgautils
from concurrent.futures import TimeoutError
from corr2 import fxcorrelator
from corr2.dsimhost_fpga import FpgaDsimHost
from katcp import KatcpClientError
from katcp import KatcpDeviceError
from katcp import KatcpSyntaxError
from katcp import ioloop_manager
from katcp import resource_client
from katcp.core import ProtocolFlags
from katcp.resource_client import KATCPSensorError
#from testconfig import config as nose_test_config
from mkat_fpga_tests.utils import ignored

# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
from memory_profiler import profile as DetectMemLeaks

try:
    get_username = os.getlogin()
except OSError:
    import pwd
    get_username = pwd.getpwuid(os.getuid()).pw_name

LOGGER = logging.getLogger(__name__)
Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - '
                              '%(pathname)s : %(lineno)d - %(message)s')
Handler = logging.FileHandler('/tmp/%s_by_%s.log' % (__name__, get_username))
Handler.setFormatter(Formatter)
LOGGER.addHandler(Handler)


_cleanups = []
"""Callables that will be called in reverse order at package teardown. Stored as a tuples of (callable,
args, kwargs)
"""
timeout = 60

def add_cleanup(_fn, *args, **kwargs):
    _cleanups.append((_fn, args, kwargs))

def teardown_package():
    """
    nose allows tests to be grouped into test packages. This allows package-level setup; for instance,
    if you need to create a test database or other data fixture for your tests, you may create it in
    package setup and remove it in package teardown once per test run, rather than having to create and
    tear it down once per test module or test case.
    To create package-level setup and teardown methods, define setup and/or teardown functions in the
    __init__.py of a test package. Setup methods may be named setup, setup_package, setUp, or
    setUpPackage; teardown may be named teardown, teardown_package, tearDown or tearDownPackage.
    Execution of tests in a test package begins as soon as the first test module is loaded from the
    test package.

    ref:https://nose.readthedocs.io/en/latest/writing_tests.html?highlight=setup_package#test-packages
    """
    while _cleanups:
        _fn, args, kwargs = _cleanups.pop()
        try:
            _fn(*args, **kwargs)
        except:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):
    def __init__(self, katcp_clt=None):
        self.katcp_clt = katcp_clt
        self.corr_config = None
        self.corr2ini_path = None
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None
        self._rct = None
        self.katcp_array_port = None
        # Assume the correlator is already started if start_correlator is False
        nose_test_config = {}
        self._correlator_started = not int(
            nose_test_config.get('start_correlator', False))
        self.test_config = self._test_config_file
        self.array_name, self.instrument = self._get_instrument

    @property
    def rct(self):
        if self._rct is not None:
            return self._rct
        else:
            self.io_manager = ioloop_manager.IOLoopManager()
            self.io_wrapper = resource_client.IOLoopThreadWrapper(
                self.io_manager.get_ioloop())
            LOGGER.info('Cleanup function \'self.io_manager\': File: %s line: %s' % (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            add_cleanup(self.io_manager.stop)
            self.io_wrapper.default_timeout = timeout
            self.io_manager.start()
            self.rc = resource_client.KATCPClientResource(
                dict(name='{}'.format(self.katcp_clt),
                     address=('{}'.format(self.katcp_clt),
                              '7147'),
                     controlled=True))
            self.rc.set_ioloop(self.io_manager.get_ioloop())
            self._rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(self.rc,
                                                                              self.io_wrapper))
            self._rct.start()
            LOGGER.info('Cleanup function \'self._rct\': File: %s line: %s' % (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            add_cleanup(self._rct.stop)
            try:
                self._rct.until_synced(timeout=timeout)
            except TimeoutError:
                self._rct.stop()
        return self._rct

    @property
    def dhost(self, program=False):
        if self._dhost is not None:
            return self._dhost
        else:
            self.config_filename = '/etc/corr/{}-{}'.format(self.array_name, self.instrument)
            if os.path.exists(self.config_filename):
                LOGGER.info('Retrieving dsim engine info from config file: %s' % self.config_filename)
                self.corr_config = corr2.utils.parse_ini_file(self.config_filename)
                self.dsim_conf = self.corr_config['dsimengine']
            elif self.instrument is not None:
                self.corr2ini_path = '/etc/corr/templates/{}'.format(self.instrument)
                LOGGER.info('Setting CORR2INI system enviroment to point to %s' % self.corr2ini_path)
                os.environ['CORR2INI'] = self.corr2ini_path
                self.corr_config = corr2.utils.parse_ini_file(self.corr2ini_path)
                self.dsim_conf = self.corr_config['dsimengine']
            else:
                errmsg = ('Could not retrieve dsim information from running config file in /etc/corr, '
                          'Perhaps, restart CBF manually and ensure dsim is running.\n'
                          'File:%s Line:%s' % (getframeinfo(currentframe()).filename.split('/')[-1],
                                               getframeinfo(currentframe()).lineno))
                LOGGER.error(errmsg)
                sys.exit(errmsg)
            try:
                dig_host = self.dsim_conf['host']
                self._dhost = FpgaDsimHost(dig_host, config=self.dsim_conf)
            except Exception:
                errmsg = 'Digitiser Simulator failed to retrieve information'
                LOGGER.exception(errmsg)
                sys.exit(errmsg)
            else:
                # Check if D-eng is running else start it.
                if self._dhost.is_running():
                    LOGGER.info('D-Eng is already running.')
                # Disabled DSim programming as it would alter the systems sync epoch
                elif program and not self._dhost.is_running():
                    LOGGER.info('Programming and starting the Digitiser Simulator.')
                    self._dhost.initialise()
                    self._dhost.enable_data_output(enabled=True)
                    self._dhost.registers.control.write(gbe_txen=True)
                else:
                    LOGGER.info('D-Eng started succesfully')
                return self._dhost

    @property
    def correlator(self):
        if self._correlator is not None:
            LOGGER.info('Using cached correlator instance')
            return self._correlator
        else:  # include a check, if correlator is running else start it.
            if not self._correlator_started:
                LOGGER.info('Correlator not running, now starting.')
                self.start_correlator()

            # We assume either start_correlator() above has been called, or the
            # instrument was started with the name contained in self.array_name
            # before running the test.

            self.config_filename = '/etc/corr/{}-{}'.format(self.array_name, self.instrument)
            if os.path.exists(self.config_filename):
                LOGGER.info('Making new correlator instance')
                try:
                    self._correlator = fxcorrelator.FxCorrelator(
                        'test correlator', config_source=self.config_filename)
                    self.correlator.initialise(program=False)
                    return self._correlator
                except Exception:
                    LOGGER.error('Failed to create new correlator instance, Will now try to '
                                 'start correlator with config: %s-%s' % (self.array_name,
                                                                          self.instrument))
                    self.start_correlator(instrument=self.instrument)
            else:
                LOGGER.error('No Config file (/etc/corr/array*-instrument), '
                             'Starting correlator with default instrument: %s' % (self.instrument))
                self.start_correlator(instrument=self.instrument)

    @property
    def halt_array(self):
        """
        Halting of primary and secondary katcp arrays and ensure that the correlator
        object is teared-down
        """
        LOGGER.info('Halting primary array: %s.' % self.array_name)
        self.katcp_rct.stop()
        self.rct.req.array_halt(self.array_name)
        self.rct.stop()
        self._rct = None
        self._katcp_rct = None
        # TODO: MM(2015-09-11) Proper teardown of corr object(katcp connections etc.)
        # Must still be implemented.
        self._correlator_started = False
        self._correlator = None
        LOGGER.info('Array %s halted and teared-down' % (self.array_name))

    @property
    def katcp_rct(self):
        try:
            katcp_prot = self.test_config['inst_param']['katcp_protocol']
        except TypeError:
            LOGGER.error('Failed to read katcp protocol from test config file')
        else:
            _major, _minor, _flags = katcp_prot.split(',')
            protocol_flags = ProtocolFlags(int(_major), int(_minor), _flags)
        multicast_ip = self.get_multicast_ips
        if not multicast_ip:
            LOGGER.error('Failed to calculate multicast IP\'s')
            self._katcp_rct = None
            return

        if self._katcp_rct is None:
            reply, informs = self.rct.req.array_list(self.array_name)
            # If no sub-array present create one, but this could cause problems
            # if more than one sub-array is present. Update this to check for
            # required sub-array.
            if reply.reply_ok():
                self.katcp_array_port = int(informs[0].arguments[1])
            else:
                LOGGER.info('Array has not been assigned yet, will try to assign.'
                            ' File:%s Line:%s' % (
                                getframeinfo(currentframe()).filename.split('/')[-1],
                                getframeinfo(currentframe()).lineno))
                try:
                    reply, _informs = self.rct.req.array_assign(self.array_name,
                                                                *multicast_ip)
                    assert reply.reply_ok()
                except (ValueError, TypeError, AssertionError):
                    LOGGER.exception('Failed to assign multicast ip on array: %s' % (self.array_name))
                else:
                    if len(reply.arguments) == 2:
                        try:
                            self.katcp_array_port = int(reply.arguments[-1])
                            LOGGER.info('Array %s assigned successfully' % (self.katcp_array_port))
                        except ValueError:
                            # self.rct.req.array_halt(self.array_name)
                            # self.rct.stop()
                            # self.rct.start()
                            # self.rct.until_synced(timeout=timeout)
                            # reply, informs = self.rct.req.array_assign(self.array_name,
                            # *multicast_ip)
                            errmsg = 'Investigate as to why this thing failed.'
                            LOGGER.exception(errmsg)
                            sys.exit(errmsg)

            katcp_rc = resource_client.KATCPClientResource(
                dict(name='{}'.format(self.katcp_clt),
                     address=('{}'.format(self.katcp_clt),
                              '{}'.format(self.katcp_array_port)),
                     preset_protocol_flags=protocol_flags, controlled=True))
            katcp_rc.set_ioloop(self.io_manager.get_ioloop())
            self._katcp_rct = (
                resource_client.ThreadSafeKATCPClientResourceWrapper(
                    katcp_rc, self.io_wrapper))
            self._katcp_rct.start()
            try:
                self._katcp_rct.until_synced(timeout=timeout)
            except TimeoutError:
                self._katcp_rct.stop()
            LOGGER.info('Cleanup function \'self._katcp_rct\': File: %s line: %s' % (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            # add_cleanup(self._katcp_rct.stop)
        else:
            self._katcp_rct.start()
            try:
                self._katcp_rct.until_synced(timeout=timeout)
            except TimeoutError:
                self._katcp_rct.stop()
        return self._katcp_rct

    @property
    def issue_metadata(self):
        """Issue Spead metadata"""
        try:
            reply, informs = self.katcp_rct.req.capture_meta(self.output_product, timeout=timeout)
            assert reply.reply_ok()
        except Exception:
            LOGGER.exception('Failed to issue new metadata: File:%s Line:%s' % (
                                            getframeinfo(currentframe()).filename.split('/')[-1],
                                            getframeinfo(currentframe()).lineno))
            return False
        else:
            return True

    @property
    def start_x_data(self):
        """
        Enable/Start output product capture
        """
        LOGGER.info('Start X data capture')
        try:
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply = self.katcp_rct.req.capture_list(timeout=timeout)
            assert reply.succeeded
        except IndexError:
            LOGGER.error('Config file does not contain Xengine output products.:'
                         ': File:%s Line:%s' % (getframeinfo(currentframe()).filename.split('/')[-1],
                                                getframeinfo(currentframe()).lineno))
            return False
        except (AttributeError, AssertionError):
            LOGGER.error('KATCP recourse client might not have any attributes: \nFile:%s Line:%s'
                         % (getframeinfo(currentframe()).filename.split('/')[-1],
                            getframeinfo(currentframe()).lineno))
            return False
        else:
            try:
                self.output_product = reply.informs[0].arguments[0]
            except IndexError:
                self.output_product = (
                    self.correlator.configd['xengine']['output_products'][0])

                LOGGER.error('KATCP reply does not contain a capture list: '
                             '\nFile:%s Line:%s'
                             % (getframeinfo(currentframe()).filename.split('/')[-1],
                                getframeinfo(currentframe()).lineno))

        try:
            reply = self.katcp_rct.req.capture_start(self.output_product)
            assert reply.succeeded
        except Exception, e:
            LOGGER.exception('Failed to capture start: %s due to %s' %str(reply, str(e)))
            return False
        else:
            LOGGER.info('Capture started on %s product' % self.output_product)
            return True

    def stop_x_data(self):
        """
        Disable/Stop output product capture
        """
        try:
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply = self.katcp_rct.req.capture_stop(self.output_product, timeout=timeout)
            assert reply.succeeded
        except IndexError:
            LOGGER.error('Failed to capture stop, might be because config file does not contain '
                         'Xengine output products.\n: File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
            return False
        except (AttributeError, AssertionError):
            LOGGER.error('KATCP recourse client might not have any attributes: \nFile:%s Line:%s' % (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            return False
        else:
            LOGGER.info('Stop X data capture')
            return True

    def deprogram_fpgas(self, instrument):
        """
        Deprogram CASPER devices listed on config file
        :param: object
        :param: str instrument
        """
        hostclass = katcp_fpga.KatcpFpga

        def get_hosts(self):
            LOGGER.info('Deprogram CASPER devices listed on config file.')
            try:
                get_running_instrument = self.get_running_instrument()
                assert get_running_instrument.values()[0]
                _running_instrument = get_running_instrument.keys()[0]
                LOGGER.info('Retrieved instrument %s from CAM Sensors' %_running_instrument)
            except AssertionError:
                _running_instrument = self.instrument
                LOGGER.info('Instrument %s to deprogram' %_running_instrument)

            try:
                assert isinstance (self.corr_config, dict)
                fengines = self.corr_config['fengine']['hosts'].split(',')
                xengines = self.corr_config['xengine']['hosts'].split(',')
                hosts = fengines + xengines
                return hosts
            except AssertionError:
                if len(_running_instrument) > 4:
                    config_file = '/etc/corr/{}-{}'.format(self.array_name,
                                                           _running_instrument)
                else:
                    config_file = '/etc/corr/{}-{}'.format(self.array_name,
                                                           self.instrument)
                if os.path.exists(config_file) or os.path.isfile(config_file):
                    LOGGER.info('Retrieving running hosts from running config: %s' % config_file)
                    fhosts = corr2.utils.parse_hosts(config_file, section='fengine')
                    xhosts = corr2.utils.parse_hosts(config_file, section='xengine')
                    hosts = fhosts + xhosts
                    return hosts
                else:
                    LOGGER.info('Could not get instrument from sensors and config file does '
                             'not exist: %s, Resorting to plan B - retrieving roach list from'
                             ' CORR2INI, In order to deprogram: File:%s Line:%s' % (config_file,
                                 getframeinfo(currentframe()).filename.split('/')[-1],
                                 getframeinfo(currentframe()).lineno))
                    with ignored(Exception):
                        if self.corr2ini_path is not None:
                            fhosts = corr2.utils.parse_hosts(self.corr2ini_path, section='fengine')
                            xhosts = corr2.utils.parse_hosts(self.corr2ini_path, section='xengine')
                            hosts = fhosts + xhosts
                            return hosts
                        else:
                            LOGGER.error('Failed to retrieve hosts from CORR2INI \n\t '
                                         'File:%s Line:%s' % (
                                             getframeinfo(currentframe()).filename.split('/')[-1],
                                             getframeinfo(currentframe()).lineno))


        try:
            hosts = list(set(get_hosts(self)))
        except TypeError:
            LOGGER.exception('Failed to deprogram all hosts, might be they\'ve been deprogrammed.')
            return

        if hosts:
            try:
                with ignored(Exception):
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(hostclass, hosts,
                                                                             timeout=timeout)
                    hosts = [host.host for host in connected_fpgas if host.ping() is True]
                    LOGGER.info('Confirm that all hosts are up and running: %s' %hosts)
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(hostclass, hosts)
                    fpgautils.threaded_fpga_function(connected_fpgas, 120, 'deprogram')
                LOGGER.info('%s Deprogrammed successfully.' % hosts)
                return True
            except Exception:
                errmsg = 'Failed to connect to roaches and deprogram, reboot devices to fix.'
                LOGGER.exception(errmsg)
                return False
        else:
            LOGGER.error('Failed to deprogram FPGAs no hosts available'
                         '\n\t File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
            return False

    def get_running_instrument(self):
        """
        Returns currently running instrument listed on the sensor(s)
        """
        try:
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
            if reply.istatus:
                return {reply.value: True}
        except AttributeError:
            LOGGER.exception('KATCP Request does not contain attributes '
                         '\n\t File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
            return False

        except KATCPSensorError:
            LOGGER.exception('KATCP Error polling sensor\n\t File:%s Line:%s' % (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            return False
        else:
            LOGGER.error('Sensor request failed: %s \n\t File:%s Line:%s' % (reply,
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            return {reply.value: False}

    def ensure_instrument(self, instrument, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required
        :param instrument: Correlator object

        """
        self.instrument = instrument
        success = False
        retries = 5
        while retries and not success:
            retries -= 1
            check_ins = self.check_instrument(self.instrument)
            if check_ins is True:
                success = True
                LOGGER.info('Return true if named instrument is enabled on correlator array after '
                        '%s retries' %retries)
                return success

        if self.check_instrument(self.instrument) is False:
            LOGGER.info('Correlator not running requested instrument, will restart.')
            deprogram_status = self.deprogram_fpgas(self.instrument)
            if not deprogram_status:
                LOGGER.info('Could not deprogram the hosts')
            corr_success = self.start_correlator(self.instrument, **kwargs)
            if corr_success is True:
                return True
            else:
                return False

    def check_instrument(self, instrument):
        """Return true if named instrument is enabled on correlator array

        Uses the correlator array KATCP interface to check if the requested
        instrument is active
        :param instrument: Correlator

        """
        try:
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
        except AssertionError:
            # This probably means that no array has been defined yet and therefore the
            # katcp_rct client cannot be created. IOW, the desired instrument would
            # not be available
            LOGGER.error('katcp rct has no attribute or no correlator '
                         'instance is running.\n\t File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
            return False
        else:
            if self.katcp_rct.state == 'synced':
                try:
                    reply = self.katcp_rct.req.instrument_list(instrument, timeout=timeout)
                except Exception:
                    LOGGER.error('Array request failed might have timedout\n\tFile:%s Line:%s' % (
                        getframeinfo(currentframe()).filename.split('/')[-1],
                        getframeinfo(currentframe()).lineno))
                else:
                    if not reply.succeeded:
                        LOGGER.error('Array request failed: %s\n\tFile:%s Line:%s' % (
                            str(reply), getframeinfo(currentframe()).filename.split('/')[-1],
                            getframeinfo(currentframe()).lineno))
                        return False
            else:
                return False

            instruments_available = [instrument_avail.arguments[0]
                                     for instrument_avail in reply.informs]
            # Test to see if requested instrument is available on the instrument list
            if instrument not in instruments_available:
                LOGGER.error('Instrument: %s is not in instrument list: %s'
                             '\n\t File:%s Line:%s' % (instrument, instruments_available,
                               getframeinfo(currentframe()).filename.split('/')[-1],
                               getframeinfo(currentframe()).lineno))
                return False

            # Get currently running instrument listed on the sensor(s)
            try:
                reply = self.katcp_rct.sensor.instrument_state.get_reading()
            except AttributeError:
                LOGGER.error('Instrument state could not be retrieved from the '
                             'sensors\n\t File:%s Line:%s' % (
                                 getframeinfo(currentframe()).filename.split('/')[-1],
                                 getframeinfo(currentframe()).lineno))
                return False
            else:
                if not reply.istatus:
                    LOGGER.error('Sensor request failed: %s \n\t File:%s Line:%s' % (
                        reply, getframeinfo(currentframe()).filename.split('/')[-1],
                        getframeinfo(currentframe()).lineno))
                    return False
                running_intrument = reply.value

                instrument_present = instrument == running_intrument
                if instrument_present:
                    self.instrument = instrument
                    LOGGER.info('Confirmed that the named instrument %s is enabled on '
                                'correlator %s.' % (self.instrument, self.array_name))
                return instrument_present

    @property
    def _get_instrument(self):
        """
        Retrieve currently running instrument from /etc/corr
        return: List
        """
        try:
            running_instr = max(glob.iglob('/etc/corr/*'), key=os.path.getctime).split('/')[-1]
            array, instrument = running_instr.split('-')
            if (instrument.startswith('bc') or instrument.startswith('c')) and array.startswith('array'):
                LOGGER.info('Currenly running instrument %s as per /etc/corr' %running_instr)
                return running_instr.split('-')
        except Exception:
            LOGGER.exception('Could not retrieve information from config file, resorting to default:\n'
                         'File:%s Line:%s' % (getframeinfo(currentframe()).filename.split('/')[-1],
                            getframeinfo(currentframe()).lineno))
            return ['array0', 'bc8n856M4k']

    @property
    def _test_config_file(self):
        """
        Configuration file containing information such as dsim, pdu and dataswitch ip's
        return: Dict
        """
        path, _none = os.path.split(__file__)
        path, _none = os.path.split(path)
        try:
            assert os.uname()[1].startswith('dbelab')
        except AssertionError:
            conf_path = '/config/test_conf_site.ini'
        else:
            conf_path = '/config/test_conf.ini'

        config_file = path + conf_path
        if os.path.isfile(config_file) or os.path.exists(config_file):
            try:
                config = corr2.utils.parse_ini_file(config_file)
                return config
            except (IOError, ValueError, TypeError):
                errmsg = ('Failed to read test config file %s, Test will exit'
                          '\n\t File:%s Line:%s' % (config_file,
                              getframeinfo(currentframe()).filename.split('/')[-1],
                              getframeinfo(currentframe()).lineno))
                LOGGER.exception(errmsg)
                return False
        else:
            LOGGER.error('Test config path: %s does not exist' % config_file)
            return False

    @property
    def get_multicast_ips(self):
        """
        Retrieves multicast ips from test configuration file and calculates the number
        of inputs depending on which instrument is being initialised
        :param instrument: Correlator
        """
        if self.instrument is None:
            return False
        try:
            multicast_ip_inp = (
                self.test_config['inst_param']['source_mcast_ips'].split(','))
        except TypeError:
            msg = ('Could not read and split the multicast ip\'s in the test config file')
            LOGGER.exception(msg)
            return False
        else:
            if self.instrument.startswith('bc') or self.instrument.startswith('c'):
                if self.instrument[0] == 'b':
                    try:
                        # multicast_ip = multicast_ip_inp * (int(self.instrument[2]) / 2)
                        multicast_ip = multicast_ip_inp * (
                            int(self.instrument.replace('bc', '').split('n')[0]) / 2)
                        return multicast_ip
                    except Exception:
                        LOGGER.error('Could not calculate multicast ips from config file')
                        return False
                else:
                    try:
                        # multicast_ip = multicast_ip_inp * (int(self.instrument[1]) / 2)
                        multicast_ip = multicast_ip_inp * (
                            int(self.instrument.replace('c', '').split('n')[0]) / 2)
                        return multicast_ip
                    except Exception:
                        LOGGER.error('Could not calculate multicast ips from config file')
                        return False

    @property
    def subscribe_multicast(self):
        """Automated multicasting subscription"""
        if self.config_filename is None:
            return

        config = self.corr_config
        if config is None:
            LOGGER.error('Failed to retrieve correlator config file, ensure that the cbf is running')
            return False
        outputIPs = {
            'beam0_ip': [tengbe.IpAddress(config['beam0']['data_ip']), int(
                                                    config['beam0']['data_port'])],
            'beam1_ip' : [tengbe.IpAddress(config['beam1']['data_ip']), int(
                                                    config['beam1']['data_port'])],
            'xengine_ip' : [tengbe.IpAddress(config['xengine']['output_destination_ip']), int(
                                                    config['xengine']['output_destination_port'])]
            }

        for prodct, data_output in outputIPs.iteritems():
            multicastIP, DataPort = data_output
            if multicastIP.is_multicast():
                LOGGER.info('%s: source is multicast %s.' % (prodct, multicastIP))
                # look up multicast group address in name server and find out IP version
                addrinfo = socket.getaddrinfo(str(multicastIP), None)[0]
                # create a socket
                try:
                    mcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                except Exception:
                    mcast_sock = socket.socket(addrinfo[0], socket.SOCK_DGRAM)
                mcast_sock.setblocking(False)
                # Do not bind as this will cause a conflict when instantiating a receiver on the main script
                # Join group
                # mcast_sock.bind(('', DataPort))
                add_cleanup(mcast_sock.close)
                group_bin = socket.inet_pton(addrinfo[0], addrinfo[4][0])
                if addrinfo[0] == socket.AF_INET:  # IPv4
                    mreq = group_bin + struct.pack('=I', socket.INADDR_ANY)
                    mcast_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                    LOGGER.info('Successfully subscribed to %s:%s.' % (str(multicastIP), DataPort))
                else:
                    mreq = group_bin + struct.pack('@I', 0)
                    mcast_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
                    LOGGER.info('Successfully subscribed to %s:%s.' % (str(multicastIP), DataPort))
            else:
                mcast_sock = None
                LOGGER.info('%s :Source is not multicast: %s:%s' % (prodct, str(multicastIP), DataPort))
                return False
        return True

    def start_correlator(self, instrument=None, retries=10):
        LOGGER.info('Will now try to start the correlator')
        success = False
        self.katcp_array_port = None
        retries_requested = retries
        if instrument is not None:
            self.instrument = instrument
        self._correlator = None  # Invalidate cached correlator instance
        LOGGER.info('Confirm DEngine is running before starting correlator')
        if not self.dhost.is_running():
            raise RuntimeError('DEngine: %s not running.' % (self.dhost.host))

        multicast_ip = self.get_multicast_ips
        if not multicast_ip:
            LOGGER.error('Failed to calculate multicast IP\'s, investigate')
            self._katcp_rct = None
        self.rct.start()
        try:
            self.rct.until_synced(timeout=timeout)
            reply, informs = self.rct.req.array_list(self.array_name)
            assert reply.reply_ok()
        except TimeoutError:
            self.rct.stop()
            LOGGER.exception('Resource client timed-out after %s s' % timeout)
            return False
        except AssertionError:
            LOGGER.exception('Failed to get array list,'
                             '\n\t File:%s Line:%s' % (
                                                   getframeinfo(currentframe()).filename.split('/')[-1],
                                                   getframeinfo(currentframe()).lineno))
        else:
            try:
                informs = informs[0]
                self.katcp_array_port = int(
                    [i for i in informs.arguments if len(i) == 5][0])
            except ValueError:
                LOGGER.exception('Failed to assign katcp port: Reply: %s' % (reply))
                reply = self.rct.req.array_halt(self.array_name)
                if not reply.succeeded:
                    LOGGER.error('Unable to halt array %s: %s \n\t File:%s Line:%s' % (
                        self.array_name, reply,
                        getframeinfo(currentframe()).filename.split('/')[-1],
                        getframeinfo(currentframe()).lineno))
                    return False

        while retries and not success:
            try:
                if self.katcp_array_port is None:
                    LOGGER.info('Assigning array port number')
                    # self.rct.start()
                    # self.rct.until_synced(timeout=timeout)
                    try:
                        reply, _informs = self.rct.req.array_assign(self.array_name,
                                                                    *multicast_ip, timeout=timeout)
                        assert reply.reply_ok()
                    except Exception:
                        self.katcp_array_port = None
                        LOGGER.exception('Failed to assign new array: %s' %self.array_name)
                    else:
                        self.katcp_array_port = int(reply.arguments[-1])
                        LOGGER.info('Successfully created %s-%s' % (self.array_name, self.katcp_array_port))

                # try:
                #     #self.katcp_array_port = int(reply.arguments[-1])
                #     LOGGER.info('Array port assigned: {}'.format(self.katcp_array_port))
                # except ValueError:
                #     LOGGER.fatal('Failed to assign array port number on {}'.format(
                #           self.array_name))
                # else:
                #     if not reply.reply_ok():
                #         LOGGER.fatal('Failed to assign array port number on {}'.format(
                #                self.array_name))
                #         return False

                instrument_param = (
                    [int(i) for i in self.test_config['inst_param']['instrument_param']
                     if i != ','])
                LOGGER.info('Starting %s with %s parameters. Try #%s' % (self.instrument,
                                                                         instrument_param,
                                                                         retries))

                reply = self.katcp_rct.req.instrument_activate(self.instrument,
                                                               *instrument_param,
                                                               timeout=500)
                success = reply.succeeded
                retries -= 1

                try:
                    assert success
                    LOGGER.info('Instrument %s started succesfully' % (self.instrument))
                except AssertionError:
                    LOGGER.warn('Failed to start correlator, %s attempts left. '
                                'Restarting Correlator. Reply:%s' % (retries, reply))
                    self.deprogram_fpgas(self.instrument)
                    self.halt_array
                    success = False
                    LOGGER.info('Katcp teardown and restarting correlator.')

            except Exception:
                try:
                    self.rct.req.array_halt(self.array_name)
                except Exception:
                    LOGGER.exception('Unable to halt array: Empty Array number: '
                                 'File:%s Line:%s' % (
                                     getframeinfo(currentframe()).filename.split('/')[-1],
                                     getframeinfo(currentframe()).lineno))
                try:
                    assert isinstance(self.katcp_rct,
                                      resource_client.ThreadSafeKATCPClientResourceWrapper)
                except AssertionError:
                    return False
                else:
                    try:
                        self.katcp_rct.stop()
                    except AttributeError:
                        LOGGER.error('KATCP request does not contain attributes: '
                                     'File:%s Line:%s' % (
                                         getframeinfo(currentframe()).filename.split('/')[-1],
                                         getframeinfo(currentframe()).lineno))
                        return False
                    else:
                        retries -= 1
                        LOGGER.warn('Failed to start correlator, %s attempts left.\n' % (retries))
            if retries < 0:
                success = False
                return success

        if success:
            self._correlator_started = True
            return self._correlator_started
        else:
            try:
                self.halt_array
                self._correlator_started = False
                self.katcp_rct.stop()
                self.rct.stop()
                self._katcp_rct = None
                self._correlator = None
            except:
                self.deprogram_fpgas(self.instrument)
                LOGGER.critical('Could not successfully start correlator within %s retries' % (
                    retries_requested))
                return False
            return False

correlator_fixture = CorrelatorFixture()
