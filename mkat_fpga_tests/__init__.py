import logging
import os
import socket
import struct
import subprocess
import sys
import time

from casperfpga import tengbe
from concurrent.futures import TimeoutError
from corr2 import fxcorrelator
from corr2.data_stream import StreamAddress
from corr2.dsimhost_fpga import FpgaDsimHost
from corr2.utils import parse_ini_file
from getpass import getuser as getusername
from glob import iglob
from katcp import ioloop_manager
from katcp import resource_client
from katcp.core import ProtocolFlags
from katcp.resource_client import KATCPSensorError
from nosekatreport import Aqf

# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
# from memory_profiler import profile as DetectMemLeaks

try:
    get_username = getusername()
except OSError:
    import pwd
    get_username = pwd.getpwuid(os.getuid()).pw_name

LOGGER = logging.getLogger(__name__)
Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - '
                              '%(pathname)s : %(lineno)d - %(message)s')
Handler = logging.FileHandler('/tmp/%s_by_%s.log' % (__name__, get_username))
Handler.setFormatter(Formatter)
LOGGER.addHandler(Handler)

# For Debugging
# logging.getLogger('katcp').setLevel('DEBUG')

_cleanups = []
"""Callables that will be called in reverse order at package teardown. Stored as a tuples of (callable,
args, kwargs)
"""

# Global katcp timeout
_timeout = 60

def add_cleanup(_fn, *args, **kwargs):
    LOGGER.info('Cleanup function: %s, %s, %s' %(_fn, args, kwargs))
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
        LOGGER.info('Cleanup in progress: %s' %_cleanups)
        _fn, args, kwargs = _cleanups.pop()
        try:
            _fn(*args, **kwargs)
        except:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):
    def __init__(self, katcp_clt=None, product_name=None):
        self.katcp_clt = katcp_clt
        self.corr_config = None
        self.corr2ini_path = None
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None
        self._rct = None
        self.katcp_array_port = None
        self.product_name = product_name
        self.halt_wait_time = 5
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
            add_cleanup(self.io_manager.stop)
            self.io_wrapper.default_timeout = _timeout
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
            add_cleanup(self._rct.stop)
            try:
                self._rct.until_synced(timeout=_timeout)
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
                self.corr_config = parse_ini_file(self.config_filename)
                self.dsim_conf = self.corr_config['dsimengine']
            elif self.instrument is not None:
                self.corr2ini_path = '/etc/corr/templates/{}'.format(self.instrument)
                LOGGER.info('Setting CORR2INI system environment to point to %s' % self.corr2ini_path)
                os.environ['CORR2INI'] = self.corr2ini_path
                self.corr_config = parse_ini_file(self.corr2ini_path)
                self.dsim_conf = self.corr_config['dsimengine']
            else:
                errmsg = ('Could not retrieve dsim information from running config file in /etc/corr, '
                          'Perhaps, restart CBF manually and ensure dsim is running.\n')
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
                    LOGGER.info('D-Eng started successfully')
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
            _retries = 3
            if os.path.exists(self.config_filename):
                LOGGER.info('Making new correlator instance')
                while True:
                    _retries -= 1
                    try:
                        self._correlator = fxcorrelator.FxCorrelator(
                            'test correlator', config_source=self.config_filename)
                        time.sleep(1)
                        try:
                            self.correlator.initialise(program=False, configure=False)
                        except TypeError:
                            self.correlator.initialise(program=False)
                        return self._correlator
                    except Exception as e:
                        LOGGER.exception('Failed to create new correlator instance with error: %s, '
                                         'Will now try to start correlator with config: %s-%s' % (
                                            str(e), self.array_name, self.instrument))
                        continue
                    if _retries == 0:
                        break
                if _retries == 0:
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
        try:
            reply, informs = self.katcp_rct.req.halt(timeout=_timeout)
            LOGGER.info(str(reply))
            assert reply.reply_ok()
            assert self._katcp_rct.is_active()
        except AssertionError:
            msg = 'Failed to halt katcp connection'
            LOGGER.error(msg)
        except AttributeError:
            raise RuntimeError('Failing to halt array, investigate halt array function.')

        self._katcp_rct.stop()
        self._katcp_rct = None

        try:
            reply, informs = self.rct.req.subordinate_list(timeout=_timeout)
            assert reply.reply_ok()
            if informs:
                informs = informs[0]
                if len(informs.arguments) >= 10 and self.array_name == informs.arguments[0]:
                    reply, informs = self.rct.req.subordinate_halt(self.array_name, timeout=_timeout)
                    assert reply.reply_ok()
        except AssertionError:
            msg = 'Failed to halt array: %s, STOPPING resource client' %self.array_name
            LOGGER.exception(msg)
        except IndexError:
            pass

        self._correlator_started = False
        self._correlator = None
        LOGGER.info('Array %s halted and teared-down' % (self.array_name))
        time.sleep(self.halt_wait_time)

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
            try:
                reply, informs = self.rct.req.subordinate_list(self.array_name)
            except TypeError:
                msg = 'Failed to list all arrays with name: %s' %self.array_name
                LOGGER.exception(msg)
            # If no sub-array present create one, but this could cause problems
            # if more than one sub-array is present. Update this to check for
            # required sub-array.
            if reply.reply_ok():
                self.katcp_array_port = int(informs[0].arguments[1])
            else:
                LOGGER.info('Array has not been assigned yet, will try to assign.')
                try:
                    reply, _informs = self.rct.req.subordinate_create(self.array_name, *multicast_ip)
                    assert reply.reply_ok()
                except (ValueError, TypeError, AssertionError):
                    try:
                        reply, informs = self.rct.req.subordinate_list()
                        assert reply.reply_ok()
                        informs = informs[0]
                        if len(informs.arguments) >= 10 and self.array_name == informs.arguments[0]:
                            msg = 'Array assigned successfully: %s' %str(informs)
                            LOGGER.info(msg)
                        else:
                            LOGGER.error('Halting array.')
                            reply, informs = self.rct.req.subordinate_halt(self.array_name,
                                timeout=_timeout)
                            assert reply.reply_ok()
                    except AssertionError:
                        LOGGER.exception('Failed to assign multicast ip on array: %s: \n\nReply: %s' % (
                            self.array_name, str(reply)))
                else:
                    if len(reply.arguments) == 2:
                        try:
                            self.katcp_array_port = int(reply.arguments[-1])
                            LOGGER.info('Array %s assigned successfully' % (self.katcp_array_port))
                        except ValueError:
                            errmsg = 'Investigate as to why this thing failed.'
                            LOGGER.exception(errmsg)
                            sys.exit(errmsg)

            katcp_rc = resource_client.KATCPClientResource(
                dict(name='{}'.format(self.katcp_clt),
                     address=('{}'.format(self.katcp_clt), '{}'.format(self.katcp_array_port)),
                     preset_protocol_flags=protocol_flags,
                     controlled=True))
            katcp_rc.set_ioloop(self.io_manager.get_ioloop())
            self._katcp_rct = (resource_client.ThreadSafeKATCPClientResourceWrapper(katcp_rc,
                self.io_wrapper))
            self._katcp_rct.start()
            try:
                self._katcp_rct.until_synced(timeout=_timeout)
            except Exception as e:
                self._katcp_rct.stop()
                LOGGER.exception('Failed to connect to katcp due to %s' %str(e))
        else:
            self._katcp_rct.start()
            try:
                time.sleep(1)
                self._katcp_rct.until_synced(timeout=_timeout)
            except Exception as e:
                self._katcp_rct.stop()
                LOGGER.exception('Failed to connect to katcp due to %s' %str(e))
        return self._katcp_rct

    @property
    def issue_metadata(self):
        """Issue Spead metadata"""
        try:
            reply, informs = self.katcp_rct.req.capture_meta(self.product_name, timeout=_timeout)
            assert reply.reply_ok()
        except Exception:
            LOGGER.exception('Failed to issue new metadata')
            return False
        else:
            return True

    @property
    def start_x_data(self):
        """
        Enable/Start output product capture
        """
        try:
            assert isinstance(self.katcp_rct, resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply, informs = self.katcp_rct.req.capture_list(timeout=_timeout)
            assert reply.reply_ok()
            self.product_name = [i.arguments[0] for i in informs
                                   if self.corr_config['xengine']['output_products'] in i.arguments][0]
            assert self.product_name is not None
            LOGGER.info('Capturing %s product' % self.product_name)
        except Exception:
            self.product_name = self.corr_config['xengine']['output_products']
            LOGGER.exception('Failed to retrieve capture list via CAM interface, '
                             'got it from config file.')


        try:
            reply, informs = self.katcp_rct.req.capture_start(self.product_name)
            assert reply.reply_ok()
            LOGGER.info(' %s' % str(reply))
            Aqf.progress(str(reply)+'\n')
            return True
        except Exception:
            LOGGER.exception('Failed to capture start: %s' %str(reply))
            return False

    def stop_x_data(self):
        """
        Disable/Stop output product capture
        """
        try:
            assert self.product_name is not None
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply, informs = self.katcp_rct.req.capture_stop(self.product_name, timeout=_timeout)
            assert reply.reply_ok()
            LOGGER.info(' %s' %str(reply))
            Aqf.progress(str(reply))
            return True
        except Exception:
            LOGGER.exception('Failed to capture stop, might be because config file does not contain '
                             'Xengine output products.')
            return False


    def get_running_instrument(self):
        """
        Returns currently running instrument listed on the sensor(s)
        """
        try:
            reply = None
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
            assert reply.istatus
            return reply.value
        except Exception as e:
            LOGGER.exception('KATCP Request failed due to error: %s/%s' % (str(e), str(reply)))
            return False

        except KATCPSensorError:
            LOGGER.exception('KATCP Error polling sensor\n')
            return False
        except AssertionError:
            LOGGER.exception('Sensor request failed: %s, Forcefully Halting the Array' % (str(reply)))
            return False

    def ensure_instrument(self, instrument, retries=5, force_reinit=False, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required
        :param self: Object
        :param instrument: CBF Instrument
        :param retries: No of instrument validation retries
        :param force_reinit: Force an instrument re-initialisation
        :rtype: Boolean
        """
        self.instrument = instrument
        if force_reinit:
            LOGGER.info('Forcing an instrument(%s) re-initialisation' %self.instrument)
            corr_success = self.start_correlator(self.instrument, **kwargs)
            return corr_success

        success = False
        while retries and not success:
            check_ins = self.check_instrument(self.instrument)
            msg = 'Will retry to check if the instrument is up: #%s retry.' % retries
            LOGGER.info(msg)
            if check_ins is True:
                success = True
                LOGGER.info('Named instrument (%s) is currently running' % instrument)
                return success
            retries -= 1

        if self.check_instrument(self.instrument) is False:
            LOGGER.info('Correlator not running requested instrument, will restart.')
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
            if reply.value == self.instrument:
                self.halt_array
            corr_success = self.start_correlator(self.instrument, **kwargs)
            return True if corr_success is True else False

    def check_instrument(self, instrument):
        """Return true if named instrument is enabled on correlator array

        Uses the correlator array KATCP interface to check if the requested
        instrument is active
        :param instrument: Correlator

        """
        self.instrument = instrument
        self._errmsg = None
        try:
            self._errmsg = 'Instrument cannot be None.'
            assert instrument is not None, self._errmsg
            self._errmsg = 'katcp client is not an instance of resource client'
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper), self._errmsg
            self._errmsg = 'katcp client failed to establish a connection'
            assert self.katcp_rct.wait_connected(), self._errmsg
            self._errmsg = 'katcp client failed to sync after establishing a connection'
            assert self.katcp_rct.until_state('synced', timeout=60), self._errmsg
        except AssertionError:
            # This probably means that no array has been defined yet and therefore the
            # katcp_rct client cannot be created. IOW, the desired instrument would
            # not be available
            LOGGER.exception(self._errmsg)
            return False
        else:
            try:
                reply, informs = self.katcp_rct.req.instrument_list()
                assert reply.reply_ok()
                instruments_available = [instrument_avail.arguments[0] for instrument_avail in informs]
                # Test to see if requested instrument is available on the instrument list
                assert instrument in instruments_available
            except Exception:
                LOGGER.exception('Array request failed might have timed-out')
            except AssertionError:
                LOGGER.exception('Array request failed: %s or Instrument: %s is not in instrument '
                                 'list: %s' % (instrument, instruments_available, str(reply)))
                return False

            try:
                reply = self.katcp_rct.sensor.instrument_state.get_reading()
                assert reply.istatus
            except AttributeError:
                LOGGER.exception('Instrument state could not be retrieved from the sensors')
                return False
            except AssertionError:
                LOGGER.error('%s: No running instrument' % str(reply))
                return False

            else:
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
            # ToDo (MM) 06-10-2017 Hardcoded array, fix it
            running_instr = max(iglob('/etc/corr/array0-*'), key=os.path.getctime).split('/')[-1]
            self.array_name, self.instrument = running_instr.split('-')
            if (self.instrument.startswith('bc') or self.instrument.startswith('c')) and \
                self.array_name.startswith('array'):
                LOGGER.info('Currently running instrument %s as per /etc/corr' % running_instr)
                return running_instr.split('-')
        except Exception:
            LOGGER.exception('Could not retrieve information from config file, resorting to default')
            return ['array0', 'bc8n856M4k']
        except ValueError:
            LOGGER.exception('Directory missing array config file.')

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
            conf_path = 'config/test_conf_site.ini'
        else:
            conf_path = 'config/test_conf_lab.ini'

        config_file = os.path.join(path, conf_path)
        if os.path.isfile(config_file) or os.path.exists(config_file):
            try:
                config = parse_ini_file(config_file)
                return config
            except (IOError, ValueError, TypeError):
                errmsg = ('Failed to read test config file %s, Test will exit' % (config_file))
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
            msg = ('Could not read and split the multicast IPs in the test config file')
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
                        LOGGER.error('Could not calculate multicast IPs from config file')
                        return False
                else:
                    try:
                        multicast_ip = multicast_ip_inp * (
                            int(self.instrument.replace('c', '').split('n')[0]) / 2)
                        return multicast_ip
                    except Exception:
                        LOGGER.error('Could not calculate multicast IPs from config file')
                        return False


    @property
    def subscribe_multicast(self):
        """Automated multicasting subscription"""
        parse_address = StreamAddress._parse_address_string
        try:
            n_xengs = self.katcp_rct.sensor.n_xengs.get_value()
        except Exception:
            n_xengs = len(self.get_multicast_ips) * 2

        if self.config_filename is None:
            return
        config = self.corr_config
        if config is None:
            LOGGER.error('Failed to retrieve correlator config file, ensure that the cbf is running')
            return False

        def confirm_multicast_subs(mul_ip='239.100.0.10'):
            """"""
            # or use [netstat -g | grep eth2]
            list_inets = subprocess.check_output(['ip', 'maddr', 'show'])
            return True if mul_ip in list_inets else False

        outputIPs = {}
        for i in [key for key, value in config.items() if 'output_destinations_base' in value]:
            _IP, _num, _Port = list(parse_address(config[i]['output_destinations_base']))
            outputIPs[i] = [tengbe.IpAddress(_IP), int(_Port)]
        if outputIPs.get('xengine'):
            LOGGER.info('Multicast subscription is only valid for xengines')
            multicastIP, DataPort = outputIPs.get('xengine')
            if multicastIP.is_multicast():
                LOGGER.info('source is multicast %s.' % (multicastIP))
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
                def join_mcast_group(address):
                    group_bin = socket.inet_pton(socket.AF_INET, address)
                    if addrinfo[0] == socket.AF_INET:  # IPv4
                        mreq = group_bin + struct.pack('=I', socket.INADDR_ANY)
                        mcast_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                        LOGGER.info('Successfully subscribed to %s:%s.' % (str(multicastIP), DataPort))
                    else:
                        mreq = group_bin + struct.pack('@I', 0)
                        mcast_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
                        LOGGER.info('Successfully subscribed to %s:%s.' % (str(multicastIP), DataPort))
                for addcntr in range(n_xengs):
                    _address = tengbe.IpAddress(multicastIP.ip_int + addcntr)
                    join_mcast_group(str(_address))
            else:
                mcast_sock = None
                LOGGER.info('Source is not multicast: %s:%s' % (str(multicastIP), DataPort))
                return False
        return confirm_multicast_subs(mul_ip=str(_address))

    def start_correlator(self, instrument=None, retries=10):
        LOGGER.debug('CBF instrument(%s) re-initialisation.' %instrument)
        success = False
        self.katcp_array_port = None
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
            self.rct.until_synced(timeout=_timeout)
            reply, informs = self.rct.req.subordinate_list(self.array_name)
            assert reply.reply_ok()
        except TimeoutError:
            self.rct.stop()
            LOGGER.exception('Resource client timed-out after %s s' % _timeout)
            return False
        except AssertionError:
            LOGGER.exception('Failed to get subordinate-list, might not have been assigned, '
                             'Will try to assign.')
        else:
            try:
                informs = informs[0]
                self.katcp_array_port = int(
                    [i for i in informs.arguments if len(i) == 5][0])
            except ValueError:
                LOGGER.exception('Failed to assign katcp port: Reply: %s' % (reply))
                reply = self.rct.req.subordinate_halt(self.array_name)
                if not reply.succeeded:
                    LOGGER.error('Unable to halt array %s: %s' % (self.array_name, reply))
                    return False

        while retries and not success:
            try:
                if self.katcp_array_port is None:
                    LOGGER.info('Assigning array port number')
                    try:
                        reply, _informs = self.rct.req.subordinate_create(self.array_name,
                                                                    *multicast_ip, timeout=_timeout)
                        assert reply.reply_ok()
                    except Exception:
                        self.katcp_array_port = None
                        LOGGER.exception('Failed to assign new array: %s' %self.array_name)
                    else:
                        self.katcp_array_port = int(reply.arguments[-1])
                        LOGGER.info('Successfully created %s-%s' % (self.array_name, self.katcp_array_port))

                instrument_param = (
                    [int(i) for i in self.test_config['inst_param']['instrument_param'] if i != ','])
                LOGGER.info('Starting %s with %s parameters. Try #%s' % (self.instrument,
                                                                         instrument_param,
                                                                         retries))

                # TODO add timeout
                reply = self.katcp_rct.req.instrument_activate(self.instrument, *instrument_param,
                                                               timeout=500)
                success = reply.succeeded
                retries -= 1

                try:
                    assert success
                    LOGGER.info('Instrument %s started successfully' % (self.instrument))
                except AssertionError:
                    LOGGER.exception('Failed to start correlator, %s attempts left. '
                                     'Restarting Correlator. Reply:%s' % (retries, reply))
                    self.halt_array
                    success = False


            except Exception:
                try:
                    self.rct.req.subordinate_halt(self.array_name)
                    assert isinstance(self.katcp_rct,
                                      resource_client.ThreadSafeKATCPClientResourceWrapper)
                except Exception:
                    LOGGER.exception('Unable to halt array: Empty Array number')
                except AssertionError:
                    LOGGER.exception('self.katcp_rct has not been initiated successfully')
                    return False
                else:
                    try:
                        self.katcp_rct.stop()
                    except AttributeError:
                        LOGGER.error('KATCP request does not contain attributes')
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
                self.halt_array
                msg = ('Could not successfully start correlator within %s retries' % (retries))
                LOGGER.critical(msg)
                return False
            return False

correlator_fixture = CorrelatorFixture()
