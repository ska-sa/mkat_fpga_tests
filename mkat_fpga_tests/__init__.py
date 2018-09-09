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
    LOGGER.info('Cleanup function: %s, %s, %s' % (_fn, args, kwargs))
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
        LOGGER.info('Cleanup in progress: %s' % _cleanups)
        _fn, args, kwargs = _cleanups.pop()
        try:
            _fn(*args, **kwargs)
        except:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):
    def __init__(self, katcp_client=None, product_name=None):
        self.katcp_client = katcp_client
        self.prim_port = '7147'
        self.corr_config = None
        self.corr2ini_path = None
        self._correlator = None
        self._dhost = None
        self._katcp_rct = None
        self._katcp_rct_sensor = None
        self._rct = None
        self.katcp_array_port = None
        self.katcp_sensor_port = None
        self.product_name = product_name
        self.halt_wait_time = 5
        # Assume the correlator is already started if start_correlator is False
        nose_test_config = {}
        self._correlator_started = not int(
            nose_test_config.get('start_correlator', False))
        self.test_config = self._test_config_file
        # ToDo get array name from file...instead of test config file
        self.subarray = self.test_config['instrument_params']['subarray']
        self.config_filename = max(
            iglob('/etc/corr/{}-*'.format(self.subarray)), key=os.path.getctime)
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
                dict(name='{}'.format(self.katcp_client),
                     address=('{}'.format(self.katcp_client),
                              self.prim_port),
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
            if os.path.exists(self.config_filename):
                LOGGER.info(
                    'Retrieving dsim engine info from config file: %s' % self.config_filename)
                self.corr_config = parse_ini_file(self.config_filename)
                self.dsim_conf = self.corr_config['dsimengine']
            elif self.instrument is not None:
                self.corr2ini_path = '/etc/corr/templates/{}'.format(
                    self.instrument)
                LOGGER.info(
                    'Setting CORR2INI system environment to point to %s' % self.corr2ini_path)
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
                self._dhost = None
                self._dhost = FpgaDsimHost(dig_host, config=self.dsim_conf)
            except Exception:
                errmsg = 'Digitiser Simulator failed to retrieve information'
                LOGGER.exception(errmsg)
                sys.exit(errmsg)
            else:
                # Check if D-eng is running else start it.
                if self._dhost.is_running():
                    LOGGER.info('D-Eng is already running.')
                    return self._dhost

    @property
    def correlator(self):
        if self._correlator is not None:
            LOGGER.info('Using cached correlator instance')
            return self._correlator
        else:  # include a check, if correlator is running else start it.
            if not self._correlator_started:
                LOGGER.info('Correlator not running, now starting.')
                print 'Correlator not running: This shouldnt happen, fix it\n'*10
                # self.start_correlator()

            # We assume either start_correlator() above has been called, or the
            # instrument was started with the name contained in self.array_name
            # before running the test.
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
                            self.correlator.initialise(
                                program=False, configure=False)
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
                # if _retries == 0:
                #     self.start_correlator(instrument=self.instrument)
            else:
                LOGGER.error('No Config file (/etc/corr/array*-instrument), '
                             'Starting correlator with default instrument: %s' % (self.instrument))
                # self.start_correlator(instrument=self.instrument)

    @property
    def katcp_rct(self):
        if self._katcp_rct is None:
            try:
                katcp_prot = self.test_config['instrument_params']['katcp_protocol']
                _major, _minor, _flags = katcp_prot.split(',')
                protocol_flags = ProtocolFlags(
                    int(_major), int(_minor), _flags)
                LOGGER.info('katcp protocol flags %s' % protocol_flags)

                LOGGER.info('Getting running array.')
                reply, informs = self.rct.req.subordinate_list(self.array_name)
                assert reply.reply_ok()
                # If no sub-array present create one, but this could cause problems
                # if more than one sub-array is present. Update this to check for
                # required sub-array.
            except Exception:
                LOGGER.exception(
                    'Failed to list all arrays with name: %s' % self.array_name)
            else:
                try:
                    try:
                        self.katcp_array_port = int(informs[0].arguments[1])
                        LOGGER.info('Current running array name: %s, port: %s' % (self.array_name,
                                                                                  self.katcp_array_port))
                    except ValueError:
                        self.katcp_array_port, self.katcp_sensor_port = informs[0].arguments[1].split(
                            ',')
                        LOGGER.info('Current running array name: %s, port: %s, sensor port: %s' % (
                            self.array_name, self.katcp_array_port, self.katcp_sensor_port))
                except Exception:
                    errmsg = (
                        'Failed to retrieve running array, ensure one has been created and running')
                    LOGGER.exception(errmsg)
                    sys.exit(errmsg)
                else:
                    katcp_rc = resource_client.KATCPClientResource(
                        dict(name='{}'.format(self.katcp_client),
                             address=('{}'.format(self.katcp_client),
                                      '{}'.format(self.katcp_array_port)),
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
                        LOGGER.exception(
                            'Failed to connect to katcp due to %s' % str(e))
                    else:
                        return self._katcp_rct
        else:
            if not self._katcp_rct.is_active():
                LOGGER.info(
                    'katcp resource client wasnt running, hence we need to start it.')
                self._katcp_rct.start()
                try:
                    time.sleep(1)
                    self._katcp_rct.until_synced(timeout=_timeout)
                    return self._katcp_rct
                except Exception:
                    self._katcp_rct.stop()
                    LOGGER.exception('Failed to connect to katcp')
            else:
                return self._katcp_rct

    @property
    def katcp_rct_sensor(self):
        if self._katcp_rct_sensor is None:
            try:
                katcp_prot = self.test_config['instrument_params']['katcp_protocol']
                _major, _minor, _flags = katcp_prot.split(',')
                protocol_flags = ProtocolFlags(
                    int(_major), int(_minor), _flags)
                LOGGER.info('katcp protocol flags %s' % protocol_flags)

                LOGGER.info('Getting running array.')
                reply, informs = self.rct.req.subordinate_list(self.array_name)
                assert reply.reply_ok()
                # If no sub-array present create one, but this could cause problems
                # if more than one sub-array is present. Update this to check for
                # required sub-array.
            except Exception:
                LOGGER.exception(
                    'Failed to list all arrays with name: %s' % self.array_name)
            else:
                try:
                    try:
                        self.katcp_array_port = int(informs[0].arguments[1])
                        LOGGER.info('Current running array name: %s, port: %s' % (self.array_name,
                                                                                  self.katcp_array_port))
                    except ValueError:
                        self.katcp_array_port, self.katcp_sensor_port = informs[0].arguments[1].split(
                            ',')
                        LOGGER.info('Current running array name: %s, port: %s, sensor port: %s' % (
                            self.array_name, self.katcp_array_port, self.katcp_sensor_port))
                except Exception:
                    errmsg = (
                        'Failed to retrieve running array, ensure one has been created and running')
                    LOGGER.exception(errmsg)
                    sys.exit(errmsg)
                else:
                    katcp_rc = resource_client.KATCPClientResource(
                        dict(name='{}'.format(self.katcp_client),
                             address=('{}'.format(self.katcp_client),
                                      '{}'.format(self.katcp_sensor_port)),
                             preset_protocol_flags=protocol_flags,
                             controlled=True))
                    katcp_rc.set_ioloop(self.io_manager.get_ioloop())
                    self._katcp_rct_sensor = (resource_client.ThreadSafeKATCPClientResourceWrapper(
                        katcp_rc, self.io_wrapper))
                    self._katcp_rct_sensor.start()
                    try:
                        self._katcp_rct_sensor.until_synced(timeout=_timeout)
                    except Exception as e:
                        self._katcp_rct_sensor.stop()
                        LOGGER.exception(
                            'Failed to connect to katcp due to %s' % str(e))
                    else:
                        return self._katcp_rct_sensor
        else:
            if not self._katcp_rct_sensor.is_active():
                LOGGER.info(
                    'katcp resource client wasnt running, hence we need to start it.')
                self._katcp_rct_sensor.start()
                try:
                    time.sleep(1)
                    self._katcp_rct_sensor.until_synced(timeout=_timeout)
                    return self._katcp_rct_sensor
                except Exception:
                    self._katcp_rct_sensor.stop()
                    LOGGER.exception('Failed to connect to katcp')
            else:
                return self._katcp_rct_sensor

    @property
    def start_x_data(self):
        """
        Enable/Start output product capture
        """
        try:
            assert isinstance(
                self.katcp_rct, resource_client.ThreadSafeKATCPClientResourceWrapper)
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
            reply, informs = self.katcp_rct.req.capture_start(
                self.product_name)
            assert reply.reply_ok()
            LOGGER.info(' %s' % str(reply))
            Aqf.progress(str(reply)+'\n')
            return True
        except Exception:
            LOGGER.exception('Failed to capture start: %s' % str(reply))
            return False

    def stop_x_data(self):
        """
        Disable/Stop output product capture
        """
        try:
            assert self.product_name is not None
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply, informs = self.katcp_rct.req.capture_stop(
                self.product_name, timeout=_timeout)
            assert reply.reply_ok()
            LOGGER.info(' %s' % str(reply))
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
            LOGGER.exception(
                'KATCP Request failed due to error: %s/%s' % (str(e), str(reply)))
            return False

        except KATCPSensorError:
            LOGGER.exception('KATCP Error polling sensor\n')
            return False
        except AssertionError:
            LOGGER.exception(
                'Sensor request failed: %s, Forcefully Halting the Array' % (str(reply)))
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
        try:
            assert '_' in self.instrument
            self.instrument = self.instrument.split('_')[0]
        except AssertionError:
            pass

        if force_reinit:
            LOGGER.info('Forcing an instrument(%s) re-initialisation' %
                        self.instrument)
            corr_success = self.start_correlator(self.instrument, **kwargs)
            return corr_success

        success = False
        while retries and not success:
            check_ins = self.check_instrument(self.instrument)
            msg = 'Will retry to check if the instrument is up: #%s retry.' % retries
            LOGGER.info(msg)
            if check_ins is True:
                success = True
                LOGGER.info(
                    'Named instrument (%s) is currently running' % self.instrument)
                return success
            retries -= 1

        if self.check_instrument(self.instrument) is False:
            LOGGER.info(
                'Correlator not running requested instrument, will restart.')
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
            if reply.value == self.instrument:
                pass
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
            assert self.katcp_rct.until_state(
                'synced', timeout=60), self._errmsg
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
                instruments_available = [instrument_avail.arguments[0].split('_')[0]
                                         for instrument_avail in informs]
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
                LOGGER.exception(
                    'Instrument state could not be retrieved from the sensors')
                return False
            except AssertionError:
                LOGGER.error('%s: No running instrument' % str(reply))
                return False

            else:
                running_intrument = reply.value.split('_')[0]
                instrument_present = instrument == running_intrument
                if instrument_present:
                    self.instrument = instrument
                    LOGGER.info(
                        'Confirmed that the named instrument %s is running' % self.instrument)
                return instrument_present

    @property
    def _get_instrument(self):
        """
        Retrieve currently running instrument from /etc/corr
        return: List
        """
        try:
            running_instr = self.config_filename.split('/')[-1]
            self.array_name, self.instrument = running_instr.split('-')
            try:
                assert '_' in self.instrument
                self.instrument = self.instrument.split('_')[0]
            except AssertionError:
                pass

            if (self.instrument.startswith('bc') or self.instrument.startswith('c')) and \
                    self.array_name.startswith('array'):
                LOGGER.info(
                    'Currently running instrument %s as per /etc/corr' % self.instrument)
                return [self.array_name, self.instrument]
        except Exception:
            LOGGER.exception(
                'Could not retrieve information from config file, resorting to default')
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
                errmsg = (
                    'Failed to read test config file %s, Test will exit' % (config_file))
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
            multicast_ip_inp = [self.corr_config['dsimengine'].get('pol0_destination_start_ip',
                                                                   '239.101.0.64'),
                                self.corr_config['dsimengine'].get('pol1_destination_start_ip',
                                                                   '239.101.0.66')
                                ]

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
                        LOGGER.error(
                            'Could not calculate multicast IPs from config file')
                        return False
                else:
                    try:
                        multicast_ip = multicast_ip_inp * (
                            int(self.instrument.replace('c', '').split('n')[0]) / 2)
                        return multicast_ip
                    except Exception:
                        LOGGER.error(
                            'Could not calculate multicast IPs from config file')
                        return False

    def start_correlator(self, instrument=None, retries=10):
        LOGGER.debug('CBF instrument(%s) re-initialisation.' % instrument)
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
                LOGGER.exception(
                    'Failed to assign katcp port: Reply: %s' % (reply))
                reply = self.rct.req.subordinate_halt(self.array_name)
                if not reply.succeeded:
                    LOGGER.error('Unable to halt array %s: %s' %
                                 (self.array_name, reply))
                    return False

        while retries and not success:
            try:
                if self.katcp_array_port is None:
                    LOGGER.info('Assigning array port number')
                    try:
                        pass
                        # reply, _informs = self.rct.req.subordinate_create(self.array_name,
                        #                                             *multicast_ip, timeout=_timeout)
                        # assert reply.reply_ok()
                    except Exception:
                        self.katcp_array_port = None
                        LOGGER.exception(
                            'Failed to assign new array: %s' % self.array_name)
                    else:
                        self.katcp_array_port = int(reply.arguments[-1])
                        LOGGER.info('Successfully created %s-%s' %
                                    (self.array_name, self.katcp_array_port))

                instrument_param = (
                    [int(i) for i in self.test_config['instrument_params']['instrument_param'] if i != ','])
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
                    LOGGER.info('Instrument %s started successfully' %
                                (self.instrument))
                except AssertionError:
                    LOGGER.exception('Failed to start correlator, %s attempts left. '
                                     'Restarting Correlator. Reply:%s' % (retries, reply))
                    success = False

            except Exception:
                try:
                    self.rct.req.subordinate_halt(self.array_name)
                    assert isinstance(self.katcp_rct,
                                      resource_client.ThreadSafeKATCPClientResourceWrapper)
                except Exception:
                    LOGGER.exception(
                        'Unable to halt array: Empty Array number')
                except AssertionError:
                    LOGGER.exception(
                        'self.katcp_rct has not been initiated successfully')
                    return False
                else:
                    try:
                        self.katcp_rct.stop()
                    except AttributeError:
                        LOGGER.error(
                            'KATCP request does not contain attributes')
                        return False
                    else:
                        retries -= 1
                        LOGGER.warn(
                            'Failed to start correlator, %s attempts left.\n' % (retries))

            if retries < 0:
                success = False
                return success

        if success:
            self._correlator_started = True
            return self._correlator_started
        else:
            try:
                self._correlator_started = False
                self.katcp_rct.stop()
                self.rct.stop()
                self._katcp_rct = None
                self._correlator = None
            except:
                msg = (
                    'Could not successfully start correlator within %s retries' % (retries))
                LOGGER.critical(msg)
                return False
            return False


correlator_fixture = CorrelatorFixture()
