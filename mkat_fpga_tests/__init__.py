import logging
import os
from inspect import currentframe, getframeinfo

import corr2
from casperfpga import katcp_fpga
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
from testconfig import config as nose_test_config

LOGGER = logging.getLogger(__name__)

_cleanups = []
"""Callables that will be called in reverse order at package teardown

Stored as a tuples of (callable, args, kwargs)
"""
timeout = 60

# Set katcp.inspect_lient logger to only report error messages
katcp_logger = logging.getLogger('katcp.inspect_client')
katcp_logger.setLevel(logging.ERROR)


def add_cleanup(_fn, *args, **kwargs):
    _cleanups.append((_fn, args, kwargs))


def teardown_package():
    while _cleanups:
        _fn, args, kwargs = _cleanups.pop()
        try:
            _fn(*args, **kwargs)
        except:
            LOGGER.exception('Exception calling cleanup fn')


class CorrelatorFixture(object):
    def __init__(self, array=None, instrument=None, resource_clt=None):

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

    @property
    def rct(self):
        if self._rct is not None:
            return self._rct
        else:
            self.io_manager = ioloop_manager.IOLoopManager()
            self.io_wrapper = resource_client.IOLoopThreadWrapper(
                self.io_manager.get_ioloop())
            LOGGER.info('Cleanup function: File: %s line: %s' % (
                        getframeinfo(currentframe()).filename.split('/')[-1],
                        getframeinfo(currentframe()).lineno))
            add_cleanup(self.io_manager.stop)
            self.io_wrapper.default_timeout = timeout
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
            LOGGER.info('Cleanup function: File: %s line: %s' % (
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
                self.dsim_conf = corr2.utils.parse_ini_file(
                    self.config_filename)['dsimengine']
                dig_host = self.dsim_conf['host']
            else:
                LOGGER.error('Could not retrieve information from config file, '
                             'resorting to test_conf.ini: File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
                self.dsim_conf = self._test_config_file['dsimengine']
                dig_host = self.dsim_conf['host']

            self._dhost = FpgaDsimHost(dig_host, config=self.dsim_conf)
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
                    LOGGER.error(
                        'Failed to create new correlator instance, Will now try to '
                        'start correlator with config: %s-%s' % (self.array_name, self.instrument))
                    self.start_correlator(instrument=self.instrument)
            else:
                LOGGER.error('No Config file (/etc/corr/array*-instrument), '
                             'Starting correlator with default instrument: %s' % (self.instrument))
                self.start_correlator(instrument=self.instrument)

    def halt_array(self):
        """
        Halting of primary and secondary katcp arrays and ensure that the correlator
        object is teared-down
        """
        # if not self._correlator:
        # raise RuntimeError('Array not yet initialised')
        LOGGER.info('Halting primary array.')
        self.katcp_rct.stop()
        self.rct.req.array_halt(self.array_name)
        self.rct.stop()
        self._rct = None
        self._katcp_rct = None
        # TODO: MM(2015-09-11) Proper teardown of corr object(katcp connections etc.)
        # Must still be implemented.
        self._correlator_started = False
        self._correlator = None
        LOGGER.info('Array %s halted and teared-down' % ( self.array_name))

    @property
    def katcp_rct(self):
        try:
            katcp_prot = self._test_config_file['inst_param']['katcp_protocol']
        except TypeError:
            LOGGER.error('Failed to read katcp protocol from test config file')
        else:
            _major, _minor, _flags = katcp_prot.split(',')
            protocol_flags = ProtocolFlags(int(_major), int(_minor), _flags)
        multicast_ip = self.get_multicast_ips()
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
                except (ValueError, TypeError):
                    LOGGER.exception('Failed to assign multicast ip on array: %s' % (self.array_name))
                else:
                    if len(reply.arguments) == 2:
                        try:
                            self.katcp_array_port = int(reply.arguments[-1])
                            LOGGER.info('Array %s assigned successfully' % (self.katcp_array_port))
                        except ValueError:
                            LOGGER.exception('Array assign failed: %s' % (reply))
                            # self.rct.req.array_halt(self.array_name)
                            # self.rct.stop()
                            # self.rct.start()
                            # self.rct.until_synced(timeout=timeout)
                            # reply, informs = self.rct.req.array_assign(self.array_name,
                            # *multicast_ip)

            katcp_rc = resource_client.KATCPClientResource(
                dict(name='{}'.format(self.resource_clt),
                     address=('{}'.format(self.resource_clt),
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
            LOGGER.info('Cleanup function: File: %s line: %s' % (
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

    def start_x_data(self):
        """
        Enable/Start output product capture
        """
        LOGGER.info('Start X data capture')
        try:
            assert isinstance(self.katcp_rct,
                              resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply = self.katcp_rct.req.capture_list()
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
            if reply.succeeded:
                try:
                    self.output_product = reply.informs[0].arguments[0]
                except IndexError:
                    LOGGER.error('KATCP reply does not contain a capture list: '
                                 '\nFile:%s Line:%s'
                                    % (getframeinfo(currentframe()).filename.split('/')[-1],
                                       getframeinfo(currentframe()).lineno))
                    return False
            else:
                self.output_product = (
                    self.correlator.configd['xengine']['output_products'][0])
        try:
            reply = self.katcp_rct.req.capture_start(self.output_product)
        except:
            LOGGER.exception('Failed to capture start')
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
        LOGGER.info('Stop X data capture')
        try:
            assert isinstance(self.katcp_rct,
                  resource_client.ThreadSafeKATCPClientResourceWrapper)
            reply = self.katcp_rct.req.capture_stop(self.output_product)
        except IndexError:
            LOGGER.error('Failed to capture stop, might be because config file does not contain '
                         'Xengine output products.\n: File:%s Line:%s' % (
                            getframeinfo(currentframe()).filename.split('/')[-1],
                            getframeinfo(currentframe()).lineno))
            return False
        except (AttributeError, AssertionError):
            LOGGER.error('KATCP recourse client might not have any attributes: \nFile:%s Line:%s'% (
                getframeinfo(currentframe()).filename.split('/')[-1],
                getframeinfo(currentframe()).lineno))
            return False
        else:
            if not reply.succeeded:
                return False
            else:
                return True

    def deprogram_fpgas(self, instrument):
        """
        Deprogram CASPER devices listed on config file or dnsmasq leases
        :param instrument: Correlator
        """
        hostclass = katcp_fpga.KatcpFpga
        self.instrument = instrument
        try:
            try:
                LOGGER.info('Retrieving running instrument from sensors')
                _running_instrument = (
                    self.katcp_rct.sensor.instrument_state.get_value())
            except:
                if self.instrument is not None:
                    _running_instrument = self.instrument

            if len(_running_instrument) > 4:
                config_file = '/etc/corr/{}-{}'.format(self.array_name,
                                                       _running_instrument)
            else:
                config_file = '/etc/corr/{}-{}'.format(self.array_name,
                                                       self.instrument)

            if os.path.exists(config_file):
                LOGGER.info('Retrieving running hosts from running config')
                fhosts = corr2.utils.parse_hosts(config_file, section='fengine')
                xhosts = corr2.utils.parse_hosts(config_file, section='xengine')
                hosts = fhosts + xhosts
            else:
                raise Exception
        except:
            LOGGER.error('Could not get instrument from sensors and config file does '
                         'not exist, Resorting to plan B - retreiving roach list from'
                         ' CORR2INI, In order to deprogram: File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))

            corr2ini_link = os.environ.get('CORR2INI')
            if corr2ini_link is not None:
                fhosts = corr2.utils.parse_hosts(corr2ini_link, section='fengine')
                xhosts = corr2.utils.parse_hosts(corr2ini_link, section='xengine')
                hosts = fhosts + xhosts
            else:
                LOGGER.error('Failed to retrieve hosts from CORR2INI \n\t '
                             'File:%s Line:%s' % (
                                getframeinfo(currentframe()).filename.split('/')[-1],
                                getframeinfo(currentframe()).lineno))
                return False

        if not len(hosts) == 0:
            try:
                try:
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(
                        hostclass, list(set(hosts)), timeout=timeout)
                    hosts = [host.host for host in connected_fpgas
                         if host.ping() is True]
                except Exception:
                    return False
                try:
                    connected_fpgas = fpgautils.threaded_create_fpgas_from_hosts(
                        hostclass, hosts)
                    fpgautils.threaded_fpga_function(connected_fpgas, 60, 'deprogram')
                except Exception:
                    return False
                LOGGER.info('FPGAs in dnsmasq all deprogrammed')
                return True
            except (katcp_fpga.KatcpRequestFail, KatcpClientError, KatcpDeviceError,
                    KatcpSyntaxError):
                errmsg = 'Failed to connect to roaches, reboot devices to fix.'
                LOGGER.exception(errmsg)
                return False
        else:
            LOGGER.error('Failed to deprogram FPGAs no hosts available'
                         '\n\t File:%s Line:%s' % (
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))
            return False

    def get_running_intrument(self):
        """
        Returns currently running instrument listed on the sensor(s)
        """
        try:
            reply = self.katcp_rct.sensor.instrument_state.get_reading()
        except AttributeError:
            LOGGER.error('KATCP Request does not contain attributes '
                         '\n\t File:%s Line:%s' % (
                         getframeinfo(currentframe()).filename.split('/')[-1],
                         getframeinfo(currentframe()).lineno))
            return False

        except KATCPSensorError:
            LOGGER.error('KATCP Error polling sensor\n\t File:%s Line:%s' % (
                         getframeinfo(currentframe()).filename.split('/')[-1],
                         getframeinfo(currentframe()).lineno))
            return False

        if reply.istatus:
            return {reply.value: True}
        else:
            LOGGER.error('Sensor request failed: %s \n\t File:%s Line:%s' % ( reply,
                         getframeinfo(currentframe()).filename.split('/')[-1],
                         getframeinfo(currentframe()).lineno))
            return {reply.value: False}

    def ensure_instrument(self, instrument, **kwargs):
        """Ensure that named instrument is active on the correlator array

        Will pass `kwargs` to self.start_correlator if a start is required
        :param instrument: Correlator object

        """
        success = False
        retries = 5
        while retries and not success:
            retries -= 1
            check_ins = self.check_instrument(instrument)
            if check_ins is True:
                success = True
                return success

        if self.check_instrument(instrument) is False:
            LOGGER.info('Correlator not running requested instrument, will restart.')
            deprogram_status = self.deprogram_fpgas(instrument)
            if not deprogram_status:
                LOGGER.info('Could not deprogram the hosts')
            self.instrument = instrument
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
        except:
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
                    reply = self.katcp_rct.req.instrument_list(instrument)
                except Exception:
                    LOGGER.error('Array request failed: %s\n\tFile:%s Line:%s' % (
                                 reply, getframeinfo(currentframe()).filename.split('/')[-1],
                                 getframeinfo(currentframe()).lineno))
                else:
                    if not reply.succeeded:
                        LOGGER.error('Array request failed: %s\n\tFile:%s Line:%s' % (
                                     reply, getframeinfo(currentframe()).filename.split('/')[-1],
                                     getframeinfo(currentframe()).lineno))
                        return False
            else:
                return False

            instruments_available = [instrument_avail.arguments[0]
                                     for instrument_avail in reply.informs]
            # Test to see if requested instrument is available on the instrument list
            if instrument not in instruments_available:
                LOGGER.error('Instrument: %s is not in instrument list: %s'
                             '\n\t File:%s Line:%s' % ( instrument, instruments_available,
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
                                'correlator %s.' % ( self.instrument, self.array_name))
                return instrument_present

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
            LOGGER.info('Using site test config file: %s' %conf_path)
        else:
            conf_path = '/config/test_conf.ini'
            LOGGER.info('Using Lab test config file on %s' % conf_path)

        config_file = path + conf_path
        if os.path.exists(config_file):
            try:
                return corr2.utils.parse_ini_file(config_file)
            except (IOError, ValueError, TypeError):
                errmsg = ('Failed to read test config file, Test will exit'
                          '\n\t File:%s Line:%s' % (
                          getframeinfo(currentframe()).filename.split('/')[-1],
                          getframeinfo(currentframe()).lineno))
                LOGGER.error(errmsg)
                return False
        else:
            return False

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
                self._test_config_file['inst_param']['source_mcast_ips'].split(','))
        except TypeError:
            msg = ('Could not read and split the multicast ip\'s in the test config file')
            LOGGER.error(msg)
            return 0
        else:
            if self.instrument.startswith('bc') or self.instrument.startswith('c'):
                if self.instrument[0] == 'b':
                    try:
                        # multicast_ip = multicast_ip_inp * (int(self.instrument[2]) / 2)
                        multicast_ip = multicast_ip_inp * int(
                                                    self.instrument.replace('bc', '').split('n')[0])
                    except Exception:
                        LOGGER.error('Could not calculate multicast ips from config file')
                        return 0
                else:
                    try:
                        # multicast_ip = multicast_ip_inp * (int(self.instrument[1]) / 2)
                        multicast_ip = multicast_ip_inp * int(
                                                    self.instrument.replace('c', '').split('n')[0])
                    except Exception:
                        LOGGER.error('Could not calculate multicast ips from config file')
                        return 0

            return multicast_ip

    def start_correlator(self, instrument=None, retries=10):
        LOGGER.info('Will now try to start the correlator')
        success = False
        retries_requested = retries
        if instrument is not None:
            self.instrument = instrument
        self._correlator = None  # Invalidate cached correlator instance
        LOGGER.info('Confirm DEngine is running before starting correlator')
        if not self.dhost.is_running():
            raise RuntimeError('DEngine: %s not running.' % (self.dhost.host))

        multicast_ip = self.get_multicast_ips()
        self.rct.start()
        try:
            self.rct.until_synced(timeout=timeout)
        except TimeoutError:
            self.rct.stop()
            return False
        else:
            reply, informs = self.rct.req.array_list(self.array_name)

        if not reply.reply_ok():
            LOGGER.error('Failed to halt down the array in primary interface: reply: %s'
                         '\n\t File:%s Line:%s' % (reply,
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
            self.katcp_array_port = None
            try:
                if self.katcp_array_port is None:
                    LOGGER.info('Assigning array port number')
                    # self.rct.start()
                    # self.rct.until_synced(timeout=timeout)
                    reply, _informs = self.rct.req.array_assign(self.array_name,
                                                                *multicast_ip)
                    if reply.reply_ok():
                        self.katcp_array_port = int(reply.arguments[-1])
                    else:
                        self.katcp_array_port = None
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
                    [int(i) for i in self._test_config_file['inst_param']['instrument_param']
                     if i != ','])
                LOGGER.info('Starting %s with %s parameters. Try #%s' % (self.instrument,
                                                                        instrument_param,
                                                                        retries))

                reply = self.katcp_rct.req.instrument_activate(self.instrument,
                                                               *instrument_param,
                                                               timeout=500)
                success = reply.succeeded
                retries -= 1

                if success is True:
                    LOGGER.info('Instrument %s started succesfully' % (self.instrument))
                else:
                    LOGGER.warn('Failed to start correlator, %s attempts left. '
                                'Restarting Correlator. Reply:%s' % (retries, reply))
                    self.halt_array()
                    success = False
                    LOGGER.info('Katcp teardown and restarting correlator.')

            except Exception:
                try:
                    self.rct.req.array_halt(self.array_name)
                except IndexError:
                    LOGGER.error('Unable to halt array: Empty Array number: '
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
                self.halt_array()
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

    def issue_metadata(self):
        """Issue Spead metadata"""
        try:
            self.katcp_rct.req.capture_meta(self.output_product)
            LOGGER.info('New metadata issued')
            return True
        except:
            LOGGER.error('Failed to issue new metadata: File:%s Line:%s' % (
                         getframeinfo(currentframe()).filename.split('/')[-1],
                         getframeinfo(currentframe()).lineno))
            return False


correlator_fixture = CorrelatorFixture()
