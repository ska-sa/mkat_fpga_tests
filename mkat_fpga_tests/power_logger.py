import csv
import logging
import os
import telnetlib
import threading
import time
from inspect import currentframe, getframeinfo
from telnetlib import IAC, NOP
from corr2.utils import parse_ini_file
from utils import decode_passwd


class NetworkError(RuntimeError):
    """
    Raised to singal connection error to PDUs
    """
    pass


class PowerLogger(threading.Thread):
    REQ_PWR = ['phReading id:all power', 'kW']
    REQ_CRNT = ['phReading id:all current', 'A']

    def __init__(self, config_info, conn_retry=10, console_log_level=logging.ERROR,
                 file_log_level=logging.INFO):
        """
            PowerLogger reads PDU IP addresses from a config file and starts logging
            current and power from each PDU. Values are written to a CSV file. This
            class is threaded and must be started with instance.start()

            params:
                config_info: Dictionary parsed with corr2.utils.parse_ini_file or
                             a config file.
                conn_retry:  Number of connection attempts for initial connection
                             and if contact is lost during logging to PDUs
                console_log_level: Log level for print to std_out
                file_log_level: Log level for logging to file
        """

        # **************************************************************************
        #
        # Add a file handler to log to file and a console handler to print out
        # debug messages. The console handler is added first, to print messages
        # to console change log level for the console handler.
        # logger.handlers[0].setLevel(logging.DEBUG)
        #
        # **************************************************************************
        # create logger
        self.logger = logging.getLogger('power_logger')
        self.logger.setLevel(logging.INFO)
        # create a file handler
        file_handler = logging.FileHandler('power_logger.log')
        file_handler.setLevel(file_log_level)
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        arb = '1234567890123456'

        threading.Thread.__init__(self)
        if isinstance(config_info, dict):
            test_conf = config_info
        elif os.path.exists(config_info):
            try:
                test_conf = parse_ini_file(config_info)
            except (IOError, ValueError, TypeError):
                errmsg = ('Failed to read test config file, Test will exit'
                          '\n\t File:%s Line:%s' % (
                              getframeinfo(currentframe()).filename.split('/')[-1],
                              getframeinfo(currentframe()).lineno))
                self.logger.error(errmsg)
                raise
        else:
            raise IOError
        pdu_names = test_conf['pdu_hosts']['pdus'].split(',')
        pdu_names = [x.replace(' ', '') for x in pdu_names]
        pdu_host_domain = test_conf['pdu_hosts']['pdu_host_domain']
        pdu_hosts = [x + '.' + pdu_host_domain for x in pdu_names]
        self._pdu_hosts = [x.replace(' ', '') for x in pdu_hosts]
        self._pdu_port = test_conf['pdu_hosts']['telnet_port']
        pdu_username = test_conf['pdu_hosts']['username']
        self._pdu_username = decode_passwd(pdu_username, arb)
        pdu_password = test_conf['pdu_hosts']['passwd']
        self._pdu_password = decode_passwd(pdu_password, arb)
        self._stop = threading.Event()
        self._conn_retry = conn_retry
        self.start_timestamp = None
        self.log_file_name = 'pdu_log.csv'
        self.logger.info('PDUs logged: %s' % (pdu_names))

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        self.logger.info('Starting power_logger %s' % self.name)
        try:
            self.write_pdu_log()
        except:
            self.logger.info('Stopping power_logger %s' % self.name)
            raise
        self.logger.info('Stopping power_logger %s' % self.name)

    def open_telnet_conn(self, host, port=23, timeout=60):
        try:
            self.logger.debug('Opening connection to {}.'.format(host))
            telnet_handle = telnetlib.Telnet(host, port, timeout=timeout)
        except Exception:
            raise
        else:
            try:
                telnet_handle.read_until('User Name :', timeout=timeout)
                telnet_handle.write(self._pdu_username + '\r\n')
                telnet_handle.read_until('Password  :', timeout=timeout)
                telnet_handle.write(self._pdu_password + '\r\n')
                telnet_handle.read_until('apc>', timeout=timeout)
                self.logger.debug('Connection to {} successful.'.format(host))
                return telnet_handle
            except Exception:
                raise

    def close_telnet_conn(self, telnet_handle, timeout=20):
        try:
            self.logger.debug('Closing connection to {}.'.format(telnet_handle.host))
            telnet_handle.write('quit\r\n')
            stdout = telnet_handle.read_until('Connection Closed', timeout=timeout)
            telnet_handle.close()
        except Exception:
            telnet_handle.close()
            raise

    def _get_stdout(self, data, unit):
        return str([float(i.split()[1]) for i in data.splitlines()
                    if i.endswith(unit) if len(i.split()) == 3]).strip('[]')

    def read_from_pdu(self, telnet_handle, cmd, timeout=20):
        try:
            self.logger.debug('Sending {} to {}.'.format(cmd, telnet_handle.host))
            telnet_handle.write(cmd[0] + '\r\n')
            try:
                stdout = telnet_handle.read_until('apc>', timeout=timeout)
            except Exception as e:
                raise
            smpl_time = str(int(time.time()))
            value = self._get_stdout(stdout, cmd[1])
            self.logger.debug('Read {}.'.format(value))
            return smpl_time, value
        except Exception:
            raise

    def write_pdu_log(self):
        # Open all the pdus
        telnet_handles = []
        rem_hosts = self._pdu_hosts
        hosts = rem_hosts[:]
        conn_good = False
        retry = self._conn_retry
        while retry:
            for host in hosts:
                try:
                    telnet_handles.append(self.open_telnet_conn(host, self._pdu_port))
                    rem_hosts.remove(host)
                except Exception as e:
                    self.logger.error('Exception ({}) occured while connecting to PDU {}.'.format(
                        host, str(e)))
            hosts = rem_hosts[:]
            if not hosts:
                retry = 0
                conn_good = True
            else:
                retry -= 1
        # Only proceed if connected to all PDUs
        if conn_good:
            if os.path.isfile(self.log_file_name):
                open_mode = 'a'
            else:
                open_mode = 'wb'
            self.logger.info('Opening {} for writing pdu data'.format(self.log_file_name))
            with open(self.log_file_name, open_mode) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t')
                if open_mode == 'wb':
                    csv_writer.writerow(['Sample Time', 'PDU Host', 'Phase Current', 'Phase Power'])
                con_err_dict = {x: 0 for x in self._pdu_hosts}
                while not self._stop.isSet():
                    for idx, th in enumerate(telnet_handles):
                        try:
                            th.sock.sendall(IAC + NOP)
                        except:
                            self.logger.warning('Connection lost to PDU {}.'.format(th.host))
                            try:
                                self.logger.info('Trying to reconnect to PDU {}'.format(th.host))
                                telnet_handles[idx] = self.open_telnet_conn(host, th.host)
                                th = telnet_handles[idx]
                                con_err_dict[th.host] = 0
                            except KeyError:
                                break
                            except Exception as e:
                                con_err_dict[th.host] += 1
                                self.logger.error('Exception occured while connecting to PDU {}.'.format(th.host))
                                self.logger.error('Exception: {}'.format(e))
                                break
                        power = self.read_from_pdu(th, self.REQ_PWR)
                        current = self.read_from_pdu(th, self.REQ_CRNT)
                        data = [power[0], th.host, current[1], power[1]]
                        if self.start_timestamp is None:
                            self.start_timestamp = power[0]
                        csv_writer.writerow(data)
                        csvfile.flush()
                        for key in con_err_dict:
                            if con_err_dict[key] > self._conn_retry:
                                self.logger.error('Connection lost to PDU {}... exiting.'.format(key))
                                self.logger.info('Closing telnet connections to PDUs.')
                                for th in telnet_handles:
                                    self.close_telnet_conn(th)
                                raise NetworkError("Connection lost to PDU {}.".format(key))
        else:
            self.logger.error('Unable to connect to the following PDUs: {}'.format(hosts))
            self.logger.info('Closing telnet connections to PDUs.')
            for th in telnet_handles:
                self.close_telnet_conn(th)
            raise NetworkError("Unable to connect to the following PDUs: {}".format(hosts))

        self.logger.info('Logging stopped, closing telnet connections to PDUs.')
        for th in telnet_handles:
            self.close_telnet_conn(th)
