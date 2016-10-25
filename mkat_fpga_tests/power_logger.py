import telnetlib,time,csv,threading,logging,os
from telnetlib import IAC, NOP
from datetime import datetime
from corr2.utils import parse_ini_file

class PowerLogger(threading.Thread):

    REQ_PWR  = ['phReading id:all power','kW']
    REQ_CRNT = ['phReading id:all current','A']

    def __init__(self, config_file, console_log_level = logging.CRITICAL, 
                 file_log_level = logging.INFO):

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
        self.logger.setLevel(logging.DEBUG)
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

        threading.Thread.__init__(self)
        test_conf = parse_ini_file(config_file)
        self._pdu_host_ips = test_conf['pdu_hosts']['pdu_ips'].split(',')
        self._pdu_port = test_conf['pdu_hosts']['telnet_port']
        self._pdu_username = test_conf['pdu_hosts']['username']
        self._pdu_password = test_conf['pdu_hosts']['passwd']
        self._stop = threading.Event()
        self.log_file_name = 'pdu_log.csv'
        self.logger.info('PDUs logged: {}'.format(self._pdu_host_ips))

    def stop(self):
        self._stop.set()
    def stopped(self):
        return self._stop.isSet()
    def run(self):
        self.logger.info('Starting power_logger ' + self.name)
        self.write_pdu_log()
        self.logger.info('Stopping power_logger ' + self.name)

    def open_telnet_conn(self, host_ip, port=23, timeout=5):
        try:
            self.logger.debug('Opening connection to {}.'.format(host_ip))
            telnet_handle = telnetlib.Telnet(host_ip, port, timeout=timeout)
        except Exception as e:
            raise
        else:
            try:
                _None = telnet_handle.read_until('User Name :', timeout=timeout)
                telnet_handle.write(self._pdu_username + '\r\n')
                _None = telnet_handle.read_until('Password  :', timeout=timeout)
                telnet_handle.write(self._pdu_password  + '\r\n')
                _None = telnet_handle.read_until('apc>', timeout=timeout)
                self.logger.debug('Connection to {} successful.'.format(host_ip))
                return telnet_handle
            except Exception as e:
                raise

    def close_telnet_conn(self, telnet_handle, timeout=5):
        try:
            self.logger.debug('Closing connection to {}.'.format(telnet_handle.host))
            telnet_handle.write('quit\r\n')
            stdout = telnet_handle.read_until('Connection Closed', timeout=timeout)
            telnet_handle.close()
        except Exception as e:
            telnet_handle.close()
            raise


    def _get_stdout(self, data, unit):
        return str([float(i.split()[1]) for i in data.splitlines()
                 if i.endswith(unit) if len(i.split()) == 3]).strip('[]')

    def read_from_pdu(self, telnet_handle, cmd, timeout=5):
        try:
            self.logger.debug('Sending {} to {}.'.format(cmd, telnet_handle.host))
            telnet_handle.write(cmd[0] + '\r\n')
            stdout = telnet_handle.read_until('apc>', timeout=timeout)
            smpl_time = str(int(time.time()))
            value = self._get_stdout(stdout, cmd[1])
            self.logger.debug('Read {}.'.format(value))
            return smpl_time, value
        except Exception as e:
            raise

    def write_pdu_log(self, retry = 5):
        # Open all the pdus
        telnet_handles = []
        rem_ips = self._pdu_host_ips
        host_ips = rem_ips[:]
        conn_good = False
        while retry:
            for host in host_ips:
                try:
                    telnet_handles.append(self.open_telnet_conn(host, self._pdu_port))
                    rem_ips.remove(host)
                except Exception as e:
                    self.logger.error('Exception occured while connecting to PDU {}.'.format(host))
                    self.logger.error('Exception: {}'.format(e))
            host_ips = rem_ips[:]
            if not host_ips:
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
                    csv_writer.writerow(['Sample Time','PDU Host IP','Phase Current','Phase Power'])
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
                            except Exception as e:
                                self.logger.error('Exception occured while connecting to PDU {}.'.format(th.host))
                                self.logger.error('Exception: {}'.format(e))
                                break
                        power = self.read_from_pdu(th, self.REQ_PWR)
                        current = self.read_from_pdu(th, self.REQ_CRNT)
                        data = [power[0], th.host, current[1], power[1]]
                        csv_writer.writerow(data)
                        csvfile.flush()
        else:
            self.logger.error('Unable to connect to the following PDUs: {}'.format(host_ips))
        self.logger.info('Logging stopped, closing telnet connections to PDUs.')
        for th in telnet_handles:
            self.close_telnet_conn(th)

