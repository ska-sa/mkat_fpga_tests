#!/usr/bin/env python

import argparse
import argcomplete
import atexit
import coloredlogs
import katcp
import logging
import sys
import time
import traceback
import random
import numpy as np
import matplotlib.pyplot as plt


from corr2 import utils
from corr2 import fxcorrelator
from corr2.dsimhost_fpga import FpgaDsimHost

from mkat_fpga_tests.Corr_RX import CorrRx
from mkat_fpga_tests.utils import init_dsim_sources
from mkat_fpga_tests.utils import loggerise
from mkat_fpga_tests.utils import normalised_magnitude



def katcp_request(katcp_ip='127.0.0.1', katcp_port=7147, katcprequest='subordinate-list',
                katcprequestArg=None, timeout=10):
    """
    Katcp requests

    Parameters
    =========
    katcp_ip: str
        IP to connect to! [Defaults: 127.0.0.1]
    katcp_port: int
        Port to connect to! [Defaults: 7147]
    katcprequest: str
        Katcp requests messages [Defaults: 'subordinate-list']
    katcprequestArg: str
        katcp requests messages arguments eg. subordinate-list array0 [Defaults: None]
    timeout: int
        katcp timeout [Defaults :10]

    Return
    ======
    reply, informs : tuple
        katcp request messages
    """
    client = katcp.BlockingClient(katcp_ip, katcp_port)
    client._logger.setLevel(logging.INFO)
    client.setDaemon(True)
    client.start()
    time.sleep(.1)
    is_connected = client.wait_running(timeout)
    time.sleep(.1)
    if not is_connected:
        client.stop()
        logger.error('Could not connect to katcp, timed out.')
        return
    try:
        if katcprequestArg:
            reply, informs = client.blocking_request(katcp.Message.request(katcprequest, katcprequestArg),
                timeout=timeout)
        else:
            reply, informs = client.blocking_request(katcp.Message.request(katcprequest),
                timeout=timeout)

        assert reply.reply_ok()
    except Exception:
        logger.exception('Failed to execute katcp command')
        return None
    else:
        client.stop()
        client = None
        return reply, informs

def cleanup_atexit(receiver):
    logger.info("Running clean-up...")
    sys.stdout.flush()
    katcp_request(katcp_ip, katcp_array_port, katcprequest='capture-stop',
                        katcprequestArg='%s' % product_name)
    try:
        receiver.stop()
        time.sleep(0.1)
        sys.stdout.flush()
        assert receiver.stopped()
    except AssertionError:
        logger.error('Ohhhh!!!! **** thread is still active')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receive data from a CBF and play.')
    parser.add_argument('--corr', dest='enable_corr', action='store', default=True,
                        help='Play with correlator!')
    parser.add_argument('--config', dest='config', action='store', default='', required=True,
                        help='Specify a config file.')
    parser.add_argument('--rx-port', dest='port', action='store', default=7148, type=int,
                        help='Port the receiver will listen on? [Default: 7148]')
    parser.add_argument('--katcp', dest='katcp_con', action='store', default='127.0.0.1:7147',
                        help='IP:Port primary interface [Default: 127.0.0.1:7147]')
    parser.add_argument('--dsim_start', dest='dsim', action='store', default=True,
                        help='Enable DEngine transmission.')
    parser.add_argument('--product', dest='data_product', action='store',
                        default='baseline-correlation-products', help='name of correlation product')
    parser.add_argument('--capture_start', dest='capture_start', action='store', default=True,
                        help='Start capture or not? [Default: True]')
    parser.add_argument('--start_stop_channels', dest='selected_channels', default=(0, 4095), type=int,
                        nargs='+', help='Which channels to capture data from? [Default: 0-4095]')
    parser.add_argument('--loglevel', dest='log_level', action='store', default='INFO',
                        help='log level to use, default INFO, options INFO, DEBUG, ERROR')
    parser.add_argument('--speadloglevel', dest='spead_log_level', action='store', default='ERROR',
                        help='log level to use in spead receiver, options INFO, DEBUG, ERROR [Default: ERROR, ]')
    argcomplete.autocomplete(parser)
    args = vars(parser.parse_args())
    log_level = None
    if args.get("log_level", 'INFO'):
        log_level = args.get("log_level", 'INFO')
        try:
            logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(pathname)s : '
                           '%(lineno)d - %(message)s')
            logging.getLogger("mkat_fpga_tests.corr_rx").setLevel(eval('logging.%s' % log_level))
        except AttributeError:
            raise RuntimeError('No such log level: %s' % log_level)
        else:
            logger = logging.getLogger('spead2_rx_debug')
            coloredlogs.install(level=log_level)

    spead_log_level = None
    if args.get("spead_log_level", 'ERROR'):
        spead_log_level = args.get("spead_log_level", 'ERROR')
        logging.getLogger("casperfpga").setLevel(logging.ERROR)
        logging.getLogger("spead2").setLevel(eval('logging.%s' % spead_log_level))
        logging.getLogger("katcp.ioloop_manager").setLevel(eval('logging.%s' % spead_log_level))

    config_file = args.get('config')
    product_name = args.get('data_product', 'baseline-correlation-products')

    if args.get('enable_corr', False):
        logger.info('Enabling correlator instance')
        correlator = fxcorrelator.FxCorrelator('LetsPlay', config_source=config_file)
        correlator.initialise(program=False, configure=False, require_epoch=False)
        try:
            assert correlator._initialised
            logger.info('Correlator is running!!!')
        except AssertionError:
            logger.error('Correlator is not running!!!')
        else:
            f_engines = dict((f.host, f) for f in correlator.fhosts)
            x_engines = dict((x.host, x) for x in correlator.xhosts)
            for fpga in correlator.fhosts:
                if fpga.is_running():
                    logger.info('%s: host is running' % fpga.host)
                    fpga.get_system_information(correlator.configd['fengine']['bitstream'])
            for fpga in correlator.xhosts:
                if fpga.is_running():
                    logger.info('%s: host is running' % fpga.host)
                    fpga.get_system_information(correlator.configd['xengine']['bitstream'])
            fhost = random.choice(correlator.fhosts)
            xhost = random.choice(correlator.xhosts)

    if args.get("dsim", False):
        corr_conf = utils.parse_ini_file(config_file, ['dsimengine'])
        logger.info('Enabling dsim instance')
        dhost = FpgaDsimHost(corr_conf.get('dsimengine').get('host'), config=corr_conf.get('dsimengine'))
        try:
            assert dhost.is_running()
            dhost.get_system_information(filename=dhost.config.get('bitstream'))
            logger.info('Confirmed that the dsimengine is currently running.')
            init_dsim_sources(dhost)
        except Exception as e:
            logger.exception('DEngine Not Running!')

    if args.get('katcp_con'):
        try:
            katcp_ip, katcp_port = args.get('katcp_con').split(':')
            _, informs = katcp_request(katcp_ip, katcp_port)
            assert isinstance(informs, list)
            katcp_array_list = informs[0].arguments
            katcp_array_name = katcp_array_list[0]
            katcp_array_port, katcp_sensor_port = katcp_array_list[1].split(',')
        except Exception as e:
            logger.exception(e.message)

    if args.get('capture_start'):
        try:
            reply = katcp_request(katcp_ip, katcp_array_port, katcprequest='capture-start',
                                        katcprequestArg='%s' % product_name)
            assert reply
            logger.info(str(reply[0]))
        except Exception as e:
            logging.exception(e, exc_info=True)
        else:
            logger.info('Starting the receiver.')
            receiver = CorrRx(katcp_ip=katcp_ip, katcp_port=katcp_array_port,
                            channels=args.get('selected_channels'))
            receiver.daemon = True
            atexit.register(cleanup_atexit, receiver)
            receiver.start(timeout=10)
            try:
                assert receiver.isAlive()
                assert receiver.running_event.wait(timeout=10)
                logger.info('Receiver ready!!!')
            except AssertionError:
                logger.exception('Receiver is not running!!!!')
                sys.stdout.flush()

    msg = ('Time to play, \nNOTE: Available Attributes/instances and etc\n'
           'np, plt, receiver, correlator, dhost, fhost, xhost and katcp_request\n'
           'Other: loggerise, normalised_magnitude')
    logger.info('Lets Play!!!')
    logger.info(dir())
    import IPython; globals().update(locals()); IPython.embed(header=msg)

