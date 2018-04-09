#!/usr/bin/env python

import argparse
import argcomplete
# import atexit
import coloredlogs
import katcp
import logging
import sys
import time
import traceback
import random
import json

# from addict import Dict
from collections import OrderedDict
from itertools import izip_longest
from pprint import PrettyPrinter

def katcp_request(katcp_ip='127.0.0.1', katcp_port=7147, katcprequest='array-list',
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
        Katcp requests messages [Defaults: 'array-list']
    katcprequestArg: str
        katcp requests messages arguments eg. array-list array0 [Defaults: None]
    timeout: int
        katcp timeout [Defaults :10]

    Return
    ======
    reply, informs : tuple
        katcp request messages
    """
    client = katcp.BlockingClient(katcp_ip, katcp_port)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Receive data from a CBF and play.')
    parser.add_argument('--katcp', dest='katcp_con', action='store', default='127.0.0.1:7147',
                        help='IP:Port primary interface [Default: 127.0.0.1:7147]')
    parser.add_argument('--poll-sensors', dest='poll', action='store_true', default=False,
                        help='Poll the sensors every 10 seconds')
    parser.add_argument('--json', dest='sensor_json', action='store_true', default=False,
                        help='Write sensors to jsonFile')
    parser.add_argument('--loglevel', dest='log_level', action='store', default='INFO',
                        help='log level to use, default INFO, options INFO, DEBUG, ERROR')

    argcomplete.autocomplete(parser)
    args = vars(parser.parse_args())

    pp = PrettyPrinter(indent=4)
    log_level = None
    if args.get("log_level", 'INFO'):
        log_level = args.get("log_level", 'INFO')
        try:
            logging.basicConfig(level=getattr(logging, log_level),
                                format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(pathname)s : '
                                '%(lineno)d - %(message)s')
        except AttributeError:
            raise RuntimeError('No such log level: %s' % log_level)
        else:
            logger = logging.getLogger(__name__)
            coloredlogs.install(level=log_level)

    if args.get('katcp_con'):
        try:
            katcp_ip, katcp_port = args.get('katcp_con').split(':')
            _, informs = katcp_request(katcp_ip, katcp_port)
            assert isinstance(informs, list)
            katcp_array_list = informs[0].arguments
            katcp_array_name = katcp_array_list[0]
            katcp_array_port, katcp_sensor_port = katcp_array_list[1].split(',')
            logger.info("Katcp connection established: IP %s, Array Port: %s, Sensor Port %s" %
                        (katcp_ip, katcp_array_port, katcp_sensor_port))
        except Exception as e:
            logger.exception(e.message)
            sys.exit(1)

    def get_sensor_values(i=1):
        logger.info('Connecting to running sensors servlet and getting sensors')
        for i in xrange(i):
            reply, informs = katcp_request(
                katcp_ip, katcp_sensor_port, katcprequest='sensor-value')
        logger.info('Retrieved sensors successfully')
        try:
            assert int(reply.arguments[-1])
            yield [inform.arguments for inform in informs]
        except AssertionError:
            logger.exception("No Sensors!!! Exiting!!!")
            sys.exit(1)

    unordered_sensor_values = next(get_sensor_values())

    def get_sensor_dict(sensor_value_informs):
        sensor_dict = dict((x[0], x[1:]) for x in [i[2:]
                                                   for i in sensor_value_informs])
        return sensor_dict

    unordered_sensor_dict = get_sensor_dict(unordered_sensor_values)

    def ordered_sensor_values(_sensor_values):
        return OrderedDict(sorted(_sensor_values.items()))

    ordered_sensor_dict = ordered_sensor_values(unordered_sensor_dict)

    def sort_by_host(_ordered_sensor_dict):
        """
        {
            "FHOSTS": [
                [
                    {
                        "fhost00": [
                            "cd",
                            "delay0-updating"
                        ]
                    },
                    [
                        "unknown",
                        ""
                    ]
                ],
                [
                    {
                        "fhost00": [
                            "cd",
                            "delay1-updating"
                        ]
                    },
        """
        # see addict doc
        # mapping = Dict()
        mapping = {'FHOSTS': [], 'XHOSTS': [], 'SYSTEM': []}
        for i, v in ordered_sensor_dict.iteritems():
            i = i.split('.')
            if i[0].startswith('fhost'):
                new_i = dict(izip_longest(
                    *[iter([i[0], i[1:]])] * 2, fillvalue=""))
                mapping['FHOSTS'].append([new_i, v])
            elif i[0].startswith('xhost'):
                new_i = dict(izip_longest(
                    *[iter([i[0], i[1:]])] * 2, fillvalue=""))
                mapping['XHOSTS'].append([new_i, v])
            else:
                new_i = dict(izip_longest(
                    *[iter([i[0], i[1:]])] * 2, fillvalue=""))
                mapping['SYSTEM'].append([new_i, v])
        return mapping

    sorted_by_host = sort_by_host(ordered_sensor_dict)

    if args.get('sensor_json', False):
        _filename = 'sensor_values.json'
        logger.info('Writing sensors to file: %s' % _filename)
        with open(_filename, 'w') as outfile:
            json.dump(sorted_by_host, outfile, indent=4, sort_keys=True)

    # pretty print
    # pp.pprint(sorted_by_host)

    # pretty print json dumps
    # print json.dumps(sorted_by_host, indent=4)

    try:
        poll = args.get('poll', False)
        assert poll
        logger.info('Begin sensor polling!!!')
        while poll:
            # Upload sensors to dashboard
            sensor_values = (next(get_sensor_values()))
            pp.pprint(sensor_values)
            logger.debug('Sensors')
            time.sleep(10)
    except Exception:
        import IPython
        globals().update(locals())
        IPython.embed(header='Lets Play')
