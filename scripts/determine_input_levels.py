#!/usr/bin/python

import os
import logging
import time
import casperfpga
import corr2
import argparse
import argcomplete

from corr2 import fxcorrelator
from corr2 import utils
from corr2.dsimhost_fpga import FpgaDsimHost

import unittest
import logging
import time
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unittest.util import strclass

from katcp.testutils import start_thread_with_cleanup
from corr2.dsimhost_fpga import FpgaDsimHost
#from corr2.corr_rx import CorrRx

from collections import namedtuple

from corr2 import utils
from casperfpga import utils as fpgautils

LOGGER = logging.getLogger(__name__)


def ip2int(ipstr): return struct.unpack('!I', socket.inet_aton(ipstr))[0]


def int2ip(n): return socket.inet_ntoa(struct.pack('!I', n))


class AttrDict(dict):
    """
    Based on JSObject : Python Objects that act like Javascript Objects
    based on James Robert blog entry:
    Making Python Objects that act like Javascript Objects
    http://jiaaro.com/making-python-objects-that-act-like-javascrip
    """


def check_x_rx_reorder(c):
    for x in c.xhosts:
        stat = x.get_rx_reorder_status()
        for i, xeng in enumerate(stat):
            for key, value in xeng.iteritems():
                if key.find('err') != -1 and value != 0:
                    print('{} Xeng {}: {}: {}'.format(x.host, i, key, value))


def check_f_rx_reorder(c):
    for f in c.fhosts:
        stat = f.get_rx_reorder_status()
        for key, value in stat.iteritems():
            if key.find('err') != -1 and value != 0:
                print('{}: {}: {}'.format(f.host, key, value))


def clear_all(c):
    for f in c.fhosts:
        f.clear_status()
    for x in c.xhosts:
        x.clear_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, action='store', default='',
        help='Instrument config file')
    parser.add_argument(
        '-d', '--dsim_present', action='store_true', default=False,
        help='Initialise DSIM')
    parser.add_argument(
        '-p', '--program', action='store_true', default=False,
        help='Program SKARABS during initialise')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.config:
        if os.path.isfile(args.config):
            ini_file = utils.parse_ini_file(args.config)
        else:
            parser.error('Specified config file does not exist.')
    if args.dsim_present:
        dsim_conf = ini_file['dsimengine']
        dig_host = dsim_conf['host']
        dhost = FpgaDsimHost(dig_host, config=dsim_conf)
        if dhost.is_running():
            dhost.get_system_information(dsim_conf['bitstream'])
            print 'Dsim is running'
    #def get_dsim_clk():
    #    feng_mcount = f.get_local_time()
    #    return (time.time() - feng_mcount/float(correlator.sample_rate_hz))

    c = fxcorrelator.FxCorrelator('steven', config_source=args.config)
    c.initialise(program=args.program, configure=args.program, require_epoch=False)
    f_engines = AttrDict({f.host: f for f in c.fhosts})
    x_engines = AttrDict({x.host: x for x in c.xhosts})
    for fpga in c.fhosts:
        if fpga.is_running():
            fpga.get_system_information(ini_file['fengine']['bitstream'])
    for fpga in c.xhosts:
        if fpga.is_running():
            fpga.get_system_information(ini_file['xengine']['bitstream'])

    print 'correlator is running'
    f = c.fhosts[0]
    fhost = c.fhosts[0]

    xhost = c.xhosts[0]
    x = c.xhosts[0]
    try:
        while True:
            quant_snap = np.abs(f.get_quant_snapshots()['inp000x'])
            print('Quantiser magnitude for channels 95 to 105:\n{}'.format(quant_snap[95:106]))
            print('PFB status: {}'.format(f.get_pfb_status()))
            print('ADC status:\n{}'.format(f.get_adc_status()))
            time.sleep(1)
    except KeyboardInterrupt:
        pass

