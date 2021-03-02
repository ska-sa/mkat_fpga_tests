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

def get_quant_ss(offset = 0):
    ss = f.snapshots.snap_quant0_ss
    sdata = ss.read(offset=offset)['data']
    compl = []
    for ctr in range(0, len(sdata['real0'])):
        compl.append(complex(sdata['real0'][ctr], sdata['imag0'][ctr]))
        compl.append(complex(sdata['real1'][ctr], sdata['imag1'][ctr]))
        compl.append(complex(sdata['real2'][ctr], sdata['imag2'][ctr]))
        compl.append(complex(sdata['real3'][ctr], sdata['imag3'][ctr]))
    compl_np = np.abs(np.asarray(compl))
    max_val = compl_np.max()
    max_idx = compl_np.argmax()
    print'Offset: {}, Index: {}, Val: {}'.format(offset,max_idx,max_val)

def quant_setup_spectrum():
    ss = f.snapshots.snap_quant0_ss
    ss.arm(man_trig=False, man_valid=False)
    f.registers.quant_snap_ctrl.write(single_channel=0)


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

    import IPython
    IPython.embed()
