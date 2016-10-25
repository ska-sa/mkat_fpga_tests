import os
import logging
import time
import casperfpga
import corr2

from casperfpga import katcp_fpga
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
from corr2.corr_rx import CorrRx

from collections import namedtuple

from corr2 import utils
from casperfpga import utils as fpgautils

config = os.environ['CORR2INI']



LOGGER = logging.getLogger(__name__)

class AttrDict(dict):
    """
    Based on JSObject : Python Objects that act like Javascript Objects
    based on James Robert blog entry:
    Making Python Objects that act like Javascript Objects
    http://jiaaro.com/making-python-objects-that-act-like-javascrip
    """

#logging.basicConfig(
#    format='%(asctime)s %(name)s %(levelname)s %(filename)s:%(lineno)s %(message)s',
#    level=logging.INFO)

corr_conf = utils.parse_ini_file(config, ['dsimengine'])
dsim_conf = corr_conf['dsimengine']
dig_host = dsim_conf['host']

dhost = FpgaDsimHost(dig_host, config=dsim_conf)
if dhost.is_running():
    dhost.get_system_information()
    print 'Dsim is running'

config_link = '/etc/corr/array0-bc8n856M4k'
config_link2 = '/etc/corr/array0-bc8n856M32k'
config_link3 = '/etc/corr/templates/bc8n856M4k'
config_link4 = '/etc/corr/templates/bc8n856M32k'
if os.path.exists(config_link):
    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link)
elif os.path.exists(config_link2):
    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link2)
elif os.path.exists(config_link3):
    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link3)
else:
    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link4)

correlator.initialise(program=False)
f_engines = AttrDict({f.host: f for f in correlator.fhosts})
x_engines = AttrDict({x.host: x for x in correlator.xhosts})
for fpga in correlator.fhosts + correlator.xhosts:
    if fpga.is_running():
        fpga.get_system_information()

print 'correlator is running'
f = correlator.fhosts[0]
fhost = correlator.fhosts[0]

xhost = correlator.xhosts[0]
x = correlator.xhosts[0]
try:
    receiver = CorrRx(port=8888, queue_size=100)
except:
    print 'Could not instantiate receiver'
else:
    receiver.start()
