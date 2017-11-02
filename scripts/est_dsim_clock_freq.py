#!/usr/bin/python
import os
import logging
import time

from corr2 import fxcorrelator
from corr2 import utils
from corr2.dsimhost_fpga import FpgaDsimHost

import numpy as np
from optparse import OptionParser

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = OptionParser()
    parser.set_usage('%prog [options]')
    parser.add_option('-d', '--time_delta', type=int, default=60,
                      help='Time in seconds to estimate DSIM clock frequency')
    parser.add_option('-c', '--config_file', type=str, default=None,
                      help='Correlator config file to use, if ommitted CORR2INI '
                           'environment variable will be used.')
    opts, args = parser.parse_args()
    if opts.config_file:
        if os.path.isfile(opts.config_file):
            config = opts.config_file
        else:
            parser.error('Specified file does not exist.')
    else:            
        config = os.environ['CORR2INI']

    #corr_conf = utils.parse_ini_file(config, ['dsimengine'])
    #dsim_conf = corr_conf['dsimengine']
    #dig_host = dsim_conf['host']
    #dhost = FpgaDsimHost(dig_host, config=dsim_conf)
    #if dhost.is_running():
    #    dhost.get_system_information()
    #    print 'Dsim is running'

    print 'Initialising correlator'
    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config)
    correlator.initialise(program=False)

    # Done during correlator init
    #for fpga in correlator.fhosts + correlator.xhosts:
    #    if fpga.is_running():
    #        fpga.get_system_information()
    print 'correlator is running'

    f = correlator.fhosts[0]

    try:
        #20ms in clock ticks
        ticks_20ms = 0.02*1712000000
        ticks_20ms = pow(2,32)-ticks_20ms
        def get_ts():
            lsw = f.registers.local_time_lsw.read()
            msw = f.registers.local_time_msw.read()
            lsw = lsw['data']['timestamp_lsw']
            msw = msw['data']['timestamp_msw']
            #Check that lsw is not within 20 ms of wrapping
            if lsw > ticks_20ms:
                return (msw<<32)+lsw
            else:
                return False
        #Function to check freq using fhost.get_local_time
        #This is unreliable. get_local_time takes too long to return
        def freq_get_local_time(delay=10):
            start_time = time.time()
            fhost_st_ts = f.get_local_time()
            time.sleep(delay)
            end_time = time.time()
            fhost_end_ts = f.get_local_time()
            loc_time_diff = end_time-start_time
            ts_diff = fhost_end_ts-fhost_st_ts
            return ts_diff/loc_time_diff

        #Function to check freq reading raw local time registers
        def freq_get_ts(delay=10):
            fhost_st_ts = False
            fhost_end_ts = False
            while not fhost_st_ts:
                start_time = time.time()
                fhost_st_ts = get_ts()
            time.sleep(delay)
            while not fhost_end_ts:
                end_time = time.time()
                fhost_end_ts = get_ts()
            loc_time_diff = end_time-start_time
            ts_diff = fhost_end_ts-fhost_st_ts
            return ts_diff/loc_time_diff
        freq_ts_list = []
        delay = opts.time_delta
        print 'Estimating dsim clock frequency using direct register method with delay of {}'.format(delay)
        while True:
            freq_ts = freq_get_ts(delay)
            freq_ts_list.append(freq_ts)
            print ('Frequency using get timestamp = {}'.format(freq_ts))

    except KeyboardInterrupt:
        freq_list = np.asarray(freq_ts_list)
        print ('Standard deviation of samples: {}').format(freq_list.std())
        pass


