#!/usr/bin/python
import logging
import time
import numpy as np
from corr2 import fxcorrelator
from corr2 import utils
from corr2.dsimhost_fpga import FpgaDsimHost
from optparse import OptionParser
from os import remove, close, path
from shutil import move
from tempfile import mkstemp

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = OptionParser()
    parser.set_usage('%prog [options]')
    parser.add_option('-s', '--sample_clock_freq', type=float, default=1712000000.,
                      help='Sample clock frequency to set in configuration file')
    parser.add_option('-c', '--config_file', type=str, default=None,
                      help='Correlator config file to modify.')
    opts, args = parser.parse_args()
    if opts.config_file:
        if path.isfile(opts.config_file):
            config = opts.config_file
        else:
            parser.error('Specified file does not exist.')
    else:
        parser.error('Specify a valid config file.')

    bandwidth = int(opts.sample_clock_freq / 2)
    true_cfreq = bandwidth / 2
    sample_clock = bandwidth * 2
    print('Setting Sample frequency = {}Hz, Bandwidth = {}Hz, True center frequency = {}Hz'
          .format(sample_clock, bandwidth, true_cfreq))

    pattern = {'sample_rate_hz': sample_clock,
               'bandwidth': bandwidth,
               'true_cf': true_cfreq,
               'center_freq': true_cfreq}
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path, 'w') as new_file:
        with open(opts.config_file) as old_file:
            for line in old_file:
                key = [x for x in pattern if x in line]
                if key and (line.find('#') == -1):
                    key = key[0]
                    new_file.write(key + ' = {}\n'.format(pattern[key]))
                else:
                    new_file.write(line)
    close(fh)
    #Remove original file
    remove(opts.config_file)
    #Move new file
    move(abs_path, opts.config_file)
