#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from mkat_fpga_tests import correlator_fixture
from optparse import OptionParser
from corr2 import utils

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = OptionParser()
    parser.set_usage('%prog [options]')
    parser.add_option('--hist', dest='hist', action='store_true',
                      help='Plot histogram of ADC data')
    parser.add_option('--raw', dest='raw', action='store_true',
                      help='Plot raw ADC data')
    parser.add_option('--fft', dest='fft', action='store_true',
                      help='Plot FFT of ADC data')
    parser.add_option('-p', dest='pol', type=str, default='x',
                      help='Polarisation to plot (x or y)')
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

    instrument = config[config.find('bc'):]
    corr_fix = correlator_fixture
    conf_file = corr_fix.test_config
    conf = conf_file['inst_param']
    corr_fix.array_name = conf['subarray']
    corr_fix.resource_clt = conf['katcp_client']
    instrument_state = corr_fix.ensure_instrument(instrument)
    if not instrument_state:
        errmsg = ('Could not initialise instrument or ensure running instrument: {}'.format(
            instrument))
        print errmsg
        quit()
    reply, informs = corr_fix.katcp_rct.req.input_labels()
    if reply.reply_ok():
        inputs = reply.arguments[1:]
        found = False
        for inp in inputs:
            if inp.find(opts.pol) != -1:
                found = True
                break
        if not found:
            parser.error('Specify an input polarisation (x or y)')
    else:
        print ('Could not get input labels, error message: {}'.format(reply))
        quit()
    reply, informs = corr_fix.katcp_rct.req.adc_snapshot(inp)
    if reply.reply_ok():
        msg = informs[0]
        adc_data = msg.arguments[1]
        adc_data = adc_data[1:-1]
        adc_data = map(float, adc_data.split(','))
    if opts.raw:
        plt.figure()
        plt.plot(adc_data)
    if opts.hist:
        plt.figure()
        plt.hist(adc_data, bins=256, range=(-1, 1))
    if opts.fft:
        plt.figure()
        chans = len(adc_data) / 2
        plt.plot(np.log10(np.abs(np.fft.fft(adc_data)))[0:chans])
    plt.show()
