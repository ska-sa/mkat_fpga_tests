#!/usr/bin/python

import casperfpga,logging,sys,time,argparse
from casperfpga import utils as fpgautils
from corr2 import utils
from corr2.dsimhost_fpga import FpgaDsimHost
from os import path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config', type=str, action='store', default='/etc/corr/array0-bc8n856M4k',
            help='a Corr2 config file')
    parser.add_argument(
            '-f', '--frequency', type=float, default=0,
            help='Dsim frequency')
    parser.add_argument(
            '-w', '--cw_scale', type=float, default=0.1,
            help='Constant wave output scale')
    parser.add_argument(
            '-n', '--noise_scale', type=float, default=0.05,
            help='Noise output scale')

    args = parser.parse_args()
    cw_scale = args.cw_scale
    n_scale = args.noise_scale
    freq = args.frequency

    #Connect to DSIM
    corr_conf = utils.parse_ini_file(args.config, ['dsimengine'])
    dsim_conf = corr_conf['dsimengine']
    dig_host = dsim_conf['host']
    dsim_fpg = dsim_conf['bitstream']
    dhost = FpgaDsimHost(dig_host, config=dsim_conf)
    if dig_host.lower().find('skarab') != -1:
        dhost.get_system_information(filename=dsim_fpg)
    else:
        dhost.get_system_information()


    #Clear all noise sources
    sources_names = dhost.noise_sources.names()
    for source in sources_names:
        try:
            noise_source = getattr(dhost.noise_sources, '{}'.format(source))
            noise_source.set(0)
            print("noise source {}, set to {}.".format(noise_source.name,
                                                       noise_source.scale))
        except:
            print("An error occured.")
            sys.exit(1)

    #Clear all sine sources
    sources_names = dhost.sine_sources.names()
    for source in sources_names:
        try:
            sine_source = getattr(dhost.sine_sources, '{}'.format(source))
            sine_source.set(0, 0)
            print("sine source {}, set to {}.".format(sine_source.name,
                                                      sine_source.scale))
        except:
            print("An error occured.")
            sys.exit(1)

    #Set noise level
    try:
        source_from = getattr(dhost.noise_sources, 'noise_corr')
    except AttributeError:
        print("You can only select between noise sources:"
              " %s" % dhost.noise_sources.names())
        sys.exit(1)
    try:
        source_from.set(scale=n_scale)
    except ValueError:
        print("Valid scale input is between 0 - 1.")
        sys.exit(1)
    print("")
    print("noise source: %s" % source_from.name)
    print("noise scale: %s" % source_from.scale)

    # Set output scale to 1 (indivitual noise sources will be used to scale signal)
    for output_scale, output_scale_s in ([1,'0'],[1,'1']):
        scale_value = float(output_scale)
        try:
            scale_from = getattr(dhost.outputs, 'out_{}'.format(output_scale_s))
        except AttributeError:
            print("You can only select between, %s" % dhost.outputs.names())
            sys.exit(1)
        try:
            scale_from.scale_output(scale_value)
        except ValueError:
            print("Valid scale input is between 0 - 1.")
            sys.exit(1)
        """Check if it can read what was written to it!"""
        print("")
        print("output selected: %s" % scale_from.name)
        print("output scale: %s" % scale_from.scale_register.read()['data']['scale'])
    
    # Set the output source to signal
    for output_type, output_type_s in ([0,'signal'],[1,'signal']):
        try:
            type_from = getattr(dhost.outputs, 'out_{}'.format(output_type))
        except AttributeError:
            print("You can only select between, Output_0 or Output_1.")
            sys.exit(1)
        try:
            type_from.select_output(output_type_s)
        except ValueError:
            print("Valid output_type values: 'test_vectors' and 'signal'")
            sys.exit(1)
        print("")
        print("output selected: %s" % type_from.name)
        print("output type: %s" % type_from.output_type)

    # Stepping through frequencies
    sine_source = getattr(dhost.sine_sources,'sin_0')
    #sine_source = getattr(dhost.sine_sources,'sin_1')
    #sine_source = getattr(dhost.sine_sources,'sin_corr')
    try:
        sine_source.set(scale=cw_scale, frequency=freq)
    except ValueError:
        print("\nError, verify your inputs for sin_%s" % sine_source.name)
        print("Max Frequency should be {}MHz".format(
            sine_source.max_freq/1e6))
        print("Scale should be between 0 and 1")
        sys.exit(1)
    print ('Dsim cw set to {:.2f}Hz'.format(sine_source.frequency))
