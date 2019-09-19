#!/usr/bin/python
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Calculate baseband center frequencies for 1k, 4k and 32k channels.')
    parser.add_argument(
            '-c', '--channel', type=int, default=100,
            help='Channel for which to display baseband center freqency.')
    parser.add_argument(
            '-f', '--sample_freq', type=float, default=1714298408,
            help='Sample frequency, default: 1714298408 Hz')
    parser.add_argument(
            '-n', '--nominal_sample_freq', type=float, default=1712e6,
            help='Nominal sample frequency, default: 1712000000 Hz')
    parser.add_argument(
            '-b', '--narrow_band_centre_freq', type=float,
            help='Narrow band centre frequency, default: half the bandwidth')

    args = parser.parse_args()
    chan_1k = float(2**10)
    chan_4k = float(2**12)
    chan_32k = float(2**15)
    sample_freq = float(args.sample_freq)
    nom_smpl_freq = float(args.nominal_sample_freq)
    nb_bw = 107e6 * sample_freq/nom_smpl_freq
    if args.narrow_band_centre_freq == None:
        nb_c_f = sample_freq/4
    else:
        nb_c_f = float(args.narrow_band_centre_freq)


    freqs_1k  = np.arange(chan_1k)*sample_freq/2/chan_1k
    freqs_4k  = np.arange(chan_4k)*sample_freq/2/chan_4k
    freqs_32k = np.arange(chan_32k)*sample_freq/2/chan_32k
    freqs_32k_nb = (np.arange(chan_32k)*nb_bw/chan_32k) + (nb_c_f - nb_bw/2)

    print ('Sample frequency set to: {} Hz'.format(sample_freq))
    try:
        print ('1k center frequency for channel {}: {} Hz'.format(args.channel, freqs_1k[args.channel]))
    except IndexError:
        pass
    try:
        print ('4k center frequency for channel {}: {} Hz'.format(args.channel, freqs_4k[args.channel]))
    except IndexError:
        pass
    try:
        print ('32k center frequency for channel {}: {} Hz'.format(args.channel, freqs_32k[args.channel]))
    except IndexError:
        pass
    try:
        print ('32k narrow band center frequency for channel {}: {} Hz'.format(args.channel, freqs_32k_nb[args.channel]))
    except IndexError:
        pass
