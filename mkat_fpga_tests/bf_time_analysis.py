#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
SKARAB System Level Test 1
- This test takes in beamformer data for a particular F-engine and plots the spectral as well time domain response
Created on Wed Mar 14 13:56:04 2018
@author: ssalie
"""

import copy
import os
import pickle

import h5py
import matplotlib.cm as CM
import matplotlib.pylab as py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# %% Import relevant packages
# import sys
import numpy as np
import scipy as sp
from pylab import figure, hist, legend, plot, subplot, title, xlabel, xlim, ylabel, ylim
from scipy import signal

# %% Define useful functions


def clear_plots():
    plt.close("all")


clear_plots()

# Function to convert linear values to dB


def pow2dB(val_lin):
    val_dB = 10 * np.log10(val_lin)
    return val_dB


# Function to convert dB values to linear


def dB2pow(val_dB):
    val_lin = 10 ** (val_dB / 10)
    return val_lin


# %% Analyse beam data function


def analyse_beam_data(
    bf_raw,
    skarab_or_roach=True,
    do_save=False,
    dsim_settings=[54832000, 0.08, 0.05],
    cbf_settings=[8191, 110],
    spectra_use="all",
    chans_to_use=2 ** 9,
    xlim=[20, 21],
    dsim_factor=1.0,
    ref_input_label="dummy",
    bandwidth=856e6,
    n_chans=4096,
):
    """
    SKARAB System Level Test 1:
        - This function takes in a list of beam test data for a particular polarisation and analyses (displays/plots) the spectral as well as time domain data response.
        Parameters:
            bf_raw: Beamformer capture of shape[channels, num_captures, polarisations]
            skarab_or_roach: True for skarab data, False for roach data, e.g. and default skarab_or_roach = True
            do_save: True to save image plots to cd, False to display figures, e.g. and default do_save = False
            dsim_settings: list of dsim settings used in format [baseband frequency in Hz, cw signal scale, noise scael], e.g. and default dsim_settings = [54832000, 0.08, 0.05]
            cbf_settings: list of fft-shift and eq-gain settings used, e.g. and default cbf_settings = [8191, 110]
            spectra_use: choose number of spectra samples to use from the beam data, either use 'all' for all spectra samples, or aninteger value. e.g. and default spectra_use = 'all' # or 40960
            chans_to_use: integer number of channels to use from 0 - value chosen, e.g. and default chans_to_use = 2**9 (first 512 channels)
            xlim: limit x-axis in zoom plots by this range in ms, e.g. and default xlim = [20,21]
            dsim_factor: Dsim clock factor = real_dsim_sample_frequency / nominal_sample_frequency
            ref_input_label: Input for which weight was set to 1, all other input weights set to zero
        Returns:
            Plots of:
            - Waterfall plot of dataset (arb dB of magnitude data)
            - mean, median and max across frequency channels
            - Time series data output (channel with tone plus adjacent)
            - Time series data output (channel with tone plus adjacent) N.B Data time samples re-ordered
            - Time series data output, Magnitude and Phase of centre and adjacent channels
            - Time series data output, Magnitude and Phase of centre and adjacent channels, N.B Data time samples re-ordered
            - Power Spectral Density (PSD) of centre and adjacent channels for real and complex output (row 1), and reconstituted data output (row 2)
            - Power Spectral Density (PSD) of centre and adjacent channels for real and complex output (row 1), and reconstituted data output (row 2), N.B Data time samples re-ordered

            prints to screen of various system information such as, system settings, expected and achived channel of tone, channel offsets etc
    """
    # %
    if do_save:
        plt.ioff()  # dont plot to screen

    sample_clock = bandwidth * 2
    ch_bw = bandwidth / n_chans
    dsim_baseband_freq = dsim_settings[0]
    dsim_cw_scale = dsim_settings[1]
    dsim_noise_scale = dsim_settings[2]
    fft_shift = cbf_settings[0]
    re_quant_gain = cbf_settings[1]

    # print out setup info
    dsim_freq = dsim_baseband_freq + bandwidth
    print "D-sim baseband frequency: %f [Hz]" % dsim_baseband_freq
    print "D-sim frequency: %f [MHz]" % (dsim_freq / 1e6)
    print "D-sim cw scale: %f" % (dsim_cw_scale)
    print "D-sim noise scale: %f" % (dsim_noise_scale)
    print "fft shift: ", fft_shift
    fft_shift_stages = bin(int(fft_shift)).count("1")
    print "fftshift stages: ", fft_shift_stages
    print "re-quant_gain: ", re_quant_gain

    dn0 = np.asarray(bf_raw[:chans_to_use, :])
    Feng_data = dn0[:, :, :]

    # Create a time vector array
    t0_or = range(0, len(Feng_data[0, :, 0]))
    t0_or = np.asarray(t0_or)
    t0_or = t0_or * (n_chans / bandwidth)

    try:
        nsamp_use = range(0, len(t0_or[:spectra_use]))
        print "spectra_use no. samples in range"
    except BaseException:
        nsamp_use = range(0, len(t0_or))
        print "spectra_use no. samples not in in range, using full spectra length"

    print "choosing approximately %.5f seconds of data (NB, this is the extent of the data file)" % (
        len(nsamp_use) * t0_or[1]
    )
    print "total spectra samples: %d " % len(nsamp_use)
    print "total spectra time: %.4f [seconds]" % (len(nsamp_use) * t0_or[1])

    t0 = t0_or[nsamp_use]
    dn_lim0 = copy.deepcopy(Feng_data[:, nsamp_use, :])
    dn_lim_real0 = copy.deepcopy(dn_lim0[:, :, 0])
    dn_lim_imag0 = copy.deepcopy(dn_lim0[:, :, 1])
    dn_lim_cmplx0 = np.zeros([len(dn_lim0[:, 0, 0]), len(dn_lim0[0, :, 0])])

    # print out info
    Bw = np.float(bandwidth / 1e6)
    print "-Bandwidth [MHz] ", Bw

    # F_b = np.float(1284)
    F_b = np.float(Bw / 2.0 + Bw)
    print "-F centre beamformer frequency [MHz] ", F_b

    Sg = dsim_freq / 1e6
    print "-Simulated (Dsim) SG Signal generator frequency [MHz] ", Sg

    freqs_or = tmp = np.linspace(bandwidth, sample_clock - ch_bw, n_chans)
    bw_each = np.average(np.diff(tmp))
    print "-Channel bandwidth [KHz] ", bw_each
    freqs = np.arange(bandwidth, bandwidth + ((len(dn_lim_cmplx0[:, 0])) * (bw_each)), bw_each)

    idx_cnt_channel = np.argmin(np.abs(freqs - (Sg * 1e6)))
    print "centre channel number integer", idx_cnt_channel

    # Calculate and print the expected in channel tone frequeencies
    channel = (Sg - Bw) * n_chans / Bw + 0.5

    base_f_minus_one_half = Sg - Bw - (np.floor(channel) - 1.5) * (Bw) / n_chans
    base_f_minus_one = Sg - Bw - (np.floor(channel) - 1) * (Bw) / n_chans
    base_f_minus_half = Sg - Bw - (np.floor(channel) - 0.5) * (Bw) / n_chans
    base_f = Sg - Bw - (np.floor(channel)) * (Bw) / n_chans
    base_f_plus_half = Sg - Bw - (np.floor(channel) + 0.5) * (Bw) / n_chans
    base_f_plus_one = Sg - Bw - (np.floor(channel) + 1) * (Bw) / n_chans
    base_f_plus_one_half = Sg - Bw - (np.floor(channel) + 1.5) * (Bw) / n_chans

    print "centre freq channel number full, ", channel
    print "################################"

    print "base_f_minus_one_half channel [kHz]___: ", base_f_minus_one_half * 1e3
    print "base_f_minus_one channel [kHz]________: ", base_f_minus_one * 1e3
    print "base_f_minus_half channel [kHz]_______: ", base_f_minus_half * 1e3
    print "base_f channel [kHz]__________________: ", base_f * 1e3
    print "base_f_plus_half channel [kHz]________: ", base_f_plus_half * 1e3
    print "base_f_plus_one channel [kHz]_________: ", base_f_plus_one * 1e3
    print "base_f_plus_one_half channel [kHz]____: ", base_f_plus_one_half * 1e3

    # % creating the complex matrix
    # create complex beamformer output data for H pol
    dn_lim_cmplx0 = dn_lim_real0 + dn_lim_imag0 * 1j

    # % create frequency channel vector
    # frequency channel numbers
    freq_chans_plt = range(0, len(dn_lim_cmplx0[:, 0]))

    # % do waterfall plot
    a_r = "auto"

    figure(figsize=(20, 12))
    plt.imshow(pow2dB(np.abs(dn_lim_cmplx0)), cmap=CM.jet, aspect=a_r, alpha=1)
    plt.colorbar()
    title("Waterfall plot of dataset (arb dB of magnitude data). Dsim @%f MHz\nInput: %s" % (Sg, ref_input_label))
    xlabel("Spectra Time sample")
    ylabel("Frequency channel number for this F-eng subband")
    if do_save:
        plt.savefig(ref_input_label + "_fengChunk-" + "_waterfall.png")
        plt.close()

    figure(figsize=(20, 12))
    plt.plot(freq_chans_plt, pow2dB(np.mean(np.abs(dn_lim_cmplx0), axis=1)), "-", label="mean")
    plt.plot(freq_chans_plt, pow2dB(np.median(np.abs(dn_lim_cmplx0), axis=1)), "-^", label="median")
    plt.plot(freq_chans_plt, pow2dB(np.max(np.abs(dn_lim_cmplx0), axis=1)), "-x", label="max")
    plt.plot(freq_chans_plt, pow2dB(np.min(np.abs(dn_lim_cmplx0), axis=1)), "-o", label="min")
    plt.legend()
    plt.grid()
    plt.title("mean, median and max across frequency channels. Dsim @%f MHz\nInput: %s" % (Sg, ref_input_label))
    plt.xlabel("Channel number")
    plt.ylabel("Raw voltage output [dBW]")

    if do_save:
        plt.savefig(ref_input_label + "_fengChunk-" + "_across_freq_view.png")
        plt.close()

    # find the channel where the tone is most likely to be present
    max_channel = np.argmax(np.abs(dn_lim_cmplx0).sum(axis=1))
    print "tone is located at channel number: ", max_channel
    dn_lim_cmplx0_copy = dn_lim_cmplx0  # create copy of data for reset purposes

    # %
    # import IPython;IPython.embed()
    for cond in (True, False):
        reset_data = True
        if reset_data:
            dn_lim_cmplx0 = copy.deepcopy(dn_lim_cmplx0_copy)

        if skarab_or_roach:
            test_sig = dn_lim_cmplx0[max_channel, 0:8]
        if not (skarab_or_roach):
            test_sig = dn_lim_cmplx0[max_channel, 0:2]

        test_sig = np.asarray(test_sig)
        test_sig = np.atleast_2d(test_sig)

        do_flip = cond  # flip samples to correct order
        if do_flip:
            num_chan = len(dn_lim_cmplx0[:, 0])
            num_spectra = len(dn_lim_cmplx0[0, :])
            if skarab_or_roach:
                samps_2_shift = 8
                blocks_2_shift = 2
                block_size = samps_2_shift
            if not (skarab_or_roach):
                samps_2_shift = 2
            dn_lim_cmplx0 = np.reshape(dn_lim_cmplx0, [num_chan, num_spectra / samps_2_shift, samps_2_shift])
            dn_lim_cmplx0 = np.fliplr(dn_lim_cmplx0)
            dn_lim_cmplx0 = np.reshape(dn_lim_cmplx0, [num_chan, num_spectra])
            dn_lim_cmplx0 = np.fliplr(dn_lim_cmplx0)
            dn_lim_cmplx0 = np.squeeze(dn_lim_cmplx0)
            if skarab_or_roach is None:
                truncate_spectra = (num_spectra / blocks_2_shift / block_size) * blocks_2_shift * block_size
                dn_lim_cmplx0 = dn_lim_cmplx0[:, :truncate_spectra]
                dn_lim_cmplx0 = dn_lim_cmplx0.reshape(
                    num_chan, num_spectra / block_size / blocks_2_shift, blocks_2_shift, block_size
                )
                dn_lim_cmplx0 = np.flip(dn_lim_cmplx0, axis=2)
                dn_lim_cmplx0 = dn_lim_cmplx0.reshape(num_chan, truncate_spectra)
                dn_lim_cmplx0 = np.squeeze(dn_lim_cmplx0)

        if skarab_or_roach:
            test_sig_after_reorder = dn_lim_cmplx0[max_channel, 0:8]
        if not (skarab_or_roach):
            test_sig_after_reorder = dn_lim_cmplx0[max_channel, 0:2]
        test_sig_after_reorder = np.asarray(test_sig_after_reorder)
        test_sig_after_reorder = np.atleast_2d(test_sig_after_reorder)
        test_sig_after_reorder = np.fliplr(test_sig_after_reorder)
        if cond:
            if np.array_equal(test_sig, test_sig_after_reorder):
                print "True:  re-order correct"
            else:
                print "False:  re-order incorrect"

        # create reconstituted data
        reconstituted = 1 * (dn_lim_cmplx0 + np.conjugate(dn_lim_cmplx0))

        if idx_cnt_channel != max_channel:
            print "expected cw tone channel NOT equal to achieved, channel offset by: %d" % np.abs(
                idx_cnt_channel - max_channel
            )
            idx_cnt_channel = max_channel  # mod for checking roach data SS 19 Jan 2018
        else:
            print "expected cw tone channel equal to achieved"

        # identify centre, adjacent left and adjacent right locations
        cnt_l = idx_cnt_channel - 1
        cnt = idx_cnt_channel
        cnt_r = idx_cnt_channel + 1

        # % time domain series plots
        f0 = figure(figsize=(20.5, 14.5))
        a0 = plt.subplot2grid((3, 2), (0, 0))
        for i in [cnt_l, cnt, cnt_r]:
            a0.plot(t0 * 1e3, np.real(dn_lim_cmplx0[i, :]), label="%d_real" % i)

        a0.legend()
        a0.set_title("Real component of centre and adjacent channels")
        a0.set_ylabel("Raw voltage output")
        a0.grid()

        az0 = plt.subplot2grid((3, 2), (0, 1))
        for i in [cnt_l, cnt, cnt_r]:
            az0.plot(t0 * 1e3, np.real(dn_lim_cmplx0[i, :]), label="%d_real" % i)

        az0.legend()
        az0.set_title("(zoomed), Real component of centre and adjacent channels")
        az0.set_ylabel("Raw voltage output")
        az0.set_xlim(xlim)
        az0.grid()

        a1 = plt.subplot2grid((3, 2), (1, 0))
        for i in [cnt_l, cnt, cnt_r]:
            a1.plot(t0 * 1e3, np.imag(dn_lim_cmplx0[i, :]), label="%d_imag" % i)

        a1.legend()
        a1.set_title("Imaginary component of centre and adjacent channels")
        a1.set_ylabel("Raw voltage output")
        a1.grid()

        az1 = plt.subplot2grid((3, 2), (1, 1))
        for i in [cnt_l, cnt, cnt_r]:
            az1.plot(t0 * 1e3, np.imag(dn_lim_cmplx0[i, :]), label="%d_imag" % i)

        az1.legend()

        az1.set_title("(zoomed), Imaginary component of centre and adjacent channels")
        az1.set_ylabel("Raw voltage output")
        az1.set_xlim(xlim)
        az1.grid()

        a2 = plt.subplot2grid((3, 2), (2, 0))
        for i in [cnt_l, cnt, cnt_r]:
            a2.plot(t0 * 1e3, np.real(reconstituted[i, :]), label="%d_recon_real" % i)

        a2.legend()
        a2.set_title("Reconstituted real signal of centre and adjacent channels")
        a2.set_xlabel("Time [ms]")
        a2.set_ylabel("Raw voltage output")
        a2.grid()

        az2 = plt.subplot2grid((3, 2), (2, 1))
        for i in [cnt_l, cnt, cnt_r]:
            az2.plot(t0 * 1e3, np.real(reconstituted[i, :]), label="%d_recon_real" % i)

        az2.legend()

        az2.set_title("(zoomed), Reconstituted real signal of centre and adjacent channels")
        az2.set_xlabel("Time [ms]")
        az2.set_ylabel("Raw voltage output")
        az2.set_xlim(xlim)
        az2.grid()

        if cond:
            plt.suptitle("Time series data output (channel with tone plus adjacent) N.B Data time samples re-ordered")
        if not (cond):
            plt.suptitle("Time series data output (channel with tone plus adjacent)")

        if do_save:
            if cond:
                plt.savefig(ref_input_label + "_fengChunk-" + "_time_series_reordered.png")
            if not (cond):
                plt.savefig(ref_input_label + "_fengChunk-" + "_time_series_original.png")
            plt.close()

        # % Plot magnitude and phase
        fh0 = figure(figsize=(20.5, 14.5))
        ah0 = plt.subplot2grid((2, 2), (0, 0))
        for i in [cnt]:  # [cnt_l,cnt,cnt_r]:
            abs_vals = np.abs(dn_lim_cmplx0[i, :])
            ah0.plot(t0 * 1e3, pow2dB(abs_vals), label="%d_magnitude" % i)

        ah0.legend()
        ah0.set_title("Magnitude signal of centre channel")
        ah0.set_xlabel("Time [ms]")
        ah0.set_ylabel("Raw voltage output [dB]")
        ah0.grid()

        ahz0 = plt.subplot2grid((2, 2), (0, 1))
        for i in [cnt]:  # [cnt_l,cnt,cnt_r]:
            ahz0.plot(t0 * 1e3, pow2dB(np.abs(dn_lim_cmplx0[i, :])), label="%d_magnitude" % i)

        ahz0.legend()
        ahz0.set_title("(zoomed), Magnitude signal of centre channel")
        ahz0.set_xlabel("Time [ms]")
        ahz0.set_ylabel("Raw voltage output [dB]")
        ahz0.set_xlim(xlim)
        ahz0.grid()

        ah1 = plt.subplot2grid((2, 2), (1, 0))
        for i in [cnt]:  # [cnt_l,cnt,cnt_r]:
            phase = np.angle(dn_lim_cmplx0[i, :], deg=1)
            ah1.plot(t0 * 1e3, phase, label="%d_phase" % i)

        ah1.legend()
        ah1.set_title("Phase signal of centre channel")
        ah1.set_xlabel("Time [ms]")
        ah1.set_ylabel("Phase [deg]")
        ah1.grid()

        ahz1 = plt.subplot2grid((2, 2), (1, 1))
        for i in [cnt]:  # [cnt_l,cnt,cnt_r]:
            phase = np.angle(dn_lim_cmplx0[i, :], deg=1)
            ahz1.plot(t0 * 1e3, phase, "x-", label="%d_phase" % i)

        ahz1.legend()
        ahz1.set_title("(zoomed), Phase signal of centre channel")
        ahz1.set_xlabel("Time [ms]")
        ahz1.set_ylabel("Phase [deg]")
        ahz1.set_xlim(xlim)
        ahz1.grid()

        if not (cond):
            plt.suptitle("Time series data output, Magnitude and Phase of centre and adjacent channels")
        if cond:
            plt.suptitle(
                "Time series data output, Magnitude and Phase of centre and adjacent channels, N.B Data time samples re-ordered"
            )

        if do_save:
            if not (cond):
                plt.savefig(ref_input_label + "_fengChunk-" + "_mag_phase_original.png")
            if cond:
                plt.savefig(ref_input_label + "_fengChunk-" + "_mag_phase_reordered.png")
            plt.close()

        # % Frequency domain plots
        # first reduce spectra data to only 8192 samples
        nsamp_spectra_use = range(2 ** 10, (2 ** 10) + (2 ** 13))  # approximately 50ms
        dn_lim_cmplx0 = dn_lim_cmplx0[:, nsamp_spectra_use]

        fs_txt = 8
        y_lim = [-80, 50]
        fs = 1 / (t0_or[1] - t0_or[0])
        print "fs = ", fs
        fc = 0  # fs/2.; #0
        y_ticks = np.arange(-100, 10, 20)

        cnt_l = idx_cnt_channel - 1
        cnt_c = idx_cnt_channel
        cnt_r = idx_cnt_channel + 1

        # adjacent left channel
        f = figure(figsize=(20.5, 12.5))

        ax0 = plt.subplot2grid((2, 3), (0, 0))
        ax0.psd(np.real(dn_lim_cmplx0[cnt_l, :]), Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="real_lft")
        ax0.psd(dn_lim_cmplx0[cnt_l, :], Fs=fs, NFFT=len(t0), label="complex_lft", linestyle="--", marker="x")
        ax0.legend(loc="lower right")
        ax0.set_title("(adj left channel),real and complex")
        ax0.set_ylim(y_lim)
        ax0.yaxis.set_ticks(y_ticks)
        ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        # centre channel
        ax1 = plt.subplot2grid((2, 3), (0, 1))
        ax1.psd(np.real(dn_lim_cmplx0[cnt_c, :]), Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="real_cnt")
        ax1.psd(dn_lim_cmplx0[cnt_c, :], Fs=fs, NFFT=len(t0), label="complex_cnt", linestyle="--", marker="x")
        ax1.legend(loc="lower right")
        ax1.set_title("(centre channel),real and complex")
        ax1.yaxis.set_ticks(y_ticks)
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        # adjacent right channel
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax2.psd(np.real(dn_lim_cmplx0[cnt_r, :]), Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="real_rht")
        ax2.psd(dn_lim_cmplx0[cnt_r, :], Fs=fs, NFFT=len(t0), label="complex_rht", linestyle="--", marker="x")
        ax2.legend(loc="lower right")
        ax2.set_title("(adj right channel),real and complex")
        ax2.set_ylim(y_lim)
        ax2.yaxis.set_ticks(y_ticks)
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        # reconstituted
        # adjacent left channel
        ax3 = plt.subplot2grid((2, 3), (1, 0))
        tmp = ax3.psd(reconstituted[cnt_l, :], Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="reconstituted_lft")
        ax3.legend(loc="lower right")
        ax3.set_title("(adj left channel),reconstituted data")
        ax3.set_ylim(y_lim)
        ax3.yaxis.set_ticks(y_ticks)
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        # centre channel
        ax4 = plt.subplot2grid((2, 3), (1, 1))
        tmp = ax4.psd(reconstituted[cnt_c, :], Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="reconstituted_cnt")
        ax4.legend(loc="lower right")
        ax4.set_title("(centre channel),reconstituted data")
        ax4.set_ylim(y_lim)
        ax4.yaxis.set_ticks(y_ticks)
        ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        # adjacent right channel
        ax5 = plt.subplot2grid((2, 3), (1, 2))
        tmp = ax5.psd(reconstituted[cnt_r, :], Fs=fs, Fc=fc, sides="twosided", NFFT=len(t0), label="reconstituted_rht")
        ax5.legend(loc="lower right")
        ax5.set_title("(adj right channel),reconstituted data")
        ax5.set_ylim(y_lim)
        ax5.yaxis.set_ticks(y_ticks)
        ax5.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))

        if cond:
            plt.suptitle(
                "Power Spectral Density (PSD) of centre and adjacent channels for real and complex output (row 1), and reconstituted data output (row 2), N.B Data time samples re-ordered"
            )
        if not (cond):
            plt.suptitle(
                "Power Spectral Density (PSD) of centre and adjacent channels for real and complex output (row 1), and reconstituted data output (row 2)"
            )

        if do_save:
            if not (cond):
                plt.savefig(ref_input_label + "_fengChunk-" + "_PSD_original.png")
            if cond:
                plt.savefig(ref_input_label + "_fengChunk-" + "_PSD_reordered.png")
            plt.close()


# %%
if __name__ == "__main__":
    # Load the .npy beam data file

    fname = [
        "bf_raw_1000.np.npy",
        "bf_raw_0100.np.npy",
        "bf_raw_0010.np.npy",
        "bf_raw_0001.np.npy",
    ]  # place all beam data files in list to analyse
    path = "./"  # path to files
    feng_num = 3  # choose which one in fname to plot
    spectra_use = 40960  # or 'all'
    # file_used = path+ fname[feng_num]
    file_used = "skarab_bf_data01.np.npy"
    bf_raw = np.load(file_used)
    analyse_beam_data(
        bf_raw,
        skarab_or_roach=True,
        do_save=True,
        dsim_settings=[13372910.15625, 0.675, 0.45],
        cbf_settings=[8191, 5],
        spectra_use="all",
        chans_to_use=2 ** 9,
        xlim=[50, 52],
        dsim_factor=1.0,
        ref_input_label="dummy",
    )
