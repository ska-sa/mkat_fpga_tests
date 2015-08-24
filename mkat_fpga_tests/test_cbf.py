from __future__ import division

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

import corr2.fxcorrelator_fengops as fengops
import corr2.fxcorrelator_xengops as xengops

from corr2 import utils

from mkat_fpga_tests import correlator_fixture
from mkat_fpga_tests.utils import normalised_magnitude, loggerise, complexise
from mkat_fpga_tests.utils import init_dsim_sources, get_dsim_source_info
from mkat_fpga_tests.utils import nonzero_baselines, zero_baselines, all_nonzero_baselines
from mkat_fpga_tests.utils import CorrelatorFrequencyInfo, TestDataH5
from mkat_fpga_tests.utils import get_snapshots
from mkat_fpga_tests.utils import set_coarse_delay

LOGGER = logging.getLogger(__name__)

DUMP_TIMEOUT = 10              # How long to wait for a correlator dump to arrive in tests

def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0, figure out if VACC is roated by 1"""
    b0 = np.abs(complexise(xeng_raw[:,0]))
    b1 = np.abs(complexise(xeng_raw[:,1]))
    if np.max(b0) > 0 and np.max(b1) == 0:
        # We expect autocorr in baseline 0 to be nonzero if the vacc is properly aligned,
        # hence no offset
        return 0
    elif np.max(b1) > 0 and np.max(b0) == 0:
        return 1
    else:
        raise ValueError('Could not determine VACC offset')

class test_CBF(unittest.TestCase):
    def setUp(self):
        self.receiver = CorrRx(port=8888)
        start_thread_with_cleanup(self, self.receiver, start_timeout=1)
        self.correlator = correlator_fixture.correlator
        self.corr_fix = correlator_fixture
        self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
        dsim_conf = self.correlator.configd['dsimengine']
        dig_host = dsim_conf['host']
        self.dhost = FpgaDsimHost(dig_host, config=dsim_conf)
        self.dhost.get_system_information()
        # Increase the dump rate so tests can run faster
        xengops.xeng_set_acc_time(self.correlator, 0.2)
        # Remove once JasonM has fixed the vacc_rsync in corr2 package
        xengops.xeng_vacc_sync(self.correlator)
        self.addCleanup(self.corr_fix.stop_x_data)
        self.corr_fix.start_x_data()
        self.corr_fix.issue_metadata()
        # Threshold: -70dB
        self.threshold = 1e-7

    # TODO 2015-05-27 (NM) Do test using get_vacc_offset(test_dump['xeng_raw']) to see if
    # the VACC is rotated. Run this test first so that we know immediately that other
    # tests will be b0rked.
    def test_channelisation(self):
        """(TP.C.1.19) CBF Channelisation Wideband Coarse L-band"""
        test_name = '{}.{}'.format(strclass(self.__class__), self._testMethodName)
        test_data_h5 = TestDataH5(test_name + '.h5')
        self.addCleanup(test_data_h5.close)
        test_chan = 1500

        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=101, chans_around=2)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]

        def get_fftoverflow_qdrstatus():
            fhosts = {}
            xhosts = {}
            dicts = {}
            dicts['fhosts'] = {}
            dicts['xhosts'] = {}
            fengs = self.correlator.fhosts
            xengs = self.correlator.xhosts
            for fhost in fengs:
                fhosts[fhost.host] = {}
                fhosts[fhost.host]['QDR_okay'] = fhost.qdr_okay()
                for pfb, value in fhost.registers.pfb_ctrs.read()['data'].iteritems():
                    fhosts[fhost.host][pfb] = value
                for xhost in xengs:
                    xhosts[xhost.host] = {}
                    xhosts[xhost.host]['QDR_okay'] = xhost.qdr_okay()
            dicts['fhosts'] = fhosts
            dicts['xhosts'] = xhosts
            return dicts

        init_dsim_sources(self.dhost)
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=0.25)
        # Put some noise on output
        # self.dhost.noise_sources.noise_0.set(scale=1e-3)
        # The signal source is going to quantise the requested freqency, so see what we
        # actually got
        source_fc = self.dhost.sine_sources.sin_0.frequency
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0

        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None

        def get_pfb_counts(status_dict):
            pfb_list = {}
            for host, pfb_value in status_dict:
                pfb_list[host] = (pfb_value['pfb_of0_cnt'],
                    pfb_value['pfb_of1_cnt'])
            return pfb_list

        last_pfb_counts = get_pfb_counts(
            get_fftoverflow_qdrstatus()['fhosts'].items())

        QDR_error_roaches = set()
        def test_fftoverflow_qdrstatus():
            fftoverflow_qdrstatus = get_fftoverflow_qdrstatus()
            curr_pfb_counts = get_pfb_counts(
                fftoverflow_qdrstatus['fhosts'].items())
            # Test FFT Overflow status
            self.assertEqual(last_pfb_counts, curr_pfb_counts)
            # Test QDR error flags
            for hosts_status in fftoverflow_qdrstatus.values():
                for host, hosts_status in hosts_status.items():
                    if hosts_status['QDR_okay'] is False:
                        QDR_error_roaches.add(host)
            # Test QDR status
            self.assertFalse(QDR_error_roaches)

        # Test fft overflow and qdr status before
        test_fftoverflow_qdrstatus()

        for i, freq in enumerate(requested_test_freqs):
            # LOGGER.info('Getting channel response for freq {}/{}: {} MHz.'.format(
            #     i+1, len(requested_test_freqs), freq/1e6))
            print ('Getting channel response for freq {}/{}: {} MHz.'.format(
                i+1, len(requested_test_freqs), freq/1e6))

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            if this_source_freq == last_source_freq:
                LOGGER.info('Skipping channel response for freq {}/{}: {} MHz.\n'
                            'Digitiser frequency is same as previous.'.format(
                                i+1, len(requested_test_freqs), freq/1e6))
                continue    # Already calculated this one
            else:
                last_source_freq = this_source_freq

            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            try:
                snapshots = get_snapshots(self.correlator)
            except Exception:
                print ("Error retrieving snapshot at {}/{}: {} MHz.\n".format(
                    i+1, len(requested_test_freqs), freq/1e6))
                LOGGER.exception("Error retrieving snapshot at {}/{}: {} MHz.".format(
                    i+1, len(requested_test_freqs), freq/1e6))
                if i == 0:
                    # The first snapshot must work properly to give us the data structure
                    raise
                else:
                    snapshots['all_ok'] = False
            else:
                snapshots['all_ok'] = True
            source_info = get_dsim_source_info(self.dhost)
            test_data_h5.add_result(this_freq_dump, source_info, snapshots)
            this_freq_data = this_freq_dump['xeng_raw']
            this_freq_response = normalised_magnitude(
                this_freq_data[:, test_baseline, :])
            actual_test_freqs.append(this_source_freq)
            chan_responses.append(this_freq_response)

        # Test fft overflow and qdr status after
        test_fftoverflow_qdrstatus()
        self.corr_fix.stop_x_data()
        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)

        def plot_and_save(freqs, data, plot_filename, show=False):
            df = self.corr_freqs.delta_f
            fig = plt.plot(freqs, data)[0]
            axes = fig.get_axes()
            ybound = axes.get_ybound()
            yb_diff = abs(ybound[1] - ybound[0])
            new_ybound = [ybound[0] - yb_diff*1.1, ybound[1] + yb_diff * 1.1]
            plt.vlines(expected_fc, *new_ybound, colors='r', label='chan fc')
            plt.vlines(expected_fc - df / 2, *new_ybound, label='chan min/max')
            plt.vlines(expected_fc - 0.8*df / 2, *new_ybound, label='chan +-40%',
                       linestyles='dashed')
            plt.vlines(expected_fc + df / 2, *new_ybound, label='_chan max')
            plt.vlines(expected_fc + 0.8*df / 2, *new_ybound, label='_chan +40%',
                       linestyles='dashed')
            plt.legend()
            plt.title('Channel {} ({} MHz) response'.format(
                test_chan, expected_fc/1e6))
            axes.set_ybound(*new_ybound)
            plt.grid(True)
            plt.ylabel('dB relative to VACC max')
            # TODO Normalise plot to frequency bins
            plt.xlabel('Frequency (Hz)')
            plt.savefig(plot_filename)
            if show:
                plt.show()
            plt.close()

        graph_name_all = test_name + '.channel_response.svg'
        plot_data_all  = loggerise(chan_responses[:, test_chan], dynamic_range=90)
        plot_and_save(actual_test_freqs, plot_data_all, graph_name_all)

        # Get responses for central 80% of channel
        df = self.corr_freqs.delta_f
        central_indices = (
            (actual_test_freqs <= expected_fc + 0.4*df) &
            (actual_test_freqs >= expected_fc - 0.4*df))
        central_chan_responses = chan_responses[central_indices]
        central_chan_test_freqs = actual_test_freqs[central_indices]

        graph_name_central = test_name + '.channel_response_central.svg'
        plot_data_central  = loggerise(central_chan_responses[:, test_chan], dynamic_range=90)
        plot_and_save(central_chan_test_freqs, plot_data_central, graph_name_central)

        # Test responses in central 80% of channel
        for i, freq in enumerate(central_chan_test_freqs):
            max_chan = np.argmax(np.abs(central_chan_responses[i]))
            self.assertEqual(max_chan, test_chan, 'Source freq {} peak not in channel '
                             '{} as expected but in {}.'
                             .format(freq, test_chan, max_chan))

        self.assertLess(
            np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
            'VACC output at > 99% of maximum value, indicates that '
            'something, somewhere, is probably overranging.')
        max_central_chan_response = np.max(10*np.log10(central_chan_responses[:, test_chan]))
        min_central_chan_response = np.min(10*np.log10(central_chan_responses[:, test_chan]))
        chan_ripple = max_central_chan_response - min_central_chan_response
        acceptable_ripple_lt = 0.3

        self.assertLess(chan_ripple, acceptable_ripple_lt,
                        'ripple {} dB within 80% of channel fc is >= {} dB'
                        .format(chan_ripple, acceptable_ripple_lt))

        # from matplotlib import pyplot
        # colour_cycle = 'rgbyk'
        # style_cycle = ['-', '--']
        # linestyles = itertools.cycle(itertools.product(style_cycle, colour_cycle))
        # for i, freq in enumerate(actual_test_freqs):
        #     style, colour = linestyles.next()
        #     pyplot.plot(loggerise(chan_responses[:, i], dynamic_range=60), color=colour, ls=style)
        # pyplot.ion()
        # pyplot.show()


    def test_product_baselines(self):
        """(TP.C.1.30) CBF Baseline Correlation Products - AR1"""

        init_dsim_sources(self.dhost)
        # Put some correlated noise on both outputs
        self.dhost.noise_sources.noise_corr.set(scale=0.5)
        test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)

        # Get list of all the correlator input labels
        input_labels = sorted(tuple(test_dump['input_labelling'][:,0]))
        # Get list of all the baselines present in the correlator output
        bls_ordering = test_dump['bls_ordering']
        baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(
            bls_ordering)}
        present_baselines = sorted(baseline_lookup.keys())

        # Make a list of all possible baselines (including redundant baselines) for the
        # given list of inputs
        possible_baselines = set()
        for li in input_labels:
            for lj in input_labels:
                possible_baselines.add((li, lj))

        test_bl = sorted(list(possible_baselines))
        # Test that each baseline (or its reverse-order counterpart) is present in the
        # correlator output
        baseline_is_present = {}

        for test_bl in possible_baselines:
           baseline_is_present[test_bl] = (test_bl in present_baselines or
                                           test_bl[::-1] in present_baselines)
        self.assertTrue(all(baseline_is_present.values()),
                        "Not all baselines are present in correlator output.")

        test_data = test_dump['xeng_raw']
        # Expect all baselines and all channels to be non-zero
        self.assertFalse(zero_baselines(test_data))
        self.assertEqual(nonzero_baselines(test_data),
                         all_nonzero_baselines(test_data))

        # Save initial f-engine equalisations
        initial_equalisations = {input: eq_info['eq'] for input, eq_info
                                in fengops.feng_eq_get(self.correlator).items()}
        def restore_initial_equalisations():
            for input, eq in initial_equalisations.items():
                fengops.feng_eq_set(self.correlator, source_name=input, new_eq=eq)
        self.addCleanup(restore_initial_equalisations)

        # Set all inputs to zero, and check that output product is all-zero
        for input in input_labels:
            fengops.feng_eq_set(self.correlator, source_name=input, new_eq=0)
        test_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw']
        self.assertFalse(nonzero_baselines(test_data))
        #-----------------------------------
        all_inputs = sorted(set(input_labels))
        zero_inputs = set(input_labels)
        nonzero_inputs = set()

        def calc_zero_and_nonzero_baselines(nonzero_inputs):
            nonzeros = set()
            zeros = set()
            for inp_i in all_inputs:
                for inp_j in all_inputs:
                    if (inp_i, inp_j) not in baseline_lookup:
                        continue
                    if inp_i in nonzero_inputs and inp_j in nonzero_inputs:
                        nonzeros.add((inp_i, inp_j))
                    else:
                        zeros.add((inp_i, inp_j))
            return zeros, nonzeros

        for inp in input_labels:
            old_eq = initial_equalisations[inp]
            fengops.feng_eq_set(self.correlator, source_name=inp, new_eq=old_eq)
            zero_inputs.remove(inp)
            nonzero_inputs.add(inp)
            expected_z_bls, expected_nz_bls = (
                calc_zero_and_nonzero_baselines(nonzero_inputs))
            test_data = self.receiver.get_clean_dump()['xeng_raw']
            actual_nz_bls_indices = all_nonzero_baselines(test_data)
            actual_nz_bls = set(tuple(bls_ordering[i])
                for i in actual_nz_bls_indices)
            actual_z_bls_indices = zero_baselines(test_data)
            actual_z_bls = set(tuple(bls_ordering[i])
                for i in actual_z_bls_indices)

            self.assertEqual(actual_nz_bls, expected_nz_bls)
            self.assertEqual(actual_z_bls, expected_z_bls)

    def test_back2back_consistency(self):
        """1. Check that back-to-back dumps with same input are equal"""
        test_name = '{}.{}'.format(strclass(self.__class__), self._testMethodName)
        init_dsim_sources(self.dhost)
        test_chan = 1500

        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=0.25)

        for i, freq in enumerate(requested_test_freqs):
            print ('Testing dump consistancy {}/{} @ {} MHz.'.format(
                i+1, len(requested_test_freqs), freq/1e6))
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
            dumps_data = []
            for dump_no in range(3):
                if dump_no == 0:
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    initial_max_freq = np.max(this_freq_dump['xeng_raw'])
                else:
                    this_freq_dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
                this_freq_data = this_freq_dump['xeng_raw']
                dumps_data.append(this_freq_data)

            diff_dumps = []
            for comparison in range(1, len(dumps_data)):
                d0 = dumps_data[0]
                d1 = dumps_data[comparison]
                diff_dumps.append(np.max(d0 - d1))

            dumps_comp = np.max(np.array(diff_dumps)/initial_max_freq)
            self.assertLess(dumps_comp, self.threshold,
                'dump comparison ({}) is >= {} threshold[dB].'
                    .format(dumps_comp, self.threshold))

    def test_freq_scan_consistency(self):
        """2. Check that identical frequency scans produce equal results"""
        test_name = '{}.{}'.format(strclass(self.__class__), self._testMethodName)
        init_dsim_sources(self.dhost)
        test_chan = 1500

        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=0.25)
        init_dsim_sources(self.dhost)

        scans = []
        initial_max_freq_list = []
        for scan_i in range(3):
            scan_dumps = []
            scans.append(scan_dumps)
            for i, freq in enumerate(requested_test_freqs):
                #print ('{} of {}: Testing frequency scan consistancy {}/{} @ {} MHz.'.format(
                #scan_i+1, len(range(3)), i+1, len(requested_test_freqs), freq/1e6))
                if scan_i == 0:
                    self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    initial_max_freq = np.max(this_freq_dump['xeng_raw'])
                    this_freq_data = this_freq_dump['xeng_raw']
                    initial_max_freq_list.append(initial_max_freq)
                else:
                    self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    this_freq_data = this_freq_dump['xeng_raw']
                scan_dumps.append(this_freq_data)

        for scan_i in range(1, len(scans)):
            for freq_i in range(len(scans[0])):
                s0 = scans[0][freq_i]
                s1 = scans[scan_i][freq_i]
                norm_fac = initial_max_freq_list[freq_i]

                self.assertLess(np.max(np.abs(s1 - s0))/norm_fac, self.threshold,
                    'frequency scan comparison({}) is >= {} threshold[dB].'
                        .format(np.max(np.abs(s1 - s0))/norm_fac, self.threshold))

    @unittest.skip('Correlator startup is currently unreliable')
    def test_restart_consistency(self):
        """3. Check that results are consequent on correlator restart"""
        # Removed test as correlator startup is currently unreliable,
        # will only add test method onces correlator startup is reliable.
        pass

    def test_delay_tracking(self):
        """
        (TP.C.1.27) CBF Delay Compensation/LO Fringe stopping polynomial
        """
        test_name = '{}.{}'.format(strclass(self.__class__), self._testMethodName)

        # Select dsim signal output, zero all sources, output scalings to 0.5
        init_dsim_sources(self.dhost)
        # Put some correlated noise on both outputs
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        # Get list of all the baselines present in the correlator output
        bls_ordering = initial_dump['bls_ordering']
        baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(
            bls_ordering)}
        # Choose baseline for phase comparison
        baseline_index = baseline_lookup[('m000_x', 'm000_y')]

        sampling_period = self.corr_freqs.sample_period
        test_delays = [0, sampling_period, #1.5*sampling_period,
            3*sampling_period]

        def expected_phases():
            sampling_period = self.corr_freqs.sample_period
            expected_chan_phase = []
            for channel in self.corr_freqs.chan_freqs:
                phases = channel * 2 * np.pi * sampling_period
                expected_chan_phase.append(phases)
            return np.array(expected_chan_phase)

        def actual_phases():
            actual_phases_list = []
            for delay in test_delays:
                # set coarse delay on correlator input m000_y
                delay_samples = int(np.floor(delay/sampling_period))
                set_coarse_delay(self.correlator, 'm000_y', value=delay_samples)

                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                data = complexise(this_freq_dump['xeng_raw']
                    [:, baseline_index, :])
                phases = np.unwrap(np.angle(data))
                actual_phases_list.append(phases)
            return actual_phases_list

        def actual_phases(plot=False):
            actual_phases_list = []
            for delay in test_delays:
                # set coarse delay on correlator input m000_y
                delay_samples = int(np.floor(delay/sampling_period))
                set_coarse_delay(self.correlator, 'm000_y', value=delay_samples)

                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                data = complexise(this_freq_dump['xeng_raw']
                    [:, baseline_index, :])

                phases = np.unwrap(np.angle(data))
                actual_phases_list.append(phases)
                plt.plot(self.corr_freqs.chan_freqs, phases)
                if plot:
                    plt.show()
            return actual_phases_list

        def plot_and_save(freqs, data, plot_filename, show=False):
            lab = plot_filename.split(".")[-2].title()
            fig = plt.plot(freqs, data, label='{}'.format(lab))[0]
            axes = fig.get_axes()
            ybound = axes.get_ybound()
            yb_diff = abs(ybound[1] - ybound[0])
            new_ybound = [ybound[0] - yb_diff*1.1, ybound[1] + yb_diff*1.1]
            plt.vlines(np.max(freqs), *new_ybound,
                label='{} MHz (max)'.format(self.corr_freqs.bandwidth/1e6),
                    linestyles='dashed')
            plt.legend().draggable()
            plt.title('Correlation Phase Slope for {}ns delay '.format(
                np.around(self.corr_freqs.sample_period/1e-9,
                    decimals=3)))
            axes.set_ybound(*new_ybound)
            plt.grid(True)
            plt.ylabel('Phase [radians]')
            plt.xlabel('Frequency (Hz)')
            plt.savefig(plot_filename)
            if show:
                plt.show()
            plt.close()

        graph_name_all = test_name + '.expected_phases.svg'
        plot_and_save(self.corr_freqs.chan_freqs,
            expected_phases(), graph_name_all)

        graph_name_all = test_name + '.actual_phases.svg'
        plot_and_save(self.corr_freqs.chan_freqs,
            actual_phases()[1], graph_name_all)

        # Compare Actual and Expected phases and check if their equal
        # upto 3 decimal places
        np.testing.assert_almost_equal(actual_phases()[1],
            expected_phases(), decimal=3)
        # Check if the phases at test delay = 0 are all zeros.
        self.assertTrue(np.min(actual_phases()[0]) == np.max(actual_phases()[0]))

    def test_channel_peaks(self):
        """4. Test that the correct channels have the peak response to each frequency"""
        test_name = '{}.{}'.format(strclass(self.__class__), self._testMethodName)

        init_dsim_sources(self.dhost)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel no with max response for each frequency
        max_channels = []
        # Channel responses higher than -20 dB relative to expected channel
        extra_peaks = []

        start_chan = 1 # skip DC channel since dsim puts out zeros
        for channel, channel_f0 in enumerate(
                self.corr_freqs.chan_freqs[start_chan:], start_chan):
            print ('Getting channel response for freq {}/{}: {} MHz.'.format(
                channel, len(self.corr_freqs.chan_freqs), channel_f0/1e6))
            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=0.125)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            this_freq_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw']
            this_freq_response = (
                normalised_magnitude(this_freq_data[:, test_baseline, :]))
            max_chan = np.argmax(this_freq_response)
            max_channels.append(max_chan)
            # Find responses that are more than -20 dB relative to max
            unwanted_cutoff = this_freq_response[max_chan] / 10e2
            extra_responses = [i for i, resp in enumerate(this_freq_response)
                               if i != max_chan and resp >= unwanted_cutoff]
            extra_peaks.append(extra_responses)

        # Check that the correct channels have the peak response to each frequency
        self.assertEqual(max_channels, range(start_chan,
            len(max_channels) + start_chan))
        # Check that no other channels responded > -20 dB
        self.assertEqual(extra_peaks, [[]]*len(max_channels))

    def test_sensor_values(self):
        """
        (TP.C.1.16) Report sensor values (AR1)
        """
        pass
