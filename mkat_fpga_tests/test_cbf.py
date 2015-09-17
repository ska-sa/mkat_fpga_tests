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
from casperfpga import utils as fpgautils

from katcp import resource_client
from katcp import ioloop_manager
from nosekatreport import Aqf, aqf_vr

from mkat_fpga_tests import correlator_fixture
from mkat_fpga_tests.aqf_utils import cls_end_aqf, aqf_numpy_almost_equal
from mkat_fpga_tests.utils import normalised_magnitude, loggerise, complexise
from mkat_fpga_tests.utils import init_dsim_sources, get_dsim_source_info
from mkat_fpga_tests.utils import nonzero_baselines, zero_baselines, all_nonzero_baselines
from mkat_fpga_tests.utils import CorrelatorFrequencyInfo, TestDataH5
from mkat_fpga_tests.utils import get_snapshots
from mkat_fpga_tests.utils import set_coarse_delay, get_quant_snapshot
from mkat_fpga_tests.utils import get_source_object_and_index

LOGGER = logging.getLogger(__name__)

DUMP_TIMEOUT = 10              # How long to wait for a correlator dump to arrive in tests

def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0, figure out if VACC is roated by 1"""
    b0 = np.abs(complexise(xeng_raw[:,0]))
    b1 = np.abs(complexise(xeng_raw[:,1]))
    if np.max(b0) > 0 and np.max(b1) == 0:
        # We expect autocorr in baseline 0 to be nonzero if the vacc is
        # properly aligned, hence no offset
        return 0
    elif np.max(b1) > 0 and np.max(b0) == 0:
        return 1
    else:
        raise ValueError('Could not determine VACC offset')

def get_and_restore_initial_eqs(test_instance, correlator):
    initial_equalisations = {input: eq_info['eq'] for input, eq_info
                             in fengops.feng_eq_get(correlator).items()}
    def restore_initial_equalisations():
        for input, eq in initial_equalisations.items():
            fengops.feng_eq_set(correlator, source_name=input, new_eq=eq)
    test_instance.addCleanup(restore_initial_equalisations)
    return initial_equalisations


@cls_end_aqf
class test_CBF(unittest.TestCase):
    def setUp(self):
        self.correlator = correlator_fixture.correlator
        self.corr_fix = correlator_fixture
        self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
        dsim_conf = self.correlator.configd['dsimengine']
        dig_host = dsim_conf['host']
        self.dhost = FpgaDsimHost(dig_host, config=dsim_conf)
        self.dhost.get_system_information()
        # Initialise dsim sources.
        init_dsim_sources(self.dhost)
        # Increase the dump rate so tests can run faster
        xengops.xeng_set_acc_time(self.correlator, 0.2)
        self.addCleanup(self.corr_fix.stop_x_data)
        self.receiver = CorrRx(port=8888)
        start_thread_with_cleanup(self, self.receiver, start_timeout=1)
        self.corr_fix.start_x_data()
        self.corr_fix.issue_metadata()
        # Threshold: -70dB
        self.threshold = 1e-7

    @aqf_vr('TP.C.1.19')
    def test_channelisation(self):
        """CBF Channelisation Wideband Coarse L-band"""
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

        # Put some noise on output
        # self.dhost.noise_sources.noise_0.set(scale=1e-3)
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
            Aqf.equals(last_pfb_counts, curr_pfb_counts,
                "Pfb FFT is not overflowing")
            # Test QDR error flags
            for hosts_status in fftoverflow_qdrstatus.values():
                for host, hosts_status in hosts_status.items():
                    if hosts_status['QDR_okay'] is False:
                        QDR_error_roaches.add(host)
            # Test QDR status
            Aqf.is_false(QDR_error_roaches,
                         'Check that none of the roaches have QDR errors')

        # Test fft overflow and qdr status before
        test_fftoverflow_qdrstatus()

        for i, freq in enumerate(requested_test_freqs):
            LOGGER.info('Getting channel response for freq {}/{}: {} MHz.'.format(
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
                LOGGER.exception("Error retrieving snapshot at {}/{}: {} MHz."
                    .format(i+1, len(requested_test_freqs), freq/1e6))
                if i == 0:
                    # The first snapshot must work properly to give us the data
                    # structure
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

        def plot_and_save(freqs, data, plot_filename, caption="", show=False):
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
            Aqf.matplotlib_fig(plot_filename, caption=caption, close_fig=False)
            if show:
                plt.show()
            plt.close()

        graph_name_all = test_name + '.channel_response.svg'
        plot_data_all  = loggerise(chan_responses[:, test_chan], dynamic_range=90)
        plot_and_save(actual_test_freqs, plot_data_all, graph_name_all,
                      caption='Channel 1500 response vs source frequency')

        # Get responses for central 80% of channel
        df = self.corr_freqs.delta_f
        central_indices = (
            (actual_test_freqs <= expected_fc + 0.4*df) &
            (actual_test_freqs >= expected_fc - 0.4*df))
        central_chan_responses = chan_responses[central_indices]
        central_chan_test_freqs = actual_test_freqs[central_indices]

        graph_name_central = test_name + '.channel_response_central.svg'
        plot_data_central  = loggerise(central_chan_responses[:, test_chan],
            dynamic_range=90)
        plot_and_save(central_chan_test_freqs, plot_data_central, graph_name_central,
                      caption='Channel 1500 central response vs source frequency')

        # Test responses in central 80% of channel
        for i, freq in enumerate(central_chan_test_freqs):
            max_chan = np.argmax(np.abs(central_chan_responses[i]))
            # TODO Aqf conversion
            self.assertEqual(max_chan, test_chan,
                'Source freq {} peak in correct channel.'
                    .format(freq, test_chan, max_chan))
        Aqf.less(
            np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
            'Check that VACC output is at < 99% of maximum value, otherwise '
            'something, somewhere, is probably overranging.')
        max_central_chan_response = np.max(10*np.log10(
            central_chan_responses[:, test_chan]))
        min_central_chan_response = np.min(10*np.log10(
            central_chan_responses[:, test_chan]))
        chan_ripple = max_central_chan_response - min_central_chan_response
        acceptable_ripple_lt = 0.3

        Aqf.less(chan_ripple, acceptable_ripple_lt,
                 'Check that ripple within 80% of channel fc is < {} dB'
                 .format(acceptable_ripple_lt))

    @aqf_vr('TP.C.1.30')
    def test_product_baselines(self):
        """CBF Baseline Correlation Products - AR1"""
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
        # Make a list of all possible baselines (including redundant baselines)
        # for the given list of inputs
        possible_baselines = set()
        for li in input_labels:
            for lj in input_labels:
                possible_baselines.add((li, lj))

        test_bl = sorted(list(possible_baselines))
        # Test that each baseline (or its reverse-order counterpart) is present
        # in the correlator output
        baseline_is_present = {}

        for test_bl in possible_baselines:
           baseline_is_present[test_bl] = (test_bl in present_baselines or
                                           test_bl[::-1] in present_baselines)
        Aqf.is_true(all(baseline_is_present.values()),
                    "Check that all baselines are present in correlator output.")

        test_data = test_dump['xeng_raw']
        # Expect all baselines and all channels to be non-zero
        Aqf.is_false(zero_baselines(test_data),
                     'Check that no baselines have all-zero visibilities')
        Aqf.equals(nonzero_baselines(test_data), all_nonzero_baselines(test_data),
                  "Check that all baseline visibilities are non-zero accross "
                    "all channels")

        # Save initial f-engine equalisations, and ensure they are restored
        # at the end of the test
        initial_equalisations = get_and_restore_initial_eqs(self, self.correlator)

        # Set all inputs to zero, and check that output product is all-zero
        for input in input_labels:
            fengops.feng_eq_set(self.correlator, source_name=input, new_eq=0)
        test_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw']
        Aqf.is_false(nonzero_baselines(test_data),
                     "Check that all baseline visibilities are zero")
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

            Aqf.equals(
                actual_nz_bls, expected_nz_bls,
                "Check that expected baseline visibilities are nonzero with "
                    "non-zero inputs {}."
                .format(sorted(nonzero_inputs)))
            Aqf.equals(
                actual_z_bls, expected_z_bls,
                "Also check that expected baselines visibilities are zero.")

    @aqf_vr('TP.C.dummy_vr_1')
    def test_back2back_consistency(self):
        """Check that back-to-back dumps with same input are equal"""
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
            Aqf.less(dumps_comp, self.threshold,
                     'Check that back-to-back dumps with the same frequency '
                     'input differ by no more than {} threshold[dB].'
                     .format(10*np.log10(self.threshold)))

    @aqf_vr('TP.C.dummy_vr_2')
    def test_freq_scan_consistency(self):
        """Check that identical frequency scans produce equal results"""
        test_chan = 1500
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=0.25)

        scans = []
        initial_max_freq_list = []
        for scan_i in range(3):
            scan_dumps = []
            scans.append(scan_dumps)
            for i, freq in enumerate(requested_test_freqs):
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

                # TODO Convert to a less-verbose comparison for Aqf.
                # E.g. test all the frequencies and only save the error cases,
                # then have a final Aqf-check so that there is only one step
                # (not n_chan) in the report.
                self.assertLess(np.max(np.abs(s1 - s0))/norm_fac, self.threshold,
                    'frequency scan comparison({}) is >= {} threshold[dB].'
                        .format(np.max(np.abs(s1 - s0))/norm_fac, self.threshold))

    # @unittest.skip('Correlator startup is currently unreliable')
    @aqf_vr('TP.C.dummy_vr_3')
    def test_restart_consistency(self):
        """3. Check that results are consequent on correlator restart"""
        # Removed test as correlator startup is currently unreliable,
        # will only add test method onces correlator startup is reliable.
        pass

    @aqf_vr('TP.C.1.27')
    def test_delay_tracking(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial
        """
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
        test_delays = [0, sampling_period, 1.5*sampling_period,
            2*sampling_period]

        def get_expected_phases():
            expected_phases = []
            for delay in test_delays:
                phases = self.corr_freqs.chan_freqs * 2 * np.pi * delay
                expected_phases.append(phases)

            return zip(test_delays, expected_phases)

        def get_actual_phases():
            actual_phases_list = []
            for delay in test_delays:
                # set coarse delay on correlator input m000_y
                # use correlator_fixture.corr_conf[]
                # correlator_fixture.katcp_rct.req.delays time.time+somethign
                # See page 22 on ICD ?delays on CBF-CAM ICD
                fengops.feng_set_delay(self.correlator, 'm000_y', delay=delay,
                    delta_delay=0, phase_offset=0, delta_phase_offset=0,
                        ld_time=None, ld_check=True)

                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                data = complexise(this_freq_dump['xeng_raw']
                    [:, baseline_index, :])

                phases = np.unwrap(np.angle(data))
                actual_phases_list.append(phases)
            return zip(test_delays, actual_phases_list)

        def plot_and_save(freqs, actual_data, expected_data, plot_filename,
            show=False):
            plt.gca().set_color_cycle(None)
            for delay, phases in actual_data:
                plt.plot(freqs, phases, label='{}ns'.format(delay*1e9))
            plt.gca().set_color_cycle(None)
            for delay, phases in expected_data:
                fig = plt.plot(freqs, phases, '--')[0]

            axes = fig.get_axes()
            ybound = axes.get_ybound()
            yb_diff = abs(ybound[1] - ybound[0])
            new_ybound = [ybound[0] - yb_diff*1.1, ybound[1] + yb_diff*1.1]
            plt.legend()
            plt.title('Unwrapped Correlation Phase')
            axes.set_ybound(*new_ybound)
            plt.grid(True)
            plt.ylabel('Phase [radians]')
            plt.xlabel('Frequency (Hz)')
            caption=("Actual and expected Unwrapped Correlation Phase, "
                     "dashed line indicates expected value.")
            Aqf.matplotlib_fig(plot_filename, caption=caption, close_fig=False)
            if show:
                plt.show()
            plt.close()

        actual_phases = get_actual_phases()
        expected_phases = get_expected_phases()
        for i, delay in enumerate(test_delays):
            delta_actual = round(np.max(actual_phases[i][1]) - np.min(
                actual_phases[i][1]),2)
            delta_expected = round(np.max(expected_phases[i][1]) - np.min(
                expected_phases[i][1]),2)
            LOGGER.debug( "delay: {}ns, expected phase delta: {},"
                " actual_phase_delta: {}".format(
                delay*1e9, delta_expected, delta_actual))
            Aqf.equals(delta_expected,delta_actual,
                'Check if difference expected({0:.3f}) and actual({1:.3f}) '
                    'phases are equal at delay {2:.3f}ns.'
                        .format(delta_expected, delta_actual, delay*1e9))

        plot_and_save(self.corr_freqs.chan_freqs, actual_phases, expected_phases,
                      'delay_phase_response.svg', show=False)
        # TODO NM 2015-09-04: We are only checking one of the results here?
        # This structure needs a bit of unpacking :)
        Aqf.equals(np.min(actual_phases[0][0]), np.max(actual_phases[0][0]),
            "Check if the phase-slope with delay = 0 is zero.")
        aqf_numpy_almost_equal(actual_phases[1][0], expected_phases[1][0],
            'Check that when one clock cycle is introduced (0.584ns),'
                ' the is a change in phases at 180 degrees as expected '
                    'to within 3 decimal places', decimal=3)
        aqf_numpy_almost_equal(actual_phases[2][1], expected_phases[2][1],
            'Check that when 1.5 clock cycle is introduced (0.876ns),'
                ' the is a change in phases at 270 degrees as expected '
                    'to within 3 decimal places', decimal=3)
        aqf_numpy_almost_equal(actual_phases[3][1], expected_phases[3][1],
            'Check that when 2 clock cycle is introduced (1.168ns),'
                ' the is a change in phases at 360 degrees as expected '
                    'to within 3 decimal places', decimal=3)

    @aqf_vr('TP.C.1.19')
    def test_sfdr_peaks(self):
        """Test spurious free dynamic range

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.
        """
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel no with max response for each frequency
        max_channels = []
        # Spurious response cutoff in dB
        cutoff = 20
        # Channel responses higher than -cutoff dB relative to expected channel
        extra_peaks = []

        # Checking for all channels.
        start_chan = 1  # skip DC channel since dsim puts out zeros
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
            # Find responses that are more than -cutoff relative to max
            unwanted_cutoff = this_freq_response[max_chan] / 10**(cutoff/10.)
            extra_responses = [i for i, resp in enumerate(this_freq_response)
                               if i != max_chan and resp >= unwanted_cutoff]
            extra_peaks.append(extra_responses)

        Aqf.equals(max_channels, range(start_chan, len(max_channels) + start_chan),
                  "Check that the correct channels have the peak response to each "
                  "frequency")
        Aqf.equals(extra_peaks, [[]]*len(max_channels),
                   "Check that no other channels responded > -{cutoff} dB"
                   .format(**locals()))

    @aqf_vr('TP.C.1.16')
    def test_sensor_values(self):
        """
        Report sensor values (AR1)
        """
        iom = ioloop_manager.IOLoopManager()
        iow = resource_client.IOLoopThreadWrapper(iom.get_ioloop())
        iom.start()
        self.addCleanup(iom.stop)

        rc = resource_client.KATCPClientResource(
            dict(name='localhost', address=('localhost', '7147'),
                controlled=True))
        rc.set_ioloop(iom.get_ioloop())
        rct = resource_client.ThreadSafeKATCPClientResourceWrapper(rc, iow)
        rct.start()
        rct.until_synced()

        # 1. Request a list of available sensors using KATCP command
        # 2. Confirm the CBF replies with a number of sensor-list inform messages
        LOGGER.info (rct.req.sensor_list())

        # 3. Confirm the CBF replies with "!sensor-list ok numSensors"
        #    where numSensors is the number of sensor-list informs sent.
        list_reply, list_informs = rct.req.sensor_list()
        sens_lst_stat, numSensors = list_reply.arguments
        numSensors = int(numSensors)
        Aqf.equals(numSensors, len(list_informs),
            'Check that the number of sensors are equal to the'
                 'number of sensors in the list.')

        # 4.1 Test that ?sensor-value and ?sensor-list agree about the number
        # of sensors.
        sens_val_stat, sens_val_cnt = rct.req.sensor_value().reply.arguments
        Aqf.equals(int(sens_val_cnt), numSensors,
            'Check that the sensor-value and sensor-list counts are the same')

        # 4.2 Request the time synchronisation status using KATCP command
        # "?sensor-value time.synchronised
        self.assertTrue(rct.req.sensor_value('time.synchronised').reply.reply_ok(),
                msg='Reading time synchronisation sensor failed!')

        # 5. Confirm the CBF replies with " #sensor-value <time>
        # time.synchronised [status value], followed by a "!sensor-value ok 1"
        # message.
        Aqf.equals(str(rct.req.sensor_value('time.synchronised')[0]),
            '!sensor-value ok 1', 'Check that the time synchronised sensor values'
                ' replies with !sensor-value ok 1')

        # Check all sensors statuses
        for sensor in rct.sensor.values():
            LOGGER.info(sensor.name + ':'+ str(sensor.get_value()))
            self.assertEqual(sensor.get_status(), 'nominal',
                msg='Sensor status fail: {}, {} '
                    .format(sensor.name, sensor.get_status()))

        roaches = self.correlator.fhosts + self.correlator.xhosts

        for roach in roaches:
            values_reply, sensors_values = roach.katcprequest('sensor-value')
            list_reply, sensors_list = roach.katcprequest('sensor-list')

            # Verify the number of sensors received with
            # number of sensors in the list.
            Aqf.is_true((values_reply.reply_ok() == list_reply.reply_ok())
                , '{}: Verify that ?sensor-list and ?sensor-value agree'
                ' about the number of sensors.'.format(roach.host))

            # Check the number of sensors in the list is equal to the list
            # of values received.
            Aqf.equals(len(sensors_list), int(values_reply.arguments[1])
                , 'Check the number of sensors in the list is equal to the '
                    'list of values received for {}'.format(roach.host))

    @aqf_vr('TP.C.dummy_vr_4')
    def test_roach_sensors_status(self):
        """ Test all roach sensors status are not failing """
        for roach in (self.correlator.fhosts + self.correlator.xhosts):
            values_reply, sensors_values = roach.katcprequest('sensor-value')
            for sensor in sensors_values[1:]:
                sensor_name, sensor_status, sensor_value = (
                    sensor.arguments[2:])
                # Check if roach sensors are failing
                Aqf.is_false((sensor_status == 'fail'),
                    'Roach {}, Sensor name: {}, status: {}'
                        .format(roach.host, sensor_name, sensor_status))

    @aqf_vr('TP.C.1.31')
    def test_vacc(self):
        """Test vector accumulator"""
        # Choose a test freqency around the centre of the band.
        test_freq = 856e6/2
        test_input = 'm000_x'
        eq_scaling = 30
        acc_times = [0.05, 0.1, 0.5, 1]

        internal_accumulations = int(
            self.correlator.configd['xengine']['xeng_accumulation_len'])
        delta_acc_t = self.corr_freqs.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = np.argmin(
            np.abs(self.corr_freqs.chan_freqs - test_freq))
        eqs = np.zeros(self.corr_freqs.n_chans, dtype=np.complex)
        eqs[test_freq_channel] = eq_scaling
        get_and_restore_initial_eqs(self, self.correlator)
        fengops.feng_eq_set(self.correlator, source_name=test_input,
                            new_eq=list(eqs))
        self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=0.125,
        # Make dsim output periodic in FFT-length so that each FFT is identical
                                          repeatN=self.corr_freqs.n_chans*2)
        # The re-quantiser outputs signed int (8bit), but the snapshot code
        # normalises it to floats between -1:1. Since we want to calculate the
        # output of the vacc which sums integers, denormalise the snapshot
        # output back to ints.
        q_denorm = 128
        quantiser_spectrum = get_quant_snapshot(
            self.correlator, test_input) * q_denorm

        # Check that the spectrum is not zero in the test channel
        Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
            'Check that the spectrum is not zero in the test channel')
        # Check that the spectrum is zero except in the test channel
        Aqf.is_true(np.all(quantiser_spectrum[0:test_freq_channel] == 0),
            'Check that the spectrum is zero except in the test channel')
        Aqf.is_true(np.all(quantiser_spectrum[test_freq_channel+1:] == 0),
            'Check that the spectrum is zero except in the test channel')

        for vacc_accumulations in test_acc_lens:
            xengops.xeng_set_acc_len(self.correlator, vacc_accumulations)
            no_accs = internal_accumulations * vacc_accumulations
            expected_response = np.abs(quantiser_spectrum)**2  * no_accs
            response = complexise(
                self.receiver.get_clean_dump(dump_timeout=5)['xeng_raw'][:, 0, :])
            # Check that the accumulator response is equal to the expected response
            Aqf.is_true(np.array_equal(expected_response, response),
                'Check that the accumulator response is equal'
                    ' to the expected response for {} accumulation length'
                        .format(vacc_accumulations))

    @aqf_vr('TP.C.1.40')
    def test_product_switch(self):
        """(TP.C.1.40) CBF Data Product Switching Time"""
        init_dsim_sources(self.dhost)
        # 1. Configure one of the ROACHs in the CBF to generate noise.
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        # Confirm that SPEAD packets are being produced,
        # with the selected data product(s).
        initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        # TODO NM 2015-09-14: Do we need to validate the shape of the data to
        # ensure the product is correct?

        # Deprogram CBF
        xhosts = self.correlator.xhosts
        fhosts = self.correlator.fhosts
        hosts = xhosts + fhosts
        # Deprogramming xhosts first then fhosts avoid reorder timeout errors
        fpgautils.threaded_fpga_function(xhosts, 10, 'deprogram')
        fpgautils.threaded_fpga_function(fhosts, 10, 'deprogram')
        [Aqf.is_false(host.is_running(),'{} Deprogrammed'.format(host.host))
            for host in hosts]
        # Confirm that SPEAD packets are either no longer being produced, or
        # that the data content is at least affected.
        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
            Aqf.failed('SPEAD parkets are still being produced.')
        except Exception:
            Aqf.passed('Check that SPEAD parkets are nolonger being produced.')

        # Start timer and re-initialise the instrument and, start capturing data.
        start_time = time.time()
        correlator_fixture.halt_array()
        correlator_fixture.start_correlator()
        self.corr_fix.start_x_data()
        # Confirm that the instrument is initialised by checking if roaches
        # are programmed.
        [Aqf.is_true(host.is_running(),'{} programmed and running'
            .format(host.host)) for host in hosts]

        # Confirm that SPEAD packets are being produced, with the selected data
        # product(s) The receiver won't return a dump if the correlator is not
        # producing well-formed SPEAD data.
        re_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        Aqf.is_true(re_dump,'Check that SPEAD parkets are being produced after '
            ' instrument re-initialisation.')

        # Stop timer.
        end_time = time.time()
        # Data Product switching time = End time - Start time.
        final_time =  round((end_time - start_time), 2)
        minute = 60.0
        # Confirm data product switching time is less than 60 seconds
        Aqf.less(final_time, minute,
            'Check that product switching time is less than one minute')

        # TODO: MM 2015-09-14, Still need more info

        # 6. Repeat for all combinations of available data products,
        # including the case where the "new" data product is the same as the
        # "old" one.
