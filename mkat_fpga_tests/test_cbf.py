from __future__ import division

import unittest
import logging
import time
import itertools
import subprocess
import threading
import os
import telnetlib
import paramiko
import subprocess
import colors as clrs
import operator
import Queue

from functools import partial
from random import randrange

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
from nosekatreport import Aqf, aqf_vr

from mkat_fpga_tests import correlator_fixture

from mkat_fpga_tests.aqf_utils import cls_end_aqf, aqf_numpy_almost_equal
from mkat_fpga_tests.aqf_utils import aqf_array_abs_error_less, aqf_plot_phase_results
from mkat_fpga_tests.aqf_utils import aqf_plot_channels, Aqf_is_not_equals

from mkat_fpga_tests.utils import normalised_magnitude, loggerise, complexise
from mkat_fpga_tests.utils import init_dsim_sources, get_dsim_source_info
from mkat_fpga_tests.utils import nonzero_baselines, zero_baselines, all_nonzero_baselines
from mkat_fpga_tests.utils import CorrelatorFrequencyInfo, TestDataH5
from mkat_fpga_tests.utils import get_snapshots, clear_all_delays
from mkat_fpga_tests.utils import get_fftoverflow_qdrstatus, check_fftoverflow_qdrstatus
from mkat_fpga_tests.utils import rearrange_snapblock, get_feng_snapshots
from mkat_fpga_tests.utils import set_coarse_delay, get_quant_snapshot
from mkat_fpga_tests.utils import get_source_object_and_index, get_baselines_lookup
from mkat_fpga_tests.utils import get_and_restore_initial_eqs, get_bit_flag, get_set_bits
from mkat_fpga_tests.utils import get_vacc_offset, get_pfb_counts
from mkat_fpga_tests.utils import get_default_instrument, check_host_okay

LOGGER = logging.getLogger(__name__)

DUMP_TIMEOUT = 10  # How long to wait for a correlator dump to arrive in tests

# From
# https://docs.google.com/spreadsheets/d/1XojAI9O9pSSXN8vyb2T97Sd875YCWqie8NY8L02gA_I/edit#gid=0
# SPEAD Identifier listing we see that the field flags_xeng_raw is a bitfield
# variable with bits 0 to 31 reserved for internal debugging and
#
# bit 34 - corruption or data missing during integration
# bit 33 - overrange in data path
# bit 32 - noise diode on during integration
#
# Also see the digitser end of the story in table 4, word 7 here:
# https://drive.google.com/a/ska.ac.za/file/d/0BzImdYPNWrAkV1hCR0hzQTYzQlE/view

flags_xeng_raw_bits = namedtuple('FlagsBits', 'corruption overrange noise_diode')(
    34, 33, 32)
#NOTE TP.C.1.20 for AR1 maps to TP.C.1.46 for RTS

# TODO NM (2015-12-10) Use steal_docstring decorator form KATCP for sub-tests to re-use
# docstring from implementaion. Perhaps do some trickery with string templates and parsing
# the mode name from the function to make an all singing all dancing decorator that does
# everything automagically?


@cls_end_aqf
class test_CBF(unittest.TestCase):
    DEFAULT_ACCUMULATION_TIME = 0.2

    def setUp(self):
        self.DEFAULT_INSTRUMENT = get_default_instrument()
        self.corr_fix = correlator_fixture
        dsim_conf = self.corr_fix.test_conf['dsimengine']
        dig_host = dsim_conf['host']
        self.dhost = FpgaDsimHost(dig_host, config=dsim_conf)
        self.dhost.get_system_information()
        # Initialise dsim sources.
        init_dsim_sources(self.dhost)
        self.receiver = None

    def set_instrument(self, instrument):
        if self.receiver:
            self.receiver.stop()
            self.receiver = None
        self.corr_fix.ensure_instrument(instrument)
        self.correlator = correlator_fixture.correlator
        self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
        self.xengops = self.correlator.xops
        self.fengops = self.correlator.fops
        try:
            reply = correlator_fixture.katcp_rct.req.accumulation_length(
                self.DEFAULT_ACCUMULATION_TIME, timeout=10)
            if reply.succeeded:
                Aqf.step('Accumulation time set: {}s'.format(reply.reply.arguments[-1]))
                self.addCleanup(self.corr_fix.stop_x_data)
                self.receiver = CorrRx(port=8888, queue_size=1000)
                start_thread_with_cleanup(self, self.receiver, start_timeout=1)
                self.corr_fix.start_x_data()
                self.corr_fix.issue_metadata()
                self._systems_tests()
                self.correlator.est_sync_epoch()
            else:
                Aqf.failed('Failed to set Accumulation time. Reply: {}'.format(reply))
        except Exception:
            Aqf.failed('Timed-out: Could not set Accumulation time.')

    #####################################################################
    #                          4k Test Method                           #
    #####################################################################

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_c8n856M4k_channelisation(self):
        """CBF Channelisation Wideband Coarse L-band"""
        Aqf.step('CBF Channelisation Wideband Coarse L-band\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_channelisation(required_chan_spacing=209e3, test_chan=1500)

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_c8n856M4k_channelisation_sfdr_peaks(self):
        """Test spurious free dynamic range for wideband coarse (c8n856M4k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.
        """
        Aqf.step('Test Spurious Free Dynamic Range for Wideband Coarse: {}\n'
                 .format(self.DEFAULT_INSTRUMENT))
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_sfdr_peaks(required_chan_spacing=209e3, cutoff=53)

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_c8n856M4k_product_baselines(self):
        """CBF Baseline Correlation Products - AR1"""
        Aqf.step('CBF Baseline Correlation Products - AR1\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_product_baselines()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_c8n856M4k_back2back_consistency(self):
        """
        Check that back-to-back SPEAD dumps with same input are equal.
        """
        Aqf.step("Check that back-to-back SPEAD dumps with same input are equal.\n")
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_back2back_consistency()

    @aqf_vr('TP.C.dummy_vr_2')
    def test_c8n856M4k_freq_scan_consistency(self):
        """Frequency Scan Consistency Test"""
        Aqf.step('Frequency Scan Consistency Test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_freq_scan_consistency()

    @aqf_vr('TP.C.dummy_vr_3')
    def test_c8n856M4k_restart_consistency(self):
        """Check that results are consistent on correlator restart"""
        Aqf.step('Correlator Restart Consistency Test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_restart_consistency()

    @aqf_vr('TP.C.1.27')
    def test_c8n856M4k_delay_tracking(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking
        """
        Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                 '-- Delay tracking\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_delay_tracking()

    @aqf_vr('TP.C.1.16')
    def test_c8n856M4k_sensor_values(self):
        """Report sensor values (AR1)"""
        Aqf.step('Report sensor values (AR1)\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_sensor_values()

    @aqf_vr('TP.C.1.16')
    def test_c8n856M4k_roach_sensors_status(self):
        """ Test all roach sensors status are not failing and count verification."""
        Aqf.step('Sensors Status Verification Test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_roach_sensors_status()

    @aqf_vr('TP.C.1.31')
    def test_c8n856M4k_vacc(self):
        """Vector Accumulator Test"""
        Aqf.step('Vector Accumulator Test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_vacc()

    @aqf_vr('TP.C.1.40')
    def test_product_switch_c8n856M4k(self):
        """CBF Data Product Switching Time"""
        Aqf.step('CBF Data Product Switching Time: {}\n'.format(
                 self.DEFAULT_INSTRUMENT))
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_product_switch(self.DEFAULT_INSTRUMENT, no_channels=4096)

    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_overflow_flag(self):
        """CBF flagging of data -- ADC overflow"""
        Aqf.step('CBF flagging of data -- ADC overflow\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_adc_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_noise_diode_flag(self):
        """CBF flagging of data -- noise diode fired"""
        Aqf.step('CBF flagging of data -- noise diode fired\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_noise_diode_flag()

    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_fft_overflow_flag(self):
        """CBF flagging of data -- FFT overflow"""
        Aqf.step('CBF flagging of data -- FFT overflow\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_roach_qdr_sensors(self):
        """QDR Memory Corruption Sensors Test"""
        Aqf.step('QDR Memory Corruption Sensors Test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_roach_qdr_sensors()

    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_roach_pfb_sensors(self):
        """PFB Error Test: Sensors"""
        Aqf.step('PFB Error Test: Sensors\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_roach_pfb_sensors()

    @aqf_vr('TP.C.5.5')
    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_deng_link_error(self):
        """Link Error :D-Engine to F-engine"""
        Aqf.step('Link Error :D-Engine to F-engine\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_deng_link_error()

    @aqf_vr('TP.C.5.5')
    @aqf_vr('TP.C.1.38')
    def test_c8n856M4k_feng_link_error(self):
        """Link Error :F-Engine to X-engine"""
        Aqf.step('Link Error :F-Engine to X-engine\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_feng_link_error()

    @aqf_vr('TP.C.1.27')
    def test_c8n856M4k_delay_rate(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate"""
        Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                 '-- Delay Rate\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    def test_c8n856M4k_fringe_offset(self):
        """CBF per-antenna phase error -- Fringe offset"""
        Aqf.step('CBF per-antenna phase error -- Fringe offset\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_fringe_offset()

    @aqf_vr('TP.C.1.28')
    def test_c8n856M4k_fringe_rate(self):
        """CBF per-antenna phase error -- Fringe rate"""
        Aqf.step('CBF per-antenna phase error -- Fringe rate\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_fringe_rate()

    @aqf_vr('TP.C.1.28')
    def test_c8n856M4k_all_delays(self):
        """
        CBF per-antenna phase error
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        Aqf.step('CBF per-antenna phase error -- Delays, Delay Rate, '
                 'Fringe Offset and Fringe Rate.\n')
        Aqf.tbd('Values still needs to be defined.')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        # self._test_all_delays()

    @aqf_vr('TP.C.1.17')
    def test_c8n856M4k_config_report(self):
        """CBF Report configuration"""
        Aqf.step('CBF Report configuration\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_config_report()

    @aqf_vr('TP.C.1.5.1')
    @aqf_vr('TP.C.1.18')
    def test_c8n856M4k_overtemperature(self):
        """ROACH2 overtemperature display test """
        Aqf.step('ROACH2 overtemperature display test\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_overtemp()

    @aqf_vr('TP.C.1.5.2')
    @aqf_vr('TP.C.1.18')
    def test_c8n856M4k_overvoltage(self):
        """ROACH2 overvoltage display test"""
        Aqf.step('ROACH2 overvoltage display test\n')
        Aqf.waived('Manual Test instead of being Automated.')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        # self._test_overvoltage()

    @aqf_vr('TP.C.1.5.3')
    @aqf_vr('TP.C.1.18')
    def test_c8n856M4k_overcurrent(self):
        """ROACH2 overcurrent display test"""
        Aqf.step('ROACH2 overcurrent display test\n')
        Aqf.waived('Manual Test instead of being Automated.')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        # self._test_overcurrent()

    @aqf_vr('TP.C.1.27')
    def test_c8n856M4k_delay_inputs(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial
           Delay applied to the correct input
        """
        Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial -- '
                 'Delay applied to the correct input\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_delay_inputs()

    @aqf_vr('TP.C.1.47')
    def test_c8n856M4k_data_product(self):
        """CBF Imaging Data Product Set"""
        Aqf.step('Imaging Data Product Set\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_data_product(self.DEFAULT_INSTRUMENT, no_channels=4096)

    @aqf_vr('TP.C.1.41')
    def test_c8n856M4k_control_init(self):
        """CBF Control - initial release"""
        Aqf.step('CBF Control - initial release\n')
        self.set_instrument(self.DEFAULT_INSTRUMENT)
        self._test_control_init()

    #################################################################
    #                       32K Mode Tests                          #
    #################################################################

    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_c8n856M32k_channelisation(self, instrument='bc8n856M32k'):
        """CBF Channelisation Wideband Fine L-band"""
        Aqf.step('CBF Channelisation Wideband Fine L-band\n')
        #self.set_instrument(instrument)
        #self._test_channelisation(required_chan_spacing=26e3, test_chan=15000)

    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_c8n856M32k_channelisation_sfdr_peaks(self, instrument='bc8n856M32k'):
        """Test spurious free dynamic range for wideband fine (c8n856M32k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

        """
        Aqf.step('Test spurious free dynamic range for wideband fine: {}\n'
                 .format(instrument))
        #self.set_instrument(instrument)
        #self._test_sfdr_peaks(required_chan_spacing=26e3, cutoff=53)

    @aqf_vr('TP.C.1.47')
    def test_c8n856M32k_data_product(self, instrument='bc8n856M32k'):
        """CBF Imaging Data Product Set"""
        Aqf.step('Imaging Data Product Set: {}\n'.format(instrument))
        #self.set_instrument(instrument)
        #self._test_data_product(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.40')
    def test_product_switch_c8n856M32k(self, instrument='bc8n856M32k'):
        """CBF Data Product Switching Time"""
        Aqf.step('CBF Data Product Switching Time: {}\n'.format(instrument))
        #self.set_instrument(instrument)
        #self._test_product_switch(instrument, no_channels=32768)

    #################################################################
    #                       Functions                               #
    #################################################################
    def _systems_tests(self):
        """Run tests fft overflow and qdr status before and after."""
        excluded_tests = [i for i in dir(self)
                        if i.startswith('test') and not i.startswith('test_c8n856M')
                        or i.startswith('test_c8n856M4k_fft_overflow_flag')
                        or i.startswith('test_c8n856M4k_roach_qdr_sensors')]

        if self._testMethodName not in excluded_tests:
            self.last_pfb_counts = get_pfb_counts(
                get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
            self.addCleanup(check_fftoverflow_qdrstatus, self.correlator,
                            self.last_pfb_counts)
            self.addCleanup(check_host_okay, self.correlator)

    def get_flag_dumps(self, flag_enable_fn, flag_disable_fn, flag_description,
                       accumulation_time=1.):
        Aqf.step('Setting  accumulation time to {}.'.format(accumulation_time))
        self.xengops.set_acc_time(accumulation_time)
        Aqf.step('Getting correlator dump 1 before setting {}.'
                 .format(flag_description))
        dump1 = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        start_time = time.time()
        Aqf.wait(0.1 * accumulation_time, 'Waiting 10% of accumulation length')
        Aqf.step('Setting {}'.format(flag_description))
        flag_enable_fn()
        # Ensure that the flag is disabled even if the test fails to avoid
        # contaminating other tests
        self.addCleanup(flag_disable_fn)
        elapsed = time.time() - start_time
        wait_time = accumulation_time * 0.8 - elapsed
        Aqf.is_true(wait_time > 0, 'Check that wait time {} is larger than zero'
                    .format(wait_time))
        Aqf.wait(wait_time, 'Waiting until 80% of accumulation length has elapsed')
        Aqf.step('Clearing {}'.format(flag_description))
        flag_disable_fn()
        Aqf.step('Getting correlator dump 2 after setting and clearing {}.'
                 .format(flag_description))
        dump2 = self.receiver.data_queue.get(DUMP_TIMEOUT)
        Aqf.step('Getting correlator dump 3.')
        dump3 = self.receiver.data_queue.get(DUMP_TIMEOUT)
        return (dump1, dump2, dump3)

    def _delays_setup(self, test_source_idx = 2):
        # Put some correlated noise on both outputs
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        local_src_names = ['input0', 'input1', 'input2', 'input3', 'input4',
                           'input5', 'input6', 'input7']
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(
            *local_src_names)
        Aqf.step('Source names changed to: ' + str(reply))
        Aqf.step('Clearing all coarse and fine delays for all inputs.')
        clear_all_delays(self.correlator, self.receiver)
        self.addCleanup(clear_all_delays, self.correlator, self.receiver)
        Aqf.step('Issuing metadata')
        self.corr_fix.issue_metadata()
        Aqf.step('Getting initial SPEAD dump.')
        initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        # TODO: (MM) 2015-10-21 get sync time from digitiser
        # We believe that sync time should be the digitiser sync epoch but
        # in the dsim this is not an int, so we using correlator value for now
        sync_time = initial_dump['sync_time'].value
        # sync_time = self.correlator.synchronisation_epoch
        scale_factor_timestamp = initial_dump['scale_factor_timestamp'].value
        time_stamp = initial_dump['timestamp'].value
        n_accs = initial_dump['n_accs'].value
        int_time = initial_dump['int_time'].value
        # 3ms added for the network round trip
        roundtrip = 0.003
        dump_1_timestamp = (sync_time + roundtrip +
                            time_stamp / scale_factor_timestamp)
        t_apply = dump_1_timestamp + 10 * int_time
        no_chans = range(self.corr_freqs.n_chans)
        reply, informs = self.corr_fix.katcp_rct.req.input_labels()
        source_names = reply.arguments[1:]
        # Get name for test_source_idx
        test_source = source_names[test_source_idx]
        ref_source = source_names[0]
        Aqf.step('Source input selected: {}'.format(test_source))
        num_inputs = len(source_names)
        # Get list of all the baselines present in the correlator output
        baseline_lookup = get_baselines_lookup(initial_dump)
        # Choose baseline for phase comparison
        # baseline_index = baseline_lookup[('m000_x', 'm000_y')]
        baseline_index = baseline_lookup[(ref_source, test_source)]
        # Get input (m000_y) index number
        #test_source_ind = source_names.index(test_source)

        return {
            'baseline_index': baseline_index,
            'baseline_lookup': baseline_lookup,
            'initial_dump': initial_dump,
            'sync_time': sync_time,
            'scale_factor_timestamp': scale_factor_timestamp,
            'time_stamp': time_stamp,
            'int_time': int_time,
            'dump_1_timestamp': dump_1_timestamp,
            't_apply': t_apply,
            'no_chans': no_chans,
            'test_source': test_source,
            'n_accs': n_accs,
            'sample_period': self.corr_freqs.sample_period,
            'num_inputs': num_inputs,
            'test_source_ind': test_source_idx
        }

    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients,
                         max_wait_dumps=20):

        try:
            cmd_start_time = time.time()
            reply = self.corr_fix.katcp_rct.req.delays(
                setup_data['t_apply'], *delay_coefficients)
            Aqf.is_true(reply.reply.reply_ok(),
                        'Delays reply: {}'.format(reply.reply.arguments[1]))
            final_cmd_time = time.time() - cmd_start_time
            Aqf.passed('Time it takes to load delays {} ms with intergration time {} ms'
                       .format(final_cmd_time / 100e-3, setup_data['int_time'] / 100e-3))

        except Exception as e:
            Aqf.failed('Failed to set delays with error: {}.'.format(e))

        last_discard = setup_data['t_apply'] - setup_data['int_time']

        num_discards = 0
        while True:
            num_discards += 1
            dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
            dump_timestamp = (setup_data['sync_time'] + dump['timestamp'].value /
                              setup_data['scale_factor_timestamp'])

            if (np.abs(dump_timestamp - last_discard) < 0.05 * setup_data['int_time']):
                Aqf.step('Received final accumulation before fringe '
                         'application with timestamp {}.'.format(dump_timestamp))
                break

            if num_discards > max_wait_dumps:
                Aqf.failed('Could not get accumulation with corrrect '
                           'timestamp within {} accumulation periods.'
                           .format(max_wait_dumps))
                break
            else:
                Aqf.step('Discarding accumulation with timestamp {}.'
                         .format(dump_timestamp))

        fringe_dumps = []
        for i in range(dump_counts):
            Aqf.step('Getting subsequent dump {}.'.format(i + 1))
            fringe_dumps.append(self.receiver.data_queue.get(DUMP_TIMEOUT))

        chan_resp = []
        phases = []
        for acc in fringe_dumps:
            dval = acc['xeng_raw'].value
            freq_response = normalised_magnitude(
                            dval[:, setup_data['baseline_index'], :])
            chan_resp.append(freq_response)

            data = complexise(dval[:, setup_data['baseline_index'], :])
            phases.append(np.angle(data))
            amp = np.mean(np.abs(data)) / setup_data['n_accs']

        return zip(phases, chan_resp)

    def _get_expected_data(self, setup_data, dump_counts, delay_coefficients,
                           actual_phases):

        def calc_actual_delay (setup_data):
            no_ch = len(setup_data['no_chans'])
            first_dump = np.unwrap(actual_phases[0])
            actual_slope = np.polyfit(range(0,no_ch), first_dump,1)[0]*no_ch
            actual_delay = setup_data['sample_period']*actual_slope/(np.pi)
            return actual_delay

        def gen_delay_vector(delay, setup_data):
            res = []
            no_ch = len(setup_data['no_chans'])
            delay_slope = np.pi * (delay / setup_data['sample_period'])
            c = delay_slope / 2
            for i in range(0, no_ch):
                m = i / float(no_ch)
                res.append(delay_slope * m - c)
            return res

        def gen_delay_data(delay, delay_rate, dump_counts, setup_data):
            expected_phases = []
            for dump in range(1, dump_counts + 1):
                tot_delay = (delay + dump * delay_rate * setup_data['int_time'] -
                             .5 * delay_rate * setup_data['int_time'])
                # The delay does not get applied on dump boundaries. This function
                # calculates the offset between the expected delay and the actual delay
                # and then adds this offset to all subsequent calculations.
                if dump == 1:
                    delay_offset = calc_actual_delay(setup_data) - tot_delay
                tot_delay = tot_delay + delay_offset
                expected_phases.append(gen_delay_vector(tot_delay, setup_data))
            return expected_phases

        def calc_actual_offset(setup_data):
            no_ch = len(setup_data['no_chans'])
            mid_ch = no_ch/2
            first_dump = actual_phases[0]
            # Determine average offset around 5 middle channels
            actual_offset = np.average(first_dump)#[mid_ch-3:mid_ch+3])
            return actual_offset

        def gen_fringe_vector(offset, setup_data):
            return [offset] * len(setup_data['no_chans'])

        def gen_fringe_data(fringe_offset, fringe_rate, dump_counts, setup_data):
            expected_phases = []
            for dump in range(1, dump_counts + 1):
                offset = -(fringe_offset + dump * fringe_rate * setup_data['int_time'])
                # The delay does not get applied on dump boundaries. This function
                # calculates the offset between the expected delay and the actual delay
                # and then adds this offset to all subsequent calculations.
                if dump == 1:
                    delta_offset = calc_actual_offset(setup_data) - offset
                offset = offset + delta_offset
                expected_phases.append(gen_fringe_vector(offset, setup_data))
            return expected_phases

        ant_delay = []
        for delay in delay_coefficients:
            bits = delay.strip().split(':')
            if len(bits) != 2:
                raise ValueError('%s is not a valid delay setting' % delay)
            delay = bits[0]
            delay = delay.split(',')
            delay = (float(delay[0]), float(delay[1]))
            fringe = bits[1]
            fringe = fringe.split(',')
            fringe = (float(fringe[0]), float(fringe[1]))
            ant_delay.append((delay, fringe))

        delay = ant_delay[setup_data['test_source_ind']][0][0]
        delay_rate = ant_delay[setup_data['test_source_ind']][0][1]
        fringe_offset = ant_delay[setup_data['test_source_ind']][1][0]
        fringe_rate = ant_delay[setup_data['test_source_ind']][1][1]

        delay_data = np.array((gen_delay_data(delay, delay_rate, dump_counts,
                                              setup_data)))
        fringe_data = np.array(gen_fringe_data(fringe_offset, fringe_rate,
                                               dump_counts, setup_data))
        result = delay_data + fringe_data
        wrapped_results = ((result + np.pi) % (2 * np.pi) - np.pi)

        if (fringe_offset or fringe_rate) != 0:
            fringe_phase = [np.abs((np.min(phase) + np.max(phase)) / 2.)
                            for phase in fringe_data]
            return zip(fringe_phase, wrapped_results)
        else:
            delay_phase = [np.abs((np.min(phase) - np.max(phase)) / 2.)
                           for phase in delay_data]
            return zip(delay_phase, wrapped_results)

    #################################################################
    #                       Test Methods                            #
    #################################################################
    def _test_channelisation(self, required_chan_spacing, test_chan):

        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=101, chans_around=3)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        # Put some noise on output
        # self.dhost.noise_sources.noise_0.set(scale=1e-3)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0

        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None
        print_counts = 3

        for i, freq in enumerate(requested_test_freqs):
            if i < print_counts:
                Aqf.step ('Getting channel response for freq {}/{}: {} MHz.'
                        .format(i + 1, len(requested_test_freqs), freq / 1e6))
            elif i >= (len(requested_test_freqs) - print_counts) :
                Aqf.step ('Getting channel response for freq {}/{}: {} MHz.'
                        .format(i + 1, len(requested_test_freqs), freq / 1e6))
            else:
                LOGGER.info ('Getting channel response for freq {}/{}: {} MHz.'
                             .format(i + 1, len(requested_test_freqs), freq / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.6)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency

            if this_source_freq == last_source_freq:
                LOGGER.info('Skipping channel response for freq {}/{}: {} MHz.\n'
                            'Digitiser frequency is same as previous.'
                            .format(i + 1, len(requested_test_freqs), freq / 1e6))
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq

            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            this_freq_data = this_freq_dump['xeng_raw'].value
            this_freq_response = normalised_magnitude(
                this_freq_data[:, test_baseline, :])
            actual_test_freqs.append(this_source_freq)
            chan_responses.append(this_freq_response)
            # Plot an overall frequency response at the centre frequency just as
            # a sanity check
            if np.abs(freq - expected_fc) < 0.1:
                aqf_plot_channels(this_freq_response, 'fc_channel_resp_log_{}.svg'.format(
                                  self.corr_freqs.n_chans),
                                  'Log channel response at {} MHz.\n'
                                  .format(this_source_freq/1e6), log_dynamic_range=90,
                                  caption='This is merely a sanity check to plot '
                                          'an overrall frequency response at the '
                                          'center frequency.')

        # Test fft overflow and qdr status after
        check_fftoverflow_qdrstatus(self.correlator, self.last_pfb_counts)
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
            new_ybound = [ybound[0] - yb_diff * 1.1, ybound[1] + yb_diff * 1.1]
            plt.vlines(expected_fc, *new_ybound, colors='r', label='chan fc')
            plt.vlines(expected_fc - df / 2, *new_ybound, label='chan min/max')
            plt.vlines(expected_fc - 0.8 * df / 2, *new_ybound, label='chan +-40%',
                       linestyles='dashed')
            plt.vlines(expected_fc + df / 2, *new_ybound, label='_chan max')
            plt.vlines(expected_fc + 0.8 * df / 2, *new_ybound, label='_chan +40%',
                       linestyles='dashed')
            plt.legend()
            plt.title('Channel {} ({} MHz) response'.format(
                test_chan, expected_fc / 1e6))
            axes.set_ybound(*new_ybound)
            plt.grid(True)
            plt.ylabel('dB relative to VACC max')
            # TODO Normalise plot to frequency bins
            plt.xlabel('Frequency (Hz)')
            Aqf.matplotlib_fig(plot_filename, caption=caption, close_fig=False)
            if show:
                plt.show()
            plt.close()

        graph_name_all = 'channel_response_{}.svg'.format(self.corr_freqs.n_chans)
        plot_data_all = loggerise(chan_responses[:, test_chan],
                                  dynamic_range=90)
        plot_and_save(actual_test_freqs, plot_data_all, graph_name_all,
                      caption='Channel {} response vs source frequency'
                      .format(test_chan))


        # Get responses for central 80% of channel
        df = self.corr_freqs.delta_f
        central_indices = (
            (actual_test_freqs <= expected_fc + 0.4 * df) &
            (actual_test_freqs >= expected_fc - 0.4 * df))
        central_chan_responses = chan_responses[central_indices]
        central_chan_test_freqs = actual_test_freqs[central_indices]

        graph_name_central = 'channel_response_central.svg'
        plot_data_central = loggerise(central_chan_responses[:, test_chan],
                                      dynamic_range=90)
        plot_and_save(central_chan_test_freqs, plot_data_central, graph_name_central,
                      caption='Channel {} central response vs source frequency on max channels {}'
                      .format(test_chan, self.corr_freqs.n_chans))


        Aqf.step('Test that the peak channeliser response to input frequencies in '
                 'central 80% of the test channel frequency band are all in the '
                 'test channel')
        fault_freqs = []
        fault_channels = []
        for i, freq in enumerate(central_chan_test_freqs):
            max_chan = np.argmax(np.abs(central_chan_responses[i]))
            if max_chan != test_chan:
                fault_freqs.append(freq)
                fault_channels.append(max_chan)
        if fault_freqs:
            Aqf.failed('The following input frequencies: {!r} respectively had '
                       'peak channeliser responses in channels {!r}, not '
                       'channel {} as expected.'
                       .format(fault_freqs, fault_channels, test_chan))

        Aqf.less(
            np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
            'Check that VACC output is at < 99% of maximum value, otherwise '
            'something, somewhere, is probably overranging.')
        max_central_chan_response = np.max(10 * np.log10(
            central_chan_responses[:, test_chan]))
        min_central_chan_response = np.min(10 * np.log10(
            central_chan_responses[:, test_chan]))
        chan_ripple = max_central_chan_response - min_central_chan_response
        acceptable_ripple_lt = 0.3
        Aqf.less(chan_ripple, acceptable_ripple_lt,
                 'Check that ripple within 80% of channel fc is < {} dB'
                 .format(acceptable_ripple_lt))

        # Get frequency samples closest channel fc and crossover points
        co_low_freq = expected_fc - df/2
        co_high_freq = expected_fc + df/2

        def get_close_result(freq):
            ind = np.argmin(np.abs(actual_test_freqs - freq))
            source_freq = actual_test_freqs[ind]
            response = chan_responses[ind, test_chan]
            return ind, source_freq, response

        fc_ind, fc_src_freq, fc_resp = get_close_result(expected_fc)
        co_low_ind, co_low_src_freq, co_low_resp = get_close_result(co_low_freq)
        co_high_ind, co_high_src_freq, co_high_resp = get_close_result(co_high_freq)

        Aqf.step('Check that response at channel-edges are -3 dB relative to '
                 'the channel centre at {} Hz, actual source freq '
                 '{} Hz'.format(expected_fc, fc_src_freq))

        desired_cutoff_resp = -3 # dB
        acceptable_co_var = 0.1 # dB, TODO 2015-12-09 NM: thumbsuck number
        co_low_rel_resp = 10*np.log10(co_low_resp / fc_resp)
        co_high_rel_resp = 10*np.log10(co_high_resp / fc_resp)
        Aqf.less(
            np.abs(desired_cutoff_resp - co_low_rel_resp), acceptable_co_var,
            'Check that relative response at the low band-edge ({co_low_rel_resp} '
            'dB @ {co_low_freq} Hz, actual source freq {co_low_src_freq}) '
            'is {desired_cutoff_resp} +- {acceptable_co_var} dB relative to '
            'channel centre response.'.format(**locals()) )
        Aqf.less(
            np.abs(desired_cutoff_resp - co_high_rel_resp), acceptable_co_var,
            'Check that relative response at the high band-edge ({co_high_rel_resp} '
            'dB @ {co_high_freq} Hz, actual source freq {co_high_src_freq}) '
            'is {desired_cutoff_resp} +- {acceptable_co_var} dB relative to '
            'channel centre response.'.format(**locals()))

    def _test_sfdr_peaks(self, required_chan_spacing, cutoff):
        """Test channel spacing and out-of-channel response

        Will loop over all the channels, placing the source frequency as close to the
        centre frequency of that channel as possible.

        Parameters
        ----------
        required_chan_spacing : float
            Maximum inter-channel spacing in Hz
        cutoff : float
            Responses in other channels must be at least `-cutoff` dB below the response
            of the channel with centre frequency corresponding to the source frequency

        """
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel no with max response for each frequency
        max_channels = []
        # Channel responses higher than -cutoff dB relative to expected channel
        extra_peaks = []

        # Checking for all channels.
        start_chan = 1  # skip DC channel since dsim puts out zeros for freq=0
        n_chans = self.corr_freqs.n_chans
        print_counts = 4

        for channel, channel_f0 in enumerate(
                self.corr_freqs.chan_freqs[start_chan:], start_chan):
            if channel < print_counts:
                Aqf.step ('Getting channel response for freq {}/{}: {} MHz.'
                   .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            elif channel > (n_chans - print_counts):
                Aqf.step ('Getting channel response for freq {}/{}: {} MHz.'
                   .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            else:
                LOGGER.info ('Getting channel response for freq {}/{}: {} MHz.'
                   .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=0.125)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            this_freq_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw'].value
            this_freq_response = (
                normalised_magnitude(this_freq_data[:, test_baseline, :]))
            if channel in (n_chans//10, n_chans//2, 9*n_chans//10):
                aqf_plot_channels(
                    this_freq_response, '{}_channel_resp.svg'.format(self._testMethodName),
                    'Log channel response at {} MHz'.format(this_source_freq/1e6),
                    log_dynamic_range=90)

            max_chan = np.argmax(this_freq_response)
            max_channels.append(max_chan)
            # Find responses that are more than -cutoff relative to max
            unwanted_cutoff = this_freq_response[max_chan] / 10 ** (cutoff / 10.)
            extra_responses = [i for i, resp in enumerate(this_freq_response)
                               if i != max_chan and resp >= unwanted_cutoff]
            extra_peaks.append(extra_responses)

        channel_range = range(start_chan, len(max_channels) + start_chan)
        if max_channels == channel_range:
            Aqf.passed('Check that the correct channels have the peak '
                       'response to each frequency')
        else:
            LOGGER.info('Expected: {}\n\nGot: {}'.format(max_channels, channel_range))
            Aqf.failed('Check that the correct channels have the peak '
                       'response to each frequency')

        if extra_peaks == [[]] * len(max_channels):
            Aqf.passed("Check that no other channels response more than -{cutoff} dB"
                   .format(**locals()))
        else:
            LOGGER.info('Expected: {}\n\nGot: {}'.format(extra_peaks,
                                                       [[]] * len(max_channels)))
            Aqf.failed("Check that no other channels responded > -{cutoff} dB"
                   .format(**locals()))

        df = self.corr_freqs.delta_f
        Aqf.less(df, required_chan_spacing,
                 'Test that computed channel spacing {} HZ is less than {} Hz. '
                 'This comparison is only valid if the peak response test '
                 'above passed, since a failure may imply that the computed '
                 'spacing is invalid.'.format(df, required_chan_spacing))

    def _test_product_baselines(self):
        # Put some correlated noise on both outputs
        self.dhost.noise_sources.noise_corr.set(scale=0.5)
        # Set list for all the correlator input labels
        local_src_names = self.correlator.configd['fengine']['source_names'].split(',')
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
        self.corr_fix.issue_metadata()
        test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        # TODO (MM) 2015-10-22
        # Get bls ordering from get baseline lookup helper functions
        bls_ordering = test_dump['bls_ordering'].value
        # Get list of all the correlator input labels
        input_labels = sorted(tuple(test_dump['input_labelling'].value[:, 0]))
        # Get list of all the baselines present in the correlator output
        baselines_lookup = get_baselines_lookup(test_dump)
        present_baselines = sorted(baselines_lookup.keys())

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

        # Select some baselines to plot
        plot_baselines = ((input_labels[0], input_labels[0]),
                          (input_labels[0], input_labels[1]),
                          (input_labels[0], input_labels[2]),
                          (input_labels[-1], input_labels[-1]),
                          (input_labels[-1], input_labels[-2]))
        plot_baseline_inds = tuple((baselines_lookup[bl] if bl in baselines_lookup
                                    else baselines_lookup[bl[::-1]])
                                    for bl in plot_baselines)

        plot_baseline_legends = tuple(
            '{bl[0]}, {bl[1]}: {ind}'.format(bl=bl, ind=ind)
            for bl, ind in zip(plot_baselines, plot_baseline_inds))

        Aqf.is_true(all(baseline_is_present.values()),
                    'Check that all baselines are present in correlator output.')

        test_data = test_dump['xeng_raw'].value
        # Expect all baselines and all channels to be non-zero
        Aqf.is_false(zero_baselines(test_data),
                     'Check that no baselines have all-zero visibilities')
        Aqf.equals(nonzero_baselines(test_data), all_nonzero_baselines(test_data),
                   'Check that all baseline visibilities are non-zero accross '
                   'all channels')

        # Save initial f-engine equalisations, and ensure they are restored
        # at the end of the test
        initial_equalisations = get_and_restore_initial_eqs(self, self.correlator)

        # Set all inputs to zero, and check that output product is all-zero
        for input in input_labels:
            self.fengops.eq_set(source_name=input, new_eq=0)
        test_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw'].value
        Aqf.is_false(nonzero_baselines(test_data),
                     "Check that all baseline visibilities are zero")
        # -----------------------------------
        all_inputs = sorted(set(input_labels))
        zero_inputs = set(input_labels)
        nonzero_inputs = set()

        def calc_zero_and_nonzero_baselines(nonzero_inputs):
            nonzeros = set()
            zeros = set()
            for inp_i in all_inputs:
                for inp_j in all_inputs:
                    if (inp_i, inp_j) not in present_baselines:
                        continue
                    if inp_i in nonzero_inputs and inp_j in nonzero_inputs:
                        nonzeros.add((inp_i, inp_j))
                    else:
                        zeros.add((inp_i, inp_j))
            return zeros, nonzeros

        for inp in input_labels:
            old_eq = initial_equalisations[inp]
            self.fengops.eq_set(source_name=inp, new_eq=old_eq)
            zero_inputs.remove(inp)
            nonzero_inputs.add(inp)
            expected_z_bls, expected_nz_bls = (
                calc_zero_and_nonzero_baselines(nonzero_inputs))
            test_data = self.receiver.get_clean_dump()['xeng_raw'].value
            plot_data = [normalised_magnitude(test_data[:,i,:])
                         for i in plot_baseline_inds]
            aqf_plot_channels(zip(plot_data, plot_baseline_legends),
                              plot_filename='product_baselines_channel_resp_{}.svg'
                                            .format(inp),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='Baseline channel response with the '
                              'following non-zero inputs: {}'
                              .format(sorted(nonzero_inputs)))
            actual_nz_bls_indices = all_nonzero_baselines(test_data)
            actual_nz_bls = set(tuple(bls_ordering[i])
                                for i in actual_nz_bls_indices)

            actual_z_bls_indices = zero_baselines(test_data)
            actual_z_bls = set(tuple(bls_ordering[i])
                               for i in actual_z_bls_indices)

            Aqf.equals(actual_nz_bls, expected_nz_bls,
                       'Check that expected baseline visibilities are nonzero with '
                       'non-zero inputs {}.'.format(sorted(nonzero_inputs)))

            Aqf.equals(actual_z_bls, expected_z_bls,
                       "Also check that expected baselines visibilities are zero.")

    def _test_back2back_consistency(self):
        threshold = 1e-7  # Threshold: -70dB
        test_chan = randrange(0, self.corr_freqs.n_chans)
        test_baseline = 0
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Check that back-to-back dumps with same input are equal on '
                 'channel({}) @ {}MHz.'.format(test_chan, expected_fc / 1e6))
        source_period_in_samples = self.corr_freqs.n_chans*2

        for i, freq in enumerate(requested_test_freqs):
            Aqf.step ('Testing dump consistency {}/{} @ {} MHz.'.format(
                i + 1, len(requested_test_freqs), freq / 1e6))
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125,
                                              repeatN=source_period_in_samples)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            dumps_data = []
            chan_responses = []
            for dump_no in range(3):
                if dump_no == 0:
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                else:
                    this_freq_dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
                this_freq_data = this_freq_dump['xeng_raw'].value
                dumps_data.append(this_freq_data)
                this_freq_response = normalised_magnitude(
                    this_freq_data[:, test_baseline, :])
                chan_responses.append(this_freq_response)

            diff_dumps = []
            for comparison in range(1, len(dumps_data)):
                d0 = dumps_data[0]
                d1 = dumps_data[comparison]
                diff_dumps.append(np.max(d0 - d1))

            dumps_comp = np.max(np.array(diff_dumps) / initial_max_freq)
            if not Aqf.less(
                    dumps_comp, threshold,
                    'Check that back-to-back dumps({}) with the same frequency '
                    'input differ by no more than {} threshold[dB].'
                    .format(dumps_comp, 10 * np.log10(threshold))):

                legends = ['dump #{}'.format(x) for x in range(len(chan_responses))]
                aqf_plot_channels(
                        zip(chan_responses, legends),
                        plot_filename='back2back_chan_resp_{}.svg'.format(i + 1),
                        log_dynamic_range=90, log_normalise_to=1,
                        caption='Comparison of back-to-back channelisation results with '
                                'source periodic every {} samples and sine frequency of '
                                '{} MHz.'.format(source_period_in_samples, this_source_freq))

    def _test_freq_scan_consistency(self):
        """Check that identical frequency scans produce equal results"""
        threshold = 1e-7  # Threshold: -70dB
        test_chan = randrange(0, self.corr_freqs.n_chans)
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Selected test channel {} and Frequency {}MHz'
                 .format(test_chan, expected_fc / 1e6))
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        for scan_i in range(3):
            scan_dumps = []
            scans.append(scan_dumps)
            for i, freq in enumerate(requested_test_freqs):
                if scan_i == 0:
                    self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                    this_freq_data = this_freq_dump['xeng_raw'].value
                    initial_max_freq_list.append(initial_max_freq)
                else:
                    self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    this_freq_data = this_freq_dump['xeng_raw'].value

                this_freq_response = normalised_magnitude(
                    this_freq_data[:, test_baseline, :])
                chan_responses.append(this_freq_response)
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
                max_freq_scan = np.max(np.abs(s1 - s0)) / norm_fac
                if not Aqf.less(max_freq_scan, threshold,
                         'Check that the frequency scan on SPEAD dump'
                         ' comparison({}) is less than {} dB.'
                         .format(max_freq_scan, threshold)):
                    legends = ['Freq scan #{}'.format(x) for x in range(len(chan_responses))]
                    aqf_plot_channels(
                            zip(chan_responses, legends),
                            plot_filename='freq_scan_cons_chan_resp.svg',
                            log_dynamic_range=90, log_normalise_to=1,
                            caption='Comparison of frequency sweeping from {}Mhz '
                            'to {}Mhz scan channelisation.'
                            .format(requested_test_freqs[0]/1e6, requested_test_freqs[-1]/1e6, expected_fc))

    def _test_restart_consistency(self):
        """Check that results are consistent on correlator restart"""
        # Removed test as correlator startup is currently unreliable,
        # will only add test method onces correlator startup is reliable.
        Aqf.tbd('Correlator restart consistency test not implemented yet.')

    def _test_delay_tracking(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking"""
        setup_data = self._delays_setup()
        sampling_period = self.corr_freqs.sample_period
        no_chans = range(len(self.corr_freqs.chan_freqs))

        test_delays = [0., sampling_period, 1.5 * sampling_period,
                       2 * sampling_period]
        test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
        delays = [0] * setup_data['num_inputs']

        def get_expected_phases():
            expected_phases = []
            for delay in test_delays:
                phases = self.corr_freqs.chan_freqs * 2 * np.pi * delay
                phases -= np.max(phases) / 2.
                expected_phases.append(phases)
            return zip(test_delays_ns, expected_phases)

        def get_actual_phases():
            actual_phases_list = []
            chan_responses = []
            for delay in test_delays:
                delays[setup_data['test_source_ind']] = delay
                delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT,
                                                              discard=0)

                future_time = 900e-3
                settling_time = 600e-3
                dump_timestamp = (this_freq_dump['sync_time'].value +
                                  this_freq_dump['timestamp'].value /
                                  this_freq_dump['scale_factor_timestamp'].value)
                t_apply = (dump_timestamp + this_freq_dump['int_time'].value +
                           future_time)
                reply = self.corr_fix.katcp_rct.req.delays(
                    t_apply, *delay_coefficients)
                Aqf.is_true(reply.reply.reply_ok(),
                            'Delays Reply: {}, Intergration time:{}s'.format(reply.reply.arguments[1], this_freq_dump['int_time'].value))
                Aqf.wait(settling_time,
                         'Settling time in order to set delay: {} ns.'.format(delay * 1e9))

                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)

                this_freq_data = this_freq_dump['xeng_raw'].value
                this_freq_response = normalised_magnitude(
                    this_freq_data[:, setup_data['test_source_ind'], :])
                chan_responses.append(this_freq_response)

                data = complexise(dump['xeng_raw'].value
                                  [:, setup_data['baseline_index'], :])

                phases = np.angle(data)
                actual_phases_list.append(phases)

            actual_channel_responses = zip(test_delays, chan_responses)
            return zip(actual_phases_list, actual_channel_responses)

        # actual_phases = get_actual_phases()
        actual_data  = get_actual_phases()
        actual_phases = [phases for phases, response in actual_data]
        actual_response = [response for phases, response in actual_data]
        expected_phases = get_expected_phases()
        title = 'CBF Delay Compensation/LO Fringe stopping polynomial'
        caption = ('Actual and expected Unwrapped Correlation Phase, '
                 'dashed line indicates expected value.')
        file_name = 'Delay_Phases_Response.svg'
        units = 'secs'

        aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                               units, file_name, title, caption)
        expected_phases = [phase for rads, phase in get_expected_phases()]
        tolerance = 1e-2
        decimal = len(str(tolerance).split('.')[-1])

        for i, delay in enumerate(test_delays):
            delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
            delta_expected = np.max(expected_phases[i]) - np.min(expected_phases[i])

            Aqf.almost_equals(delta_expected, delta_actual, tolerance,
                              'Check if difference expected({0:.5f}) and actual({1:.5f}) '
                              'phases are equal at delay {2:.5f}ns within {3} tolerance.'
                              .format(delta_expected, delta_actual, delay * 1e9, tolerance))
            try:
                delta_actual_s = delta_actual - (delta_actual % tolerance)
                delta_expected_s = delta_expected - (delta_expected % tolerance)
                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
            except AssertionError:
                Aqf.step('Difference expected({0:.5f}) and actual({1:.5f}) '
                         'phases are not equal at delay {2:.5f}ns within {3} tolerance.'
                         .format(delta_expected, delta_actual, delay * 1e9, tolerance))
                chan_response = [responses
                                for test_delays_, responses in actual_response
                                    if test_delays_ == delay][0]
                aqf_plot_channels(
                    chan_response, '{}_{}.svg'.format(self._testMethodName, delay),
                    'Log channel response at {}ns'.format(delay),
                    log_dynamic_range=90,
                    caption='Difference expected({0:.5f}) and actual({1:.5f}) '
                         'phases are not equal at delay {2:.5f}ns within {3} tolerance.'
                         .format(delta_expected, delta_actual, delay * 1e9, tolerance))

        for delay, count in zip(test_delays[1:], range(1, len(expected_phases))):
            aqf_array_abs_error_less(actual_phases[count], expected_phases[count],
                                     'Check that when a delay of {0} clock cycle({1:.5f} ns) is introduced '
                                     'there is a change of phase of {2:.5f} degrees as expected to within '
                                     '{3} tolerance.'
                                     .format((count + 1) * .5, delay * 1e9, np.rad2deg(np.pi) * (count + 1) * .5,
                                             tolerance), tolerance)

            # TODO MM 2016-01-06 Is it viable to plot the phases as they have
            # already been plotted above?
            #try:
                #np.testing.assert_array_almost_equal(actual_phases[count], expected_phases[count], decimal=decimal)
            #except AssertionError:
                #actual_phases_c = actual_phases[count]
                #expected_phases_c = expected_phases[count]
                #Aqf.step('Delay of {0} clock cycle({1:.5f} ns) was introduced '
                         #'and there was no phase change of {2:.5f} degrees as expected to within '
                         #'{3} tolerance.'
                         #.format(count_s, delay_ns, cur_degrees, tolerance))
                #aqf_plot_phase_results(no_chans, actual_phases_c, expected_phases_c,
                               #'', '{}_{}deg.svg'.format(self._testMethodName, cur_degrees),
                               #'Actual vs Expected Phases @ {} deg'.format(cur_degrees),
                               #caption='Delay of {0} clock cycle({1:.5f} ns) was introduced '
                                       #'and there was no phase change of {2:.5f} degrees '
                                       #'as expected to within {3} tolerance.'
                                       #.format(count_s, delay_ns, cur_degrees, tolerance))

    def _test_sensor_values(self):
        """
        Report sensor values (AR1)
        """
        # Request a list of available sensors using KATCP command
        try:
            sensors_req = self.corr_fix.rct.req
            array_sensors_req = self.corr_fix.katcp_rct.req

            list_reply, list_informs = sensors_req.sensor_list()
            # Confirm the CBF replies with a number of sensor-list inform messages
            LOGGER.info(list_reply, list_informs)
            sens_lst_stat, numSensors = list_reply.arguments

            array_list_reply, array_list_informs = array_sensors_req.sensor_list()
            array_sens_lst_stat, array_numSensors = array_list_reply.arguments
        except Exception as errmsg:
            Aqf.failed('KATCP connection encountered errors.: {}'.format(errmsg))

        # Confirm the CBF replies with "!sensor-list ok numSensors"
        # where numSensors is the number of sensor-list informs sent.
        numSensors = int(numSensors)
        Aqf.equals(numSensors, len(list_informs),
                   "Check that the instrument's number of sensors are equal to the "
                   "number of sensors in the list.")

        array_numSensors = int(array_numSensors)
        Aqf.equals(array_numSensors, len(array_list_informs),
                   'Check that the number of array sensors are equal to the '
                   'number of sensors in the list.')

        # Check that ?sensor-value and ?sensor-list agree about the number
        # of sensors.
        sensor_value = sensors_req.sensor_value()
        sens_val_stat, sens_val_cnt = sensor_value.reply.arguments
        Aqf.equals(int(sens_val_cnt), numSensors,
                   'Check that the instrument sensor-value and sensor-list counts are the same')

        array_sensor_value = array_sensors_req.sensor_value()
        array_sens_val_stat, array_sens_val_cnt = array_sensor_value.reply.arguments
        Aqf.equals(int(array_sens_val_cnt), array_numSensors,
                   'Check that the array sensor-value and sensor-list counts are the same')

        # Request the time synchronisation status using KATCP command
        # "?sensor-value time.synchronised
        Aqf.is_true(sensors_req.sensor_value('time.synchronised').reply.reply_ok(),
                    'Reading time synchronisation sensor failed!')

        # Confirm the CBF replies with " #sensor-value <time>
        # time.synchronised [status value], followed by a "!sensor-value ok 1"
        # message.
        Aqf.equals(str(sensors_req.sensor_value('time.synchronised')[0]), '!sensor-value ok 1',
                   'Check that the time synchronised sensor values replies with !sensor-value ok 1')

        # Check all sensors statuses if they are nominal
        for sensor in self.corr_fix.rct.sensor.values():
            LOGGER.info(sensor.name + ' :' + str(sensor.get_value()))
            Aqf.equals(sensor.get_status(), 'nominal',
                       'Sensor status: {}, {} '
                       .format(sensor.name, sensor.get_status()))

    def _test_roach_qdr_sensors(self):

        def roach_qdr(corr_hosts, engine_type, sensor_timeout=30):
            try:
                array_sensors = self.corr_fix.katcp_rct.sensor
                hosts = corr_hosts
                host = hosts[randrange(len(hosts))]
                Aqf.step("Selected host: {} - {}".format(host.host, engine_type))
                host_sensor = getattr(array_sensors, '{}_{}_qdr'.format(
                                      host.host.lower(), engine_type))

                # Check if qdr is okay
                Aqf.is_true(host_sensor.get_value(),
                            'Confirm that sensor indicates QDR status: {} on {}.'
                            .format(host_sensor.status, host.host))
                host_sensor.set_strategy('auto')
                self.addCleanup(host_sensor.set_sampling_strategy, 'none')

                Aqf.step("Writing junk to {} memory.".format(host.host))
                # Write junk to memory
                for i in range(100):
                    host.blindwrite('qdr0_memory', 'write_junk_to_memory')
                    host.blindwrite('qdr1_memory', 'write_junk_to_memory')

                if host_sensor.wait(False, timeout=sensor_timeout):
                    # Verify that qdr corrupted or unreadable
                    Aqf.equals(host_sensor.get_status(), 'error',
                               'Confirm that sensor indicates that the memory on {} '
                               'is unreadable/corrupted.'
                               .format(host.host))
                else:
                    Aqf.failed('Confirm that sensor indicates that memory is unreadable/corrupted.')

                if engine_type == 'xeng':
                    current_errors = host.registers.vacc_errors1.read()['data']['parity']
                else:
                    ct_ctrs = host.registers.ct_ctrs.read()['data']
                    current_errors = (ct_ctrs['ct_parerr_cnt0'] + ct_ctrs['ct_parerr_cnt1'])

                Aqf_is_not_equals(current_errors, 0, "Confirm that the error counters incremented.")

                if engine_type == 'xeng':
                    if current_errors == host.registers.vacc_errors1.read()['data']['parity']:
                        Aqf.passed('Confirm that the error counters have stopped incrementing: '
                                   '{} increments.'.format(current_errors))

                else:
                    new_errors = (host.registers.ct_ctrs.read()['data']['ct_parerr_cnt0'] +
                                  host.registers.ct_ctrs.read()['data']['ct_parerr_cnt1'])
                    if current_errors == new_errors:
                        Aqf.passed('Confirm that the error counters have stopped incrementing: '
                                   '{} increments.'.format(current_errors))
                # Clear and confirm error counters
                host.clear_status()
                if engine_type == 'xeng':
                    final_errors = host.registers.vacc_errors1.read()['data']['parity']
                else:
                    final_errors = (host.registers.ct_ctrs.read()['data']['ct_parerr_cnt0'] +
                                    host.registers.ct_ctrs.read()['data']['ct_parerr_cnt1'])

                Aqf.is_false(final_errors,
                             'Confirm that the counters have been reset, count {} to {}'
                             .format(current_errors, final_errors))

                if host_sensor.wait(True, timeout=sensor_timeout):
                    Aqf.is_true(host_sensor.get_value(),
                                'Confirm that sensor indicates that the QDR memory recovered. '
                                'Status: {} on {}.\n\n'.format(host_sensor.status, host.host))
                else:
                    Aqf.failed('QDR sensor failed to recover. '
                               'Status: {} on {}.\n\n'.format(host_sensor.status, host.host))
            except Exception:
                Aqf.failed('KATCP connection encountered errors.')

        roach_qdr(self.correlator.fhosts, 'feng')
        roach_qdr(self.correlator.xhosts, 'xeng')

    def _test_roach_pfb_sensors(self):
        """Sensor PFB error"""
        array_sensors = self.corr_fix.katcp_rct.sensor
        Aqf.skipped('PFB sensor test not yet implemented.')


    def _test_feng_link_error(self):
        # Select an F-engine that is being used to produce the test data product on
        # which to trigger the link error.
        # To set: Using the roach2 KATCP interface
        # Record the current multicast desitination of one of the F-engine data
        # ethernet ports,
        # configure the same port multicast destination to an unused address,
        # effectively dropping that data.
        # To clear: Restore the data ethernet port multicast destination to the saved
        # value.
        Aqf.tbd('Link Error: F-engine to X-engine test not yet implemented.')

    def _test_deng_link_error(self):
        # To set: Disable data output on one of the digitiser simulator's outputs
        # To clear: Enabkle data output on all the digitiser simulator outputs
        Aqf.tbd('Link Error: D-engine to F-engine test not yet implemented.')

    def _test_roach_sensors_status(self):
        """ Test all roach sensors status are not failing and count verification."""
        for roach in (self.correlator.fhosts + self.correlator.xhosts):
            values_reply, sensors_values = roach.katcprequest('sensor-value')
            list_reply, sensors_list = roach.katcprequest('sensor-list')
            # Verify the number of sensors received with
            # number of sensors in the list.
            Aqf.is_true((values_reply.reply_ok() == list_reply.reply_ok()),
                        '{}: Verify that ?sensor-list and ?sensor-value agree '
                        'about the number of sensors.'.format(roach.host))

            # Check the number of sensors in the list is equal to the list
            # of values received.
            Aqf.equals(len(sensors_list), int(values_reply.arguments[1]),
                       'Check the number of sensors in the list is equal to the '
                       'list of values received for {}\n'.format(roach.host))

            for sensor in sensors_values[1:]:
                sensor_name, sensor_status, sensor_value = (
                    sensor.arguments[2:])
                # Check if roach sensors are failing
                Aqf.is_false((sensor_status == 'fail'),
                             'Roach {}, Sensor name: {}, status: {}'
                             .format(roach.host, sensor_name, sensor_status))

    def _test_vacc(self):
        """Test vector accumulator"""
        # Choose a test freqency around the centre of the band.
        test_freq = self.corr_freqs.bandwidth / 2.
        sources = [input['source'].name
                   for input in self.correlator.fengine_sources]
        test_input = sources[0]
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
        reply, informs = self.corr_fix.katcp_rct.req.gain(test_input, *list(eqs))
        Aqf.step('Gain factors set {}.'.format(reply.arguments[0]))
        self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=0.125,
                                          # Make dsim output periodic in FFT-length
                                          # so that each FFT is identical
                                          repeatN=self.corr_freqs.n_chans * 2)
        # The re-quantiser outputs signed int (8bit), but the snapshot code
        # normalises it to floats between -1:1. Since we want to calculate the
        # output of the vacc which sums integers, denormalise the snapshot
        # output back to ints.
        q_denorm = 128
        quantiser_spectrum = get_quant_snapshot(
            self.correlator, test_input) * q_denorm
        Aqf.step('Test input: {}, Test Channel :{}'.format(test_input,
                                                           test_freq_channel))
        # Check that the spectrum is not zero in the test channel
        Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
                    'Check that the spectrum is not zero in the test channel')
        # Check that the spectrum is zero except in the test channel
        Aqf.is_true(np.all(quantiser_spectrum[0:test_freq_channel] == 0),
                    'Check that the spectrum is zero except in the test channel: [0:{}(test_freq_channel)]'.format(test_freq_channel))
        Aqf.is_true(np.all(quantiser_spectrum[test_freq_channel + 1:] == 0),
                    'Check that the spectrum is zero except in the test channel: [(test_freq_channel){}+1:]'.format(test_freq_channel))

        chan_response = []
        success = False
        for vacc_accumulations in test_acc_lens:
            self.xengops.set_acc_len(vacc_accumulations)
            no_accs = internal_accumulations * vacc_accumulations
            expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
            d = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            response = complexise(d['xeng_raw'].value[:, 0, :])
            chan_response.append(normalised_magnitude(d['xeng_raw'].value[:, 0, :]))

            # Check that the accumulator response is equal to the expected response
            if not Aqf.is_true(np.array_equal(expected_response, response),
                        'Check that the accumulator response is equal '
                        'to the expected response for {} accumulation length'
                        .format(vacc_accumulations)):
                            success = True
        if not success:
            legends = ['Acc Len: {}'.format(x) for x in test_acc_lens]
            aqf_plot_channels(
                            zip(chan_response, legends),
                            plot_filename='chan_resp_vacc.svg',
                            plot_title='Vector Accumulation Length', log_dynamic_range=90,
                            log_normalise_to=1,
                            caption='Check that the accumulator response is equal '
                                    'to the expected response for {} accumulation length '
                                    'with source periodic every {} samples and sine frequency of '
                                    '{} MHz. '
                                    .format(test_acc_lens, self.corr_freqs.n_chans * 2, test_freq))

    def _test_product_switch(self, instrument, no_channels):
        Aqf.step('Confirm that SPEAD packets are being produced')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        Aqf.step('Stopped Xengine data capturing/receiving.')
        self.corr_fix.stop_x_data()
        Aqf.step('Deprogramming xhosts first then fhosts avoid reorder timeout errors')
        xhosts = self.correlator.xhosts
        fhosts = self.correlator.fhosts

        fpgautils.threaded_fpga_function(xhosts, 10, 'deprogram')
        fpgautils.threaded_fpga_function(fhosts, 10, 'deprogram')
        [Aqf.is_false(host.is_running(), '{} Deprogrammed'.format(host.host))
         for host in xhosts + fhosts]
        # Confirm that SPEAD packets are either no longer being produced, or
        # that the data content is at least affected.
        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
            Aqf.failed('SPEAD packets are still being produced.')
        except Queue.Empty:
            Aqf.passed('Check that SPEAD packets are nolonger being produced.')

        self.corr_fix.halt_array()
        Aqf.step('Initialising {instrument} instrument'.format(**locals()))
        corr_init = False
        retries = 5
        start_time = time.time()
        while retries and not corr_init:
            try:
                self.set_instrument(instrument)
                self.corr_fix.start_x_data()
                corr_init = True
                retries -= 1
                if corr_init == True:
                    LOGGER.info('Correlator started successfully after {} retries'
                                .format(retries))
            except Exception:
                retries -= 1
                if retries == 0:
                    errmsg = 'Could not restart the correlator after 5 tries.'
                    Aqf.failed(errmsg)
                    LOGGER.info(errmsg)

        if corr_init == True:
            [Aqf.is_true(host.is_running(),
                         'Confirm that the instrument is initialised by checking if '
                         'roaches are programmed.\n{} programmed and running'
                         .format(host.host)) for host in xhosts + fhosts]

            Aqf.hop('Waiting to receive SPEAD data')
            re_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            end_time = time.time()

            Aqf.is_true(re_dump,
                        'Confirm that SPEAD packets are being produced after instrument '
                        're-initialisation.')

            Aqf.equals(re_dump['xeng_raw'].value.shape[0], no_channels,
                       'Check that data product has the number of frequency '
                       'channels {no_channels} corresponding to the {instrument} '
                       'instrument product'.format(**locals()) )

            final_time = end_time - start_time
            minute = 60.0
            Aqf.less(final_time, minute,
                     'Confirm that instrument switching to {instrument} time is '
                     'less than one minute'.format(**locals()) )

    def _test_adc_overflow_flag(self):
        """CBF flagging of data -- ADC overflow"""

        # TODO 2015-09-22 (NM): Test is currently failing since the noise diode flag is
        # also set when the overange occurs. Needs to check if the dsim is doing this or
        # if it is an error in the CBF. 2015-09-30 update: Nope, Dsim seems to be fine,
        # only the adc bit is set in the SPEAD header, checked many packets by network
        # packet capture.
        def enable_adc_overflow():
            self.dhost.registers.flag_setup.write(adc_flag=1, load_flags='pulse')

        def disable_adc_overflow():
            self.dhost.registers.flag_setup.write(adc_flag=0, load_flags='pulse')

        condition = 'ADC overflow flag on the digitiser simulator'
        dump1, dump2, dump3, = self.get_flag_dumps(enable_adc_overflow,
                                                   disable_adc_overflow, condition)
        flag_bit = flags_xeng_raw_bits.overrange
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(flags_xeng_raw_bits)
        other_bits = all_bits - set([flag_bit])
        flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
        flag_condition = 'ADC overrange'

        set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value, consider_bits=all_bits)
        Aqf.is_false(flag_bit in set_bits1,
                     'Check that {} is not set in dump 1 before setting {}.'
                     .format(flag_descr, condition))
        # Bits that should not be set
        other_set_bits1 = set_bits1.intersection(other_bits)
        Aqf.equals(other_set_bits1, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits2 = get_set_bits(dump2['flags_xeng_raw'].value, consider_bits=all_bits)
        other_set_bits2 = set_bits2.intersection(other_bits)
        Aqf.is_true(flag_bit in set_bits2,
                    'Check that {} is set in dump 2 while toggeling {}.'
                    .format(flag_descr, condition))
        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value, consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in dump 3 after clearing {}.'
                     .format(flag_descr, condition))
        Aqf.equals(other_set_bits3, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

    def _test_noise_diode_flag(self):
        """CBF flagging of data -- noise diode fired"""

        def enable_noise_diode():
            self.dhost.registers.flag_setup.write(ndiode_flag=1, load_flags='pulse')

        def disable_noise_diode():
            self.dhost.registers.flag_setup.write(ndiode_flag=0, load_flags='pulse')

        condition = 'Noise diode flag on the digitiser simulator'
        dump1, dump2, dump3, = self.get_flag_dumps(
            enable_noise_diode, disable_noise_diode, condition)

        flag_bit = flags_xeng_raw_bits.noise_diode
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(flags_xeng_raw_bits)
        other_bits = all_bits - set([flag_bit])
        flag_descr = 'noise diode fired, bit {},'.format(flag_bit)
        flag_condition = 'digitiser noise diode fired flag'

        set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value, consider_bits=all_bits)
        Aqf.is_false(flag_bit in set_bits1,
                     'Check that {} is not set in dump 1 before setting {}.'
                     .format(flag_descr, condition))

        # Bits that should not be set
        other_set_bits1 = set_bits1.intersection(other_bits)
        Aqf.equals(other_set_bits1, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits2 = get_set_bits(dump2['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits2 = set_bits2.intersection(other_bits)
        Aqf.is_true(flag_bit in set_bits2,
                    'Check that {} is set in dump 2 while toggeling {}.'
                    .format(flag_descr, condition))

        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in dump 3 after clearing {}.'
                     .format(flag_descr, condition))

        Aqf.equals(other_set_bits3, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

    def _test_fft_overflow_flag(self):
        """CBF flagging of data -- FFT overflow"""
        freq = self.corr_freqs.bandwidth / 2.

        def enable_fft_overflow():
            # TODO 2015-09-22 (NM) There seems to be some issue with the dsim sin_corr
            # source that results in it producing all zeros... So using sin_0 and sin_1
            # instead
            # self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=1.)
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=1.)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=1.)
            # Set FFT to never shift, ensuring an FFT overflow with the large tone we are
            # putting in.
            self.fengops.set_fft_shift_all(shift_value=0)

        def disable_fft_overflow():
            # TODO 2015-09-22 (NM) There seems to be some issue with the dsim sin_corr
            # source that results in it producing all zeros... So using sin_0 and sin_1
            # instead
            # self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=0)
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=0.)
            # Restore the default FFT shifts as per the correlator config.
            self.fengops.set_fft_shift_all()

        condition = ('FFT overflow by setting an agressive FFT shift with '
                     'a pure tone input')
        dump1, dump2, dump3, = self.get_flag_dumps(enable_fft_overflow,
                                                   disable_fft_overflow, condition)
        flag_bit = flags_xeng_raw_bits.overrange
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(flags_xeng_raw_bits)
        other_bits = all_bits - set([flag_bit])
        flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
        flag_condition = 'FFT overrange'

        set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        Aqf.is_false(flag_bit in set_bits1,
                     'Check that {} is not set in dump 1 before setting {}.'
                     .format(flag_descr, condition))
        # Bits that should not be set
        other_set_bits1 = set_bits1.intersection(other_bits)
        Aqf.equals(other_set_bits1, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits2 = get_set_bits(dump2['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits2 = set_bits2.intersection(other_bits)
        Aqf.is_true(flag_bit in set_bits2,
                    'Check that {} is set in dump 2 while toggeling {}.'
                    .format(flag_descr, condition))
        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in dump 3 after clearing {}.'
                     .format(flag_descr, condition))

        Aqf.equals(other_set_bits3, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

    def _test_fringe_offset(self):
        """CBF per-antenna phase error -- Fringe offset"""
        # TODO Randomise test values
        setup_data = self._delays_setup()
        dump_counts = 5
        delay_value = 0
        delay_rate = 0
        fringe_offset = np.pi / 2.
        fringe_rate = 0
        load_time = setup_data['t_apply']
        load_check = False
        fringe_offsets = [0] * setup_data['num_inputs']
        fringe_offsets[setup_data['test_source_ind']] = fringe_offset
        delay_coefficients = ['0,0:{},0'.format(fo) for fo in fringe_offsets]

        Aqf.step('Setting Parameters')
        Aqf.step('Time apply: {}'.format(load_time))
        Aqf.step('Delay Rate: {}'.format(delay_rate))
        Aqf.step('Delay Value: {}'.format(delay_value))
        Aqf.step('Fringe Offset: {}'.format(fringe_offset))
        Aqf.step('Fringe Rate: {}'.format(fringe_rate))

        actual_data = self._get_actual_data(setup_data, dump_counts,
                                              delay_coefficients)
        actual_phases = [phases for phases, response in actual_data]
        actual_response = [response for phases, response in actual_data]

        expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                  delay_coefficients, actual_phases)
        no_chans = setup_data['no_chans']
        graph_units = 'rads'
        graph_title = 'Fringe Offset at {} {}.'.format(fringe_offset, graph_units)
        graph_name = 'Fringe_Offset_Response.svg'
        caption = ('Actual and expected Unwrapped Correlation Phase, '
                   'dashed line indicates expected value.')

        aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                               graph_units, graph_name, graph_title, caption)

        # Ignoring first dump because the delays might not be set for full
        # intergration.
        tolerance = 1e-5
        decimal = len(str(tolerance).split('.')[-1])
        actual_phases = np.unwrap(actual_phases)
        expected_phases = np.unwrap([phase for label, phase in expected_phases])

        for i in range(1, len(expected_phases) - 1):
            delta_expected = np.max(expected_phases[i])
            delta_actual = np.max(actual_phases[i])
            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))

            Aqf.almost_equals(delta_expected, delta_actual, tolerance,
                              'Check if difference expected({}) and actual({}) '
                              'phases are equal withing {} tolerance when fringe offset is {}.'
                              .format(delta_expected, delta_actual, tolerance, fringe_offset))

            Aqf.less(abs_diff, 1,
                    'Check that the maximum degree between '
                    'expected and actual phase difference between integrations '
                    'is below 1 degree: {} degree\n'.format(abs_diff))

            try:
                delta_actual_s = delta_actual - (delta_actual % tolerance)
                delta_expected_s = delta_expected - (delta_expected % tolerance)
                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

            except AssertionError:
                Aqf.step('Difference expected({0:.5f}) and actual({1:.5f}) '
                         'phases are not equal within {2} tolerance when fringe offset is {3}.'
                         .format(delta_expected, delta_actual, tolerance, fringe_offset))
                aqf_plot_channels(
                    actual_response[-1], '{}_{}.svg'.format(self._testMethodName, fringe_offset),
                    'Log channel response of Fringe offset: {}rads'.format(fringe_offset),
                    log_dynamic_range=90, log_normalise_to=1,
                    caption='Difference expected({0:.5f}) and actual({1:.5f}) '
                            'phases are not equal within {2} tolerance when fringe offset is {3}.'
                            .format(delta_expected, delta_actual, tolerance, fringe_offset))

    def _test_delay_rate(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate"""

        setup_data = self._delays_setup()
        dump_counts = 5
        delay_value = 0
        # TODO Randomise test values
        #delay_rate = randrange(0, 7.645783361528156e-05, int=float)
        delay_rate = setup_data['sample_period'] / setup_data['int_time']
        fringe_offset = 0
        fringe_rate = 0
        load_time = setup_data['t_apply']
        load_check = False
        delay_rates = [0] * setup_data['num_inputs']
        delay_rates[setup_data['test_source_ind']] = delay_rate
        delay_coefficients = ['0,{}:0,0'.format(fr) for fr in delay_rates]
        Aqf.step('Setting Parameters')
        Aqf.step('Time apply: {}'.format(load_time))
        Aqf.step('Delay Rate: {}'.format(delay_rate))
        Aqf.step('Delay Value: {}'.format(delay_value))
        Aqf.step('Fringe Offset: {}'.format(fringe_offset))
        Aqf.step('Fringe Rate: {}'.format(fringe_rate))

        actual_data = self._get_actual_data(setup_data, dump_counts,
                                              delay_coefficients)
        actual_phases = [phases for phases, response in actual_data]
        actual_response = [response for phases, response in actual_data]

        expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                  delay_coefficients, actual_phases)
        no_chans = setup_data['no_chans']
        graph_units = ' '
        graph_title = 'Delay Rate at {} ns/s'.format(delay_rate * 1e9)
        graph_name = 'Delay_Rate_Response.svg'
        caption =  ('Actual and expected Unwrapped Correlation Delay Rate, '
                    'dashed line indicates expected value.')

        aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                               graph_units, graph_name, graph_title, caption)

        actual_phases = np.unwrap(actual_phases)
        # TODO MM 2015-10-22
        # Ignoring first dump because the delays might not be set for full
        # intergration.
        tolerance = 0.01
        decimal = len(str(tolerance).split('.')[-1])
        expected_phases = np.unwrap([phase for label, phase in expected_phases])
        for i in range(1, len(expected_phases) - 1):
            delta_expected = np.max(expected_phases[i + 1] - expected_phases[i])
            delta_actual = np.max(actual_phases[i + 1] - actual_phases[i])
            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))

            Aqf.almost_equals(delta_expected, delta_actual, tolerance,
                              'Check if difference expected({0:.5f}) and actual({1:.5f}) '
                              'phases are equal withing {2} tolerance when delay rate is {3}.'
                              .format(delta_expected, delta_actual, tolerance, delay_rate))

            Aqf.less(abs_diff, 1,
                     'Check that the maximum degree between expected and actual phase '
                     'difference between integrations is below 1 degree: {0:.3f} degree\n'
                     .format(abs_diff))
            try:
                delta_actual_s = delta_actual - (delta_actual % tolerance)
                delta_expected_s = delta_expected - (delta_expected % tolerance)
                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

            except AssertionError:
                Aqf.step('Difference expected({0:.5f}) and actual({1:.5f}) '
                         'phases are not equal withing {2} tolerance when delay rate is {3}.'
                          .format(delta_expected, delta_actual, tolerance, delay_rate))
                legends = ['Response per dump #{}'.format(x) for x in range(len(actual_response))]
                aqf_plot_channels(
                    zip(actual_response, legends), '{}.svg'.format(self._testMethodName),
                    'Log channel response of Delay actual: {}'.format(delta_actual),
                    log_dynamic_range=90,
                    caption='Difference expected({0:.5f}) and actual({1:.5f}) '
                            'phases are not equal withing {2} tolerance when delay rate is {3}.'
                            .format(delta_expected, delta_actual, tolerance, delay_rate))

    def _test_fringe_rate(self):
        """CBF per-antenna phase error -- Fringe rate"""
        # TODO Randomise test values
        setup_data = self._delays_setup()
        dump_counts = 5
        delay_value = 0
        delay_rate = 0
        fringe_offset = 0
        fringe_rate = (np.pi / 8.) / setup_data['int_time']
        load_time = setup_data['t_apply']
        load_check = False
        fringe_rates = [0] * setup_data['num_inputs']
        fringe_rates[setup_data['test_source_ind']] = fringe_rate
        delay_coefficients = ['0,0:0,{}'.format(fr) for fr in fringe_rates]

        Aqf.step('Setting Parameters')
        Aqf.step('Time apply: {}'.format(load_time))
        Aqf.step('Delay Rate: {}'.format(delay_rate))
        Aqf.step('Delay Value: {}'.format(delay_value))
        Aqf.step('Fringe Offset: {}'.format(fringe_offset))
        Aqf.step('Fringe Rate: {}'.format(fringe_rate))

        actual_data = self._get_actual_data(setup_data, dump_counts,
                                              delay_coefficients)

        actual_phases = [phases for phases, response in actual_data]
        actual_response = [response for phases, response in actual_data]

        expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                  delay_coefficients, actual_phases)

        no_chans = setup_data['no_chans']
        graph_units = 'rads/sec'
        caption = ('Actual and expected Unwrapped Correlation Phase Rate, '
                   'dashed line indicates expected value.')
        graph_title = 'Fringe Rate at {} {}.'.format(fringe_rate, graph_units)
        graph_name = 'Fringe_Rate_Response.svg'
        aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                               graph_units, graph_name, graph_title, caption)

        # NOTE: Ignoring first dump because the delays might not be set for full
        # intergration.
        tolerance = 0.01
        decimal = len(str(tolerance).split('.')[-1])
        actual_phases = np.unwrap(actual_phases)
        expected_phases = np.unwrap([phase for label, phase in expected_phases])

        for i in range(1, len(expected_phases) - 1):
            delta_expected = np.max(expected_phases[i + 1] - expected_phases[i])
            delta_actual = np.max(actual_phases[i + 1] - actual_phases[i])
            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))

            Aqf.almost_equals(delta_expected, delta_actual, tolerance,
                              'Check if difference expected({0:.5f}) and actual({1:.5f}) '
                              'phases are equal within {2} tolerance when fringe rate is {3}.'
                              .format(delta_expected, delta_actual, tolerance, fringe_rate))

            Aqf.less(abs_diff, 1,
                     'Check that the maximum degree between '
                     'expected and actual phase difference between integrations '
                     'is below 1 degree: {0:.3f} degree\n'.format(abs_diff))

            try:
                delta_actual_s = delta_actual - (delta_actual % tolerance)
                delta_expected_s = delta_expected - (delta_expected % tolerance)
                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
            except AssertionError:
                Aqf.step('Difference expected({0:.5f}) and actual({1:.5f}) '
                         'phases are not equal within {2} tolerance when fringe rate is {3}.'
                         .format(delta_expected, delta_actual, tolerance, fringe_rate))

                legends = ['Response per dump #{}'.format(x) for x in range(len(actual_response))]
                aqf_plot_channels(
                    zip(actual_response, legends), '{}.svg'.format(self._testMethodName),
                    'Log channel response of Fringe offset: {}rads'.format(fringe_offset),
                    log_dynamic_range=90,
                    caption='Difference expected({0:.5f}) and actual({1:.5f}) '
                            'phases are not equal within {2} tolerance when fringe rate is {3}.'
                            .format(delta_expected, delta_actual, tolerance, fringe_rate))

    def _test_all_delays(self):
        """
        CBF per-antenna phase error -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        # TODO Randomise test values
        setup_data = self._delays_setup()
        dump_counts = 5
        delay_value = setup_data['sample_period'] * 1.5
        delay_rate = setup_data['sample_period'] / setup_data['int_time']
        fringe_offset = np.pi / 4.
        fringe_rate = (np.pi / 4.) / setup_data['int_time']
        load_time = setup_data['t_apply']
        load_check = False

        delay_values = [0] * setup_data['num_inputs']
        delay_rates = [0] * setup_data['num_inputs']
        fringe_offsets = [0] * setup_data['num_inputs']
        fringe_rates = [0] * setup_data['num_inputs']

        delay_values[setup_data['test_source_ind']] = delay_value
        delay_rates[setup_data['test_source_ind']] = delay_rate
        fringe_offsets[setup_data['test_source_ind']] = fringe_offset
        fringe_rates[setup_data['test_source_ind']] = fringe_rate
        delay_coefficients = []
        for idx in range(len(delay_values)):
            delay_coefficients.append('{},{}:{},{}'
                                      .format(delay_values[idx], delay_rates[idx],
                                              fringe_offsets[idx], fringe_rates[idx]))

        Aqf.step('Setting Parameters')
        Aqf.step('Time apply: {}'.format(load_time))
        Aqf.step('Delay Rate: {}'.format(delay_rate))
        Aqf.step('Delay Value: {}'.format(delay_value))
        Aqf.step('Fringe Offset: {}'.format(fringe_offset))
        Aqf.step('Fringe Rate: {}'.format(fringe_rate))

        expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                  delay_coefficients)

        actual_phases = self._get_actual_data(setup_data, dump_counts,
                                              delay_coefficients)

        no_chans = setup_data['no_chans']
        graph_units = ''
        graph_title = 'All Delays Responses'
        graph_name = 'All_Delays_Response.svg'

        aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                               graph_units, graph_name, graph_title)

        # Ignoring first dump because the delays might not be set for full
        # intergration.
        tolerance = 0.01
        actual_phases = np.unwrap(actual_phases)
        expected_phases = np.unwrap([phase for label, phase in expected_phases])
        for i in range(1, len(expected_phases) - 1):
            delta_expected = np.max(expected_phases[i + 1] - expected_phases[i])
            delta_actual = np.max(actual_phases[i + 1] - actual_phases[i])
            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))

            Aqf.almost_equals(delta_expected, delta_actual, tolerance,
                              'Check if difference expected({0:.5f}) and actual({1:.5f}) '
                              'phases are equal withing {2} tolerance when delay rate is {}.'
                              .format(delta_expected, delta_actual, tolerance, delay_rate))
            # TODO Plot spectrums if test fails as per above tests
            Aqf.less(abs_diff, 1,
                     'Check that the maximum degree between expected and actual phase'
                     ' difference between intergrations is below 1 degree: {0:.3f}'
                     ' degree\n'.format(abs_diff))

    def _test_config_report(self):
        """CBF Report configuration"""
        test_config = self.corr_fix.test_conf
        def get_roach_config():

            Aqf.hop('DEngine :{}'.format(self.dhost.host))

            fhosts = [fhost.host for fhost in self.correlator.fhosts]
            Aqf.hop('Available FEngines :{}'.format(', '.join(fhosts)))

            xhosts = [xhost.host for xhost in self.correlator.xhosts]
            Aqf.hop('Available XEngines :{}\n'.format(', '.join(xhosts)))

            uboot_cmd = 'cat /dev/mtdblock5 | less | strings | head -1\n'
            romfs_cmd = 'cat /dev/mtdblock1 | less | strings | head -2 | tail -1\n'
            lnx_cmd = 'cat /dev/mtdblock0 | less | strings | head -1\n'

            for count, host in enumerate((self.correlator.fhosts +
                                              self.correlator.xhosts), start=1):
                hostname = host.host
                Aqf.step('Host {}: {}'.format(count, hostname))
                user = 'root\n'
                try:
                    tn = telnetlib.Telnet(hostname)

                    tn.read_until('login: ')
                    tn.write(user)
                    tn.write(uboot_cmd)
                    tn.write(romfs_cmd)
                    tn.write(lnx_cmd)
                    tn.write("exit\n")
                    stdout = tn.read_all()
                    tn.close()

                    Aqf.hop('Gateware :{}'.format(', Build Date: '.join(
                        host.system_info.values()[1::2])))

                    uboot_ver = stdout.splitlines()[-6]
                    Aqf.hop('Current UBoot Version: {}'.format(uboot_ver))

                    romfs_ver = stdout.splitlines()[-4]
                    Aqf.hop('Current ROMFS Version: {}'.format(romfs_ver))

                    linux_ver = stdout.splitlines()[-2]
                    Aqf.hop('Linux Version: {}\n'.format(linux_ver))
                except Exception:
                    Aqf.failed('Could not connect to host: {}'.format(hostname))

        def get_src_dir():

            import corr2
            import casperfpga
            import katcp
            import spead2

            corr2_dir, _None = os.path.split(os.path.split(corr2.__file__)[0])
            corr2_name = corr2.__name__

            casper_dir, _None = os.path.split(os.path.split(casperfpga.__file__)[0])
            casper_name = casperfpga.__name__

            katcp_dir, _None = os.path.split(os.path.split(katcp.__file__)[0])
            katcp_name = katcp.__name__

            spead2_dir, _None = os.path.split(os.path.split(spead2.__file__)[0])
            spead2_name = spead2.__name__

            bitstream_dir = self.correlator.configd['xengine']['bitstream']
            mkat_dir, _None = os.path.split(os.path.split(os.path.dirname(
                          os.path.realpath(bitstream_dir)))[0])
            _None, mkat_name = os.path.split(mkat_dir)

            test_dir, test_name = os.path.split(os.path.dirname(
                                  os.path.realpath(__file__)))

            return {corr2_name: corr2_dir,
                    casper_name: casper_dir,
                    katcp_name: katcp_dir,
                    spead2_name: spead2_dir,
                    mkat_name: mkat_dir,
                    test_name: test_dir}

        def get_package_versions():
            for name, repo_dir in get_src_dir().iteritems():
                git_hash = subprocess.check_output(['git', '--git-dir={}/.git'
                                                   .format(repo_dir), '--work-tree={}'
                                                   .format(repo_dir), 'rev-parse',
                                                   '--short', 'HEAD']).strip()

                git_branch = subprocess.check_output(['git', '--git-dir={}/.git'
                                                     .format(repo_dir), '--work-tree={}'
                                                     .format(repo_dir), 'rev-parse',
                                                     '--abbrev-ref', 'HEAD']).strip()

                Aqf.hop('Repo: {}, Branch: {}, Last Hash: {}'
                           .format(name, git_branch, git_hash))

                git_diff = subprocess.check_output(
                        ['git', '--git-dir={}/.git'.format(repo_dir),
                         '--work-tree={}'.format(repo_dir), 'diff', 'HEAD'])
                if bool(git_diff):
                    Aqf.progress('Repo: {}: Contains changes not staged for commit.\n\n'
                                 'Difference: \n\n{}'
                                 .format(name, clrs.red(git_diff)))
                else:
                    Aqf.hop('Repo: {}: Up-to-date.\n\n'.format(name))

        def get_pdu_config():
            host_ips = test_config['pdu_hosts']['pdu_ips'].split(',')
            for count, host_ip in enumerate(host_ips, start=1):
                user = 'apc\r\n'
                password = 'apc\r\n'
                model_cmd = 'prodInfo\r\n'
                about_cmd = 'about\r\n'
                try:
                    tn = telnetlib.Telnet(host_ip, timeout=10)

                    tn.read_until('User Name : ')
                    tn.write(user)
                    if password:
                        tn.read_until("Password")
                        tn.write(password)

                    tn.write(model_cmd)
                    tn.write(about_cmd)
                    tn.write("exit\r\n")
                    stdout = tn.read_all()
                    tn.close()

                    if 'Model' in stdout:
                        pdu_model = stdout[stdout.index('Model'):].split()[1]
                        Aqf.step('Checking PDU no: {}'.format(count))
                        Aqf.hop('PDU Model: {} on {}'.format(pdu_model, host_ip))

                    if 'Name' in stdout:
                        pdu_name = (' '.join(stdout[stdout.index('Name'):stdout.index(
                                    'Date')].split()[-4:]))
                        Aqf.hop('PDU Name: {}'.format(pdu_name))

                    if 'Serial' in stdout:
                        pdu_serial = (stdout[stdout.find('Hardware Factory'):]
                                     .splitlines()[3].split()[-1])
                        Aqf.hop('PDU Serial Number: {}'.format(pdu_serial))

                    if 'Revision' in stdout:
                        pdu_hw_rev = (stdout[stdout.find('Hardware Factory'):]
                                     .splitlines()[4].split()[-1])
                        Aqf.hop('PDU HW Revision: {}'.format(pdu_hw_rev))

                    if 'Application Module' and 'Version' in stdout:
                        pdu_app_ver = (stdout[stdout.find('Application Module'):]
                                      .split()[6])
                        Aqf.hop('PDU Application Module Version: {} '.format(
                                    pdu_app_ver))

                    if 'APC OS(AOS)' in stdout:
                        pdu_apc_name = (stdout[stdout.find('APC OS(AOS)'):]
                                       .splitlines()[2].split()[-1])
                        pdu_apc_ver = (stdout[stdout.find('APC OS(AOS)'):]
                                      .splitlines()[3].split()[-1])
                        Aqf.hop('PDU APC OS: {}'.format(pdu_apc_name))
                        Aqf.hop('PDU APC OS ver: {}'.format(pdu_apc_ver))

                    if 'APC Boot Monitor' in stdout:
                        pdu_apc_boot = (stdout[stdout.find('APC Boot Monitor'):]
                                        .splitlines()[2].split()[-1])
                        pdu_apc_ver = (stdout[stdout.find('APC Boot Monitor'):]
                                      .splitlines()[3].split()[-1])
                        Aqf.hop('PDU APC Boot Mon: {}'.format(pdu_apc_boot))
                        Aqf.hop('PDU APC Boot Mon Ver: {}\n'.format(pdu_apc_ver))
                except Exception:
                    Aqf.failed('Could not connect to PDU host: {}'.format(host_ip))

        def get_data_switch():
            '''Verify info on each Data Switch'''
            host_ips = test_config['data_switch_hosts']['data_switch_ips'].split(',')
            username = 'admin'
            password = 'admin'
            nbytes = 2048
            wait_time = 1
            for count, ip in enumerate(host_ips, start=1):
                try:
                    remote_conn_pre = paramiko.SSHClient()
                    # Load host keys from a system file.
                    remote_conn_pre.load_system_host_keys()
                    remote_conn_pre.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    remote_conn_pre.connect(ip, username=username, password=password,
                                            timeout=10)
                    remote_conn = remote_conn_pre.invoke_shell()
                    Aqf.step('Connected to Data switch {} on IP: {}'.format(count, ip))
                except paramiko.SSHException:
                    Aqf.failed('Failed to connect to Data switch {} on IP: {}'.format(
                                count, ip))

                remote_conn.send("\n")
                while not remote_conn.recv_ready():
                    time.sleep(wait_time)
                remote_conn.recv(nbytes)

                remote_conn.send("show inventory | include CHASSIS\n")
                while not remote_conn.recv_ready():
                    time.sleep(wait_time)
                inventory = remote_conn.recv(nbytes)
                if 'CHASSIS' in inventory:
                    part_number = inventory.split()[8]
                    Aqf.hop('Data Switch Part no: {}'.format(part_number))
                    serial_number = inventory.split()[9]
                    Aqf.hop('Data Switch Serial no: {}'.format(serial_number))

                remote_conn.send("show version\n")
                while not remote_conn.recv_ready():
                    time.sleep(wait_time)
                version = remote_conn.recv(nbytes)
                if 'version' in version:
                    prod_name = version[version.find('Product name:'):].split()[2]
                    Aqf.hop('Software Product name: {}'.format(prod_name))
                    prod_rel = version[version.find('Product release:'):].split()[2]
                    Aqf.hop('Software Product release: {}'.format(prod_rel))
                    build_date = version[version.find('Build date:'):].split()[2]
                    Aqf.hop('Software Build date: {}\n'.format(build_date))

                remote_conn.send("exit\n")
                remote_conn.close()
                remote_conn_pre.close()

        Aqf.step('CMC CBF Package Software version information.')
        reply, informs = self.corr_fix.katcp_rct.req.version_list()
        if reply.reply_ok():
            katcp_dev, katcp_lib = [i.arguments[-1].split('-')[-1][1:]
                                    for i in informs
                                    if ('katcp-device' in i.arguments or
                                        'katcp-library' in i.arguments)]
            Aqf.hop('Repo: katcp-device, Last Hash:{}\n'.format(katcp_dev))
            Aqf.hop('Repo: katcp-library, Last Hash:{}\n'.format(katcp_lib))
        get_package_versions()
        Aqf.step('CBF ROACH information.')
        get_roach_config()
        Aqf.step('CBF ROACH information on each PDU.')
        get_pdu_config()
        Aqf.step('CBF ROACH information on each Data Switch.')
        get_data_switch()
        try:
            Aqf.hop('Test ran by: {} on {}'.format(os.getlogin(), time.ctime()))
        except OSError:
            LOGGER.info('Test ran by: Jenkins on {}'.format(time.ctime()))

    def _test_overvoltage(self):
        """ROACH2 overvoltage display test

        Test waived
        """
        overvoltage_dict = {0: '1V0', 1: '1V5', 2: '1V8', 3: '2V5',
                            4: '3V3', 5: '5V',  6: '12V'}
        for port, label in overvoltage_dict.iteritems():
            Aqf.step('Trigger the {} overvoltage warning'.format(label))
            self._test_over_warning('hwmon2', 'in{}'.format(port), 'overvoltage')

    def _test_overcurrent(self):
        """ROACH2 overcurrent display test

        Test waived
        """
        overcurrent_dict = {4: '1V0', 3: '1V5', 2: '1V8', 1: '2V5', 0: '3V3'}
        for port, label in overcurrent_dict.iteritems():
            Aqf.step('Trigger the {} overcurrent current warning.'.format(label))
            self._test_over_warning('hwmon3', 'in{}'.format(port), 'overcurrent')

    def _test_over_warning(self, hwmon_dir, port, label):

        hosts = [host.host for host in self.correlator.xhosts + self.correlator.fhosts]
        user = 'root\n'
        wait_time = 1
        hwmon = '/sys/class/hwmon/{}'.format(hwmon_dir)

        def overwarning_triggers():
            # set the limit ridiculously low, the red LED should turn on
            set_limit = 'echo "10" > {}/{}_crit\n'.format(hwmon, port)
            tn.write(set_limit)
            Aqf.wait(wait_time, 'Setting the limit low, the red LED should turn on.')
            time.sleep(wait_time)
            tn.write(curr_alarm_val)
            Aqf.wait(wait_time, 'Wait for command to be received successfully.')
            time.sleep(wait_time)
            stdout = tn.read_until('#', timeout=wait_time)
            try:
                new_alarm_value = int(stdout.splitlines()[-2])
                #Aqf.is_true(new_alarm_value, 'Confirm that the alarm has been Triggered.')
                Aqf.tbd('Confirm that the alarm has been Triggered.')
                Aqf.tbd('Confirm the CBF sends an error message "#TBD"')

                Aqf.failed('PROBLEM - the driver does not read the alarm correctly,'
                        ' so the error message never gets triggered.')
            except ValueError:
                Aqf.failed('Failed to read current {} alarm value: {}.'.format(label, hostname))

            orig_alarm_val = 'echo "{}" > {}/{}_crit\n'.format(lim_val, hwmon, port)
            tn.write(orig_alarm_val)
            Aqf.step('Setting current warning limit back to default')
            time.sleep(wait_time*3)
            tn.write(curr_read_lim)
            time.sleep(wait_time)
            stdout = tn.read_until('#', timeout=wait_time)
            try:
                def_lim_val = int(stdout.splitlines()[-2])
                Aqf.equals(def_lim_val, lim_val,
                    'Confirm that the current warning limit was set back to default')
            except ValueError:
                Aqf.failed('Failed to set default value: {}.'.format(hostname))

            tn.write(curr_alarm_val)
            Aqf.wait(wait_time, 'Setting {} alarm to default state.'.format(label))
            time.sleep(wait_time)
            stdout = tn.read_until('#', timeout=wait_time)
            try:
                new_alarm_value = int(stdout.splitlines()[-2])
                Aqf.is_false(new_alarm_value, 'Confirm that the alarm was set to default')
                Aqf.tbd ('PROBLEM - the driver does not read the alarm correctly,'
                        ' so the error message never gets triggered.\n')
            except ValueError:
                Aqf.failed('Failed to read default value: {}.\n'.format(hostname))

        hostname = hosts[randrange(len(hosts))]
        tn = telnetlib.Telnet(hostname)
        Aqf.step('Connected to Host: {}'.format(hostname))
        tn.read_until('login: ', timeout=wait_time)
        tn.write(user)
        time.sleep(wait_time)

        curr_alarm_val = 'cat {}/{}_alarm\n'.format(hwmon, port)
        tn.write(curr_alarm_val)
        time.sleep(wait_time)
        stdout = tn.read_until('#', timeout=wait_time)
        try:
            alarm_value = int(stdout.splitlines()[-2])
            Aqf.is_false(alarm_value,
                'Confirm that the {} alarm has Not triggered.'.format(label))
        except ValueError:
            Aqf.failed('Failed to read current {} alarm: {}.'.format(label, hostname))

        curr_read_lim = 'cat {}/{}_crit\n'.format(hwmon, port)
        tn.write(curr_read_lim)
        time.sleep(wait_time)
        stdout = tn.read_until('#', timeout=wait_time)
        try:
            lim_val = int(stdout.splitlines()[-2])
            Aqf.passed('Confirm current {} limit : {}'.format(label, lim_val))

        except ValueError:
            Aqf.failed('Failed to read {} limit: {}.'.format(label, hostname))
        Aqf.tbd('see comments: https://skaafrica.atlassian.net/browse/CBFTASKS-282')
        Aqf.failed('PROBLEM - the driver does not read the alarm correctly,'
                    ' so the error message never gets triggered.\n')

    def _test_overtemp(self):
        """ROACH2 overtemperature display test """
        def air_temp_warn(hwmon_dir, label):

            hwmon = '/sys/class/hwmon/{}'.format(hwmon_dir)
            hostname = hosts[randrange(len(hosts))]

            try:
                tn = telnetlib.Telnet(hostname)
                Aqf.step('Connected to Host: {}'.format(hostname))
                tn.read_until('login: ', timeout=wait_time)
                tn.write(user)
                time.sleep(wait_time)
                stdout = tn.read_until('#', timeout=wait_time)
                # returns current temperature
                read_cur_temp = 'cat {}/temp1_input\n'.format(hwmon)
                tn.write(read_cur_temp)
                time.sleep(wait_time)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    cur_temp = int(stdout.splitlines()[-2])
                    Aqf.step('Current Air {} Temp: {} deg'.format(label, int(cur_temp)/1000.))
                except ValueError:
                    Aqf.failed('Failed to read current temp {}.'.format(hostname))

                # returns 1 if the roach is overtemp, it should be 0
                read_overtemp_ind = 'cat {}/temp1_max_alarm\n'.format(hwmon)
                tn.write(read_overtemp_ind)
                time.sleep(wait_time)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    # returns 1 if the roach is overtemp, it should be 0
                    overtemp_ind = int(stdout.splitlines()[-2])
                    Aqf.is_false(overtemp_ind,
                                'Confirm that the overtemp alarm is Not triggered.')
                except ValueError:
                    Aqf.failed('Failed to read overtemp alarm on {}.'.format(hostname))

                # returns 0 if the roach is undertemp, it should be 1
                read_undertemp_ind = 'cat {}/temp1_min_alarm\n'.format(hwmon)
                tn.write(read_undertemp_ind)
                time.sleep(wait_time*3)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    # returns 1 if the roach is undertemp, it should be 1
                    undertemp_ind = int(stdout.splitlines()[-2])
                    Aqf.is_true(undertemp_ind,
                                'Confirm that the undertemp alarm is Not triggered.')
                except ValueError:
                    Aqf.failed('Failed to read undertemp alarm on {}.'.format(hostname))

                # set the max temp limit to 10 degrees
                set_max_limit = 'echo "10000" > {}/temp1_max\n'.format(hwmon)
                tn.write(set_max_limit)
                Aqf.wait(wait_time, 'Setting max temp limit to 10 degrees')
                stdout = tn.read_until('#', timeout=wait_time)

                tn.write(read_overtemp_ind)
                time.sleep(wait_time)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    overtemp_ind = int(stdout.splitlines()[-2])
                    Aqf.is_true(overtemp_ind,
                                'Confirm that the overtemp alarm is Triggered.')
                except ValueError:
                    Aqf.failed('Failed to read overtemp alarm on {}.'.format(hostname))

                # set the min temp limit to below current temp
                set_min_limit = 'echo "10000" > {}/temp1_min\n'.format(hwmon)
                tn.write(set_min_limit)
                Aqf.wait(wait_time*2, 'Setting min temp limit to 10 degrees')
                stdout = tn.read_until('#', timeout=wait_time)

                tn.write(read_undertemp_ind)
                time.sleep(wait_time*3)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    undertemp_ind = int(stdout.splitlines()[-2])
                    Aqf.is_false(undertemp_ind,
                                'Confirm that the undertemp alarm is Triggered.')
                except ValueError:
                    Aqf.failed('Failed to read undertemp alarm on {}.'.format(hostname))

                # TODO MM add sensor sniffer, at the moment sensor in not implemented
                # Confirm the CBF sends an error message
                # "#log warn <> roach2hwmon Sensor\_alarm:\_Chip\_ad7414-i2c-0-4c:\_temp1:\_<>\_C\_(min\_=\_50.0\_C,\_max\_=\_10.0\_C)\_[ALARM]"

                # set the max temp limit back to 55 degrees
                default_max = 'echo "55000" > {}/temp1_max\n'.format(hwmon)
                tn.write(default_max)
                Aqf.wait(wait_time, 'Setting max temp limit back to 55 degrees')
                stdout = tn.read_until('#', timeout=wait_time)

                # set the min temp limit back to 50 degrees
                default_min = 'echo "50000" > {}/temp1_min\n'.format(hwmon)
                tn.write(default_min)
                Aqf.wait(wait_time, 'Setting min temp limit back to 50 degrees')
                stdout = tn.read_until('#', timeout=wait_time)

                tn.write(read_overtemp_ind)
                time.sleep(wait_time*3)
                overtemp_ind  = tn.read_until('#', timeout=wait_time)

                tn.write(read_undertemp_ind)
                time.sleep(wait_time*3)
                undertemp_ind  = tn.read_until('#', timeout=wait_time)

                try:
                    overtemp_ind = int(overtemp_ind.splitlines()[-2])
                    # returns 1 if the roach is overtemp, it should be 0
                    Aqf.is_false(overtemp_ind,
                                'Confirm that the overtemp alarm was set back to default.')
                    # returns 0 if the roach is undertemp, it should be 1
                    undertemp_ind = int(undertemp_ind.splitlines()[-2])
                    Aqf.is_true(undertemp_ind,
                                'Confirm that the undertemp alarm was set back to default.\n')
                except ValueError:
                    Aqf.failed('Failed to read undertemp alarm on {}.\n'.format(hostname))

                tn.write("exit\n")
                tn.close()
            except Exception:
                Aqf.failed('Could not connect to host: {}'.format(hostname))

        hosts = [host.host for host in self.correlator.xhosts + self.correlator.fhosts]
        user = 'root\n'
        wait_time = 1
        temp_dict = {4:'Inlet', 1:'Outlet'}
        for hwmon_dir, label in temp_dict.iteritems():
            Aqf.step('Trigger Air {} Temperature Warning.'.format(label))
            air_temp_warn('hwmon{}'.format(hwmon_dir), '{}'.format(label))

    def _test_delay_inputs(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial:
        Delay applied to the correct input
        """
        setup_data = self._delays_setup(test_source_idx = 0)
        sampling_period = self.corr_freqs.sample_period
        no_chans = range(len(self.corr_freqs.chan_freqs))
        test_delay = sampling_period
        expected_phases = self.corr_freqs.chan_freqs * 2 * np.pi * test_delay
        expected_phases -= np.max(expected_phases) / 2.

        test_source_idx = 2
        reply, informs = self.corr_fix.katcp_rct.req.input_labels()
        source_names = reply.arguments[1:]
        last_pfb_counts = get_pfb_counts(
            get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
        for delayed_input in source_names:
            delays = [0] * setup_data['num_inputs']
            # Get index for input to delay
            test_source_idx = source_names.index(delayed_input)
            Aqf.step('Delayed input = {}'.format(delayed_input))
            delays[test_source_idx] = test_delay
            delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT,
                                                          discard=0)

            future_time = 1000e-3
            settling_time = 600e-3
            dump_timestamp = (this_freq_dump['sync_time'].value +
                              this_freq_dump['timestamp'].value /
                              this_freq_dump['scale_factor_timestamp'].value)
            t_apply = (dump_timestamp + this_freq_dump['int_time'].value +
                       future_time)

            reply = self.corr_fix.katcp_rct.req.delays(
                t_apply, *delay_coefficients)
            Aqf.is_true(reply.reply.reply_ok(), 'Delays Reply: {}'.format(reply.reply.arguments[1]))
            Aqf.wait(settling_time,
                     'Settling time in order to set delay: {} ns.'
                     .format(test_delay * 1e9))
            QDR_error_roaches = check_fftoverflow_qdrstatus(self.correlator,
                                                           last_pfb_counts)
            if QDR_error_roaches:
                Aqf.failed(QDR_error_roaches)
            dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            baselines = get_baselines_lookup(this_freq_dump)
            sorted_bls = sorted(baselines.items(), key=operator.itemgetter(1))
            chan_response = []
            for b_line in sorted_bls:
                b_line_val = b_line[1]
                b_line_dump = (dump['xeng_raw'].value[:, b_line_val, :])
                # TODO MM 2016-08-01
                # Plot spectrum
                b_line_cplx_data = complexise(b_line_dump)
                b_line_phase = np.angle(b_line_cplx_data)
                b_line_phase_max = np.max(b_line_phase)
                if ((delayed_input in b_line[0]) and
                    b_line[0] != (delayed_input,delayed_input)):
                    aqf_array_abs_error_less(np.abs(b_line_phase),
                                             np.abs(expected_phases),
                                             'Checking baseline {0}, '
                                             'index = {1:02d}... expecting a delay. '
                                             .format(b_line[0], b_line_val), 0.01)
                else:
                    desc = ('Checking baseline {0}, index = {1:02d}... '
                            .format(b_line[0], b_line_val))
                    if b_line_phase_max != 0:
                        Aqf.failed(desc + 'phase offset found, maximum value = {0:0.8f}'
                                          .format(b_line_phase_max))
                        chan_response.append(normalised_magnitude(b_line_dump))

        if chan_response:
            Aqf.step('Delay applied to the correct input')
            legends = ['Dumps per Baseline #{}'.format(x) for x in range(len(chan_response))]
            aqf_plot_channels(zip(chan_response, legends),
                            plot_filename='{}.svg'.format(self._testMethodName),
                            plot_title='Log channel response Phase Offsets Found',
                            log_dynamic_range=90, log_normalise_to=1,
                            caption='Delay applied to the correct input')

    def _test_data_product(self, instrument, no_channels):
        """CBF Imaging Data Product Set"""
        Aqf.step('Configured DSim to produce correlated noise.')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        Aqf.step('Getting initial SPEAD data')
        test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        Aqf.equals(test_dump['xeng_raw'].value.shape[0], no_channels,
                   'Check that data product has the number of frequency '
                   'channels {no_channels} corresponding to the {instrument} '
                   'instrument product'.format(**locals()))
        response = normalised_magnitude(test_dump['xeng_raw'].value[:, test_baseline, :])

        if response.shape[0] == no_channels:
            Aqf.passed('Confirm that imaging data product set has been implemented.')
            aqf_plot_channels(response,
                              plot_filename='data_product_channel_resp_{}.svg'.format(
                                            no_channels),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='This serves merely as a record whether functionality'
                                      'has been included in this initial release, '
                                      'and what level of confidence there is in that '
                                      'functionality.')
        else:
            Aqf.failed('Imaging data product set has not been implemented.')

    def _test_control_init(self):
        Aqf.passed('List of available commands\n{}'.format(self.corr_fix.katcp_rct.req.help()))
        Aqf.progress('Polarisation correction has not been implemented yet.')
        Aqf.is_true(self.corr_fix.katcp_rct.req.gain.is_active(),
                    'Re-quantiser settings (Gain) and Complex gain correction has '
                    'been implemented')
        Aqf.is_true(self.corr_fix.katcp_rct.req.accumulation_length.is_active(),
                    'Accumulation interval has been implemented')
        Aqf.is_true(self.corr_fix.katcp_rct.req.frequency_select.is_active(),
                    'Channelisation configuration has been implemented')
