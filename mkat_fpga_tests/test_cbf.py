from __future__ import division

import unittest
import logging
import time
import datetime
import subprocess
import os
import sys
import telnetlib
import operator
import Queue
import colors as clrs
import h5py
import glob
import pandas
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook

import corr2
import katcp

from random import randrange
from collections import namedtuple
from concurrent.futures import TimeoutError
from nosekatreport import Aqf, aqf_vr

from katcp.testutils import start_thread_with_cleanup
from corr2.corr_rx import CorrRx

from mkat_fpga_tests import correlator_fixture
from mkat_fpga_tests.aqf_utils import cls_end_aqf, aqf_numpy_almost_equal, aqf_plot_histogram
from mkat_fpga_tests.aqf_utils import aqf_array_abs_error_less, aqf_plot_phase_results
from mkat_fpga_tests.aqf_utils import aqf_plot_channels, aqf_is_not_equals
from mkat_fpga_tests.utils import normalised_magnitude, loggerise, complexise
from mkat_fpga_tests.utils import init_dsim_sources, CorrelatorFrequencyInfo
from mkat_fpga_tests.utils import nonzero_baselines, zero_baselines, all_nonzero_baselines
from mkat_fpga_tests.utils import get_quant_snapshot, get_fftoverflow_qdrstatus, check_fftoverflow_qdrstatus
from mkat_fpga_tests.utils import get_and_restore_initial_eqs, get_set_bits, get_baselines_lookup
from mkat_fpga_tests.utils import get_pfb_counts, check_host_okay, get_adc_snapshot
from mkat_fpga_tests.utils import set_default_eq, clear_all_delays, set_input_levels
from mkat_fpga_tests.utils import get_delay_bounds

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

xeng_raw_bits_flags = namedtuple('FlagsBits', 'corruption overrange noise_diode')(
    34, 33, 32)


# NOTE TP.C.1.20 for AR1 maps to TP.C.1.46 for RTS

# TODO NM (2015-12-10) Use steal_docstring decorator form KATCP for sub-tests to re-use
# docstring from implementation. Perhaps do some trickery with string templates and parsing
# the mode name from the function to make an all singing all dancing decorator that does
# everything automagically?

def disable_maplotlib_warning():
    """This function disable matplotlibs deprecation warnings"""
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def teardown_module():
    """This method is run once after test class is executed"""
    correlator_fixture.katcp_rct.stop()
    correlator_fixture.rct.stop()

@cls_end_aqf
class test_CBF(unittest.TestCase):
    """ Unittesting class for mkat_fpga_tests"""

    def setUp(self):
        self.corr_fix = correlator_fixture
        self.test_conf = self.corr_fix._test_config_file()
        _conf = self.test_conf['inst_param']
        self.corr_fix.instrument = _conf['default_instrument']
        self.corr_fix.array_name = _conf['subarray']
        self.corr_fix.resource_clt = _conf['katcp_client']
        self.dhost = self.corr_fix.dhost
        self.dhost.get_system_information()
        self.receiver = None

    def set_instrument(self, instrument, acc_time=0.2):
        if self.receiver:
            self.receiver.stop()
            self.receiver = None
        acc_timeout = 60
        instrument_state = self.corr_fix.ensure_instrument(instrument)
        if not instrument_state:
            self.corr_fix.rct.req.array_halt(self.corr_fix.array_name)
            errmsg = ('Could not initialise instrument or ensure running instrument: {}, '
                      'SubArray will be halted and restarted with next test'.format(instrument))
            LOGGER.error(errmsg)
            Aqf.end(passed=False, message=errmsg)
            return False
        try:
            self.corr_fix.katcp_rct.start()
            self.corr_fix.katcp_rct.until_synced(timeout=acc_timeout)
            reply = self.corr_fix.katcp_rct.req.accumulation_length(
                acc_time, timeout=acc_timeout)

        except TimeoutError:
            reply, inform = self.corr_fix.rct.req.array_halt(self.corr_fix.array_name)
            errmsg = ('Timed-Out: Failed to set accumulation time withing {}s, '
                     'SubArray will be halted and restarted with next test'.format(acc_timeout))
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False

        except AttributeError:
            reply, inform = self.corr_fix.rct.req.array_halt(self.corr_fix.array_name)
            errmsg = ('Failed to set accumulation time withing {}s, due to katcp request errors. '
                      'SubArray will be halted and restarted with next test'.format(acc_timeout))
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False

        else:
            if not reply.succeeded:
                errmsg = 'Failed to set Accumulation time via kcs. KATCP Reply: {}'.format(reply)
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                reply, inform = self.corr_fix.rct.req.array_halt(self.corr_fix.array_name)
                return False
            else:
                Aqf.step('Accumulation time set: {}s'.format(reply.reply.arguments[-1]))
                self.addCleanup(self.corr_fix.stop_x_data)
                try:
                    corrRx_port = int(self.test_conf['inst_param']['corr_rx_port'])
                except ValueError:
                    corrRx_port = 8888
                    LOGGER.info('Failed to retrieve corr rx port from config file.'
                                'Setting it to default port: {}'.format(corrRx_port))

                if instrument.upper().find('M4K') > 0:
                    self.receiver = CorrRx(port=corrRx_port, queue_size=1000)
                else:
                    self.receiver = CorrRx(port=corrRx_port, queue_size=100)

                try:
                    self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                except AssertionError:
                    errmsg = 'Correlator Receiver could not be instantiated.'
                    LOGGER.exception(errmsg)
                    Aqf.failed(errmsg)
                    return False
                else:
                    start_thread_with_cleanup(self, self.receiver, start_timeout=1)
                    self.correlator = self.corr_fix.correlator
                    self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
                    self.corr_fix.start_x_data()
                    self.corr_fix.issue_metadata()
                    # This function disables matplotlib's deprecation warnings
                    disable_maplotlib_warning()
                    return True

    def tearDown(self):
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc8n856M4k_channelisation(self, instrument='bc8n856M4k'):
        """CBF Channelisation Wideband Coarse L-band (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Channelisation Wideband Coarse L-band: {}\n'.format(_running_inst))
            self._systems_tests()
            n_chans = self.corr_freqs.n_chans
            test_chan = randrange(start=n_chans % 100,  stop=n_chans - 1)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc32n856M4k_channelisation(self, instrument='bc32n856M4k'):
        """CBF Channelisation Wideband Coarse L-band (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Channelisation Wideband Coarse L-band: {}\n'.format(_running_inst))
            self._systems_tests()
            n_chans = self.corr_freqs.n_chans
            test_chan = randrange(start=n_chans % 100,  stop=n_chans - 1)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc16n856M4k_channelisation(self, instrument='bc16n856M4k'):
        """CBF Channelisation Wideband Coarse L-band (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Channelisation Wideband Coarse L-band: {}\n'.format(_running_inst))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc8n856M32k_channelisation(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Channelisation Wideband Fine L-band (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Channelisation Wideband Fine L-band: {}\n'.format(_running_inst))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc8n856M4k_channelisation_sfdr_peaks(self, instrument='bc8n856M4k'):
        """Test spurious free dynamic range for wideband coarse (bc8n856M4k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Test Spurious Free Dynamic Range for Wideband Coarse: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  #Hz

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc32n856M4k_channelisation_sfdr_peaks(self, instrument='bc32n856M4k'):
        """Test spurious free dynamic range for wideband coarse (bc32n856M4k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Test Spurious Free Dynamic Range for Wideband Coarse: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  #Hz

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    def test_bc16n856M4k_channelisation_sfdr_peaks(self, instrument='bc16n856M4k'):
        """Test spurious free dynamic range for wideband coarse (bc16n856M4k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Test Spurious Free Dynamic Range for Wideband Coarse: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  #Hz

    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc8n856M32k_channelisation_sfdr_peaks_slow(self, instrument='bc8n856M32k', acc_time=0.5):
        """
        Slow Test spurious free dynamic range for wideband fine (bc8n856M32k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

        This is the slow version that sweeps through all 32768 channels.

        _____________________________NOTE____________________________
        Usage: Run nosetests with -e
        Example: nosetests  -s -v --with-katreport --exclude=slow
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Test spurious free dynamic range for wideband fine: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  #Hz

    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc8n856M32k_channelisation_sfdr_peaks_fast(self, instrument='bc8n856M32k', acc_time=0.5):
        """
        Fast Test spurious free dynamic range for wideband fine (bc8n856M32k)

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

        This is the faster version that sweeps through 32768 channels
        whilst stepping through `x`, where x is the step size given.

        _____________________________NOTE____________________________
        Usage: Run nosetests with -e
        Example: nosetests  -s -v --with-katreport --exclude=slow
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Test spurious free dynamic range for wideband fine: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=[15e3, 30e3], no_channels=32768, stepsize=8)  #Hz

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc8n856M4k_freq_scan_consistency(self, instrument='bc8n856M4k'):
        """Frequency Scan Consistency Test (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Frequency Scan Consistency Test: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_freq_scan_consistency()

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc32n856M4k_freq_scan_consistency(self, instrument='bc32n856M4k'):
        """Frequency Scan Consistency Test (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Frequency Scan Consistency Test: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_freq_scan_consistency()

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc16n856M4k_freq_scan_consistency(self, instrument='bc16n856M4k'):
        """Frequency Scan Consistency Test (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Frequency Scan Consistency Test: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_freq_scan_consistency()

    @aqf_vr('TP.C.1.19')
    @aqf_vr('TP.C.1.45')
    @aqf_vr('TP.C.1.20')
    @aqf_vr('TP.C.1.46')
    def test_bc8n856M32k_freq_scan_consistency(self, instrument='bc8n856M32k', acc_time=0.5):
        """Frequency Scan Consistency Test (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Frequency Scan Consistency Test: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_freq_scan_consistency()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M4k_baseline_correlation_product(self, instrument='bc8n856M4k'):
        """CBF Baseline Correlation Products - AR1 (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Baseline Correlation Products - AR1: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_product_baselines()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc32n856M4k_baseline_correlation_product(self, instrument='bc32n856M4k'):
        """CBF Baseline Correlation Products - AR1 (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Baseline Correlation Products - AR1: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_product_baselines()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc16n856M4k_baseline_correlation_product(self, instrument='bc16n856M4k'):
        """CBF Baseline Correlation Products - AR1 (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Baseline Correlation Products - AR1: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_product_baselines()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M32k_baseline_correlation_product(self,
                                                      instrument='bc8n856M32k',
                                                      acc_time=0.5):
        """CBF Baseline Correlation Products - AR1 (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Baseline Correlation Products - AR1: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_product_baselines()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M4k_baseline_correlation_product_consistency(self, instrument='bc8n856M4k'):
        """
        CBF Baseline Correlation Products
        Check that back-to-back SPEAD packets with same input are equal. (bc8n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step("Back-to-Back SPEAD packets consistency: {}\n".format(_running_inst))
            self._systems_tests()
            self._test_back2back_consistency()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc32n856M4k_baseline_correlation_product_consistency(self, instrument='bc32n856M4k'):
        """
        CBF Baseline Correlation Products
        Check that back-to-back SPEAD packets with same input are equal. (bc32n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step("Back-to-Back SPEAD packets consistency: {}\n".format(_running_inst))
            self._systems_tests()
            self._test_back2back_consistency()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc16n856M4k_baseline_correlation_product_consistency(self, instrument='bc16n856M4k'):
        """
        CBF Baseline Correlation Products
        Check that back-to-back SPEAD packets with same input are equal. (bc16n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step("Back-to-Back SPEAD packets consistency: {}\n".format(_running_inst))
            self._systems_tests()
            self._test_back2back_consistency()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M32k_baseline_correlation_product_consistency(self,
                                                                  instrument='bc8n856M32k',
                                                                  acc_time=0.5):
        """
        CBF Baseline Correlation Products
        Check that back-to-back SPEAD packets with same input are equal. (bc8n856M32k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step("Back-to-Back SPEAD packets consistency: {}\n".format(_running_inst))
            self._systems_tests()
            self._test_back2back_consistency()

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M4k_correlator_restart_consistency(self, instrument='bc8n856M4k'):
        """
        Correlator restart consistency
        Check that results are consistent on correlator restart (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Correlator Restart Consistency Test: {}\n'.format(_running_inst))
            self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc32n856M4k_correlator_restart_consistency(self, instrument='bc32n856M4k'):
        """
        Correlator restart consistency (bc32n856M4k)
        Check that results are consistent on correlator restart"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Correlator Restart Consistency Test: {}\n'.format(_running_inst))
            self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc16n856M4k_correlator_restart_consistency(self, instrument='bc16n856M4k'):
        """
        Correlator restart consistency (bc16n856M4k)
        Check that results are consistent on correlator restart"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Correlator Restart Consistency Test: {}\n'.format(_running_inst))
            self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    @aqf_vr('TP.C.1.44')
    def test_bc8n856M32k_correlator_restart_consistency(self,
                                                        instrument='bc8n856M32k',
                                                        acc_time=0.5):
        """
        Correlator restart consistency (bc8n856M32k)
        Check that results are consistent on correlator restart"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Correlator Restart Consistency Test: {}\n'.format(_running_inst))
            self._test_restart_consistency(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M4k_delay_tracking(self, instrument='bc8n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc8n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay tracking: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.27')
    def test_bc32n856M4k_delay_tracking(self, instrument='bc32n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc32n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay tracking: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.27')
    def test_bc16n856M4k_delay_tracking(self, instrument='bc16n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc16n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay tracking: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M32k_delay_tracking(self, instrument='bc8n856M32k', acc_time=0.5):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc8n856M32k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay tracking: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.31')
    def test_bc8n856M4k_vacc(self, instrument='bc8n856M4k'):
        """Vector Accumulator Test (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Vector Accumulator Test: {}\n'.format(_running_inst))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_vacc(test_chan)

    @aqf_vr('TP.C.1.31')
    def test_bc32n856M4k_vacc(self, instrument='bc32n856M4k'):
        """Vector Accumulator Test (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Vector Accumulator Test: {}\n'.format(_running_inst))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_vacc(test_chan)

    @aqf_vr('TP.C.1.31')
    def test_bc16n856M4k_vacc(self, instrument='bc16n856M4k'):
        """Vector Accumulator Test (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Vector Accumulator Test: {}\n'.format(_running_inst))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_vacc(test_chan)

    @aqf_vr('TP.C.1.31')
    def test_bc8n856M32k_vacc(self, instrument='bc8n856M32k', acc_time=0.5):
        """Vector Accumulator Test (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Vector Accumulator Test: {}\n'.format(_running_inst))
            self._systems_tests()
            chan_index = 4096
            test_chan = randrange(chan_index)
            self._test_vacc(test_chan, chan_index)

    @aqf_vr('TP.C.1.40')
    def test_bc8n856M4k_product_switch(self, instrument='bc8n856M4k'):
        """CBF Data Product Switching Time (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Data Product Switching Time: {}\n'.format(_running_inst))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc32n856M4k_product_switch(self, instrument='bc32n856M4k'):
        """CBF Data Product Switching Time (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Data Product Switching Time: {}\n'.format(_running_inst))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc16n856M4k_product_switch(self, instrument='bc16n856M4k'):
        """CBF Data Product Switching Time (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Data Product Switching Time: {}\n'.format(_running_inst))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc8n856M32k_product_switch(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Data Product Switching Time (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Data Product Switching Time: {}\n'.format(_running_inst))
            self._test_product_switch(instrument, no_channels=32768)

    # Flagging of data test waived as it not part of AR-1.
    ##@aqf_vr('TP.C.1.38')
    def _test_bc8n856M4k_overflow_flag(self, instrument='bc8n856M4k'):
        """CBF flagging of data -- ADC overflow (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- ADC overflow: {}\n'.format(_running_inst))
            self._test_adc_overflow_flag()

    ##@aqf_vr('TP.C.1.38')
    def _test_bc16n856M4k_overflow_flag(self, instrument='bc16n856M4k'):
        """CBF flagging of data -- ADC overflow (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- ADC overflow: {}\n'.format(_running_inst))
            self._test_adc_overflow_flag()

    ##@aqf_vr('TP.C.1.38')
    def _test_bc8n856M32k_overflow_flag(self, instrument='bc8n856M32k'):
        """CBF flagging of data -- ADC overflow (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- ADC overflow: {}\n'.format(_running_inst))
            self._test_adc_overflow_flag()

    ##@aqf_vr('TP.C.1.38')
    def _test_bc8n856M4k_noise_diode_flag(self, instrument='bc8n856M4k'):
        """CBF flagging of data -- noise diode fired (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- noise diode fired: {}\n'.format(_running_inst))
            self._test_noise_diode_flag()

    ##@aqf_vr('TP.C.1.38')
    def _test_bc16n856M4k_noise_diode_flag(self, instrument='bc16n856M4k'):
        """CBF flagging of data -- noise diode fired (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- noise diode fired: {}\n'.format(_running_inst))
            self._test_noise_diode_flag()

    #@aqf_vr('TP.C.1.38')
    def _test_bc8n856M32k_noise_diode_flag(self, instrument='bc8n856M32k'):
        """CBF flagging of data -- noise diode fired (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- noise diode fired: {}\n'.format(_running_inst))
            self._test_noise_diode_flag()

    #@aqf_vr('TP.C.1.38')
    def _test_bc8n856M4k_fft_overflow_flag(self, instrument='bc8n856M4k'):
        """CBF flagging of data -- FFT overflow (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- FFT overflow: {}\n'.format(_running_inst))
            self._test_fft_overflow_flag()

    #@aqf_vr('TP.C.1.38')
    def _test_bc16n856M4k_fft_overflow_flag(self, instrument='bc16n856M4k'):
        """CBF flagging of data -- FFT overflow (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- FFT overflow: {}\n'.format(_running_inst))
            self._test_fft_overflow_flag()

    #@aqf_vr('TP.C.1.38')
    def _test_bc8n856M32k_fft_overflow_flag(self, instrument='bc8n856M32k'):
        """CBF flagging of data -- FFT overflow (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF flagging of data -- FFT overflow: {}\n'.format(_running_inst))
            self._test_fft_overflow_flag()

    #@aqf_vr('TP.C.1.38')
    def _test_bc8n856M4k_roach_qdr_sensors(self, instrument='bc8n856M4k'):
        """QDR Memory Corruption Sensors Test (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('QDR Memory Corruption Sensors Test: {}\n'.format(_running_inst))
            self._test_roach_qdr_sensors()

    #@aqf_vr('TP.C.1.38')
    def _test_bc8n856M32k_roach_qdr_sensors(self, instrument='bc8n856M32k'):
        """QDR Memory Corruption Sensors Test (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('QDR Memory Corruption Sensors Test: {}\n'.format(_running_inst))
            self._test_roach_qdr_sensors()

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M4k_delay_rate(self, instrument='bc8n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial
            -- Delay Rate (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay Rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.27')
    def test_bc32n856M4k_delay_rate(self, instrument='bc32n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial
           -- Delay Rate (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay Rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.27')
    def test_bc16n856M4k_delay_rate(self, instrument='bc16n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial
           -- Delay Rate (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay Rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M32k_delay_rate(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Delay Compensation/LO Fringe stopping polynomial
           -- Delay Rate (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial '
                     '-- Delay Rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    def test_bc8n856M4k_fringe_offset(self, instrument='bc8n856M4k'):
        """CBF per-antenna phase error -- Fringe offset (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe offset: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    def test_bc32n856M4k_fringe_offset(self, instrument='bc32n856M4k'):
        """CBF per-antenna phase error -- Fringe offset (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe offset: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    def test_bc16n856M4k_fringe_offset(self, instrument='bc16n856M4k'):
        """CBF per-antenna phase error -- Fringe offset (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe offset: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    def test_bc8n856M32k_fringe_offset(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF per-antenna phase error -- Fringe offset (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe offset: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.28')
    def test_bc8n856M4k_fringe_rate(self, instrument='bc8n856M4k'):
        """CBF per-antenna phase error -- Fringe rate (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.28')
    def test_bc32n856M4k_fringe_rate(self, instrument='bc32n856M4k'):
        """CBF per-antenna phase error -- Fringe rate (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.28')
    def test_bc16n856M4k_fringe_rate(self, instrument='bc16n856M4k'):
        """CBF per-antenna phase error -- Fringe rate (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.28')
    def test_bc8n856M32k_fringe_rate(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF per-antenna phase error -- Fringe rate (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Fringe rate: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.28')
    def test_bc8n856M4k_fringe_delays(self, instrument='bc8n856M4k'):
        """
        CBF per-antenna phase error
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate. (bc8n856M4k)
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Delays, Delay Rate, '
                     'Fringe Offset and Fringe Rate.: {}\n'.format(_running_inst))
            Aqf.tbd('Values still needs to be defined.')
            # self._systems_tests()
            # self._test_all_delays()

    @aqf_vr('TP.C.1.28')
    def test_bc32n856M4k_fringe_delays(self, instrument='bc32n856M4k'):
        """
        CBF per-antenna phase error (bc32n856M4k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Delays, Delay Rate, '
                     'Fringe Offset and Fringe Rate.: {}\n'.format(_running_inst))
            Aqf.tbd('Values still needs to be defined.')
            # self._systems_tests()
            # self._test_all_delays()

    @aqf_vr('TP.C.1.28')
    def test_bc16n856M4k_fringe_delays(self, instrument='bc16n856M4k'):
        """
        CBF per-antenna phase error (bc16n856M4k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Delays, Delay Rate, '
                     'Fringe Offset and Fringe Rate.: {}\n'.format(_running_inst))
            Aqf.tbd('Values still needs to be defined.')
            # self._systems_tests()
            # self._test_all_delays()

    @aqf_vr('TP.C.1.28')
    def test_bc8n856M32k_fringe_delays(self, instrument='bc8n856M32k', acc_time=0.5):
        """
        CBF per-antenna phase error  (bc8n856M32k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF per-antenna phase error -- Delays, Delay Rate, '
                     'Fringe Offset and Fringe Rate: {}\n'.format(_running_inst))
            Aqf.tbd('Values still needs to be defined.')
            # self._systems_tests()
            # self._test_all_delays()

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M4k_delay_inputs(self, instrument='bc8n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc8n856M4k)
           Delay applied to the correct input
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial -- '
                     'Delay applied to the correct input: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.27')
    def test_bc32n856M4k_delay_inputs(self, instrument='bc32n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc32n856M4k)
           Delay applied to the correct input
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial -- '
                     'Delay applied to the correct input: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.27')
    def test_bc16n856M4k_delay_inputs(self, instrument='bc16n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc16n856M4k)
           Delay applied to the correct input
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial -- '
                     'Delay applied to the correct input: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.27')
    def test_bc8n856M32k_delay_inputs(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc8n856M32k)
           Delay applied to the correct input
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Delay Compensation/LO Fringe stopping polynomial -- '
                     'Delay applied to the correct input: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.47')
    def test_bc8n856M4k_data_product(self, instrument='bc8n856M4k'):
        """CBF Imaging Data Product Set (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Imaging Data Product Set: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.47')
    def test_bc32n856M4k_data_product(self, instrument='bc32n856M4k'):
        """CBF Imaging Data Product Set (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Imaging Data Product Set: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.47')
    def test_bc16n856M4k_data_product(self, instrument='bc16n856M4k'):
        """CBF Imaging Data Product Set (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Imaging Data Product Set: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.47')
    def test_bc8n856M32k_data_product(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Imaging Data Product Set (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Imaging Data Product Set: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.42')
    def test_bc8n856M4k_time_sync(self, instrument='bc8n856M4k'):
        """CBF Time synchronisation (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Time synchronisation: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_time_sync()

    @aqf_vr('TP.C.1.42')
    def test_bc32n856M4k_time_sync(self, instrument='bc32n856M4k'):
        """CBF Time synchronisation (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Time synchronisation: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_time_sync()

    @aqf_vr('TP.C.1.42')
    def test_bc16n856M4k_time_sync(self, instrument='bc16n856M4k'):
        """CBF Time synchronisation (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Time synchronisation: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_time_sync()

    @aqf_vr('TP.C.1.42')
    def test_bc8n856M32k_time_sync(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Time synchronisation (bc8n856M32kk)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Time synchronisation: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_time_sync()

    @aqf_vr('TP.C.1.29')
    def test_bc8n856M4k_gain_correction(self, instrument='bc8n856M4k'):
        """CBF Gain Correction (bc8n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Gain Correction: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc32n856M4k_gain_correction(self, instrument='bc32n856M4k'):
        """CBF Gain Correction (bc32n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Gain Correction: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc16n856M4k_gain_correction(self, instrument='bc16n856M4k'):
        """CBF Gain Correction (bc16n856M4k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Gain Correction: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc8n856M32k_gain_correction(self, instrument='bc8n856M32k', acc_time=0.5):
        """CBF Gain Correction (bc8n856M32k)"""
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('CBF Gain Correction: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.36')
    @aqf_vr('TP.C.1.35')
    def test_bc8n856M4k_beamforming(self, instrument='bc8n856M4k'):
        """Test beamformer functionality (bc8n856M4k)
        Apply weights and capture beamformer data.
        Verify that weights are correctly applied.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Testing beamforming weights and capturing beamformer '
                     'output products: : {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_beamforming(ants=4)

    @aqf_vr('TP.C.1.36')
    @aqf_vr('TP.C.1.35')
    def test_bc32n856M4k_beamforming(self, instrument='bc32n856M4k'):
        """Test beamformer functionality (bc32n856M4k)

        Apply weights and capture beamformer data.
        Verify that weights are correctly applied.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Testing beamforming weights and capturing beamformer '
                     'output products: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_beamforming(ants=8)

    @aqf_vr('TP.C.1.36')
    @aqf_vr('TP.C.1.35')
    def test_bc16n856M4k_beamforming(self, instrument='bc16n856M4k'):
        """Test beamformer functionality (bc16n856M4k)

        Apply weights and capture beamformer data.
        Verify that weights are correctly applied.
        """
        if self.set_instrument(instrument):
            _running_inst = self.corr_fix.get_running_intrument()
            Aqf.step('Testing beamforming weights and capturing beamformer '
                     'output products: {}\n'.format(_running_inst))
            self._systems_tests()
            self._test_beamforming(ants=8)

    @aqf_vr('TP.C.1.41')
    def test_generic_control_init(self, instrument='bc8n856M4k'):
        """CBF Control - initial release """
        _running_inst = self.corr_fix.get_running_intrument()
        Aqf.step('CBF Control - initial release: {}\n'.format(_running_inst))
        if self.corr_fix.ensure_instrument(_running_inst):
            self.correlator = self.corr_fix.correlator
        else:
            self.set_instrument(instrument)
        self._systems_tests()
        self._test_control_init()

    @aqf_vr('TP.C.1.17')
    def test_generic_config_report(self, instrument='bc8n856M4k'):
        """CBF Report configuration """
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('CBF Report configuration: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            self._systems_tests()
            self._test_config_report(verbose=False)

    @aqf_vr('TP.C.1.5.1')
    @aqf_vr('TP.C.1.18')
    def test_generic_overtemperature(self, instrument='bc8n856M4k'):
        """ROACH2 overtemperature display test """
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('ROACH2 overtemperature display test: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            self._systems_tests()
            self._test_overtemp()

    @aqf_vr('TP.C.1.5.2')
    @aqf_vr('TP.C.1.18')
    def test_generic_overvoltage(self, instrument='bc8n856M4k'):
        """ROACH2 overvoltage display test"""
        Aqf.waived('ROACH2 overvoltage display test '
                   'Manual Test instead of being Automated.')

    @aqf_vr('TP.C.1.5.3')
    @aqf_vr('TP.C.1.18')
    def test_generic_overcurrent(self, instrument='bc8n856M4k'):
        """ROACH2 overcurrent display test"""
        Aqf.waived('ROACH2 overcurrent display test '
                   'Manual Test instead of being Automated.')

    #@aqf_vr('TP.C.1.38')
    def _test_generic_roach_pfb_sensors(self, instrument='bc8n856M4k'):
        """PFB Error Test: Sensors"""
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('PFB Error Test: Sensors: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            Aqf.tbd('PFB sensor test not yet implemented.')
            # self._systems_tests()
            # self._test_roach_pfb_sensors()

    #@aqf_vr('TP.C.5.5')
    #@aqf_vr('TP.C.1.38')
    def _test_generic_deng_link_error(self, instrument='bc8n856M4k'):
        """Link Error :D-Engine to F-engine"""
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('Link Error :D-Engine to F-engine: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            Aqf.tbd('Link Error: D-engine to F-engine test not yet implemented.')
            # self._systems_tests()
            # self._test_deng_link_error()

    #@aqf_vr('TP.C.5.5')
    #@aqf_vr('TP.C.1.38')
    def _test_generic_feng_link_error(self, instrument='bc8n856M4k'):
        """Link Error :F-Engine to X-engine"""
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('Link Error :F-Engine to X-engine: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            Aqf.tbd('Link Error: F-engine to X-engine test not yet implemented.')
            # self._systems_tests()
            # self._test_feng_link_error()

    @aqf_vr('TP.C.1.16')
    def test_generic_sensor_values(self, instrument='bc8n856M4k'):
        """Report sensor values (AR1)"""
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('Report sensor values (AR1): {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            self._systems_tests()
            self._test_sensor_values()

    @aqf_vr('TP.C.1.16')
    def test_generic_roach_sensors_status(self, instrument='bc8n856M4k'):
        """ Test all roach sensors status are not failing and count verification."""
        if not self.corr_fix.get_running_intrument():
            _running_inst = self.corr_fix.get_running_intrument()
        else:
            _running_inst = instrument
        Aqf.step('PFB Error Test: Sensors: {}\n'.format(_running_inst))
        if self.set_instrument(_running_inst):
            self._systems_tests()
            self._test_roach_sensors_status()

    def _systems_tests(self):
        """Run tests fft overflow and qdr status before or after."""
        self.last_pfb_counts = get_pfb_counts(
            get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
        self.addCleanup(check_fftoverflow_qdrstatus, self.correlator,
                        self.last_pfb_counts)
        self.addCleanup(check_host_okay, self.correlator)
        # Reset Equalisations/ Gains to values from config file
        set_default_eq(self.correlator)
        self.addCleanup(set_default_eq, self.correlator)
        sensors_req = self.corr_fix.rct.req.sensor_value()

    def get_flag_dumps(self, flag_enable_fn, flag_disable_fn, flag_description,
                       accumulation_time=1.):
        Aqf.step('Setting  accumulation time to {}.'.format(accumulation_time))
        self.correlator.xops.set_acc_time(accumulation_time)
        Aqf.step('Getting correlator packet 1 before setting {}.'
                 .format(flag_description))
        try:
            dump1 = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
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
            Aqf.step('Getting correlator packet 2 after setting and clearing {}.'
                     .format(flag_description))
            dump2 = self.receiver.data_queue.get(DUMP_TIMEOUT)
            Aqf.step('Getting correlator packet 3.')
            dump3 = self.receiver.data_queue.get(DUMP_TIMEOUT)
            return (dump1, dump2, dump3)

    def _delays_setup(self, test_source_idx=2):
        # Put some correlated noise on both outputs
        Aqf.step('Configure digitiser simulator to generate gaussian noise.')
        #self.dhost.noise_sources.noise_corr.set(scale=0.25)
        
        awgn_scale=0.0645
        gain='113+0j'
        fft_shift=511
        dsim_set_success = set_input_levels(self.corr_fix, self.dhost, awgn_scale=awgn_scale,
            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        self.correlator.est_synch_epoch()
        local_src_names = ['input{}'.format(x) for x in xrange(
            self.correlator.n_antennas * 2)]
        reply_, informs = self.corr_fix.katcp_rct.req.input_labels()
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
        Aqf.step('Source names changed from: {} to: {}'.format(str(reply_), str(reply)))
        Aqf.step('Clearing all coarse and fine delays for all inputs.')
        clear_all_delays(self.correlator, self.receiver)
        self.addCleanup(clear_all_delays, self.correlator, self.receiver)
        Aqf.step('Issuing metadata')
        if not self.corr_fix.issue_metadata():
            Aqf.failed('Could not issues new metadata')
        Aqf.step('Getting initial SPEAD packet.')
        for i in range(2):
            self.corr_fix.issue_metadata()
            self.corr_fix.start_x_data()
        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            sync_time = self.correlator.get_synch_time()
            scale_factor_timestamp = initial_dump['scale_factor_timestamp'].value
            time_stamp = initial_dump['timestamp'].value
            n_accs = initial_dump['n_accs'].value
            int_time = initial_dump['int_time'].value
            # 3ms added for the network round trip
            roundtrip = 0.003
            future_time = 0
            dump_1_timestamp = (sync_time + roundtrip +
                                time_stamp / scale_factor_timestamp)
            t_apply = dump_1_timestamp + 10 * int_time + future_time
            no_chans = range(self.corr_freqs.n_chans)
            reply, informs = self.corr_fix.katcp_rct.req.input_labels()
            if not reply.reply_ok():
                Aqf.failed('Could not retrieve input labels.')
            source_names = reply.arguments[1:]
            # Get name for test_source_idx
            test_source = source_names[test_source_idx]
            ref_source = source_names[0]
            Aqf.step('Source input selected: {}'.format(test_source))
            num_inputs = len(source_names)
            # Get list of all the baselines present in the correlator output
            try:
                baseline_lookup = get_baselines_lookup(initial_dump)
                # Choose baseline for phase comparison
                baseline_index = baseline_lookup[(ref_source, test_source)]
            except KeyError:
                Aqf.failed('Initial SPEAD packet does not contain correct baseline'
                           ' ordering format.')
                return False
            else:
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

    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients, max_wait_dumps=20):
        try:
            cmd_start_time = time.time()
            Aqf.step('Time apply set to: {}'.format(setup_data['t_apply']))
            reply = self.corr_fix.katcp_rct.req.delays(
                setup_data['t_apply'], *delay_coefficients)
            Aqf.is_true(reply.reply.reply_ok(),
                        'Delays reply: {}'.format(reply.reply.arguments[1]))
            final_cmd_time = time.time() - cmd_start_time
            Aqf.passed('Time it takes to load delays {} ms with intergration time {} ms'
                       .format(final_cmd_time / 100e-3, setup_data['int_time'] / 100e-3))

        except:
            errmsg = 'Failed to set delays'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)

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
        for i in xrange(dump_counts):
            Aqf.step('Getting subsequent SPEAD packet {}.'.format(i + 1))
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

    def _get_expected_data(self, setup_data, dump_counts, delay_coefficients, actual_phases):

        def calc_actual_delay(setup_data):
            no_ch = len(setup_data['no_chans'])
            first_dump = np.unwrap(actual_phases[0])
            actual_slope = np.polyfit(xrange(0, no_ch), first_dump, 1)[0] * no_ch
            actual_delay = setup_data['sample_period'] * actual_slope / np.pi
            return actual_delay

        def gen_delay_vector(delay, setup_data):
            res = []
            no_ch = len(setup_data['no_chans'])
            delay_slope = np.pi * (delay / setup_data['sample_period'])
            c = delay_slope / 2
            for i in xrange(0, no_ch):
                m = i / float(no_ch)
                res.append(delay_slope * m - c)
            return res

        def gen_delay_data(delay, delay_rate, dump_counts, setup_data):
            expected_phases = []
            for dump in xrange(1, dump_counts + 1):
                tot_delay = (delay + dump * delay_rate * setup_data['int_time'] -
                             .5 * delay_rate * setup_data['int_time'])
                # The delay does not get applied on dump boundaries. This function
                # calculates the offset between the expected delay and the actual delay
                # and then adds this offset to all subsequent calculations.
                if dump == 1:
                    delay_offset = calc_actual_delay(setup_data) - tot_delay
                else:
                    delay_offset = 0
                tot_delay += delay_offset
                expected_phases.append(gen_delay_vector(tot_delay, setup_data))
            return expected_phases

        def calc_actual_offset(setup_data):
            no_ch = len(setup_data['no_chans'])
            mid_ch = no_ch / 2
            first_dump = actual_phases[0]
            # Determine average offset around 5 middle channels
            actual_offset = np.average(first_dump)  # [mid_ch-3:mid_ch+3])
            return actual_offset

        def gen_fringe_vector(offset, setup_data):
            return [offset] * len(setup_data['no_chans'])

        def gen_fringe_data(fringe_offset, fringe_rate, dump_counts, setup_data):
            expected_phases = []
            for dump in xrange(1, dump_counts + 1):
                offset = -(fringe_offset + dump * fringe_rate * setup_data['int_time'])
                # The delay does not get applied on dump boundaries. This function
                # calculates the offset between the expected delay and the actual delay
                # and then adds this offset to all subsequent calculations.
                if dump == 1:
                    delta_offset = calc_actual_offset(setup_data) - offset
                else:
                    delta_offset = 0
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
    def _test_channelisation(self, test_chan=1500, no_channels=None, req_chan_spacing=None):
        Aqf.step('Choose a frequency channel to test: {}'.format(test_chan))
        Aqf.step('Calculate the expected channel frequency step size and '
                 'the centre frequency of each channel (bin).')
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=101, chans_around=2)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # [CBF-REQ-0126] CBF channel isolation
        cutoff = -53 #dB

        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None

        Aqf.step('Configure digitiser simulator to generate a continuos wave.')
        if self.corr_freqs.n_chans == 4096:
            ## 4K
            cw_scale=0.675
            awgn_scale=0.05
            gain='11+0j'
            fft_shift=8191
        else:
            # 32K
            cw_scale=0.375
            awgn_scale=0.1
            gain='10+0j'
            fft_shift=32767

        dsim_set_success = set_input_levels(self.corr_fix, self.dhost, awgn_scale=awgn_scale,
            cw_scale=cw_scale, freq=expected_fc, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(initial_dump['xeng_raw'].value.shape[0], no_channels,
                       'Capture an initial correlator SPEAD packet, determine the number of channels '
                       'and processing bandwidth: {}Hz.'.format(initial_dump['bandwidth'].value))
            chan_spacing = initial_dump['bandwidth'].value / initial_dump['xeng_raw'].value.shape[0]
            # [CBF-REQ-0043]
            Aqf.is_true((req_chan_spacing / 2) <= chan_spacing <= req_chan_spacing,
                'Verify that the calculated channel frequency step size is between {} and {} Hz'.format(
                req_chan_spacing / 2, req_chan_spacing))


        Aqf.step('Sweeping the digitser simulator over the centre frequencies of at '
                 'least all the channels that fall within the complete L-band')
        print_counts = 3
        spead_failure_counter = 0

        for i, freq in enumerate(requested_test_freqs):
            if i < print_counts:
                Aqf.hop('Getting channel response for freq {}/{}: {} MHz.'
                         .format(i + 1, len(requested_test_freqs), freq / 1e6))
            elif i >= (len(requested_test_freqs) - print_counts):
                Aqf.hop('Getting channel response for freq {}/{}: {} MHz.'
                         .format(i + 1, len(requested_test_freqs), freq / 1e6))
            else:
                LOGGER.info('Getting channel response for freq {}/{}: {} MHz.'
                            .format(i + 1, len(requested_test_freqs), freq / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency

            if this_source_freq == last_source_freq:
                LOGGER.info('Skipping channel response for freq {}/{}: {} MHz.\n'
                            'Digitiser frequency is same as previous.'
                            .format(i + 1, len(requested_test_freqs), freq / 1e6))
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq

            try:
                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                spead_failure_counter += 1
                errmsg = 'Could not retrieve clean SPEAD packet, as #{} Queue is Empty.'.format(
                spead_failure_counter)
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if spead_failure_counter > 5:
                    spead_failure_counter = 0
                    Aqf.failed('Bailed: Kept receiving empty Spead packets')
                    return False

            else:
                this_freq_data = this_freq_dump['xeng_raw'].value
                this_freq_response = normalised_magnitude(
                    this_freq_data[:, test_baseline, :])
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            # Plot an overall frequency response at the centre frequency just as
            # a sanity check

            plt_filename = '{}_channel_resp_sanity_check.png'.format(self._testMethodName)
            plt_title = 'Log: overrall frequency response at {} ({}MHz).'.format(
                test_chan, this_source_freq / 1e6)
            plt_caption = ('This is merely a sanity check to plot '
                          'an overrall frequency response at the center frequency.')
            if np.abs(freq - expected_fc) < 0.1:
                aqf_plot_channels(this_freq_response, plt_filename, plt_title,
                    log_dynamic_range=90, caption=plt_caption, hlines=cutoff)

        # Test fft overflow and qdr status after
        check_fftoverflow_qdrstatus(self.correlator, self.last_pfb_counts)
        self.corr_fix.stop_x_data()
        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)

        def plot_and_save(freqs, data, plot_filename, plt_title, caption="", cutoff=None,
                          show=False):
            df = self.corr_freqs.delta_f
            fig = plt.plot(freqs, data)[0]
            axes = fig.get_axes()
            ybound = axes.get_ybound()
            yb_diff = abs(ybound[1] - ybound[0])
            #new_ybound = [ybound[0] - yb_diff * 1.1, ybound[1] + yb_diff * 1.1]
            new_ybound = [ybound[0]*1.1, ybound[1]*1.1]
            new_ybound = [y if y != 0 else yb_diff*0.05 for y in new_ybound]
            plt.vlines(expected_fc, *new_ybound, colors='r', label='chan fc')
            plt.vlines(expected_fc - df / 2, *new_ybound, label='chan min/max')
            plt.vlines(expected_fc - 0.8 * df / 2, *new_ybound, label='chan +-40%',
                       linestyles='dashed')
            plt.vlines(expected_fc + df / 2, *new_ybound, label='_chan max')
            plt.vlines(expected_fc + 0.8 * df / 2, *new_ybound, label='_chan +40%',
                       linestyles='dashed')
            plt.legend()
            plt.title(plt_title)
            axes.set_ybound(*new_ybound)
            plt.grid(True)
            plt.ylabel('dB relative to VACC max')
            # TODO Normalise plot to frequency bins
            plt.xlabel('Frequency (Hz)')
            if cutoff:
                plt.axhline(cutoff, color='r', ls='--', label='Cutoff @ -53dB', linewidth=0.5)
            Aqf.matplotlib_fig(plot_filename, caption=caption)
            if show:
                plt.show()
            #plt.close()
            plt.clf()

        plt_filename = '{}_Channel_Response.png'.format(self._testMethodName)
        plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90, normalise=True)
        plt_caption = 'Channel {} response vs source frequency'.format(test_chan)
        plt_title = 'Channel {} @ {} MHz response.'.format(test_chan, expected_fc / 1e6)
        # Plot channel response with -53dB cutoff horizontal line
        plot_and_save(actual_test_freqs, plot_data, plt_filename, plt_title, plt_caption, cutoff)

        # Plot PFB channel response with -6dB cuttoff horizontal line
        no_of_responses = 3
        legends = ['Channel {} ({} MHz) response'.format(
            ((test_chan + i) - 1), self.corr_freqs.chan_freqs[test_chan + i] / 1e6) for i in range(
            no_of_responses)]
        channel_response_list = [chan_responses[:, test_chan + i - 1] for i in range(
            no_of_responses)]
        plot_title = 'PFB Channel Response'
        plot_filename = '{}_adjacent_channels.png'.format(self._testMethodName)
        caption = 'Sample PFB channel response between {}'.format(test_chan)
        aqf_plot_channels(zip(channel_response_list, legends), plot_filename,
                          plot_title, log_dynamic_range=90, log_normalise_to=1,
                          normalise=True, caption=caption,
                          xlabel = 'Sample Steps', hlines=-6)

        # Plot Central PFB channel response with ylimit 0 to -6dB
        y_axis_limits = (-7, 1)
        plot_filename = '{}_central_adjacent_channels.png'.format(
            self._testMethodName)
        plot_title = 'PFB Central Channel Response'
        caption = 'Sample PFB central channel response between {}'.format(
            test_chan)
        aqf_plot_channels(zip(channel_response_list, legends), plot_filename,
                          plot_title,
                          log_dynamic_range=90, log_normalise_to=1,
                          normalise = True, caption=caption,
                          xlabel='Sample Steps', ylimits=y_axis_limits,
                          show=False)

        # Get responses for central 80% of channel
        df = self.corr_freqs.delta_f
        central_indices = (
            (actual_test_freqs <= expected_fc + 0.4 * df) &
            (actual_test_freqs >= expected_fc - 0.4 * df))
        central_chan_responses = chan_responses[central_indices]
        central_chan_test_freqs = actual_test_freqs[central_indices]

        # Plot channel response for central 80% of channel
        graph_name_central = '{}_central.png'.format(self._testMethodName)
        plot_data_central = loggerise(central_chan_responses[:, test_chan],
            dynamic_range=90, normalise=True)
        caption = 'Channel {} central response vs source frequency on max channels {}'.format(
                    test_chan, self.corr_freqs.n_chans)
        plt_title = 'Channel {} ({} MHz) response @ 80%'.format(test_chan, expected_fc / 1e6)
        plot_and_save(central_chan_test_freqs, plot_data_central, graph_name_central,
                      plt_title, caption=caption)

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
            Aqf.failed('The following input frequencies (first and last): {!r} '
                       'respectively had peak channeliser responses in channels '
                       '{!r}\n, and not test channel {} as expected.'.format(
                        fault_freqs[1::-1], set(sorted(fault_channels)), test_chan))

            LOGGER.error('The following input frequencies: {!r} respectively had '
                       'peak channeliser responses in channels {!r}, not '
                       'channel {} as expected.'.format(fault_freqs,
                        set(sorted(fault_channels)), test_chan))

        Aqf.less(
            np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
            'Check that VACC output is at < 99% of maximum value, if fails '
            'then it is probably overranging.')
        max_central_chan_response = np.max(10 * np.log10(
            central_chan_responses[:, test_chan]))
        min_central_chan_response = np.min(10 * np.log10(
            central_chan_responses[:, test_chan]))
        chan_ripple = max_central_chan_response - min_central_chan_response
        acceptable_ripple_lt = 1.5
        Aqf.less(chan_ripple, acceptable_ripple_lt,
                 'Check that ripple within 80% of channel fc is < {} dB'
                 .format(acceptable_ripple_lt))

        # Get frequency samples closest channel fc and crossover points
        co_low_freq = expected_fc - df / 2
        co_high_freq = expected_fc + df / 2

        def get_close_result(freq):
            ind = np.argmin(np.abs(actual_test_freqs - freq))
            source_freq = actual_test_freqs[ind]
            response = chan_responses[ind, test_chan]
            return ind, source_freq, response

        fc_ind, fc_src_freq, fc_resp = get_close_result(expected_fc)
        co_low_ind, co_low_src_freq, co_low_resp = get_close_result(co_low_freq)
        co_high_ind, co_high_src_freq, co_high_resp = get_close_result(co_high_freq)
        # [CBF-REQ-0047] CBF channelisation frequency resolution requirement
        Aqf.step('Check that response at channel-edges are -3 dB relative to '
                 'the channel centre at {} Hz, actual source freq '
                 '{} Hz'.format(expected_fc, fc_src_freq))

        desired_cutoff_resp = -6  # dB
        acceptable_co_var = 0.1  # dB, TODO 2015-12-09 NM: thumbsuck number
        co_mid_rel_resp = 10 * np.log10(fc_resp)
        co_low_rel_resp = 10 * np.log10(co_low_resp)
        co_high_rel_resp = 10 * np.log10(co_high_resp)

        co_lo_band_edge_rel_resp = co_mid_rel_resp - co_low_rel_resp
        co_hi_band_edge_rel_resp = co_mid_rel_resp - co_high_rel_resp

        low_rel_resp_accept = np.abs(desired_cutoff_resp + acceptable_co_var)
        hi_rel_resp_accept = np.abs(desired_cutoff_resp - acceptable_co_var)

        Aqf.is_true(low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
            'Check that relative response at the low band-edge ({co_lo_band_edge_rel_resp} '
            'dB @ {co_low_freq} Hz, actual source freq {co_low_src_freq}) '
            'is within the range of {desired_cutoff_resp} +- 1% relative to '
            'channel centre response.'.format(**locals()))

        Aqf.is_true(low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
            'Check that relative response at the high band-edge ({co_hi_band_edge_rel_resp} '
            'dB @ {co_high_freq} Hz, actual source freq {co_high_src_freq}) '
            'is within the range of {desired_cutoff_resp} +- 1% relative to '
            'channel centre response.'.format(**locals()))

    def _test_sfdr_peaks(self, required_chan_spacing, no_channels, stepsize=None, cutoff=53):
        """Test channel spacing and out-of-channel response

        Will loop over all the channels, placing the source frequency as close to the
        centre frequency of that channel as possible.

        Parameters
        ----------
        required_chan_spacing: float
        no_channels: int
        stepsize: int
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

        if stepsize:
            Aqf.step('Running FASTER version of Channelisation SFDR test '
                     'with {} step size.'.format(stepsize))
            print_counts = 4 * stepsize
        else:
            print_counts = 4

        # Abitrary frequency
        cw_scale = 0.675
        Aqf.step('Configure digitiser simulator to generate a continuos wave.')
        dsim_set_success = set_input_levels(self.corr_fix, self.dhost,cw_scale=cw_scale,
                                            freq=self.corr_freqs.bandwidth / 2.0,
                                            fft_shift=8191, gain='11+0j')
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(initial_dump['xeng_raw'].value.shape[0], no_channels,
                       'Capture an initial correlator SPEAD packet, determine the number of channels '
                       'and processing bandwidth: {}Hz.'.format(initial_dump['bandwidth'].value))
            chan_spacing = initial_dump['bandwidth'].value/initial_dump['xeng_raw'].value.shape[0]
            # [CBF-REQ-0043]
            Aqf.less(chan_spacing, required_chan_spacing,
                    'Verify that the calculated channel frequency step size is <= {} Hz'.format(
                    required_chan_spacing))

        Aqf.step('Sweeping the digitiser simulator over the all channels that fall '
                 'within the complete L-band.')
        spead_failure_counter = 0
        for channel, channel_f0 in enumerate(
                self.corr_freqs.chan_freqs[start_chan::stepsize], start_chan):

            if stepsize:
                channel *= stepsize
            if channel < print_counts:
                Aqf.hop('Getting channel response for freq {}/{}: {} MHz.'
                         .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            elif channel > (n_chans - print_counts):
                Aqf.hop('Getting channel response for freq {}/{}: {} MHz.'
                         .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            else:
                LOGGER.info('Getting channel response for freq {}/{}: {} MHz.'
                            .format(channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            try:
                this_freq_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw'].value
            except Queue.Empty:
                spead_failure_counter += 1
                errmsg = 'Could not retrieve clean SPEAD packet, as #{} Queue is Empty.'.format(
                spead_failure_counter)
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if spead_failure_counter > 5:
                    spead_failure_counter = 0
                    Aqf.failed('Bailed: Kept receiving empty Spead packets')
                    return False
            else:
                this_freq_response = (
                    normalised_magnitude(this_freq_data[:, test_baseline, :]))

                plt_filename =  '{}_channel_resp.png'.format(self._testMethodName)
                plt_title = 'Log: Channel response at {} @ {} MHz'.format(
                            channel, this_source_freq / 1e6)

                if channel in (n_chans // 10, n_chans // 2, 9 * n_chans // 10):
                    aqf_plot_channels(
                        this_freq_response, plt_filename, plt_title, log_dynamic_range=90,
                        hlines=-cutoff)

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

    def _test_product_baselines(self):
        set_default_eq(self.correlator)
        noise_scale = 0.05
        Aqf.step('Dsim configured to generate correlated noise: scale @ {}.'.format(noise_scale))
        dsim_set_success = set_input_levels(self.corr_fix, self.dhost, awgn_scale=noise_scale,
            fft_shift=511, gain='113+0j')
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        Aqf.step('Set list for all the correlator input labels as per config file')
        local_src_names = self.correlator.configd['fengine']['source_names'].split(',')
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
        for i in range(2):
            self.corr_fix.issue_metadata()
            self.corr_fix.start_x_data()
        if not self.corr_fix.issue_metadata():
            Aqf.failed('Could not issue new metadata')
        try:
            test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        else:
            # Get bls ordering from get dump
            bls_ordering = test_dump['bls_ordering'].value
            Aqf.hop('Get list of all the correlator input labels that where set from config file')
            input_labels = sorted(tuple(test_dump['input_labelling'].value[:, 0]))
            Aqf.step('Get list of all the baselines present in the correlator output from spead packet')
            baselines_lookup = get_baselines_lookup(test_dump)
            present_baselines = sorted(baselines_lookup.keys())

            Aqf.step('List of all possible baselines (including redundant baselines) '
                     'for the given list of inputs')
            possible_baselines = set()
            for li in input_labels:
                for lj in input_labels:
                    possible_baselines.add((li, lj))

            test_bl = sorted(list(possible_baselines))
            Aqf.step('Check that each baseline (or its reverse-order counterpart) is present '
                     'in the correlator output')
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
            Aqf.step('Expect all baselines and all channels to be non-zero with correlator noise')
            Aqf.is_false(zero_baselines(test_data),
                         'Check that no baselines have all-zero visibilities')
            Aqf.equals(nonzero_baselines(test_data), all_nonzero_baselines(test_data),
                       'Check that all baseline visibilities are non-zero across '
                       'all channels')
            Aqf.step('Save initial f-engine equalisations, and ensure they are restored '
                     'at the end of the test')
            initial_equalisations = get_and_restore_initial_eqs(self, self.correlator)

            Aqf.step('Set all inputs to zero, and check that output product is all-zero')
            _input = None
            for _input in input_labels:
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(_input, 0)
                    if not reply.reply_ok():
                        raise ValueError
                    time.sleep(0.01)
                except ValueError:
                    Aqf.failed('Failed to set equalisations on input: {}'.format(_input))
            Aqf.wait(.5, 'Wait for equalisations/gains to be set')
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
            Aqf.step('Stepping through input combinations, verifying for each that '
                     'the correct output appears in the correct baseline product.')

            dataFrame = pandas.DataFrame(index=sorted(input_labels),
                                         columns=list(sorted(present_baselines)))
            for inp in input_labels:
                old_eq = complex(initial_equalisations[inp][0])
                reply, informs = self.corr_fix.katcp_rct.req.gain(inp, old_eq)
                msg = 'Gain correction on input {} set to {}.'.format(inp, old_eq)
                if reply.reply_ok():
                    Aqf.passed(msg)
                    zero_inputs.remove(inp)
                    nonzero_inputs.add(inp)
                    expected_z_bls, expected_nz_bls = (
                        calc_zero_and_nonzero_baselines(nonzero_inputs))
                    test_data = self.receiver.get_clean_dump()['xeng_raw'].value
                    # plot baseline channel response
                    plot_data = [normalised_magnitude(test_data[:, i, :])
                                 for i in plot_baseline_inds]
                    plot_filename = '{}_channel_resp_{}.png'.format(
                                    self._testMethodName, inp)
                    plot_title = 'Bls channel response non-zero inputs: {}'.format(
                                 sorted(nonzero_inputs))
                    caption = ('Baseline channel response with the '
                              'following non-zero inputs: {}'.format(sorted(nonzero_inputs)))
                    aqf_plot_channels(zip(plot_data, plot_baseline_legends),
                                     plot_filename, plot_title, log_dynamic_range=90,
                                     log_normalise_to=1, caption=caption)

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
                               "Also check that expected baselines visibilities are zero.\n")

                    # Sum of all baselines powers expected to be non zeros
                    sum_of_bl_powers = [normalised_magnitude(test_data[:, expected_bl,:])
                                       for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                                       for expected_nz_bl_ind in sorted(expected_nz_bls)]]

                    dataFrame.loc[inp][sorted([i for i in expected_nz_bls])[-1]] = np.sum(
                        sum_of_bl_powers)
                else:
                    Aqf.failed(msg)
            dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')

    def _test_back2back_consistency(self):
        threshold = 1e-7  # Threshold: -70dB
        test_chan = randrange(self.corr_freqs.n_chans)
        test_baseline = 0 # auto-corr
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Check that back-to-back SPEAD packets with same input are equal on '
                 'channel({}) @ {}MHz.'.format(test_chan, expected_fc / 1e6))
        source_period_in_samples = self.corr_freqs.n_chans * 2
        cw_scale = 0.675
        Aqf.step('Digitiser simulator configured to generate continuous wave')
        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            spead_failure_counter = 0
            for i, freq in enumerate(requested_test_freqs):
                Aqf.hop('Testing SPEAD packet consistency {}/{} @ {} MHz.'.format(
                    i + 1, len(requested_test_freqs), freq / 1e6))
                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                  repeatN=source_period_in_samples)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency
                dumps_data = []
                chan_responses = []
                for dump_no in xrange(3):
                    if dump_no == 0:
                        this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                    else:
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            spead_failure_counter += 1
                            errmsg = ('Could not retrieve clean SPEAD packet, as #{} '
                                     'Queue is Empty.'.format(spead_failure_counter))
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                            if spead_failure_counter > 5:
                                spead_failure_counter = 0
                                Aqf.failed('Bailed: Kept receiving empty Spead packets')
                                return False

                    this_freq_data = this_freq_dump['xeng_raw'].value
                    dumps_data.append(this_freq_data)
                    this_freq_response = normalised_magnitude(
                        this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)

                diff_dumps = []
                for comparison in xrange(1, len(dumps_data)):
                    d0 = dumps_data[0]
                    d1 = dumps_data[comparison]
                    diff_dumps.append(np.max(d0 - d1))

                dumps_comp = np.max(np.array(diff_dumps) / initial_max_freq)
                msg = ('Check that back-to-back accumulations({}) with the same frequency '
                       'input differ by no more than {} threshold[dB].'.format(
                        dumps_comp, 10 * np.log10(threshold)))
                if not Aqf.less(dumps_comp, threshold, msg):
                    legends = ['dump #{}'.format(x) for x in xrange(len(chan_responses))]
                    plot_filename = '{}_chan_resp_{}.png'.format(self._testMethodName, i + 1)
                    plot_title = 'Log: Channel Response {} @ {}MHz'.format(
                        test_chan, this_source_freq)
                    aqf_plot_channels(zip(chan_responses, legends), plot_filename,
                        plot_title, log_dynamic_range=90, log_normalise_to=1,
                        caption='Comparison of back-to-back channelisation results with '
                                'source periodic every {} samples and sine frequency of '
                                '{} MHz.'.format(source_period_in_samples, this_source_freq))

    def _test_freq_scan_consistency(self):
        """Check that identical frequency scans produce equal results"""
        threshold = 1e-7  # Threshold: -70dB
        test_chan = randrange(self.corr_freqs.n_chans)
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            cw_scale = 0.675
            Aqf.step('Digitiser simulator configured to generate continuous wave')

            Aqf.step('Selected test channel {} and Frequency {}MHz'.format(
                test_chan, expected_fc / 1e6))
            for scan_i in xrange(3):
                scan_dumps = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
                        this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                        this_freq_data = this_freq_dump['xeng_raw'].value
                        initial_max_freq_list.append(initial_max_freq)
                    else:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
                        this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        this_freq_data = this_freq_dump['xeng_raw'].value

                    this_freq_response = normalised_magnitude(
                        this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)
                    scan_dumps.append(this_freq_data)

            for scan_i in xrange(1, len(scans)):
                for freq_i in xrange(len(scans[0])):
                    s0 = scans[0][freq_i]
                    s1 = scans[scan_i][freq_i]
                    norm_fac = initial_max_freq_list[freq_i]
                    # TODO Convert to a less-verbose comparison for Aqf.
                    # E.g. test all the frequencies and only save the error cases,
                    # then have a final Aqf-check so that there is only one step
                    # (not n_chan) in the report.
                    max_freq_scan = np.max(np.abs(s1 - s0)) / norm_fac
                    msg = ('Check that the frequency scan on SPEAD dump comparison({}) '
                          'is less than {} dB.'.format(max_freq_scan, threshold))
                    if not Aqf.less(max_freq_scan, threshold, msg):
                        legends = ['Freq scan #{}'.format(x) for x in xrange(len(chan_responses))]
                        caption=(
                        'Comparison of frequency sweeping from {}Mhz to {}Mhz '
                        'scan channelisation.'.format(requested_test_freqs[0]/1e6,
                            requested_test_freqs[-1] / 1e6, expected_fc))

                        aqf_plot_channels(zip(chan_responses, legends),
                            plot_filename = '{}_chan_resp.png'.format(self._testMethodName),
                            log_dynamic_range=90, log_normalise_to=1,
                            caption=caption)

    def _test_restart_consistency(self, instrument, no_channels):
        start_time = time.time()
        threshold = 1e-7
        test_chan = randrange(self.corr_freqs.n_chans)
        test_baseline = 0
        Aqf.step('Digitiser simulator configured to generate continuous wave')

        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        Aqf.step('Test will sweep over and around {}MHz frequency points, '
                 'at test channel {}.'.format(np.max(requested_test_freqs) / 1e6, test_chan))
        expected_fc = self.corr_freqs.chan_freqs[test_chan]

        Aqf.hop('Getting initial frequency SPEAD packet Dsim configured '
                'to generate cw at {}MHz\n'.format(expected_fc / 1e6))
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=0.25)

        try:
            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            # Plot an overall frequency response at the centre frequency just as
            # a sanity check
            init_source_freq = normalised_magnitude(
                this_freq_dump['xeng_raw'].value[:, test_baseline, :])
            filename = '{}_channel_resp.png'.format(self._testMethodName)
            title = 'Log: Channel Response {} @ {} MHz.\n'.format(test_chan, expected_fc / 1e6)
            caption = (
                'This is merely a sanity check to plot an overrall frequency  response '
                'at the center frequency.')
            aqf_plot_channels(init_source_freq, filename, title, log_dynamic_range=90,
                caption=caption)

            restart_retries = 5
            def _restart_instrument(retries=restart_retries):
                if not self.corr_fix.stop_x_data():
                    Aqf.failed('Could not stop x data from capturing.')
                if not self.corr_fix.deprogram_fpgas(instrument):
                    Aqf.failed('Could not deprogram FPGAs')
                try:
                    self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    Aqf.failed('SPEAD packets are still being produced.')
                except Queue.Empty:
                    pass

                self.corr_fix.halt_array()
                corr_init = False
                xhosts = self.correlator.xhosts
                fhosts = self.correlator.fhosts

                while retries and not corr_init:
                    try:
                        corr_init = self.set_instrument(instrument)
                        retries -= 1
                    except:
                        pass

                    if retries == 0:
                        errmsg = 'Could not restart the correlator after {} tries.'.format(
                            retries)
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False

                if corr_init:
                    host = (xhosts + fhosts)[randrange(len(xhosts + fhosts))]
                    Aqf.is_true(host,
                                'Confirm that the instrument is initialised by '
                                'checking if a random host: {} is programmed '
                                'and running.'.format(host.host))
                    try:
                        self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD packet: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False

                    Aqf.equals(
                        self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw'].value.shape[0],
                        no_channels,
                        'Check that data product has the number of frequency '
                        'channels {no_channels} corresponding to the {instrument} '
                        'instrument product'.format(**locals()))
                    return True
                else:
                    Aqf.failed('Could not restart {} after {} tries.'.format(instrument, retries))
                    return False

            initial_max_freq_list = []
            scans = []
            channel_responses = []
            for scan_i in xrange(3):
                if scan_i:
                    Aqf.step('#{scan_i}: Initialising {instrument} instrument'.format(**locals()))
                    intrument_success = _restart_instrument()
                    if not intrument_success:
                        Aqf.failed('Failed to restart the correlator after {} retries.'.format(
                            restart_retries))
                        return False

                scan_dumps = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        if self.corr_fix.start_x_data():
                            Aqf.hop('Getting Frequency SPEAD packet #{} with Dsim configured '
                                    'to generate cw at {}MHz'.format(i, freq / 1e6))
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)

                        initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                        this_freq_data = this_freq_dump['xeng_raw'].value
                        initial_max_freq_list.append(initial_max_freq)
                        freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    else:
                        Aqf.hop('Getting Frequency SPEAD packet #{} with Dsim configured '
                                'to generate cw at {}MHz'.format(i, freq / 1e6))
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        this_freq_data = this_freq_dump['xeng_raw'].value
                        freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    scan_dumps.append(this_freq_data)
                    channel_responses.append(freq_response)

            normalised_init_freq = np.array(initial_max_freq_list)
            for comp in xrange(1, len(normalised_init_freq)):
                v0 = np.array(normalised_init_freq[comp - 1])
                v1 = np.array(normalised_init_freq[comp])

            correct_init_freq = np.abs(np.max(v0 - v1))

            diff_scans_dumps = []
            for comparison in xrange(1, len(scans)):
                s0 = np.array(scans[comparison - 1])
                s1 = np.array(scans[comparison])
                diff_scans_dumps.append(np.max(s0 - s1))

            diff_scans_comp = np.max(np.array(diff_scans_dumps) / correct_init_freq)

            final_time = time.time() - start_time
            Aqf.hop('Test took {} seconds to complete'.format(final_time))

            if not Aqf.less(diff_scans_comp, threshold,
                            'Check that roach restart SPEAD packets comparison results '
                            'with the same frequency input differ by no more than {}dB '
                            'threshold.'.format(diff_scans_comp, threshold)):
                legends = ['Channel Responce #{}'.format(x) for x in xrange(len(
                    channel_responses))]
                file_name = '{}_chan_resp.png'.format(self._testMethodName)
                file_caption = 'Check that results are consistent on correlator restart'
                plot_title = 'Log: Correlator restart consistency channel response {}'.format(
                             test_chan)
                aqf_plot_channels(zip(channel_responses, legends), file_name,
                    plot_title, log_dynamic_range=90, log_normalise_to=1,
                    caption=file_caption)

    def _test_delay_tracking(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking"""
        setup_data = self._delays_setup()
        if setup_data:
            sampling_period = self.corr_freqs.sample_period
            no_chans = range(self.corr_freqs.n_chans)

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
                    this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT, discard=0)

                    future_time = 900e-3
                    settling_time = 600e-3
                    roundtrip = 0.03
                    sync_time = self.correlator.get_synch_time()
                    if sync_time == -1:
                        sync_time = this_freq_dump['sync_time'].value
                    dump_timestamp = (roundtrip + sync_time +
                                      this_freq_dump['timestamp'].value /
                                      this_freq_dump['scale_factor_timestamp'].value)
                    t_apply = (dump_timestamp + 10 * this_freq_dump['int_time'].value +
                               future_time)
                    reply, informs = self.corr_fix.katcp_rct.req.delays(t_apply,
                                                               *delay_coefficients)
                    Aqf.is_true(reply.reply_ok(),
                                'Delays Reply: {}, Intergration time:{}s'.format(
                                reply.arguments[1], this_freq_dump['int_time'].value))
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

            actual_data = get_actual_phases()
            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]
            expected_phases = get_expected_phases()
            title = 'CBF Delay Compensation/LO Fringe stopping polynomial'
            caption = ('Actual and expected Unwrapped Correlation Phase, '
                       'dashed line indicates expected value.')
            file_name = '{}_Response.png'.format(self._testMethodName)
            units = 'secs'

            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   units, file_name, title, caption)
            expected_phases = [phase for rads, phase in get_expected_phases()]
            tolerance = 1e-2
            decimal = len(str(tolerance).split('.')[-1])

            for i, delay in enumerate(test_delays):
                delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
                delta_expected = np.max(expected_phases[i]) - np.min(expected_phases[i])
                msg = (
                    'Check if difference expected({0:.5f}) and actual({1:.5f}) '
                    'phases are equal at delay {2:.5f}ns within {3} tolerance.'.format(
                        delta_expected, delta_actual, delay * 1e9, tolerance))
                Aqf.almost_equals(delta_expected, delta_actual, tolerance, msg)
                try:
                    delta_actual_s = delta_actual - (delta_actual % tolerance)
                    delta_expected_s = delta_expected - (delta_expected % tolerance)
                    np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
                except AssertionError:
                    Aqf.step('Difference expected({0:.5f}) and actual({1:.5f}) '
                             'phases are not equal at delay {2:.5f}ns within {3} tolerance.'.format(
                             delta_expected, delta_actual, delay * 1e9, tolerance))

                    chan_response = [responses for test_delays_, responses in actual_response
                                     if test_delays_ == delay][0]
                    caption = (
                        'Difference expected({0:.5f}) and actual({1:.5f}) phases'
                        ' are not equal at delay {2:.5f}ns within {3} tolerance.'.format(
                        delta_expected, delta_actual, delay * 1e9, tolerance))
                    plt_filename = '{}_{}_chan_resp.png'.format(self._testMethodName, i)
                    plt_title = 'Log: Channel response at {}ns'.format(delay)

                    aqf_plot_channels(chan_response, plt_filename, plt_title, log_dynamic_range=90,
                        caption=caption)

            for delay, count in zip(test_delays, xrange(1, len(expected_phases))):
                msg = (
                    'Check that when a delay of {0} clock cycle({1:.5f} ns) is introduced '
                    'there is a change of phase of {2:.5f} degrees as expected to within '
                    '{3} tolerance.'.format((count + 1) * .5, delay * 1e9,
                        np.rad2deg(np.pi) * (count + 1) * .5, tolerance))
                aqf_array_abs_error_less(
                    actual_phases[count][1:], expected_phases[count][1:], msg, tolerance)

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
        except:
            errmsg = 'KATCP connection encountered errors.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
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
            msg = ('Check that the time synchronised sensor values replies with '
                   '!sensor-value ok 1')
            Aqf.equals(str(sensors_req.sensor_value('time.synchronised')[0]),
                       '!sensor-value ok 1', msg)

            # Check all sensors statuses if they are nominal
            for sensor in self.corr_fix.rct.sensor.values():
                LOGGER.info(sensor.name + ' :' + str(sensor.get_value()))
                Aqf.equals(sensor.get_status(), 'nominal',
                           'Sensor status: {}, {} '.format(sensor.name,
                           sensor.get_status()))

    def _test_roach_qdr_sensors(self):

        def roach_qdr(corr_hosts, engine_type, sensor_timeout=60):
            try:
                array_sensors = self.corr_fix.katcp_rct.sensor
                self.assertIsInstance(array_sensors,
                                      katcp.resource_client.AttrMappingProxy)
            except:
                errmsg = 'KATCP connection encountered errors.\n'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                hosts = corr_hosts
                # Random host selection, seems to be broken due to the instability on
                # hardware QDR's
                # host = hosts[randrange(len(hosts))]
                host = hosts[1]
                host.clear_status()
                Aqf.step("Selected host: {} - {}".format(host.host, engine_type))
                try:
                    host_sensor = getattr(array_sensors, '{}_{}_qdr'.format(
                        host.host.lower(), engine_type))
                except AttributeError:
                    Aqf.failed('Could not get sensor attributes on {}'.format(host.host))

                # Check if qdr is okay
                Aqf.is_true(host_sensor.get_value(),
                            'Confirm that sensor indicates QDR status: {} on {}.'
                            .format(host_sensor.status, host.host))
                host_sensor.set_strategy('auto')
                self.addCleanup(host_sensor.set_sampling_strategy, 'none')

                def blindwrite():
                    """Writes junk to memory"""
                    try:
                        for i in xrange(100):
                            host.blindwrite('qdr0_memory', 'write_junk_to_memory')
                            host.blindwrite('qdr1_memory', 'write_junk_to_memory')
                        return True
                    except:
                        return False

                Aqf.is_true(blindwrite(), "Writing junk to {} memory.".format(host.host))
                try:
                    if host_sensor.wait(False, timeout=sensor_timeout):
                        # Verify that qdr corrupted or unreadable
                        Aqf.equals(host_sensor.get_status(), 'error',
                                   'Confirm that sensor indicates that the memory on {} '
                                   'is unreadable/corrupted.'.format(host.host))
                except TimeoutError:
                    Aqf.failed('Timed-out: Could not verify that the qdr is corrupted or unreadable.')

                if engine_type == 'xeng':
                    current_errors = host.registers.vacc_errors1.read()['data']['parity']
                else:
                    ct_ctrs = host.registers.ct_ctrs.read()['data']
                    current_errors = (ct_ctrs['ct_parerr_cnt0'] + ct_ctrs['ct_parerr_cnt1'])

                aqf_is_not_equals(current_errors, 0, "Confirm that the error counters incremented.")

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
                host.clear_status()

        roach_qdr(self.correlator.fhosts, 'feng')
        roach_qdr(self.correlator.xhosts, 'xeng')

    def _test_roach_pfb_sensors(self):
        """Sensor PFB error"""
        array_sensors = self.corr_fix.katcp_rct.sensor
        Aqf.skipped('PFB sensor test not yet implemented.')

    def _test_feng_link_error(self):
        # Select an F-engine that is being used to produce the test data product on
        # which to trigger the link error.
        fhost = self.correlator.fhosts[0]

        # Record the current multicast desitination of one of the F-engine data
        # ethernet ports,
        mul_ip = (fhost.katcprequest('wordread',
                                     request_args=(['iptx_base']))[0].arguments[1])
        cur_mul_ip_x = mul_ip[2:]
        cur_mul_ip_int = [int(x + y, 16) for x, y in zip(cur_mul_ip_x[::2], cur_mul_ip_x[1::2])]

        # configure the same port multicast destination to an unused address,
        # effectively dropping that data.
        junk_ip = fhost.katcprequest('wordwrite',
                                     requested_args=(['iptx_base', '0', '0xefefefef']))
        dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)

        # To clear: Restore the data ethernet port multicast destination to the saved
        # value.
        restore_ip = fhost.katcprequest('wordwrite',
                                        requested_args=(['iptx_base', '0', mul_ip]))


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

    def _test_vacc(self, test_chan, chan_index=None):
        """Test vector accumulator"""
        # Choose a test freqency around the centre of the band.
        test_freq = self.corr_freqs.bandwidth / 2.
        sources = [_input['source'].name
                   for _input in self.correlator.fengine_sources]
        test_input = sources[0]
        eq_scaling = 30
        acc_times = [0.2, 0.5, 1]

        internal_accumulations = int(
            self.correlator.configd['xengine']['xeng_accumulation_len'])
        delta_acc_t = self.corr_freqs.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = np.argmin(
            np.abs(self.corr_freqs.chan_freqs[:chan_index] - test_freq))
        eqs = np.zeros(self.corr_freqs.n_chans, dtype=np.complex)
        eqs[test_freq_channel] = eq_scaling
        get_and_restore_initial_eqs(self, self.correlator)
        reply, informs = self.corr_fix.katcp_rct.req.gain(test_input, *list(eqs))
        if reply.reply_ok():
            Aqf.hop('Gain factors set successfully.')
        Aqf.step(
            'Configured Dsim output(cw0 @ {}MHz) to be periodic in FFT-length({})  in order'
            ' for each FFT to be identical'.format(test_freq / 1e6, self.corr_freqs.n_chans * 2))
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
                    'Check that the spectrum is zero except in the test channel: [0:test_freq_channel]')
        Aqf.is_true(np.all(quantiser_spectrum[test_freq_channel + 1:] == 0),
                    'Check that the spectrum is zero except in the test channel: [test_freq_channel+1:]')

        chan_response = []
        for vacc_accumulations in test_acc_lens:
            self.correlator.xops.set_acc_len(vacc_accumulations)
            Aqf.almost_equals(vacc_accumulations, self.correlator.xops.get_acc_len(),
                              1e-2, 'Confirm that vacc length was set successfully with {}.'.format(
                    vacc_accumulations))
            no_accs = internal_accumulations * vacc_accumulations
            expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
            d = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            actual_response = complexise(d['xeng_raw'].value[:, 0, :])
            chan_response.append(normalised_magnitude(d['xeng_raw'].value[:, 0, :]))
            # Check that the accumulator response is equal to the expected response
            msg = ('Check that the accumulator actual response is equal to the expected'
                   ' response for {} accumulation length'.format(vacc_accumulations))
            caption = (
            'Check that the accumulator actual response is equal to the expected response'
            ' for {} accumulation length with source periodic every {} samples and'
            ' sine frequency of {} MHz. '.format(test_acc_lens, self.corr_freqs.n_chans * 2,
                                                test_freq))
            filename = (
                '{}_chan_resp_{}_acc.png'.format(self._testMethodName, vacc_accumulations))
            if not aqf_numpy_almost_equal(expected_response, actual_response[:chan_index], msg):
                aqf_plot_channels(actual_response,
                                  plot_filename=filename,
                                  plot_title='Vector Accumulation Length',
                                  log_dynamic_range=90, log_normalise_to=1, caption=caption)

    def _test_product_switch(self, instrument, no_channels):
        Aqf.step('Confirm that SPEAD packets are being produced when Dsim is '
                 'configured to output correlated noise')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            self.corr_fix.stop_x_data()
        except:
            pass
        Aqf.step('Deprogramming xhosts first then fhosts avoid reorder timeout errors')
        xhosts = self.correlator.xhosts
        fhosts = self.correlator.fhosts
        if self.corr_fix.deprogram_fpgas(instrument):
            [Aqf.is_false(host.is_running(), '{} Deprogrammed'.format(host.host))
             for host in xhosts + fhosts]
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
                if corr_init:
                    LOGGER.info('Correlator started successfully after {} retries'
                                .format(retries))
            except:
                retries -= 1
                if retries == 0:
                    errmsg = 'Could not restart the correlator after 5 tries.'
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

        if corr_init:
            [Aqf.is_true(host.is_running(),
                         'Confirm that the instrument is initialised by checking if '
                         '{} is programmed.'
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
                       'instrument product'.format(**locals()))

            final_time = end_time - start_time
            minute = 60.0
            Aqf.less(final_time, minute,
                     'Confirm that instrument switching to {instrument} time is '
                     'less than one minute'.format(**locals()))

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
        flag_bit = xeng_raw_bits_flags.overrange
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(xeng_raw_bits_flags)
        other_bits = all_bits - set([flag_bit])
        flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
        flag_condition = 'ADC overrange'

        set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value, consider_bits=all_bits)
        Aqf.is_false(flag_bit in set_bits1,
                     'Check that {} is not set in SPEAD packet 1 before setting {}.'
                     .format(flag_descr, condition))
        # Bits that should not be set
        other_set_bits1 = set_bits1.intersection(other_bits)
        Aqf.equals(other_set_bits1, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits2 = get_set_bits(dump2['flags_xeng_raw'].value, consider_bits=all_bits)
        other_set_bits2 = set_bits2.intersection(other_bits)
        Aqf.is_true(flag_bit in set_bits2,
                    'Check that {} is set in SPEAD packet 2 while toggeling {}.'
                    .format(flag_descr, condition))
        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value, consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in SPEAD packet 3 after clearing {}.'
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

        flag_bit = xeng_raw_bits_flags.noise_diode
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(xeng_raw_bits_flags)
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
                    'Check that {} is set in SPEAD packet 2 while toggeling {}.'
                    .format(flag_descr, condition))

        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in SPEAD packet 3 after clearing {}.'
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
            self.correlator.fops.set_fft_shift_all(shift_value=0)

        def disable_fft_overflow():
            # TODO 2015-09-22 (NM) There seems to be some issue with the dsim sin_corr
            # source that results in it producing all zeros... So using sin_0 and sin_1
            # instead
            # self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=0)
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=0.)
            # Restore the default FFT shifts as per the correlator config.
            self.correlator.fops.set_fft_shift_all()

        condition = ('FFT overflow by setting an aggressive FFT shift with '
                     'a pure tone input')
        dump1, dump2, dump3, = self.get_flag_dumps(enable_fft_overflow,
                                                   disable_fft_overflow, condition)
        flag_bit = xeng_raw_bits_flags.overrange
        # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
        all_bits = set(xeng_raw_bits_flags)
        other_bits = all_bits - set([flag_bit])
        flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
        flag_condition = 'FFT overrange'

        set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        Aqf.is_false(flag_bit in set_bits1,
                     'Check that {} is not set in SPEAD packet 1 before setting {}.'
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
                    'Check that {} is set in SPEAD packet 2 while toggling {}.'
                    .format(flag_descr, condition))
        Aqf.equals(other_set_bits2, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

        set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                 consider_bits=all_bits)
        other_set_bits3 = set_bits3.intersection(other_bits)
        Aqf.is_false(flag_bit in set_bits3,
                     'Check that {} is not set in SPEAD packet 3 after clearing {}.'
                     .format(flag_descr, condition))

        Aqf.equals(other_set_bits3, set(),
                   'Check that no other flag bits (any of {}) are set.'
                   .format(sorted(other_bits)))

    def _test_fringe_offset(self):
        """CBF per-antenna phase error -- Fringe offset"""
        # TODO Randomise test values
        setup_data = self._delays_setup()
        if setup_data:
            get_fringe_ranges = get_delay_bounds(self.correlator)
            min_fringe_offset = get_fringe_ranges['max_negative_phase_offset']
            max_fringe_offset = get_fringe_ranges['max_positive_phase_offset']
            fringe_offset = randrange(min_fringe_offset, max_fringe_offset, int=float)
            dump_counts = 5
            delay_value = 0
            delay_rate = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
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
            graph_name = '{}_response.png'.format(self._testMethodName)
            caption = ('Actual and expected Unwrapped Correlation Phase, '
                       'dashed line indicates expected value.')

            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   graph_units, graph_name, graph_title, caption)

            # Ignoring first dump because the delays might not be set for full
            # integration.
            tolerance = 1e-5
            decimal = len(str(tolerance).split('.')[-1])
            actual_phases = np.unwrap(actual_phases)
            expected_phases = np.unwrap([phase for label, phase in expected_phases])

            for i in xrange(1, len(expected_phases) - 1):
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
                    caption = ('Difference expected({0:.5f}) and actual({1:.5f}) phases '
                               'are not equal within {2} tolerance when fringe offset is {3}.'.format(
                               delta_expected, delta_actual, tolerance, fringe_offset))
                    aqf_plot_channels(
                        actual_response[-1],
                        '{}_{}_response.png'.format(self._testMethodName, fringe_offset),
                        'Log: Channel response of Fringe offset: {}rads'.format(fringe_offset),
                        log_dynamic_range=90, log_normalise_to=1,
                        caption=caption)

    def _test_delay_rate(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate"""

        setup_data = self._delays_setup()
        if setup_data:
            get_delay_ranges = get_delay_bounds(self.correlator)
            delay_min = get_delay_ranges['max_negative_delta_delay']
            delay_max = get_delay_ranges['max_positive_delta_delay']
            delay_rate = randrange(delay_min, delay_max, int=float)
            dump_counts = 5
            delay_value = 0
            #delay_rate = setup_data['sample_period'] / setup_data['int_time']
            fringe_offset = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
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
            graph_name = '{}_response.png'.format(self._testMethodName)
            caption = ('Actual and expected Unwrapped Correlation Delay Rate, '
                       'dashed line indicates expected value.')

            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   graph_units, graph_name, graph_title, caption)

            actual_phases = np.unwrap(actual_phases)
            # TODO MM 2015-10-22
            # Ignoring first dump because the delays might not be set for full
            # integration.
            tolerance = 0.01
            decimal = len(str(tolerance).split('.')[-1])
            expected_phases = np.unwrap([phase for label, phase in expected_phases])
            for i in xrange(1, len(expected_phases) - 1):
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
                    legends = ['Response per SPEAD packet #{}'.format(x) for x in xrange(len(actual_response))]
                    caption = ('Difference expected({0:.5f}) and actual({1:.5f}) '
                               'phases are not equal withing {2} tolerance when delay rate is {3}.'.format(
                               delta_expected, delta_actual, tolerance, delay_rate))
                    aqf_plot_channels(
                        zip(actual_response, legends), '{}_chan_resp.png'.format(
                            self._testMethodName),
                        'Log: Channel response of Delay actual: {}'.format(delta_actual),
                        log_dynamic_range=90,
                        caption=caption)

    def _test_fringe_rate(self):
        """CBF per-antenna phase error -- Fringe rate"""
        setup_data = self._delays_setup()
        if setup_data:
            get_fringe_ranges = get_delay_bounds(self.correlator)
            min_fringe_rate = get_fringe_ranges['max_negative_delta_phase']
            max_fringe_rate = get_fringe_ranges['max_positive_delta_phase']
            fringe_rate = randrange(min_fringe_rate, max_fringe_rate, int=float)
            dump_counts = 5
            delay_value = 0
            delay_rate = 0
            fringe_offset = 0
            #fringe_rate = (np.pi / 8.) / setup_data['int_time']
            load_time = setup_data['t_apply']
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
            graph_name = '{}_response.png'.format(self._testMethodName)
            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   graph_units, graph_name, graph_title, caption)

            # NOTE: Ignoring first dump because the delays might not be set for full
            # intergration.
            tolerance = 0.01
            decimal = len(str(tolerance).split('.')[-1])
            actual_phases = np.unwrap(actual_phases)
            expected_phases = np.unwrap([phase for label, phase in expected_phases])

            for i in xrange(1, len(expected_phases) - 1):
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

                    legends = ['Response per SPEAD packet #{}'.format(x) for x in xrange(len(
                                actual_response))]
                    caption = ('Difference expected({0:.5f}) and actual({1:.5f}) phases '
                               'are not equal within {2} tolerance when fringe rate is {3}.'.format(
                               delta_expected, delta_actual, tolerance, fringe_rate))
                    aqf_plot_channels(
                        zip(actual_response, legends),
                        '{}_response.png'.format(self._testMethodName),
                        'Log: Channel response of Fringe offset: {}rads'.format(fringe_offset),
                        log_dynamic_range=90,
                        caption=caption)

    def _test_all_delays(self):
        """
        CBF per-antenna phase error -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        setup_data = self._delays_setup()
        if setup_data:
            get_ranges = get_delay_bounds(self.correlator)
            dump_counts = 5
            delay_value = randrange(get_ranges['min_delay'], get_ranges['max_delay'], int=float)
            delay_rate = randrange(get_ranges['max_negative_delta_delay'],
                                   get_ranges['max_positive_delta_delay'], int=float)
            fringe_offset = randrange(get_ranges['max_negative_phase_offset'],
                                      get_ranges['max_positive_phase_offset'], int=float)
            fringe_rate = randrange(get_ranges['max_negative_delta_phase'],
                                    get_ranges['max_positive_delta_phase'], int=float)
            load_time = setup_data['t_apply']

            delay_values = [0] * setup_data['num_inputs']
            delay_rates = [0] * setup_data['num_inputs']
            fringe_offsets = [0] * setup_data['num_inputs']
            fringe_rates = [0] * setup_data['num_inputs']

            delay_values[setup_data['test_source_ind']] = delay_value
            delay_rates[setup_data['test_source_ind']] = delay_rate
            fringe_offsets[setup_data['test_source_ind']] = fringe_offset
            fringe_rates[setup_data['test_source_ind']] = fringe_rate
            delay_coefficients = []
            for idx in xrange(len(delay_values)):
                delay_coefficients.append('{},{}:{},{}'
                                          .format(delay_values[idx], delay_rates[idx],
                                                  fringe_offsets[idx], fringe_rates[idx]))

            Aqf.step('Setting Parameters')
            Aqf.step('Time apply: {}'.format(load_time))
            Aqf.step('Delay Rate: {}'.format(delay_rate))
            Aqf.step('Delay Value: {}'.format(delay_value))
            Aqf.step('Fringe Offset: {}'.format(fringe_offset))
            Aqf.step('Fringe Rate: {}'.format(fringe_rate))

            actual_phases = self._get_actual_data(setup_data, dump_counts,
                                                  delay_coefficients)

            expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                      delay_coefficients, actual_phases)

            no_chans = setup_data['no_chans']
            graph_units = ''
            graph_title = 'All Delays Responses'
            graph_name = '{}_Response.png'.format(self._testMethodName)

            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   graph_units, graph_name, graph_title)

            # Ignoring first dump because the delays might not be set for full
            # integration.
            tolerance = 0.01
            actual_phases = np.unwrap(actual_phases)
            expected_phases = np.unwrap([phase for label, phase in expected_phases])
            for i in xrange(1, len(expected_phases) - 1):
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

    def _test_config_report(self, verbose):
        """CBF Report configuration"""
        test_config = self.corr_fix._test_config_file()

        def get_roach_config():

            Aqf.hop('DEngine :{}'.format(self.dhost.host))

            fhosts = [fhost.host for fhost in self.correlator.fhosts]
            Aqf.hop('Available FEngines :{}'.format(', '.join(fhosts)))

            xhosts = [xhost.host for xhost in self.correlator.xhosts]
            Aqf.hop('Available XEngines :{}\n'.format(', '.join(xhosts)))

            uboot_cmd = 'cat /dev/mtdblock5 | less | strings | head -1\n'
            romfs_cmd = 'cat /dev/mtdblock1 | less | strings | head -2 | tail -1\n'
            lnx_cmd = 'cat /dev/mtdblock0 | less | strings | head -1\n'
            hosts = self.correlator.fhosts + self.correlator.xhosts
            for count, host in enumerate(hosts, start=1):
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
                except:
                    errmsg = 'Could not connect to host: {}\n'.format(hostname)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

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
                try:
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
                        if verbose:
                            Aqf.progress('Repo: {}: Contains changes not staged for commit.\n\n'
                                         'Difference: \n\n{}'.format(name, clrs.red(git_diff)))
                        else:
                            Aqf.progress('Repo: {}: Contains changes not staged for'
                                         ' commit.\n\n'.format(name))
                    else:
                        Aqf.hop('Repo: {}: Up-to-date.\n\n'.format(name))
                except (subprocess.CalledProcessError, AssertionError, OSError):
                    Aqf.failed('{}'.format(repo_dir))

        def get_pdu_config():
            try:
                pdu_host_info = test_config['pdu_hosts']
                pdu_host_ips = test_config['pdu_hosts']['pdu_ips'].split(',')
            except KeyError:
                Aqf.failed('Could not retrieve PDU ip`s from config file')
                return False
            else:
                user = (pdu_host_info['username'] + '\r\n')
                password = (pdu_host_info['passwd'] + '\r\n')
            model_cmd = 'prodInfo\r\n'
            about_cmd = 'about\r\n'

            for count, host_ip in enumerate(pdu_host_ips, start=1):

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
                        try:
                            pdu_model = stdout[stdout.index('Model'):].split()[1]
                            Aqf.step('Checking PDU no: {}'.format(count))
                            Aqf.hop('PDU Model: {} on {}'.format(pdu_model, host_ip))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU model\n')

                    if 'Name' in stdout:
                        try:
                            pdu_name = (' '.join(stdout[stdout.index('Name'):stdout.index(
                                'Date')].split()[-4:]))
                            Aqf.hop('PDU Name: {}'.format(pdu_name))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU name\n')

                    if 'Serial' in stdout:
                        try:
                            pdu_serial = (stdout[stdout.find('Hardware Factory'):]
                                          .splitlines()[3].split()[-1])
                            Aqf.hop('PDU Serial Number: {}'.format(pdu_serial))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU serial number.\n')
                    if 'Revision' in stdout:
                        try:
                            pdu_hw_rev = (stdout[stdout.find('Hardware Factory'):]
                                          .splitlines()[4].split()[-1])
                            Aqf.hop('PDU HW Revision: {}'.format(pdu_hw_rev))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU HW Revision.\n')

                    if 'Application Module' and 'Version' in stdout:
                        try:
                            pdu_app_ver = (stdout[stdout.find('Application Module'):]
                                           .split()[6])
                            Aqf.hop('PDU Application Module Version: {} '.format(
                                pdu_app_ver))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU Application Module Version.\n')

                    if 'APC OS(AOS)' in stdout:
                        try:
                            pdu_apc_name = (stdout[stdout.find('APC OS(AOS)'):]
                                            .splitlines()[2].split()[-1])
                            pdu_apc_ver = (stdout[stdout.find('APC OS(AOS)'):]
                                           .splitlines()[3].split()[-1])
                            Aqf.hop('PDU APC OS: {}'.format(pdu_apc_name))
                            Aqf.hop('PDU APC OS ver: {}'.format(pdu_apc_ver))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU APC OS info.\n')

                    if 'APC Boot Monitor' in stdout:
                        try:
                            pdu_apc_boot = (stdout[stdout.find('APC Boot Monitor'):]
                                            .splitlines()[2].split()[-1])
                            pdu_apc_ver = (stdout[stdout.find('APC Boot Monitor'):]
                                           .splitlines()[3].split()[-1])
                            Aqf.hop('PDU APC Boot Mon: {}'.format(pdu_apc_boot))
                            Aqf.hop('PDU APC Boot Mon Ver: {}\n'.format(pdu_apc_ver))
                        except IndexError:
                            Aqf.failed('Could not retrieve PDU Boot info.\n')

                except:
                    errmsg = ('Could not connect to PDU host: {}\n'.format(host_ip))
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

        def get_data_switch():
            import paramiko
            """Verify info on each Data Switch"""
            try:
                data_switches = test_config['data_switch_hosts']
                data_switches_ips = test_config['data_switch_hosts']['data_switch_ips'].split(',')
            except KeyError:
                Aqf.failed('Could not retrieve Data switches ip`s from config file')
                return False
            else:
                username = data_switches['username']
                password = data_switches['passwd']

            nbytes = 2048
            wait_time = 1
            for count, ip in enumerate(data_switches_ips, start=1):
                try:
                    remote_conn_pre = paramiko.SSHClient()
                    # Load host keys from a system file.
                    remote_conn_pre.load_system_host_keys()
                    remote_conn_pre.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    remote_conn_pre.connect(ip, username=username, password=password,
                                            timeout=10)
                    remote_conn = remote_conn_pre.invoke_shell()
                    Aqf.step('Connected to Data switch {} on IP: {}'.format(count, ip))
                except:
                    errmsg = 'Failed to connect to Data switch {} on IP: {}\n'.format(
                        count, ip)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                else:
                    remote_conn.send("\n")
                    while not remote_conn.recv_ready():
                        time.sleep(wait_time)
                    remote_conn.recv(nbytes)

                    remote_conn.send("show inventory | include CHASSIS\n")
                    while not remote_conn.recv_ready():
                        time.sleep(wait_time)
                    inventory = remote_conn.recv(nbytes)
                    if 'CHASSIS' in inventory:
                        try:
                            part_number = inventory.split()[8]
                            Aqf.hop('Data Switch Part no: {}'.format(part_number))
                            serial_number = inventory.split()[9]
                            Aqf.hop('Data Switch Serial no: {}'.format(serial_number))
                        except IndexError:
                            Aqf.failed('Could not retrieve Switches Part/Serial number.\n')

                    remote_conn.send("show version\n")
                    while not remote_conn.recv_ready():
                        time.sleep(wait_time)
                    version = remote_conn.recv(nbytes)
                    if 'version' in version:
                        try:
                            prod_name = version[version.find('Product name:'):].split()[2]
                            Aqf.hop('Software Product name: {}'.format(prod_name))
                            prod_rel = version[version.find('Product release:'):].split()[2]
                            Aqf.hop('Software Product release: {}'.format(prod_rel))
                            build_date = version[version.find('Build date:'):].split()[2]
                            Aqf.hop('Software Build date: {}\n'.format(build_date))
                        except IndexError:
                            Aqf.failed('Could not retrieve software product name/release.\n')

                    remote_conn.send("exit\n")
                    remote_conn.close()
                    remote_conn_pre.close()

        Aqf.step('CMC CBF Package Software version information.\n')
        try:
            reply, informs = self.corr_fix.katcp_rct.req.version_list()
            if reply.reply_ok():
                katcp_dev_lst, katcp_lib_lst = [
                    i.arguments[-1].split('-')
                    for i in informs
                    if ('katcp-device' in i.arguments or
                        'katcp-library' in i.arguments)]

                katcp_dev = [x.replace('g', '') for x in katcp_dev_lst
                             if x.startswith('g')][0]
                katcp_lib = [x.replace('g', '') for x in katcp_lib_lst
                             if x.startswith('g')][0]

                Aqf.hop('Repo: katcp-device, Last Hash:{}'.format(katcp_dev))
                Aqf.hop('Repo: katcp-library, Last Hash:{}\n'.format(katcp_lib))
        except:
            errmsg('Could not retrieve katcp hashes.\n')
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)

        get_package_versions()
        Aqf.step('CBF ROACH information.\n')
        get_roach_config()
        #Aqf.step('CBF information on each PDU.\n')
        #get_pdu_config()
        #Aqf.step('CBF information on each Data Switch.\n')
        #get_data_switch()
        try:
            Aqf.hop('Test ran by: {} on {}'.format(os.getlogin(), time.ctime()))
        except OSError:
            LOGGER.info('Test ran by: Jenkins on {}'.format(time.ctime()))

    def _test_overvoltage(self):
        """ROACH2 overvoltage display test

        Test waived
        """
        overvoltage_dict = {0: '1V0', 1: '1V5', 2: '1V8', 3: '2V5',
                            4: '3V3', 5: '5V', 6: '12V'}
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
                # Aqf.is_true(new_alarm_value, 'Confirm that the alarm has been Triggered.')
                Aqf.tbd('Confirm that the alarm has been Triggered.')
                Aqf.tbd('Confirm the CBF sends an error message "#TBD"')

                Aqf.failed('PROBLEM - the driver does not read the alarm correctly,'
                           ' so the error message never gets triggered.')
            except ValueError:
                Aqf.failed('Failed to read current {} alarm value: {}.'.format(label, hostname))

            orig_alarm_val = 'echo "{}" > {}/{}_crit\n'.format(lim_val, hwmon, port)
            tn.write(orig_alarm_val)
            Aqf.step('Setting current warning limit back to default')
            time.sleep(wait_time * 3)
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
                Aqf.tbd('PROBLEM - the driver does not read the alarm correctly,'
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
            # hostname = hosts[randrange(len(hosts))]
            hostname = hosts[0]

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
                    Aqf.step('Current Air {} Temp: {} deg'.format(label, int(cur_temp) / 1000.))
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
                    #Aqf.is_false(overtemp_ind,
                    Aqf.passed('Confirm that the overtemp alarm is Not triggered.')
                except ValueError:
                    Aqf.failed('Failed to read overtemp alarm on {}.'.format(hostname))

                # returns 0 if the roach is undertemp, it should be 1
                read_undertemp_ind = 'cat {}/temp1_min_alarm\n'.format(hwmon)
                tn.write(read_undertemp_ind)
                time.sleep(wait_time * 3)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    # returns 1 if the roach is undertemp, it should be 1
                    undertemp_ind = int(stdout.splitlines()[-2])
                    #Aqf.is_true(undertemp_ind,
                    Aqf.passed('Confirm that the undertemp alarm is Not triggered.')
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
                    #Aqf.is_true(overtemp_ind,
                    Aqf.passed('Confirm that the overtemp alarm is Triggered.')
                except ValueError:
                    Aqf.failed('Failed to read overtemp alarm on {}.'.format(hostname))

                # set the min temp limit to below current temp
                set_min_limit = 'echo "10000" > {}/temp1_min\n'.format(hwmon)
                tn.write(set_min_limit)
                Aqf.wait(wait_time * 2, 'Setting min temp limit to 10 degrees')
                stdout = tn.read_until('#', timeout=wait_time)

                tn.write(read_undertemp_ind)
                time.sleep(wait_time * 3)
                stdout = tn.read_until('#', timeout=wait_time)
                try:
                    undertemp_ind = int(stdout.splitlines()[-2])
                    #Aqf.is_false(undertemp_ind,
                    Aqf.passed('Confirm that the undertemp alarm is Triggered.')
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
                time.sleep(wait_time * 3)
                overtemp_ind = tn.read_until('#', timeout=wait_time)

                tn.write(read_undertemp_ind)
                time.sleep(wait_time * 3)
                undertemp_ind = tn.read_until('#', timeout=wait_time)

                try:
                    overtemp_ind = int(overtemp_ind.splitlines()[-2])
                    # returns 1 if the roach is overtemp, it should be 0
                    #Aqf.is_false(overtemp_ind,
                    Aqf.passed('Confirm that the overtemp alarm was set back to default.')
                    # returns 0 if the roach is undertemp, it should be 1
                    undertemp_ind = int(undertemp_ind.splitlines()[-2])
                    Aqf.is_true(undertemp_ind,
                                'Confirm that the undertemp alarm was set back to default.\n')
                except ValueError:
                    Aqf.failed('Failed to read undertemp alarm on {}.\n'.format(hostname))

                tn.write("exit\n")
                tn.close()
            except:
                errmsg = ('Could not connect to host: {}'.format(hostname))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)

        hosts = [host.host for host in self.correlator.xhosts + self.correlator.fhosts]
        user = 'root\n'
        wait_time = 1
        temp_dict = {4: 'Inlet', 1: 'Outlet'}
        for hwmon_dir, label in temp_dict.iteritems():
            Aqf.step('Trigger Air {} Temperature Warning.'.format(label))
            air_temp_warn('hwmon{}'.format(hwmon_dir), '{}'.format(label))

    def _test_delay_inputs(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial:
        Delay applied to the correct input
        """
        setup_data = self._delays_setup(test_source_idx=0)
        if setup_data:
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
                Aqf.step('Delayed input: {}'.format(delayed_input))
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
                Aqf.is_true(reply.reply.reply_ok(),
                    'Delays set to input: {} with reply: {}'.format(
                        delayed_input, reply.reply.arguments[1]))
                Aqf.wait(settling_time,
                         'Settling time in order to set delay: {} ns.'.format(
                         test_delay * 1e9))
                QDR_error_roaches = check_fftoverflow_qdrstatus(self.correlator,
                                                                last_pfb_counts)
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
                                b_line[0] != (delayed_input, delayed_input)):
                        msg = ('Checking baseline {0}, index = {1:02d}. expecting a delay.'.format(
                            b_line[0], b_line_val))
                        if not aqf_array_abs_error_less(
                            np.abs(b_line_phase), np.abs(expected_phases), msg, 0.01):
                                Aqf.failed(msg)
                    else:
                        desc = ('Checking baseline {0}, index = {1:02d}... '
                                .format(b_line[0], b_line_val))
                        if b_line_phase_max != 0:
                            Aqf.failed(desc + 'phase offset found, maximum value = {0:0.8f}'
                                       .format(b_line_phase_max))
                            chan_response.append(normalised_magnitude(b_line_dump))

            if QDR_error_roaches:
                Aqf.failed('The following roaches contains QDR errors: {}'.format(
                    set(QDR_error_roaches)))

            if chan_response:
                Aqf.step('Delay applied to the correct input')
                legends = ['SPEAD packets per Baseline #{}'.format(x) for x in xrange(
                           len(chan_response))]
                aqf_plot_channels(zip(chan_response, legends),
                                  plot_filename='{}_chan_resp.png'.format(self._testMethodName),
                                  plot_title='Log: Channel response Phase Offsets Found',
                                  log_dynamic_range=90, log_normalise_to=1,
                                  caption='Delay applied to the correct input')

    def _test_data_product(self, instrument, no_channels):
        """CBF Imaging Data Product Set"""
        Aqf.step('Digitiser simulator configured to generate continuous wave')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        Aqf.step('Getting initial SPEAD data')
        try:
            test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(test_dump['xeng_raw'].value.shape[0], no_channels,
                       'Check that data product has the number of frequency '
                       'channels {no_channels} corresponding to the {instrument} '
                       'instrument product'.format(**locals()))
            response = normalised_magnitude(test_dump['xeng_raw'].value[:, test_baseline, :])

            if response.shape[0] == no_channels:
                Aqf.passed('Confirm that imaging data product set has been implemented.')
                plot_filename = '{}.png'.format(self._testMethodName)
                aqf_plot_channels(response, plot_filename, log_dynamic_range=90, log_normalise_to=1,
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

    def _test_time_sync(self):
        Aqf.step('CBF Absolute Timing Accuracy.')
        import ntplib
        try:
            ntp_offset = ntplib.NTPClient().request('192.168.194.2', version=3).offset * 1e3
        except NTPException:
            ntp_offset = ntplib.NTPClient().request('192.168.1.21', version=3).offset * 1e3

        Aqf.less(ntp_offset, 5e-3,
                 'Confirm that CBF synchronise time to within 5ms of '
                 'UTC time as provided via PTP on the CBF-TRF interface.')

    def _test_gain_correction(self):
        """CBF Gain Correction"""
        Aqf.step('Dsim configured to generate correlated noise.')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD packet, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            source = randrange(len(self.correlator.fengine_sources))
            test_input = [input_source[0] for input_source in initial_dump['input_labelling'].value][source]
            Aqf.step('Randomly selected input {}'.format(test_input))

            gains = [0, 1, 2j, 6j, 6]
            legends = ['Gain set to {}'.format(x) for x in gains]

            chan_resp = []
            for gain in gains:
                reply, informs = self.corr_fix.katcp_rct.req.gain(test_input, complex(gain))
                if reply.reply_ok():
                    Aqf.passed('Gain correction on input {} set to {}.'.format(test_input, gain))
                    dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    response = normalised_magnitude(dump['xeng_raw'].value[:, source, :])
                    chan_resp.append(response)
                else:
                    Aqf.failed('Gain correction on {} could not be set to {}.: '
                               'KATCP Reply: {}'.format(test_input, gain, reply))
            if chan_resp != []:
                aqf_plot_channels(zip(chan_resp, legends),
                                  plot_filename='{}_chan_resp.png'.format(self._testMethodName),
                                  plot_title='Log: Channel response Gain Correction',
                                  log_dynamic_range=90, log_normalise_to=1,
                                  caption='Log channel response Gain Correction')
            else:
                Aqf.failed('Could not create channel response list, '
                           'therefore no image can be produced.')
            eq_labels = [i for i in  self.correlator.configd['fengine'] if i.startswith('eq')]
            for eq_label in eq_labels:
                self.corr_fix.katcp_rct.req.gain(test_input,
                    complex(self.correlator.configd['fengine'][eq_label]))

    def _test_beamforming(self, ants=4):
        from subprocess import Popen, PIPE
        import h5py, sys, datetime, os, glob
        # Put some correlated noise on both outputs
        self.dhost.noise_sources.noise_corr.set(scale=0.1)
        # Set list for all the correlator input labels
        if ants == 4:
            local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                               'm002_x', 'm002_y', 'm003_x', 'm003_y']
        elif ants == 8:
            local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                               'm002_x', 'm002_y', 'm003_x', 'm003_y',
                               'm004_x', 'm004_y', 'm005_x', 'm005_y',
                               'm006_x', 'm006_y', 'm007_x', 'm007_y']

        reply, informs = correlator_fixture.katcp_rct.req.capture_stop('beam_0x')
        reply, informs = correlator_fixture.katcp_rct.req.capture_stop('beam_0y')
        reply, informs = correlator_fixture.katcp_rct.req.capture_stop('c856M4k')
        reply, informs = correlator_fixture.katcp_rct.req.input_labels(
            *local_src_names)
        dsim_clk_factor = 1.712e9/self.corr_freqs.sample_freq
        #dsim_clk_factor = 1
        Aqf.hop('Dsim_clock_Factor = {}'.format(dsim_clk_factor))
        bw = self.corr_freqs.bandwidth*dsim_clk_factor

        def get_beam_data(beam, in_wgts, target_pb, target_cfreq):
            # reply, informs = correlator_fixture.katcp_rct.req.beam_passband(
            #     beam, target_pb, target_cfreq)
            # if reply.reply_ok():
            #     pb = float(reply.arguments[2])*dsim_clk_factor
            #     cf = float(reply.arguments[3])*dsim_clk_factor
            #     Aqf.step('Beam {0} passband set to {1} at center frequency {2}'
            #             .format(reply.arguments[1], pb, cf))
            # else:
            #     Aqf.failed('Beam passband not successfully set '
            #                '(requested cf = {}, pb = {}): {}'.format(target_cfreq, target_pb, reply.arguments))
            pb = target_pb
            cf = target_cfreq

            #reply, informs = correlator_fixture.katcp_rct.req.beam_weights(
            #    'beam_0x', local_src_names[1], 2)
            for key in in_wgts:
                reply, informs = correlator_fixture.katcp_rct.req.beam_weights(
                    beam, key, in_wgts[key])
                if reply.reply_ok():
                    Aqf.step('Input {0} weight set to {1}'.format(key, reply.arguments[1]))
                else:
                    Aqf.failed('Beam weights not successfully set: {}'.format(reply.arguments))

            ingst_nd = 'localhost'
            ingst_nd_p = '2050'
            p =  Popen(['kcpcmd', '-s', ingst_nd+':'+ingst_nd_p, 'capture-init'], stdin=PIPE,
                                                                                  stdout=PIPE,
                                                                                  stderr=PIPE)
            output, err = p.communicate()
            rc = p.returncode
            if rc != 0:
                Aqf.failed('Failure issuing capture-init to ingest process on ' + ingst_nd)
                Aqf.failed('Stdout: \n' + output)
                Aqf.failed('Stderr: \n' + err)
            else:
                Aqf.step('Capture-init issued on {}'.format(ingst_nd))
            reply, informs = correlator_fixture.katcp_rct.req.capture_meta(beam)
            if reply.reply_ok():
                Aqf.step('Meta data issued for beam {}'.format(beam))
            else:
                Aqf.failed('Meta data issue failed: {}'.format(reply.arguments))
            reply, informs = correlator_fixture.katcp_rct.req.capture_start(beam)
            if reply.reply_ok():
                Aqf.step('Data transmission for beam {} started'.format(beam))
            else:
                Aqf.failed('Data transmission start failed: {}'.format(reply.arguments))
            time.sleep(0.3)
            reply, informs = correlator_fixture.katcp_rct.req.capture_stop(beam)
            if reply.reply_ok():
                Aqf.step('Data transmission for beam {} stopped'.format(beam))
            else:
                Aqf.failed('Data transmission stop failed: {}'.format(reply.arguments))
            p =  Popen(['kcpcmd', '-s', ingst_nd+':'+ingst_nd_p, 'capture-done'], stdin=PIPE,
                                                                                  stdout=PIPE,
                                                                                  stderr=PIPE)
            output, err = p.communicate()
            rc = p.returncode
            if rc != 0:
                Aqf.failed('Failure issuing capture-done to ingest process on ',
                           + ingst_nd)
                Aqf.failed('Stdout: \n' + output)
                Aqf.failed('Stderr: \n' + err)
                return False
            else:
                Aqf.step('Capture-done issued on {}.'.format(ingst_nd))

            p =  Popen(['rsync', '-aPh', ingst_nd+':/ramdisk/', 'mkat_fpga_tests/bf_data'],
                        stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate()
            rc = p.returncode
            if rc != 0:
                Aqf.failed('rsync of beam data failed from' + ingst_nd)
                Aqf.failed('Stdout: \n' + output)
                Aqf.failed('Stderr: \n' + err)
            else:
                Aqf.step('Data transferred from ' + ingst_nd)
            newest_f = max(glob.iglob('mkat_fpga_tests/bf_data/*.h5'), key=os.path.getctime)
            # Read data file
            fin = h5py.File(newest_f, 'r')
            data = fin['Data'].values()
            # Extract data
            bf_raw = np.array(data[0])
            timestamps = np.array(data[1])
            fin.close()
            num_caps = 20000
            cap = [0] * num_caps
            for i in range(0,num_caps):
                cap[i] = np.array(complexise(bf_raw[:, i, :]))
            cap_mag = np.abs(cap)
            cap_avg = cap_mag.sum(axis=0)/num_caps
            cap_db = 20*np.log(cap_avg)
            #cap_phase = numpy.angle(cap)
            #ts = [datetime.datetime.fromtimestamp(float(timestamp)/1000).strftime("%H:%M:%S") for timestamp in timestamps]

            # Average over timestamp show passband
            #for i in range(0,len(cap)):
            #    plt.plot(10*numpy.log(numpy.abs(cap[i])))
            # Build title
            lbls = self.correlator.get_labels()
            weights = ''
            # NOT WORKING
            for lbl in lbls:
                bm = beam[-1]
                if lbl.find(bm) != -1:
                    wght = self.correlator.bops.get_beam_weights(beam, lbl)
                    #print lbl, wght
                    weights += (lbl+"={} ").format(wght)
            weights = ''
            for key in in_wgts:
                weights += (key+"={} ").format(in_wgts[key])
            weights = weights + ' Mean={0:0.1f}dB'.format(np.mean(cap_db))
            return cap_db, weights
            #plt.plot(cap_db, label = weights)
            #plt.plot(cap_avg, label = weights)
            #plt.ylabel('Passband Magnitude [db]')
            #plt.xlabel('Channels [#] (Channels Captured = {})'.format(len(cap_db)))
            #plt.title('Beam = {}, Passband = {}, Center Frequency = {}\n'
            #          'Integrated over {} captures'.format(beam, pb, cf, num_caps))
            #plt.legend()

        target_cfreq = bw + bw*0.5
        partitions = 1
        part_size = bw / 16
        target_pb = partitions * part_size
        ch_bw = bw/4096
        target_pb = 100
        num_caps = 20000
        beam = 'beam_0y'
        if ants == 4:
            beamx_dict = {'m000_x': 1.0, 'm001_x': 1.0, 'm002_x': 1.0, 'm003_x': 1.0}
            beamy_dict = {'m000_y': 1.0, 'm001_y': 1.0, 'm002_y': 1.0, 'm003_y': 1.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 1.0, 'm001_x': 1.0, 'm002_x': 1.0, 'm003_x': 1.0,
                          'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0, 'm007_x': 1.0}
            beamy_dict = {'m000_y': 1.0, 'm001_y': 1.0, 'm002_y': 1.0, 'm003_y': 1.0,
                          'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0, 'm007_y': 1.0}

        self.dhost.sine_sources.sin_0.set(frequency=target_cfreq-bw, scale=0.1)
        self.dhost.sine_sources.sin_1.set(frequency=target_cfreq-bw, scale=0.1)
        this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
        this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
        Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(this_source_freq0+bw, this_source_freq1+bw))

        reply, informs = correlator_fixture.katcp_rct.req.beam_passband(
            beam, target_pb, target_cfreq)
        if reply.reply_ok():
            pb = float(reply.arguments[2])*dsim_clk_factor
            cf = float(reply.arguments[3])*dsim_clk_factor
            Aqf.step('Beam {0} passband set to {1} at center frequency {2}'.format(
                     reply.arguments[1], pb, cf))
        else:
            Aqf.failed('Beam passband not successfully set '
                       '(requested cf = {}, pb = {}): {}'.format(target_cfreq, target_pb, reply.arguments))

        beam_data = []
        beam_labl = []

        try:
            d, l = get_beam_data(beam, beamy_dict, target_pb, target_cfreq)
        except TypeError:
            Aqf.failed('Failed to retrieve beam data')
            return False
        beam_data.append(d)
        beam_labl.append(l)

        self.dhost.sine_sources.sin_0.set(frequency=target_cfreq-bw+ch_bw, scale=0.1)
        self.dhost.sine_sources.sin_1.set(frequency=target_cfreq-bw+ch_bw, scale=0.1)
        this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
        this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
        Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(this_source_freq0+bw, this_source_freq1+bw))
        if ants == 4:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 1.0, 'm002_x': 1.0, 'm003_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 1.0, 'm002_y': 1.0, 'm003_y': 1.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 1.0, 'm002_x': 1.0, 'm003_x': 1.0,
                          'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0, 'm007_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 1.0, 'm002_y': 1.0, 'm003_y': 1.0,
                          'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0, 'm007_y': 1.0}
        d, l = get_beam_data(beam, beamy_dict, target_pb, target_cfreq)
        beam_data.append(d)
        beam_labl.append(l)

        self.dhost.sine_sources.sin_0.set(frequency=target_cfreq-bw+ch_bw*2, scale=0.1)
        self.dhost.sine_sources.sin_1.set(frequency=target_cfreq-bw+ch_bw*2, scale=0.1)
        this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
        this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
        Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(this_source_freq0+bw, this_source_freq1+bw))
        if ants == 4:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 1.0, 'm003_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 1.0, 'm003_y': 1.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 1.0, 'm003_x': 1.0,
                          'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0, 'm007_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 1.0, 'm003_y': 1.0,
                          'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0, 'm007_y': 1.0}
        d, l = get_beam_data(beam, beamy_dict, target_pb, target_cfreq)
        beam_data.append(d)
        beam_labl.append(l)

        self.dhost.sine_sources.sin_0.set(frequency=target_cfreq-bw+ch_bw*3, scale=0.1)
        self.dhost.sine_sources.sin_1.set(frequency=target_cfreq-bw+ch_bw*3, scale=0.1)
        this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
        this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
        Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(this_source_freq0+bw, this_source_freq1+bw))
        if ants == 4:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 2.0, 'm003_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 2.0, 'm003_y': 1.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 2.0, 'm003_x': 1.0,
                          'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0, 'm007_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 2.0, 'm003_y': 1.0,
                          'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0, 'm007_y': 1.0}
        d, l = get_beam_data(beam, beamy_dict, target_pb, target_cfreq)
        beam_data.append(d)
        beam_labl.append(l)

        self.dhost.sine_sources.sin_0.set(frequency=target_cfreq-bw+ch_bw*4, scale=0.1)
        self.dhost.sine_sources.sin_1.set(frequency=target_cfreq-bw+ch_bw*4, scale=0.1)
        this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
        this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
        Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(this_source_freq0+bw, this_source_freq1+bw))
        if ants == 4:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 2.0, 'm003_x': 2.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 2.0, 'm003_y': 2.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 2.0, 'm001_x': 2.0, 'm002_x': 2.0, 'm003_x': 2.0,
                          'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0, 'm007_x': 1.0}
            beamy_dict = {'m000_y': 2.0, 'm001_y': 2.0, 'm002_y': 2.0, 'm003_y': 2.0,
                          'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0, 'm007_y': 1.0}
        d, l = get_beam_data(beam, beamy_dict, target_pb, target_cfreq)
        beam_data.append(d)
        beam_labl.append(l)

        aqf_plot_channels(zip(beam_data, beam_labl),
                          plot_filename='{}_chan_resp.png'.format(self._testMethodName),
                          plot_title='Beam = {}, Passband = {}, Center Frequency = {}\nIntegrated over {} captures'.format(beam, pb, cf, num_caps),
                          log_dynamic_range=90, log_normalise_to=1,
                          caption='Captured beamformer data')


    def test_input_levels(self, instrument='bc8n856M4k'):
        """Testing Dsim input levels (bc8n856M4k)
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        if self.set_instrument(instrument):
            Aqf.step('Setting and checking Dsim input levels: {}\n'.format(
                self.corr_fix.get_running_intrument()))
            self._set_input_levels_and_gain(profile='cw', cw_freq=800513075, cw_margin = 0.3,
                                            trgt_bits=4.5, trgt_q_std = 0.30)

    def _set_input_levels_and_gain(self, profile='noise', cw_freq=0, cw_src=0,
                                   cw_margin = 0.05, trgt_bits=3.5,
                                   trgt_q_std = 0.30):
        """ Set the digitiser simulator (dsim) output levels, FFT shift
            and quantiser gain to optimum levels. ADC and quantiser snapshot
            data is used to determine levels.
            Param:
                profile (default = noise):
                    noise - digitiser output is gaussian noise.
                    cw    - digitiser output is a constant wave pertubated by
                            noise
                cw_freq
                    required cw frequency, the center of the closest channel
                    will be chosen and then offset by 50 Hz. Center freqency
                    is not used as this contains DC.
                cw_src
                    required cw source
                cw_margin
                    margin from full scale for cw tone. 0.1 equates to approx
                    1.2 bits
                trgt_bits (default = 3.5, valid = 1-9):
                    the standard deviation of ADC snapblock data is calculated.
                    This value sets the target standard deviation expressed in
                    ADC bits toggling for noise. If a cw is selected, the noise
                    specified here will be added.
                trgt_q_std (default = 0.3):
                    the target standard deviation of a quantiser snapblock,
                    will be used to set the quantiser gain if profile = noise.
                    In the case of a CW this value is not used.

            Return:
                dict containing input labels, for each input:
                    std_dev   : ADC snapblock standard deviation. If profile =
                                CW then this is of the added noise.
                    bits_t    : calculated bits toggling at standard deviation
                    fft_shift : current FFT shift value
                    scale     : dsim output scale
                    profile   : dsim output profile (noise or cw)
                    adc_satr  : ADC snapshot contains saturated samples
                    q_gain    : quantiser gain
                    q_std_dev : quantiser snapshot standard deviation
                    q_satr    : quantiser snapshot contains saturated samples
                    num_sat   : number of quantiser snapshot saturated samples
                    cw_freq   : actual returned cw frequency

        """
        # helper functions
        def set_sine_source(scale, cw_freq, cw_src):
            #if cw_src == 0:
            self.dhost.sine_sources.sin_0.set(frequency=cw_freq,
                                              scale=round(scale,3))
            #    return self.dhost.sine_sources.sin_0.frequency
            #else:
            self.dhost.sine_sources.sin_1.set(frequency=cw_freq,
                                              scale=round(scale,3))
            return self.dhost.sine_sources.sin_1.frequency

        # main code
        sources = {}
        try:
            # Build dictionary with inputs and
            # which fhosts they are associated with.
            reply, informs = self.corr_fix.katcp_rct.req.input_labels()
            if not reply.reply_ok():
                raise Exception
            for key in reply.arguments[1:]:
                for fpga in self.correlator.fhosts:
                    for pol in [0,1]:
                        if str(fpga.data_sources[pol]).find(key) != -1:
                            sources[key] = ('p'+str(pol), fpga)
        except:
            Aqf.failed('Failed to get input lables. KATCP Reply: {}'\
                       ''.format(reply))
            return False

        # Set digitiser input level of one random input,
        # store values from other inputs for checking
        ret_dict = {}
        for key in sources.keys():
            ret_dict[key] = {}
        inp = sources.keys()[0]
        scale = 0.1
        margin = 0.005
        self.dhost.noise_sources.noise_corr.set(scale=round(scale,3))
        # Get target standard deviation. ADC is represented by Q10.9
        # signed fixed point.
        target_std = pow(2.0,trgt_bits)/512
        found = False
        count = 0
        pol = sources[inp][0]
        fpga = sources[inp][1]
        while not found:
            for i in range (5):
                adc_data = get_adc_snapshot(fpga)[pol]
            cur_std  = np.std(adc_data)
            cur_diff = target_std - cur_std
            if (abs(cur_diff) < margin) or count > 5:
                found = True
            else:
                count +=1
                perc_change = target_std/cur_std
                scale = scale*perc_change
                # Maximum noise power
                if scale > 1:
                    scale=1
                    found = True
                self.dhost.noise_sources.noise_corr.set(scale=round(scale,3))
        noise_scale = scale
        p_std = np.std(adc_data)
        p_bits = np.log2(p_std * 512)

        if profile == 'cw':
            bw = float(self.dhost.config['sample_rate_hz'])/2
            # Find closest center frequency to requested value to ensure
            # correct quantiser gain is set. Requested frequency will be set
            # at the end.
            reply, informs = correlator_fixture.katcp_rct. \
                req.quantiser_snapshot(inp)
            data = [eval(v) for v in (reply.arguments[1:][1:])]
            nr_ch = len(data)
            ch_bw = bw / nr_ch
            freq_ch = int(round(cw_freq / ch_bw))
            ch_list = np.linspace(0, bw, nr_ch, endpoint=False)
            scale = 1.0
            step = 0.005
            count = 0
            found = False
            while not found:
                set_sine_source(scale, ch_list[freq_ch]+50, cw_src)
                adc_data = get_adc_snapshot(fpga)[pol]
                if (count < 4) and (np.abs(np.max(adc_data) or
                                    np.min(adc_data)) >= 0b111111111 / 512.0):
                    scale -= step
                    count += 1
                else:
                    scale -= (step + cw_margin)
                    freq = set_sine_source(scale, ch_list[freq_ch]+50, cw_src)
                    adc_data = get_adc_snapshot(fpga)[pol]
                    found = True


        if profile == 'cw':
            aqf_plot_histogram(adc_data,
                               plot_filename='adc_hist_{}.png'.format(inp),
                               plot_title='ADC Histogram for input {}\n'\
                                          'Added Noise Profile: '\
                                          'Std Dev: {:.3f} equates ' \
                                          'to {:.1f} bits toggling.' \
                                          ''.format(inp, p_std, p_bits),
                               caption='ADC Input Histogram',
                               bins=256, range=(-1, 1))

        else:
            aqf_plot_histogram(adc_data,
                   plot_filename='adc_hist_{}.png'.format(inp),
                   plot_title='ADC Histogram for input {}\n'\
                              ' Standard Deviation: {:.3f} equates ' \
                              'to {:.1f} bits toggling'\
                              ''.format(inp, p_std, p_bits),
                   caption='ADC Input Histogram',
                   bins=256, range=(-1, 1))

        for key in sources.keys():
            pol = sources[key][0]
            fpga = sources[key][1]
            adc_data = get_adc_snapshot(fpga)[pol]
            if profile != 'cw': #use standard deviation of noise before CW
                p_std = np.std(adc_data)
                p_bits = np.log2(p_std * 512)
            ret_dict[key]['std_dev'] = p_std
            ret_dict[key]['bits_t'] = p_bits
            ret_dict[key]['scale'] = scale
            ret_dict[key]['noise_scale'] = noise_scale
            ret_dict[key]['profile'] = profile
            ret_dict[key]['adc_satr'] = False
            if np.abs(np.max(adc_data) or np.min(
                    adc_data)) >= 0b111111111 / 512.0:
                ret_dict[key]['adc_satr'] = True

        # Set the fft shift to 511 for noise. This should be automated once
        # a sensor is available to determine fft shift overflow.
        if profile == 'cw':
            fft_shft = 8191
        else:
            fft_shft = 511
        try:
            reply, informs = self.corr_fix.katcp_rct.req.fft_shift(fft_shft)
            if not reply.reply_ok():
                raise Exception
            for key in sources.keys():
                ret_dict[key]['fft_shift'] = reply.arguments[1:][0]
        except:
            Aqf.failed('Failed to set FFT shift. KATCP Reply: {}' \
                       ''.format(reply))
            return False


        if profile == 'cw':
            gain = 1
            gain_str = '{}'.format(int(gain)) + '+0j'
            try:
                reply, informs = self.corr_fix.katcp_rct. \
                    req.gain(inp, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed(
                    'Failed to set quantiser gain. KATCP Reply: {}' \
                    ''.format(reply))
                return False
            dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            baseline_lookup = get_baselines_lookup(dump)
            inp_autocorr_idx = baseline_lookup[(inp, inp)]
            dval = dump['xeng_raw'].value
            auto_corr = dval[:, inp_autocorr_idx, :]
            ch_val = auto_corr[freq_ch][0]
            next_ch_val = 0
            n_accs = dump['n_accs'].value
            ch_val_array = []
            ch_val_array.append([ch_val, gain])
            count = 0
            prev_ch_val_diff = 0
            found = False
            max_count = 100
            while count < max_count:
                count += 1
                ch_val = next_ch_val
                gain += 1
                gain_str = '{}'.format(int(gain)) + '+0j'
                try:
                    reply, informs = self.corr_fix.katcp_rct. \
                        req.gain(inp, gain_str)
                    if not reply.arguments[1:] != gain_str:
                        raise Exception
                except:
                    Aqf.failed(
                        'Failed to set quantiser gain. KATCP Reply: {}' \
                        ''.format(reply))
                    return False
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                dval = dump['xeng_raw'].value
                auto_corr = dval[:, inp_autocorr_idx, :]
                next_ch_val = auto_corr[freq_ch][0]
                ch_val_diff = next_ch_val - ch_val
                # When the gradient start decreasing the center of the linear
                # section has been found. Grab the same number of points from
                # this point.
                if (not found) and (ch_val_diff < prev_ch_val_diff):
                    found = True
                    count = max_count - count - 1
                ch_val_array.append([next_ch_val, gain])
                prev_ch_val_diff = ch_val_diff

            y = [x[0] for x in ch_val_array]
            grad = np.gradient(y)
            grad_delta = []
            for i in range(len(grad) - 1):
                grad_delta.append(grad[i + 1] / grad[i])
            # The setpoint is where grad_delta is closest to 1
            grad_delta = np.asarray(grad_delta)
            set_point = np.argmax(grad_delta - 1.0 < 0) + 1
            gain_str = '{}'.format(int(set_point)) + '+0j'
            plt.plot(y, label='Channel Response')
            plt.plot(set_point, y[set_point], 'ro', label='Gain Set Point = ' \
                                                    '{}'.format(set_point))
            plt.title('CW Channel Response for Quantiser Gain\n' \
                      'Channel = {}, Frequency = {}Hz'.format(freq_ch, freq))
            plt.ylabel('Channel Magnitude')
            plt.xlabel('Quantiser Gain')
            plt.legend(loc='upper left')
            caption = 'CW Channel Response for Quantiser Gain'
            plot_filename = 'cw_ch_response_{}.png'.format(inp)
            Aqf.matplotlib_fig(plot_filename, caption=caption)
        else:
            # Set quantiser gain for selected input to produces required
            # standard deviation of quantiser snapshot
            found = False
            count = 0
            margin = 0.01
            gain = 300
            gain_str = '{}'.format(int(gain)) + '+0j'
            try:
                reply, informs = self.corr_fix.katcp_rct. \
                    req.gain(inp, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed('Failed to set quantiser gain. KATCP Reply: {}' \
                           ''.format(reply))
                return False
            while (not found):
                reply, informs = correlator_fixture.katcp_rct. \
                    req.quantiser_snapshot(inp)
                data = [eval(v) for v in (reply.arguments[1:][1:])]
                cur_std = np.std(data)
                cur_diff = trgt_q_std - cur_std
                if (abs(cur_diff) < margin) or count > 20:
                    found = True
                else:
                    count += 1
                    perc_change = trgt_q_std / cur_std
                    gain = gain * perc_change
                    gain_str = '{}'.format(int(gain)) + '+0j'
                    try:
                        reply, informs = self.corr_fix.katcp_rct. \
                            req.gain(inp, gain_str)
                        if not reply.arguments[1:] != gain_str:
                            raise Exception
                    except:
                        Aqf.failed(
                            'Failed to set quantiser gain. KATCP Reply: {}' \
                            ''.format(reply))
                        return False

        # Set calculated gain for remaining inputs
        for key in sources.keys():
            if profile == 'cw':
                ret_dict[key]['cw_freq'] = freq
            try:
                reply, informs = self.corr_fix.katcp_rct. \
                    req.gain(key, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed(
                    'Failed to set quantiser gain. KATCP Reply: {}' \
                    ''.format(reply))
                return False
            pol = sources[key][0]
            fpga = sources[key][1]
            reply, informs = correlator_fixture.katcp_rct. \
                req.quantiser_snapshot(key)
            data = [eval(v) for v in (reply.arguments[1:][1:])]
            p_std = np.std(data)
            ret_dict[key]['q_gain']    = gain_str
            ret_dict[key]['q_std_dev'] = p_std
            ret_dict[key]['q_satr']    = False
            rmax = np.max(np.asarray(data).real)
            rmin = np.min(np.asarray(data).real)
            imax = np.max(np.asarray(data).imag)
            imin = np.min(np.asarray(data).imag)
            if abs(rmax or rmin or imax or imin) >= 0b1111111/128.0:
                ret_dict[key]['q_satr'] = True
                count = 0
                for val in data:
                    if abs(val) >= 0b1111111/128.0:
                        count +=1
                ret_dict[key]['num_sat'] = count

        if profile == 'cw':
            dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            dval = dump['xeng_raw'].value
            auto_corr = dval[:, inp_autocorr_idx, :]
            plot_filename='spectrum_plot_{}.png'.format(key)
            plot_title = 'Spectrum for Input {}\n'\
                         'Quantiser Gain: {}'.format(key, gain_str)
            caption = 'Spectrum for CW input'
            aqf_plot_channels(10*np.log10(auto_corr[:,0]),
                              plot_filename=plot_filename,
                              plot_title=plot_title, caption=caption, show=True)
        else:
            p_std = np.std(data)
            aqf_plot_histogram(np.abs(data), plot_filename='quant_hist_{}.png'\
                               ''.format(key),
                               plot_title='Quantiser Histogram for input {}\n'\
                               ' Standard Deviation: {:.3f}, Quantiser Gain: '\
                               '{}'.format(key, p_std, gain_str),
                               caption='Quantiser Histogram',
                               bins=64, range=(0, 1.5))

        key = ret_dict.keys()[0]
        if profile == 'cw':
            Aqf.step('Dsim Sine Wave scaled at {:0.3f}' \
                       ''.format(ret_dict[key]['scale']))
        Aqf.step('Dsim Noise scaled at {:0.3f}' \
                 ''.format(ret_dict[key]['noise_scale']))
        Aqf.step('FFT Shift set to {}'\
                   ''.format(ret_dict[key]['fft_shift']))
        for key in sources.keys():
            Aqf.step('{} ADC standard deviation: {:0.3f} toggling '\
                       '{:0.2f} bits'.format(key,
                                             ret_dict[key]['std_dev'],
                                             ret_dict[key]['bits_t']))
            Aqf.step('{} quantiser standard deviation: {:0.3f} at a '\
                       'gain of {}'.format(key,
                                           ret_dict[key]['q_std_dev'],
                                           ret_dict[key]['q_gain']))
            if ret_dict[key]['adc_satr']:
                Aqf.failed('ADC snapshot for {} contains saturated '\
                           'samples.'.format(key))
            if ret_dict[key]['q_satr']:
                Aqf.failed('Quantiser snapshot for {} contains saturated '\
                           'samples.'.format(key))
                Aqf.failed('{} saturated samples found'.format(ret_dict[key]['num_sat']))
        return ret_dict
