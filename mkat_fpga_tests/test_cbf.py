from __future__ import division

import Queue
import csv
import glob
import logging
import matplotlib
import operator
import os
import subprocess
import sys
import telnetlib
import textwrap
import time
import unittest
from collections import namedtuple
from random import randrange
from subprocess import Popen, PIPE

import corr2
import h5py
import katcp

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import ntplib
import numpy as np
import pandas
from concurrent.futures import TimeoutError
from corr2.corr_rx import CorrRx
from corr2.fxcorrelator_xengops import VaccSynchAttemptsMaxedOut
from katcp.testutils import start_thread_with_cleanup
from mkat_fpga_tests import correlator_fixture
from mkat_fpga_tests.aqf_utils import aqf_plot_channels, aqf_plot_and_save
from mkat_fpga_tests.aqf_utils import aqf_plot_phase_results
from mkat_fpga_tests.aqf_utils import cls_end_aqf, aqf_plot_histogram
from mkat_fpga_tests.utils import check_fftoverflow_qdrstatus, Style
from mkat_fpga_tests.utils import disable_warnings_messages, confirm_out_dest_ip
from mkat_fpga_tests.utils import get_and_restore_initial_eqs, get_set_bits, deprogram_hosts
from mkat_fpga_tests.utils import get_baselines_lookup, TestTimeout
from mkat_fpga_tests.utils import get_pfb_counts, check_host_okay, get_clean_dump, who_ran_test
from mkat_fpga_tests.utils import get_quant_snapshot, get_fftoverflow_qdrstatus
from mkat_fpga_tests.utils import ignored, clear_host_status, restore_src_names
from mkat_fpga_tests.utils import init_dsim_sources, CorrelatorFrequencyInfo
from mkat_fpga_tests.utils import nonzero_baselines, zero_baselines, all_nonzero_baselines
from mkat_fpga_tests.utils import normalised_magnitude, loggerise, complexise, human_readable_ip
from mkat_fpga_tests.utils import set_default_eq, clear_all_delays, set_input_levels
from mkat_fpga_tests.utils import get_local_src_names, get_input_labels
from nosekatreport import Aqf, aqf_vr
from nose.plugins.attrib import attr

LOGGER = logging.getLogger('mkat_fpga_tests')
# LOGGER = logging.getLogger(__name__)

DUMP_TIMEOUT = 60  # How long to wait for a correlator dump to arrive in tests

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

xeng_raw_bits_flags = namedtuple('FlagsBits', 'corruption overrange noise_diode')(34, 33, 32)

# NOTE TP.C.1.20 for AR1 maps to TP.C.1.46 for RTS

# TODO NM (2015-12-10) Use steal_docstring decorator form KATCP for sub-tests to re-use
# docstring from implementation. Perhaps do some trickery with string templates and parsing
# the mode name from the function to make an all singing all dancing decorator that does
# everything automagically?

disable_warnings_messages()
# protected member included in __all__
__all__ = ['correlator_fixture', '_test_config_file']


@cls_end_aqf
class test_CBF(unittest.TestCase):
    """ Unit-testing class for mkat_fpga_tests"""

    def setUp(self):
        self.corr_fix = correlator_fixture
        self.conf_file = self.corr_fix._test_config_file
        try:
            _conf = self.conf_file['inst_param']
        except (ValueError, TypeError):
            LOGGER.error('Failed to read test config file.')
        else:
            self.corr_fix.instrument = _conf['default_instrument']
            self.corr_fix.array_name = _conf['subarray']
            self.corr_fix.resource_clt = _conf['katcp_client']
        self.dhost = self.corr_fix.dhost
        try:
            self.dhost.get_system_information()
        except Exception:
            errmsg = ('Failed to connect to retrieve information from Digitiser Simulator.')
            LOGGER.exception(errmsg)
            sys.exit(errmsg)
        self.receiver = None

    def set_instrument(self, instrument, acc_time=0.5):
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)

        if self.receiver:
            self.receiver.stop()
            self.receiver = None
        acc_timeout = 60
        instrument_state = self.corr_fix.ensure_instrument(instrument)
        if not instrument_state:
            errmsg = (
                'Could not initialise instrument or ensure running instrument: {}'.format(
                    instrument))
            return {False: errmsg}
        try:
            reply = self.corr_fix.katcp_rct.req.accumulation_length(
                acc_time, timeout=acc_timeout)
            self.assertIsInstance(reply, katcp.resource.KATCPReply)

        except (TimeoutError, VaccSynchAttemptsMaxedOut):
            self.corr_fix.halt_array()
            errmsg = ('Timed-Out: Failed to set accumulation time within {}s, '
                      '[CBF-REQ-0064] SubArray will be halted and restarted with next '
                      'test'.format(acc_timeout))
            return {False: errmsg}

        except (AttributeError, AssertionError):
            self.corr_fix.halt_array()
            errmsg = (
                'Failed to set accumulation time within {}s, due to katcp request '
                'errors. [CBF-REQ-0064] SubArray will be halted and restarted with '
                'next test'.format(acc_timeout))
            return {False: errmsg}

        else:
            if not reply.succeeded:
                return {False: str(reply)}
            else:
                Aqf.step('[CBF-REQ-0071, 0096, 0089] Accumulation time set via CAM interface: '
                         '{0:.3f}s\n'.format((float(reply.reply.arguments[-1]))))
                try:
                    corrRx_port = int(self.conf_file['inst_param']['corr_rx_port'])
                except (ValueError, IOError, TypeError):
                    corrRx_port = 8888
                    LOGGER.info('Failed to retrieve corr rx port from config file.'
                                'Setting it to default port: %s' % (corrRx_port))
                try:
                    assert instrument.upper().endswith('M4K')
                    self.receiver = CorrRx(port=corrRx_port, queue_size=1000)
                except AssertionError:
                    self.receiver = CorrRx(port=corrRx_port, queue_size=100)

                try:
                    self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                except AssertionError:
                    errmsg = 'Correlator Receiver could not be instantiated.'
                    return {False: errmsg}
                else:
                    start_thread_with_cleanup(self, self.receiver, start_timeout=1)
                    self.correlator = self.corr_fix.correlator
                    try:
                        self.assertIsInstance(self.correlator, corr2.fxcorrelator.FxCorrelator)
                    except AssertionError, e:
                        errmsg = 'Failed to instantiate a correlator object: %s' % str(e)
                        return {False: errmsg}
                    else:
                        self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
                        subscribe_multicast = self.corr_fix.subscribe_multicast()
                        if subscribe_multicast.keys()[0]:
                            Aqf.step(subscribe_multicast.values()[0])
                        self.corr_fix.start_x_data()
                        self.addCleanup(self.corr_fix.stop_x_data)
                        self.corr_fix.issue_metadata()
                        try:
                            sync_time = self.corr_fix.katcp_rct.sensor.synchronisation_epoch.get_value()
                            self.correlator.set_synch_time(sync_time)
                        except:
                            errmsg = 'Sync time could not be read and/or set via CAM interface.'
                            LOGGER.error(errmsg)
                            return {False: errmsg}
                        else:
                            return {True: None}

    @aqf_vr('TP.C.1.19')
    def test_bc8n856M4k_channelisation(self, instrument='bc8n856M4k'):
        """
        CBF Channelisation Wideband Coarse L-band (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0043
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            n_chans = self.corr_freqs.n_chans
            test_chan = randrange(start=n_chans % 100, stop=n_chans - 1)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.19')
    def test_bc16n856M4k_channelisation(self, instrument='bc16n856M4k'):
        """
        CBF Channelisation Wideband Coarse L-band (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0043
            CBF-REQ-0053
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.19')
    def test_bc32n856M4k_channelisation(self, instrument='bc32n856M4k'):
        """
        CBF Channelisation Wideband Coarse L-band (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0043
        """
        instrument_success = self.set_instrument(instrument, acc_time=2)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            n_chans = self.corr_freqs.n_chans
            test_chan = randrange(start=n_chans % 100, stop=n_chans - 1)
            self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)

    @aqf_vr('TP.C.1.20')
    def test_bc8n856M32k_channelisation(self, instrument='bc8n856M32k'):
        """
        CBF Channelisation Wideband Fine L-band (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
            CBF-REQ-0053
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)

    @aqf_vr('TP.C.1.20')
    def test_bc16n856M32k_channelisation(self, instrument='bc16n856M32k'):
        """
        CBF Channelisation Wideband Fine L-band (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
            CBF-REQ-0053
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)

    @aqf_vr('TP.C.1.20')
    def test_bc32n856M32k_channelisation(self, instrument='bc32n856M32k'):
        """
        CBF Channelisation Wideband Fine L-band (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
            CBF-REQ-0053
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)

    @attr(speed='slow')
    @aqf_vr('TP.C.1.19')
    def test_bc8n856M4k_channelisation_sfdr_peaks(self, instrument='bc8n856M4k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz

    @attr(speed='slow')
    @aqf_vr('TP.C.1.19')
    def test_bc16n856M4k_channelisation_sfdr_peaks(self, instrument='bc16n856M4k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz

    @attr(speed='slow')
    @aqf_vr('TP.C.1.19')
    def test_bc32n856M4k_channelisation_sfdr_peaks(self, instrument='bc32n856M4k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz

    @attr(speed='slowwer')
    @aqf_vr('TP.C.1.20')
    def test_bc8n856M32k_channelisation_sfdr_peaks_slow(self, instrument='bc8n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Slow Test spurious free dynamic range for wideband fine (bc8n856M32k)
        # This is the slow version that sweeps through all 32768 channels.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow

        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz

    @attr(speed='slow')
    @aqf_vr('TP.C.1.20')
    def test_bc8n856M32k_channelisation_sfdr_peaks_fast(self, instrument='bc8n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Fast Test spurious free dynamic range for wideband fine (bc8n856M32k)

        # Check that the correct channels have the peak response to each
        # frequency and that no other channels have significant relative power.

        # This is the faster version that sweeps through 32768 channels
        # whilst stepping through `x`, where x is the step size given.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow

        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768, stepsize=8)  # Hz

    @attr(speed='slowwer')
    @aqf_vr('TP.C.1.20')
    def test_bc16n856M32k_channelisation_sfdr_peaks_slow(self, instrument='bc16n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Slow Test spurious free dynamic range for wideband fine
        # This is the slow version that sweeps through all 32768 channels.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow

        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz

    @attr(speed='slow')
    @aqf_vr('TP.C.1.20')
    def test_bc16n856M32k_channelisation_sfdr_peaks_fast(self, instrument='bc16n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Fast Test spurious free dynamic range for wideband fine (bc8n856M32k)

        # Check that the correct channels have the peak response to each
        # frequency and that no other channels have significant relative power.

        # This is the faster version that sweeps through 32768 channels
        # whilst stepping through `x`, where x is the step size given.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow

        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768, stepsize=8)  # Hz

    @attr(speed='slowwer')
    @aqf_vr('TP.C.1.20')
    def test_bc32n856M32k_channelisation_sfdr_peaks_slow(self, instrument='bc32n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Slow Test spurious free dynamic range for wideband fine (bc8n856M32k)
        # This is the slow version that sweeps through all 32768 channels.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz

    @attr(speed='slow')
    @aqf_vr('TP.C.1.20')
    def test_bc32n856M32k_channelisation_sfdr_peaks_fast(self, instrument='bc32n856M32k'):
        """
        CBF Channelisation Spurious Free Dynamic Range (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0126
            CBF-REQ-0047
            CBF-REQ-0046
            CBF-REQ-0053
            CBF-REQ-0050
            CBF-REQ-0049
        """
        # Fast Test spurious free dynamic range for wideband fine (bc32n856M32k)

        # Check that the correct channels have the peak response to each
        # frequency and that no other channels have significant relative power.

        # This is the faster version that sweeps through 32768 channels
        # whilst stepping through `x`, where x is the step size given.

        # _____________________________NOTE____________________________
        # Usage: Run nosetests with -e
        # Example: nosetests  -s -v --with-katreport --exclude=slow

        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768, stepsize=8)  # Hz

    @aqf_vr('TP.C.1.30')
    def test_bc8n856M4k_baseline_correlation_product(self, instrument='bc8n856M4k'):
        """
        CBF Baseline Correlation Products - AR1 (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    def test_bc16n856M4k_baseline_correlation_product(self, instrument='bc16n856M4k'):
        """
        CBF Baseline Correlation Products - AR1 (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    def test_bc32n856M4k_baseline_correlation_product(self, instrument='bc32n856M4k'):
        """
        CBF Baseline Correlation Products - AR1 (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.30')
    def test_bc8n856M32k_baseline_correlation_product(self, instrument='bc8n856M32k'):
        """
        CBF Baseline Correlation Products - AR1 (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.30')
    def test_bc16n856M32k_baseline_correlation_product(self, instrument='bc16n856M32k'):
        """
        CBF Baseline Correlation Products - AR1 (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.30')
    def test_bc32n856M32k_baseline_correlation_product(self, instrument='bc32n856M32k'):
        """
        CBF Baseline Correlation Products - AR1 (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0087
            CBF-REQ-0225
            CBF-REQ-0104
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_product_baselines()
            linelength = 100
            Aqf.addLine('-', linelength)
            self._test_back2back_consistency()
            Aqf.addLine('-', linelength)
            self._test_freq_scan_consistency()
            # Aqf.addLine('-', linelength)
            # self._test_restart_consistency(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M4k_delay_tracking(self, instrument='bc8n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M4k_delay_tracking(self, instrument='bc16n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M4k_delay_tracking(self, instrument='bc32n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M32k_delay_tracking(self, instrument='bc8n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M32k_delay_tracking(self, instrument='bc16n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M32k_delay_tracking(self, instrument='bc32n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_tracking()

    @aqf_vr('TP.C.1.31')
    def test_bc8n856M4k_accumulation_length(self, instrument='bc8n856M4k'):
        """
        Vector Accumulator Test (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.31')
    def test_bc16n856M4k_accumulation_length(self, instrument='bc16n856M4k'):
        """
        Vector Accumulator Test (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.31')
    def test_bc32n856M4k_accumulation_length(self, instrument='bc32n856M4k'):
        """
        Vector Accumulator Test (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            test_chan = randrange(self.corr_freqs.n_chans)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.31')
    def test_bc8n856M32k_accumulation_length(self, instrument='bc8n856M32k'):
        """
        Vector Accumulator Test (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            chan_index = 4096
            test_chan = randrange(chan_index)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan, chan_index)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.31')
    def test_bc16n856M32k_accumulation_length(self, instrument='bc16n856M32k'):
        """
        Vector Accumulator Test (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            chan_index = 4096
            test_chan = randrange(chan_index)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan, chan_index)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.31')
    def test_bc32n856M32k_accumulation_length(self, instrument='bc32n856M32k'):
        """
        Vector Accumulator Test (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0096
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            chan_index = 4096
            test_chan = randrange(chan_index)
            timeout_test = 1800
            try:
                with TestTimeout(timeout_test):
                    self._test_vacc(test_chan, chan_index)
            except TestTimeout.TestTimeoutError:
                errmsg = ('Could not be properly run the test, it timed-out after {} seconds.'.format(
                    timeout_test))
                Aqf.failed(errmsg)

    @aqf_vr('TP.C.1.40')
    def test_bc8n856M4k_product_switch(self, instrument='bc8n856M4k'):
        """
        CBF Data Product Switching Time (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc16n856M4k_product_switch(self, instrument='bc16n856M4k'):
        """
        CBF Data Product Switching Time (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc32n856M4k_product_switch(self, instrument='bc32n856M4k'):
        """
        CBF Data Product Switching Time (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.40')
    def test_bc8n856M32k_product_switch(self, instrument='bc8n856M32k'):
        """
        CBF Data Product Switching Time (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.40')
    def test_bc16n856M32k_product_switch(self, instrument='bc16n856M32k'):
        """
        CBF Data Product Switching Time (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument, acc_time=2.5)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.40')
    def test_bc32n856M32k_product_switch(self, instrument='bc32n856M32k'):
        """
        CBF Data Product Switching Time (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0013
        """
        instrument_success = self.set_instrument(instrument, acc_time=2.5)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_product_switch(instrument, no_channels=32768)

    # Flagging of data test waived as it not part of AR-1.
    @aqf_vr('TP.C.1.38')
    def _test_bc8n856M4k_overflow_flag(self, instrument='bc8n856M4k'):
        """CBF flagging of data"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def _test_bc16n856M4k_overflow_flag(self, instrument='bc16n856M4k'):
        """CBF flagging of data -- ADC overflow (bc16n856M4k)"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def _test_bc32n856M4k_overflow_flag(self, instrument='bc32n856M4k'):
        """CBF flagging of data -- ADC overflow (bc32n856M4k)"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def _test_bc8n856M32k_overflow_flag(self, instrument='bc8n856M32k'):
        """CBF flagging of data"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def _test_bc16n856M32k_overflow_flag(self, instrument='bc16n856M32k'):
        """CBF flagging of data"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.38')
    def _test_bc32n856M32k_overflow_flag(self, instrument='bc32n856M32k'):
        """CBF flagging of data"""
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n-- ADC overflow'])))
            self._test_adc_overflow_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--noise diode fired '])))
            self._test_noise_diode_flag()
            Aqf.step(Style.Bold(''.join([self._testMethodDoc, '\n--FFT overflow '])))
            self._test_fft_overflow_flag()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M4k_delay_rate(self, instrument='bc8n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc8n856M4k)
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M4k_delay_rate(self, instrument='bc16n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M4k_delay_rate(self, instrument='bc32n856M4k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M32k_delay_rate(self, instrument='bc8n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M32k_delay_rate(self, instrument='bc16n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M32k_delay_rate(self, instrument='bc32n856M32k'):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M4k_fringe_offset(self, instrument='bc8n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M4k_fringe_offset(self, instrument='bc16n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M4k_fringe_offset(self, instrument='bc32n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M32k_fringe_offset(self, instrument='bc8n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M32k_fringe_offset(self, instrument='bc16n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M32k_fringe_offset(self, instrument='bc32n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Offset (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_offset()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M4k_fringe_rate(self, instrument='bc8n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M4k_fringe_rate(self, instrument='bc16n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M4k_fringe_rate(self, instrument='bc32n856M4k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M32k_fringe_rate(self, instrument='bc8n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M32k_fringe_rate(self, instrument='bc16n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M32k_fringe_rate(self, instrument='bc32n856M32k'):
        """
        CBF per-antenna phase error -- Fringe Rate (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_fringe_rate()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def _test_bc8n856M4k_fringes_delays(self, instrument='bc8n856M4k'):
        """
        CBF per-antenna phase error
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate. (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_all_delays()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def _test_bc32n856M4k_fringes_delays(self, instrument='bc32n856M4k'):
        """
        CBF per-antenna phase error (bc32n856M4k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_all_delays()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def _test_bc16n856M4k_fringes_delays(self, instrument='bc16n856M4k'):
        """
        CBF per-antenna phase error (bc16n856M4k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_all_delays()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def _test_bc8n856M32k_fringes_delays(self, instrument='bc8n856M32k'):
        """
        CBF per-antenna phase error  (bc8n856M32k)
        -- Delays, Delay Rate, Fringe Offset and Fringe Rate.
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_all_delays()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M4k_delay_inputs(self, instrument='bc8n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc8n856M4k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M4k_delay_inputs(self, instrument='bc16n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc16n856M4k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M4k_delay_inputs(self, instrument='bc32n856M4k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc32n856M4k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc8n856M32k_delay_inputs(self, instrument='bc8n856M32k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc8n856M32k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc16n856M32k_delay_inputs(self, instrument='bc16n856M32k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc16n856M32k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.24')
    @aqf_vr('TP.C.1.49')
    @aqf_vr('TP.C.1.54')
    def test_bc32n856M32k_delay_inputs(self, instrument='bc32n856M32k'):
        """CBF Delay Compensation/LO Fringe stopping polynomial (bc32n856M32k)
           Delay applied to the correct input
        Test Verifies these requirements:
            CBF-REQ-0187
            CBF-REQ-0188
            CBF-REQ-0110
            CBF-REQ-0112
            CBF-REQ-0128
            CBF-REQ-0077
            CBF-REQ-0072
            CBF-REQ-0066
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_delay_inputs()

    @aqf_vr('TP.C.1.33')
    def test_bc8n856M4k_data_product(self, instrument='bc8n856M4k'):
        """
        CBF Imaging Data Product Set (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.33')
    @aqf_vr('TP.C.1.47')
    def test_bc16n856M4k_data_product(self, instrument='bc16n856M4k'):
        """
        CBF Imaging Data Product Set (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.33')
    @aqf_vr('TP.C.1.47')
    def test_bc32n856M4k_data_product(self, instrument='bc32n856M4k'):
        """
        CBF Imaging Data Product Set (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=4096)

    @aqf_vr('TP.C.1.33')
    @aqf_vr('TP.C.1.47')
    def test_bc8n856M32k_data_product(self, instrument='bc8n856M32k'):
        """
        CBF Imaging Data Product Set (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.33')
    @aqf_vr('TP.C.1.47')
    def test_bc16n856M32k_data_product(self, instrument='bc16n856M32k'):
        """
        CBF Imaging Data Product Set (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.33')
    @aqf_vr('TP.C.1.47')
    def test_bc32n856M32k_data_product(self, instrument='bc32n856M32k'):
        """
        CBF Imaging Data Product Set (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0120
            CBF-REQ-0213
            CBF-REQ-0223
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_data_product(instrument, no_channels=32768)

    @aqf_vr('TP.C.1.29')
    def test_bc8n856M4k_gain_correction(self, instrument='bc8n856M4k'):
        """CBF Gain Correction (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc16n856M4k_gain_correction(self, instrument='bc16n856M4k'):
        """CBF Gain Correction (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc32n856M4k_gain_correction(self, instrument='bc32n856M4k'):
        """
        CBF Gain Correction (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc8n856M32k_gain_correction(self, instrument='bc8n856M32k'):
        """
        CBF Gain Correction (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.49)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc16n856M32k_gain_correction(self, instrument='bc16n856M32k'):
        """
        CBF Gain Correction (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.29')
    def test_bc32n856M32k_gain_correction(self, instrument='bc32n856M32k'):
        """
        CBF Gain Correction (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0119
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_gain_correction()

    @aqf_vr('TP.C.1.37')
    @aqf_vr('TP.C.1.51')
    @aqf_vr('TP.C.1.35')
    def test_bc8n856M4k_beamforming(self, instrument='bc8n856M4k'):
        """
        CBF Beamformer functionality (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0117
            CBF-REQ-0094
            CBF-REQ-0118
            CBF-REQ-0123
            CBF-REQ-0092
            CBF-REQ-0183
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_beamforming()

    @aqf_vr('TP.C.1.37')
    @aqf_vr('TP.C.1.51')
    @aqf_vr('TP.C.1.35')
    def test_bc16n856M4k_beamforming(self, instrument='bc16n856M4k'):
        """
        CBF Beamformer functionality (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0117
            CBF-REQ-0094
            CBF-REQ-0118
            CBF-REQ-0123
            CBF-REQ-0092
            CBF-REQ-0183
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_beamforming()

    @aqf_vr('TP.C.1.37')
    @aqf_vr('TP.C.1.51')
    @aqf_vr('TP.C.1.35')
    def test_bc32n856M4k_beamforming(self, instrument='bc32n856M4k'):
        """
        CBF Beamformer functionality (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0117
            CBF-REQ-0094
            CBF-REQ-0118
            CBF-REQ-0123
            CBF-REQ-0092
            CBF-REQ-0183
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_beamforming()

    @aqf_vr('TP.C.1.48')
    def test_bc8n856M4k_bf_efficiency(self, instrument='bc8n856M4k'):
        """
        CBF Beamformer Efficiency
        Test Verifies these requirements:
            CBF-REQ-0124
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._bf_efficiency()

    @aqf_vr('TP.C.1.48')
    def test_bc16n856M4k_bf_efficiency(self, instrument='bc16n856M4k'):
        """
        CBF Beamformer Efficiency
        Test Verifies these requirements:
            CBF-REQ-0124
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._bf_efficiency()

    @aqf_vr('TP.C.1.48')
    def test_bc32n856M4k_bf_efficiency(self, instrument='bc32n856M4k'):
        """
        CBF Beamformer Efficiency
        Test Verifies these requirements:
            CBF-REQ-0124
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._bf_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc8n856M4k_corr_efficiency(self, instrument='bc8n856M4k'):
        """
        CBF L-Band Correlator Efficiency (bc8n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.05)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
        if self.set_instrument(instrument):
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._corr_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc16n856M4k_corr_efficiency(self, instrument='bc16n856M4k'):
        """
        CBF L-Band Correlator Efficiency (bc16n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.05)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._corr_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc32n856M4k_corr_efficiency(self, instrument='bc32n856M4k'):
        """
        CBF L-Band Correlator Efficiency (bc32n856M4k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.05)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._corr_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc8n856M32k_corr_efficiency(self, instrument='bc8n856M32k'):
        """
        CBF L-Band Correlator Efficiency (bc8n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.2)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._corr_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc16n856M32k_corr_efficiency(self, instrument='bc16n856M32k'):
        """
        CBF L-Band Correlator Efficiency (bc16n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=1.8)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._corr_efficiency()

    @attr(speed='slow')
    @aqf_vr('TP.C.1.23')
    def _test_bc32n856M32k_corr_efficiency(self, instrument='bc32n856M32k'):
        """
        CBF L-Band Correlator Efficiency (bc32n856M32k)
        Test Verifies these requirements:
            CBF-REQ-0127
        """
        instrument_success = self.set_instrument(instrument, acc_time=0.05)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._corr_efficiency()

    def _test_bc8n856M32k_input_levels(self, instrument='bc8n856M32k'):
        """
        Testing Digitiser simulator input levels (bc8n856M32k)
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            fft_shift = pow(2, 15) - 1
            self._set_input_levels_and_gain(profile='cw', cw_freq=200000000, cw_margin=0.6,
                                            trgt_bits=5, trgt_q_std=0.30, fft_shift=fft_shift)

    @aqf_vr('TP.C.1.41')
    @aqf_vr('TP.C.1.43')
    def test__generic_control_init(self, instrument='bc8n856M4k'):
        """
        CBF Control
        Test Varifies these requirements:
            CBF-REQ-0178
            CBF-REQ-0071
            CBF-REQ-0204
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument
        Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                     self._testMethodDoc])))
        if self.corr_fix.ensure_instrument(_running_inst):
            self.correlator = self.corr_fix.correlator
        else:
            self.set_instrument(instrument)
        self._systems_tests()
        self._test_control_init()
        who_ran_test()

    @aqf_vr('TP.C.1.17')
    def test__generic_config_report(self, instrument='bc8n856M4k'):
        """
        CBF Report Configuration
        Test Verifies these requirements:
            CBF-REQ-0060
            CBF-REQ-0178
            CBF-REQ-0204
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument
        Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                     self._testMethodDoc])))
        if self.set_instrument(_running_inst):
            self._systems_tests()
            self._test_config_report(verbose=False)
            who_ran_test()

    @aqf_vr('TP.C.1.18')
    @aqf_vr('TP.C.1.15')
    def test__generic_fault_detection(self, instrument='bc8n856M4k'):
        """
        CBF Fault Detection
        Test Verifies these requirements:
            CBF-REQ-0157
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument

        if self.set_instrument(_running_inst):
            self._systems_tests()
            msg = Style.Bold(
                '[CBF-REQ-0157] ROACH2 Temperature Fault Detection: {}\n'.format(_running_inst))
            Aqf.step(msg)
            self._test_overtemp()

            msg = Style.Bold(
                '[CBF-REQ-0157] QDR Memory Fault Detection: {}\n'.format(_running_inst))
            Aqf.step(msg)
            self._test_roach_qdr_sensors()

            msg = Style.Bold(
                '[CBF-REQ-0157] Link-Error: F-engine to X-engine: {}\n'.format(_running_inst))
            Aqf.step(msg)
            self._test_link_error()

            msg = Style.Bold(
                '[CBF-REQ-0157] PFB Fault Detection: {}\n'.format(_running_inst))
            Aqf.step(msg)
            self._test_roach_pfb_sensors()
        clear_host_status(self)
        who_ran_test()

    @aqf_vr('VR.C.14')
    @aqf_vr('TP.C.1.16')
    def test__generic_sensor_values(self, instrument='bc8n856M4k'):
        """
        CBF Report Sensor-values
        Test Verifies these requirements:
            CBF-REQ-0068
            CBF-REQ-0069
            CBF-REQ-0178
            CBF-REQ-0056
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument
        if self.set_instrument(_running_inst):
            self._systems_tests()
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._test_sensor_values()
            msg = Style.Bold('ROACH2 Sensor (Temp, Voltage, Current, Fan) Status: {}\n'.format(
                _running_inst))
            Aqf.step(msg)
            self._test_roach_sensors_status()
        clear_host_status(self)
        who_ran_test()

    @aqf_vr('TP.C.1.42')
    def test__generic_time_sync(self, instrument='bc8n856M4k'):
        """
        CBF Time synchronisation
        Test Verifies these requirements:
            CBF-REQ-0203
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument
        if self.set_instrument(_running_inst):
            self._systems_tests()
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._systems_tests()
            self._test_time_sync()

    @aqf_vr('TP.C.4.6')
    def test__generic_small_voltage_buffer(self, instrument='bc8n856M4k'):
        """
        CBF Voltage Buffer Data Product
        Test Verifies these requirements:
            CBF-REQ-0083
            CBF-REQ-0084
            CBF-REQ-0085
            CBF-REQ-0086
            CBF-REQ-0221
        """
        running_inst = self.corr_fix.get_running_instrument()
        if running_inst.values()[0]:
            _running_inst = running_inst.keys()[0]
        else:
            _running_inst = instrument
        if self.set_instrument(_running_inst, acc_time=0.99):
            self._systems_tests()
            Aqf.step(Style.Bold(''.join(['\n\tRunning instrument: {}\n\t'.format(_running_inst),
                                         self._testMethodDoc])))
            self._small_voltage_buffer()
            who_ran_test()


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

    def _systems_tests(self):
        """Checking system stability before and after use"""
        if not confirm_out_dest_ip(self):
            Aqf.failed('Output destination IP is not the same as the one stored in the register, '
                       'ie data is being spewed elsewhere.')

        set_default_eq(self)

        # ---------------------------------------------------------------
        def get_hosts_status(self, check_host_okay, list_sensor=None, engine_type=None, ):
            LOGGER.info('Retrieving PFB, LRU, QDR, PHY and reorder status on all Engines.')
            for _sensor in list_sensor:
                _status_hosts = check_host_okay(self, engine=engine_type, sensor=_sensor)
                if _status_hosts is not True:
                    for _status in _status_hosts:
                        Aqf.failed(_status)

        feng_sensors = ['pfb', 'phy', 'qdr', 'reorder']
        try:
            get_hosts_status(self, check_host_okay, feng_sensors, engine_type='feng')
        except:
            LOGGER.error('Failed to retrieve F-Eng status sensors')

        xeng_sensors = ['phy', 'qdr', 'reorder']
        try:
            get_hosts_status(self, check_host_okay, xeng_sensors, engine_type='xeng')
        except:
            LOGGER.error('Failed to retrieve X-Eng status sensors')
        # ---------------------------------------------------------------
        try:
            self.last_pfb_counts = get_pfb_counts(
                get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
        except AttributeError:
            LOGGER.error('Failed to read correlator attribute, correlator might not be running.')

    def get_flag_dumps(self, flag_enable_fn, flag_disable_fn, flag_description,
                       accumulation_time=1.):
        Aqf.step('Setting  accumulation time to {}.'.format(accumulation_time))
        max_vacc_sync_attempts = corr2.fxcorrelator_xengops.MAX_VACC_SYNCH_ATTEMPTS

        try:
            self.correlator.xops.set_acc_time(accumulation_time)
        except VaccSynchAttemptsMaxedOut:
            Aqf.failed('Failed to set accumulation time of {} after {} maximum vacc '
                       'sync attempts.'.format(accumulation_time, max_vacc_sync_attempts))
        else:
            Aqf.step('Getting SPEAD accumulation #1 before setting {}.'
                     .format(flag_description))
            try:
                dump1 = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
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
                Aqf.step('Getting SPEAD accumulation #2 after setting and clearing {}.'
                         .format(flag_description))
                dump2 = self.receiver.data_queue.get(DUMP_TIMEOUT)
                Aqf.step('Getting SPEAD accumulation #3.')
                dump3 = self.receiver.data_queue.get(DUMP_TIMEOUT)
                return (dump1, dump2, dump3)

    def _delays_setup(self, test_source_idx=2):
        # Put some correlated noise on both outputs
        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise with scale: {}, '
                 'gain: {} and fft shift: {}.'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, fft_shift=fft_shift,
                                            gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            reply_, _informs = self.corr_fix.katcp_rct.req.input_labels()
        except Exception:
            Aqf.failed('Failed to retrieve input labels via CAM interface')

        if reply_.reply_ok():
            Aqf.step('[CBF-REQ-0001, 0087, 0091, 0104]: Original source names changed from: {}'.format(
                Style.Underline(', '.join(reply_.arguments[1:]))))
        else:
            Aqf.failed('Could not retrieve original source names via CAM interface')

        local_src_names = get_local_src_names(self)
        self.addCleanup(restore_src_names, self)
        try:
            reply, _informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            if not reply.reply_ok():
                Aqf.failed('Could not retrieve new source names via CAM interface:\n {}'.format(str(
                    reply)))
                return False
        except Exception:
            Aqf.failed('Failed to rename input labels via CAM interface')
        else:
            source_names = reply.arguments[1:]
            Aqf.step('[CBF-REQ-0001, 0087, 0091, 0104]: Source names changed to: {}'.format(
                Style.Underline(', '.join(source_names))))
            # Get name for test_source_idx
            test_source = source_names[test_source_idx]
            ref_source = source_names[0]
            num_inputs = len(source_names)

            Aqf.step('[CBF-REQ-0110, 0066] Clearing all coarse and fine delays for all inputs.')
            clear_all_delays(self)
            self.addCleanup(clear_all_delays, self)

            if not self.corr_fix.issue_metadata():
                Aqf.failed('Could not issues new metadata')
            Aqf.step('Retrieving initial SPEAD accumulation.')
            self.corr_fix.issue_metadata()
            self.corr_fix.start_x_data()
            try:
                initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Exception:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch()
                    sync_time = float(reply.arguments[-1])
                    if not reply.reply_ok():
                        raise Exception
                except Exception:
                    Aqf.failed('Could not retrieve sync time via correlator object.')
                    return False

                scale_factor_timestamp = initial_dump['scale_factor_timestamp'].value
                time_stamp = initial_dump['timestamp'].value
                n_accs = initial_dump['n_accs'].value
                int_time = initial_dump['int_time'].value
                ticks_between_spectra = initial_dump['ticks_between_spectra'].value
                int_time_ticks = n_accs * ticks_between_spectra
                # roundtrip = self.corr_fix.katcp_rct.MAX_LOOP_LATENCY
                roundtrip = 0
                # Aqf.hop('Added {}s for the network round trip to dump timestamp'.format(
                #    roundtrip))
                # TODO:
                # This factor is due to an offset in the vacc sync. Must be removed
                # when fixed
                future_time = 0.001223325332135038767
                future_time = 0.02
                future_time = 0
                dump_1_timestamp = (sync_time + roundtrip +
                                    time_stamp / scale_factor_timestamp)
                t_apply = dump_1_timestamp + 10 * int_time + future_time

                t_apply_ticks = (t_apply - sync_time) * scale_factor_timestamp
                Aqf.hop('Time apply in board ticks: {:20f}'.format(t_apply_ticks))
                no_chans = range(self.corr_freqs.n_chans)
                Aqf.hop('Get list of all the baselines present in the correlator output')
                try:
                    baseline_lookup = get_baselines_lookup(initial_dump)
                    # Choose baseline for phase comparison
                    baseline_index = baseline_lookup[(ref_source, test_source)]
                except KeyError:
                    Aqf.failed('Initial SPEAD accumulation does not contain correct baseline '
                               'ordering format.')
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
                        'int_time_ticks': int_time_ticks,
                        'dump_1_timestamp': dump_1_timestamp,
                        't_apply': t_apply,
                        't_apply_ticks': t_apply_ticks,
                        'no_chans': no_chans,
                        'test_source': test_source,
                        'n_accs': n_accs,
                        'sample_period': self.corr_freqs.sample_period,
                        'num_inputs': num_inputs,
                        'test_source_ind': test_source_idx
                    }

    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients, max_wait_dumps=30):
        Aqf.step('Time apply to set delays: {}'.format(setup_data['t_apply']))
        cam_max_load_time = 1
        try:
            # Max time it takes to resync katcp (client connection)
            katcp_rsync_time = 0.5
            # Time it takes to execute the python command
            cmd_exec_time = 0.2
            # Max network latency
            network_roundtrip = self.corr_fix.katcp_rct.MAX_LOOP_LATENCY + katcp_rsync_time

            # katcp_host = self.corr_fix.katcp_rct.host
            # katcp_port = self.corr_fix.katcp_rct.port
            # cmd_start_time = time.time()
            # os.system("/usr/local/bin/kcpcmd -s {}:{} delays {} {}".format(katcp_host, katcp_port,
            # setup_data['t_apply'] + 5, ' '.join(delay_coefficients)))
            # final_cmd_time = time.time() - cmd_start_time

            ### TODO MM 2016-07-05
            ## Disabled katcp resource client setting delays, instead setting them
            ## via telnet kcs interface.
            katcp_conn_time = time.time()
            reply, _informs = self.corr_fix.katcp_rct.req.delays(setup_data['t_apply'],
                                                                 *delay_coefficients)
            cmd_end_time = time.time()
            actual_delay_coef = reply.arguments[1:]
            try:
                assert setup_data['num_inputs'] == len(actual_delay_coef)
            except:
                actual_delay_coef = None

            cmd_tot_time = katcp_conn_time + network_roundtrip + cmd_exec_time
            final_cmd_time = np.abs(cmd_end_time - cmd_tot_time)
        except:
            errmsg = ('Failed to set delays via CAM interface with loadtime: %s, '
                      'Delay coefficiencts: %s' % (setup_data['t_apply'], delay_coefficients))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            msg = ('[CBF-REQ-0077, 0187]: Time it takes to load delays is less than {}s with '
                   'integration time of {} seconds'.format(cam_max_load_time,
                                                           setup_data['int_time']))
            # Aqf.less(final_cmd_time, cam_max_load_time, msg)
            Aqf.passed(msg)
            msg = ('[CBF-REQ-0066, 0072]: Delays set via CAM interface reply : {}'.format(
                reply.arguments[1]))
            Aqf.is_true(reply.reply_ok(), msg)

        last_discard = setup_data['t_apply'] - setup_data['int_time']
        num_discards = 0
        fpga = self.correlator.xhosts[0]
        vacc_lsw = fpga.registers.vacc_time_lsw.read()
        vacc_msw = fpga.registers.vacc_time_msw.read()
        vacc_lsw = vacc_lsw['data']['lsw']
        vacc_msw = vacc_msw['data']['msw']
        vacc_sync = (vacc_msw << 32) + vacc_lsw
        fringe_dumps = []
        while True:
            num_discards += 1
            dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
            dump_ts = dump['timestamp'].value
            vacc_frac = (dump_ts - vacc_sync) / setup_data['int_time_ticks']
            Aqf.hop('Dump timestamp relative to vacc sync {:f}'.format(vacc_frac))
            dump_timestamp = (setup_data['sync_time'] + dump_ts /
                              setup_data['scale_factor_timestamp'])
            apply_dump_frac = ((setup_data['t_apply_ticks'] - dump['timestamp'].value) /
                               setup_data['int_time_ticks'])
            Aqf.hop('Apply time relative to dump: {:0.8f}'.format(apply_dump_frac))
            Aqf.hop('Dump timestamp in ticks: {:20d}'.format(dump['timestamp'].value))
            if (np.abs(dump_timestamp - last_discard) < 0.1 * setup_data['int_time']):
                Aqf.step('[CBF-REQ-0077]: Received final accumulation before fringe '
                         'application with timestamp {}.'.format(dump_timestamp))
                fringe_dumps.append(dump)
                break

            if num_discards > max_wait_dumps:
                Aqf.failed('Could not get accumulation with corrrect '
                           'timestamp within {} accumulation periods.'
                           .format(max_wait_dumps))
                break
            else:
                Aqf.step('Discarding accumulation with timestamp {}.'
                         .format(dump_timestamp))

        for i in xrange(dump_counts - 1):
            Aqf.step('Getting subsequent SPEAD accumulation {}.'.format(i + 1))
            dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
            fringe_dumps.append(dump)
            apply_dump_frac = ((setup_data['t_apply_ticks'] - dump['timestamp'].value) /
                               setup_data['int_time_ticks'])
            Aqf.hop('Apply time relative to dump: {:0.8f}'.format(apply_dump_frac))
            Aqf.hop('Dumps timestamp in ticks: {:20d}'.format(dump['timestamp'].value))

        chan_resp = []
        phases = []
        for acc in fringe_dumps:
            dval = acc['xeng_raw'].value
            freq_response = normalised_magnitude(
                dval[:, setup_data['baseline_index'], :])
            chan_resp.append(freq_response)

            data = complexise(dval[:, setup_data['baseline_index'], :])
            phases.append(np.angle(data))
            # amp = np.mean(np.abs(data)) / setup_data['n_accs']

        return zip(phases, chan_resp), actual_delay_coef

    def _get_expected_data(self, setup_data, dump_counts, delay_coefficients, actual_phases):

        def calc_actual_delay(setup_data):
            no_ch = self.corr_freqs.n_chans
            first_dump = np.unwrap(actual_phases[0])
            actual_slope = np.polyfit(xrange(0, no_ch), first_dump, 1)[0] * no_ch
            actual_delay = setup_data['sample_period'] * actual_slope / np.pi
            return actual_delay

        def gen_delay_vector(delay, setup_data):
            res = []
            no_ch = self.corr_freqs.n_chans
            delay_slope = np.pi * (delay / setup_data['sample_period'])
            c = delay_slope / 2
            for i in xrange(0, no_ch):
                m = i / float(no_ch)
                res.append(delay_slope * m - c)
            return res

        def gen_delay_data(delay, delay_rate, dump_counts, setup_data):
            expected_phases = []
            prev_delay_rate = 0
            for dump in xrange(0, dump_counts):
                # For delay rate the expected delay is the average of delays
                # applied during the integration. This is equal to the
                # delay delta over the integration divided by two
                max_delay_rate = dump * delay_rate
                avg_delay_rate = ((max_delay_rate - prev_delay_rate) / 2) + prev_delay_rate
                prev_delay_rate = max_delay_rate
                tot_delay = (delay + avg_delay_rate * setup_data['int_time'])
                expected_phases.append(gen_delay_vector(tot_delay, setup_data))
            return expected_phases

        def calc_actual_offset(setup_data):
            no_ch = self.corr_freqs.n_chans
            # mid_ch = no_ch / 2
            first_dump = actual_phases[0]
            # Determine average offset around 5 middle channels
            actual_offset = np.average(first_dump)  # [mid_ch-3:mid_ch+3])
            return actual_offset

        def gen_fringe_vector(offset, setup_data):
            return [offset] * self.corr_freqs.n_chans

        def gen_fringe_data(fringe_offset, fringe_rate, dump_counts, setup_data):
            expected_phases = []
            prev_fringe_rate = 0
            for dump in xrange(0, dump_counts):
                # For fringe rate the expected delay is the average of delays
                # applied during the integration. This is equal to the
                # delay delta over the integration divided by two
                max_fringe_rate = dump * fringe_rate
                avg_fringe_rate = ((max_fringe_rate - prev_fringe_rate) / 2) + prev_fringe_rate
                prev_fringe_rate = max_fringe_rate
                offset = -(fringe_offset + avg_fringe_rate * setup_data['int_time'])
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

        ant_idx = setup_data['test_source_ind']
        delay = ant_delay[ant_idx][0][0]
        delay_rate = ant_delay[ant_idx][0][1]
        fringe_offset = ant_delay[ant_idx][1][0]
        fringe_rate = ant_delay[ant_idx][1][1]

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

        Aqf.step('Randomly selected frequency channel to test: {}'.format(test_chan))
        requested_test_freqs = self.corr_freqs.calc_freq_samples(test_chan, samples_per_chan=101,
                                                                 chans_around=2)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # [CBF-REQ-0053]
        min_bandwithd_req = 770e6
        # [CBF-REQ-0126] CBF channel isolation
        cutoff = 53  # dB
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                 'with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}'.format(cw_scale,
                                                                                        awgn_scale,
                                                                                        gain,
                                                                                        fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=expected_fc, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(initial_dump['xeng_raw'].value.shape[0], no_channels,
                       'Capture an initial correlator SPEAD accumulation, '
                       'determine the number of frequency channels: {}'.format(
                           initial_dump['xeng_raw'].value.shape[0]))

            Aqf.is_true(initial_dump['bandwidth'].value >= min_bandwithd_req,
                        '[CBF-REQ-0053] Channelise total bandwidth {}Hz shall be >= {}Hz.'.format(
                            initial_dump['bandwidth'].value, min_bandwithd_req))

            chan_spacing = initial_dump['bandwidth'].value / initial_dump['xeng_raw'].value.shape[0]
            chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100),
                                chan_spacing + (chan_spacing * 1 / 100)]
            msg = ('[CBF-REQ-0043, 0046, 0053]: Verify that the calculated channel '
                   'frequency ({} Hz)step size is between {} and {} Hz'.format(chan_spacing,
                                                                               req_chan_spacing / 2,
                                                                               req_chan_spacing))
            Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)
            msg = ('[CBF-REQ-0047] Channelisation spacing is within maximum tolerance of 1% of the '
                   'channel spacing.')
            Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

        Aqf.step('Sweeping the digitiser simulator over the centre frequencies of at '
                 'least all the channels that fall within the complete L-band')
        print_counts = 3
        spead_failure_counter = 0

        for i, freq in enumerate(requested_test_freqs):
            if i < print_counts:
                Aqf.hop('Getting channel response for freq {0} @ {1}: {2:.3f} MHz.'.format(i + 1,
                                                                                           len(requested_test_freqs),
                                                                                           freq / 1e6))
            elif i == print_counts:
                Aqf.hop('.' * print_counts)
            elif i > (len(requested_test_freqs) - print_counts):
                Aqf.hop('Getting channel response for freq {0} @ {1}: {2:.3f} MHz.'.format(i + 1,
                                                                                           len(requested_test_freqs),
                                                                                           freq / 1e6))
            else:
                LOGGER.info('Getting channel response for freq %s @ %s: %s MHz.' % (i + 1,
                                                                                    len(requested_test_freqs),
                                                                                    freq / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency

            if this_source_freq == last_source_freq:
                LOGGER.info('Skipping channel response for freq %s @ %s: %s MHz.\n'
                            'Digitiser frequency is same as previous.' % (i + 1,
                                                                          len(requested_test_freqs), freq / 1e6))
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq

            try:
                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                spead_failure_counter += 1
                errmsg = ('Could not retrieve clean SPEAD accumulation, as # %s '
                          'Queue is Empty.' % spead_failure_counter)
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if spead_failure_counter > 5:
                    spead_failure_counter = 0
                    Aqf.failed('Bailed: Kept receiving empty SPEAD accumulations')
                    return False
            else:
                this_freq_data = this_freq_dump['xeng_raw'].value
                this_freq_response = normalised_magnitude(
                    this_freq_data[:, test_baseline, :])
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            # Plot an overall frequency response at the centre frequency just as
            # a sanity check

            if np.abs(freq - expected_fc) < 0.1:
                plt_filename = '{}_overall_channel_resolution.png'.format(self._testMethodName)
                plt_title = 'Overall frequency response at {} at {:.3f}MHz.'.format(
                    test_chan,
                    this_source_freq / 1e6)
                max_peak = np.max(loggerise(this_freq_response))
                new_cutoff = max_peak - cutoff
                y_axis_limits = (-100, 1)
                caption = ('An overrall frequency response at the center frequency, and ({:.3f}dB) '
                           'CBF channel isolation [max channel peak ({:.3f}dB) - ({}dB) cutoff] when '
                           'digitiser simulator is configured to generate a continuous wave, with '
                           'cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(
                    new_cutoff,
                    max_peak,
                    cutoff,
                    cw_scale,
                    awgn_scale,
                    gain,
                    fft_shift))
                aqf_plot_channels(this_freq_response, plt_filename, plt_title, caption=caption,
                                  hlines=new_cutoff, ylimits=y_axis_limits)

        # Test fft overflow and qdr status after
        Aqf.step('[CBF-REQ-0067] Check FFT overflow and QDR errors after channelisation.')
        check_fftoverflow_qdrstatus(self.correlator, self.last_pfb_counts)
        clear_host_status(self)
        self.corr_fix.stop_x_data()
        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        df = self.corr_freqs.delta_f

        plt_filename = '{}_Channel_Response.png'.format(self._testMethodName)
        plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90, normalise=True)
        plt_caption = ('Frequncy channel {} @ {}Mhz response vs source frequency'.format(test_chan,
                                                                                         expected_fc / 1e6))
        plt_title = 'Channel {0} @ {1:.3f}MHz response.'.format(test_chan, expected_fc / 1e6)
        # Plot channel response with -53dB cutoff horizontal line
        aqf_plot_and_save(freqs=actual_test_freqs, data=plot_data, df=df, expected_fc=expected_fc,
                          plot_filename=plt_filename, plt_title=plt_title, caption=plt_caption, cutoff=-cutoff)

        # Get responses for central 80% of channel
        df = self.corr_freqs.delta_f
        central_indices = (
            (actual_test_freqs <= expected_fc + 0.4 * df) &
            (actual_test_freqs >= expected_fc - 0.4 * df))
        central_chan_responses = chan_responses[central_indices]
        central_chan_test_freqs = actual_test_freqs[central_indices]

        # Plot channel response for central 80% of channel
        graph_name_central = '{}_central.png'.format(self._testMethodName)
        plot_data_central = loggerise(central_chan_responses[:, test_chan], dynamic_range=90,
                                      normalise=True)

        n_chans = self.corr_freqs.n_chans
        caption = ('Channel {} central response vs source frequency on max channels {}'.format(
            test_chan,
            n_chans))
        plt_title = 'Channel {0} @ {1:.3f} MHz response @ 80%'.format(test_chan, expected_fc / 1e6)

        aqf_plot_and_save(central_chan_test_freqs, plot_data_central, df, expected_fc,
                          graph_name_central, plt_title, caption=caption)

        Aqf.step('Test that the peak channeliser response to input frequencies in central 80% of '
                 'the test channel frequency band are all in the test channel')
        fault_freqs = []
        fault_channels = []
        for i, freq in enumerate(central_chan_test_freqs):
            max_chan = np.argmax(np.abs(central_chan_responses[i]))
            if max_chan != test_chan:
                fault_freqs.append(freq)
                fault_channels.append(max_chan)
        if fault_freqs:
            Aqf.failed('[CBF-REQ-0126] The following input frequencies (first and last): {!r} '
                       'respectively had peak channeliser responses in channels '
                       '{!r}\n, and not test channel {} as expected.'.format(fault_freqs[1::-1],
                                                                             set(sorted(fault_channels)), test_chan))

            LOGGER.error('The following input frequencies: %s respectively had '
                         'peak channeliser responses in channels %s, not '
                         'channel %s as expected.' % (fault_freqs, set(sorted(fault_channels)),
                                                      test_chan))

        Aqf.less(np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
                 'Check that VACC output is at < 99% of maximum value, if fails '
                 'then it is probably overranging.')

        max_central_chan_response = np.max(10 * np.log10(
            central_chan_responses[:, test_chan]))
        min_central_chan_response = np.min(10 * np.log10(
            central_chan_responses[:, test_chan]))
        chan_ripple = max_central_chan_response - min_central_chan_response
        acceptable_ripple_lt = 1.5
        Aqf.less(chan_ripple, acceptable_ripple_lt,
                 '[CBF-REQ-0126] Check that ripple within 80% of cutoff '
                 'frequency channel is < {} dB'.format(acceptable_ripple_lt))

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
        Aqf.step('[CBF-REQ-0047] Check that response at channel-edges are -3 dB '
                 'relative to the channel centre at {0:.3f} Hz, actual source freq '
                 '{1:.3f} Hz'.format(expected_fc, fc_src_freq))

        desired_cutoff_resp = -6  # dB
        acceptable_co_var = 0.1  # dB, TODO 2015-12-09 NM: thumbsuck number
        co_mid_rel_resp = 10 * np.log10(fc_resp)
        co_low_rel_resp = 10 * np.log10(co_low_resp)
        co_high_rel_resp = 10 * np.log10(co_high_resp)

        co_lo_band_edge_rel_resp = co_mid_rel_resp - co_low_rel_resp
        co_hi_band_edge_rel_resp = co_mid_rel_resp - co_high_rel_resp

        low_rel_resp_accept = np.abs(desired_cutoff_resp + acceptable_co_var)
        hi_rel_resp_accept = np.abs(desired_cutoff_resp - acceptable_co_var)

        cutoff_edge = np.abs((co_lo_band_edge_rel_resp + co_hi_band_edge_rel_resp) / 2)

        # Plot PFB channel response with -6dB cuttoff horizontal line
        # TODO MM 2016-10-04 hard coded center bins, should probably fix
        no_of_responses = 3
        center_bin = [150, 250, 350]
        y_axis_limits = (-90, 1)
        legends = ['Channel {} / Sample {} \n@ {:.3f} MHz'.format(((test_chan + i) - 1), v,
                                                                  self.corr_freqs.chan_freqs[test_chan + i] / 1e6)
                   for i, v in zip(range(no_of_responses), center_bin)]
        center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.corr_freqs.n_chans / 1e3))
        # center_bin.append('Channel spacing: {:.3f}kHz'.format(chan_spacing/1e3))

        channel_response_list = [chan_responses[:, test_chan + i - 1]
                                 for i in range(no_of_responses)]
        plot_title = 'PFB Channel Response'
        plot_filename = '{}_adjacent_channels.png'.format(self._testMethodName)

        caption = ('Sample PFB central channel response between channel {}, with '
                   'channelisation spacing of {:.3f}kHz within tolerance of 1%, with the digitiser '
                   'simulator configured to generate a continuous wave, with cw scale: {}, '
                   'awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(test_chan,
                                                                          chan_spacing / 1e3,
                                                                          cw_scale,
                                                                          awgn_scale,
                                                                          gain,
                                                                          fft_shift))

        aqf_plot_channels(zip(channel_response_list, legends), plot_filename, plot_title,
                          normalise=True, caption=caption, hlines=-cutoff_edge, vlines=center_bin,
                          xlabel='Sample Steps', ylimits=y_axis_limits)

        # Plot Central PFB channel response with ylimit 0 to -6dB
        y_axis_limits = (-7, 1)
        plot_filename = '{}_central_adjacent_channels.png'.format(self._testMethodName)
        plot_title = 'PFB Central Channel Response'
        caption = ('Sample PFB central channel response between channel {}, with the digitiser '
                   'simulator configured to generate a continuous wave, with cw scale: {}, '
                   'awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(test_chan,
                                                                          cw_scale,
                                                                          awgn_scale,
                                                                          gain,
                                                                          fft_shift))

        aqf_plot_channels(zip(channel_response_list, legends), plot_filename, plot_title,
                          normalise=True, caption=caption, xlabel='Sample Steps',
                          ylimits=y_axis_limits, )

        Aqf.is_true(low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
                    '[CBF-REQ-0126] Check that relative response at the low band-edge '
                    '({co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq '
                    '{co_low_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                    'relative to channel centre response.'.format(**locals()))

        Aqf.is_true(low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
                    '[CBF-REQ-0126] Check that relative response at the high band-edge '
                    '({co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq '
                    '{co_high_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                    'relative to channel centre response.'.format(**locals()))
        who_ran_test()

    def _test_sfdr_peaks(self, required_chan_spacing, no_channels, stepsize=None, cutoff=53):
        """Test channel spacing and out-of-channel response

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

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
        msg = ('Check that the correct channels have the peak response to each'
               'frequency and that no other channels have significant relative power')
        Aqf.step(msg)

        if stepsize:
            Aqf.step('Running FASTER version of Channelisation SFDR test '
                     'with {} step size.'.format(stepsize))
            print_counts = 4 * stepsize
        else:
            print_counts = 4

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                 'with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}'.format(cw_scale,
                                                                                        awgn_scale,
                                                                                        gain,
                                                                                        fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=self.corr_freqs.bandwidth / 2.0,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(initial_dump['xeng_raw'].value.shape[0], no_channels,
                       '[CBF-REQ-0053] Capture an initial correlator SPEAD accumulation, '
                       'determine the number of channels and processing bandwidth: '
                       '{}Hz.'.format(initial_dump['bandwidth'].value))

            chan_spacing = (initial_dump['bandwidth'].value /
                            initial_dump['xeng_raw'].value.shape[0])
            # [CBF-REQ-0043]
            calc_channel = ((required_chan_spacing / 2) <= chan_spacing <= required_chan_spacing)
            mag = ('[CBF-REQ-0043, 0046, 0053, 0198]: Verify that the calculated channel '
                   'frequency step size is between {} and {} Hz'.format(required_chan_spacing / 2,
                                                                        required_chan_spacing))
            Aqf.is_true(calc_channel, msg)

        Aqf.step('Sweeping the digitiser simulator over the all channels that fall '
                 'within the complete L-band.')
        spead_failure_counter = 0
        channel_response_lst = []
        for channel, channel_f0 in enumerate(
                self.corr_freqs.chan_freqs[start_chan::stepsize], start_chan):

            if stepsize:
                channel *= stepsize

            if channel < print_counts:
                Aqf.hop('Getting channel response for freq {0} @ {1}: {2:.3f} MHz.'.format(
                    channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            elif channel == print_counts:
                Aqf.hop('.' * 3)
            elif channel > (n_chans - print_counts):
                Aqf.hop('Getting channel response for freq {0} @ {1}: {2:.3f} MHz.'.format(
                    channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            else:
                LOGGER.info('Getting channel response for freq %s @ %s: %s MHz.' % (channel,
                                                                                    len(self.corr_freqs.chan_freqs),
                                                                                    channel_f0 / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            try:
                this_freq_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)['xeng_raw'].value
            except Queue.Empty:
                spead_failure_counter += 1
                errmsg = ('Could not retrieve clean SPEAD accumulation, as # %s Queue is'
                          ' Empty.' % (spead_failure_counter))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if spead_failure_counter > 5:
                    spead_failure_counter = 0
                    Aqf.failed('Bailed: Kept receiving empty SPEAD accumulations')
                    return False
            else:
                this_freq_response = (
                    normalised_magnitude(this_freq_data[:, test_baseline, :]))
                chans_to_plot = (n_chans // 10, n_chans // 2, 9 * n_chans // 10)
                if channel in chans_to_plot:
                    channel_response_lst.append(this_freq_response)

                max_chan = np.argmax(this_freq_response)
                max_channels.append(max_chan)
                # Find responses that are more than -cutoff relative to max

                new_cutoff = np.max(loggerise(this_freq_response)) + cutoff
                unwanted_cutoff = this_freq_response[max_chan] / 10 ** (new_cutoff / 10.)
                extra_responses = [i for i, resp in enumerate(this_freq_response)
                                   if i != max_chan and resp >= unwanted_cutoff]
                extra_peaks.append(extra_responses)

        for channel, channel_resp in zip(chans_to_plot, channel_response_lst):
            plt_filename = '{}_channel_{}_resp.png'.format(self._testMethodName, channel)

            test_freq_mega = self.corr_freqs.chan_freqs[channel] / 1e6
            plt_title = 'Frequency response at {0} @ {1:.3f} MHz'.format(channel, test_freq_mega)
            caption = ('An overrall frequency response at channel {0} @ {1:.3f}MHz, '
                       'when digitiser simulator is configured to generate a continuous wave, '
                       'with cw scale: {2}. awgn scale: {3}, eq gain: {4}, fft shift: {5}'.format(
                channel, test_freq_mega, cw_scale, awgn_scale, gain, fft_shift))

            new_cutoff = np.max(loggerise(channel_resp)) - cutoff
            aqf_plot_channels(channel_resp, plt_filename, plt_title, log_dynamic_range=90,
                              caption=caption, hlines=new_cutoff)

        channel_range = range(start_chan, len(max_channels) + start_chan)
        if max_channels == channel_range:
            Aqf.passed('[VR.C.20] Check that the correct channels have the peak '
                       'response to each frequency')
        elif stepsize is not None:
            diff = list(set(
                range(start_chan, self.corr_freqs.n_chans, stepsize)) - set(max_channels))
            if diff == []:
                Aqf.passed('[VR.C.20] Check that the correct channels have the peak '
                           'response to each frequency')
            else:
                Aqf.failed('[VR.C.20] Channel(s) {} does not have the correct peak '
                           'response to each frequency'.format(diff))
        else:
            msg = (
                '[VR.C.20] Check that the correct channels have the peak response to each frequency')
            Aqf.array_abs_error(max_channels, channel_range, msg, 1)

        msg = ("[CBF-REQ-0126] Check that no other channels response more than -{cutoff} dB".format(
            **locals()))
        if extra_peaks == [[]] * len(max_channels):
            Aqf.passed(msg)
        else:
            LOGGER.info('Expected: %s\n\nGot: %s' % (extra_peaks, [[]] * len(max_channels)))
            Aqf.failed(msg)
        who_ran_test()

    def _test_product_baselines(self):
        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise, '
                 'with awgn scale: {}, eq gain: {}, fft shift: {}'.format(awgn_scale, gain,
                                                                          fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            freq=self.corr_freqs.chan_freqs[1500],
                                            fft_shift=fft_shift, gain=gain)

        Aqf.step('Set list for all the correlator input labels as per config file')
        local_src_names = self.correlator.configd['fengine']['source_names'].split(',')
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)

        for i in range(2):
            self.corr_fix.issue_metadata()
            self.corr_fix.start_x_data()
        if not self.corr_fix.issue_metadata():
            Aqf.failed('Could not issue new metadata')

        Aqf.step('Capture an initial correlator SPEAD accumulation, and retrieve list '
                 'of all the correlator input labels from SPEAD accumulation.')

        try:
            test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Exception:
            errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        else:
            # Get bls ordering from get dump
            Aqf.step('[CBF-REQ-0001, 0087, 0091, 0104] Get list of all possible '
                     'baselines (including redundant baselines) present in the correlator '
                     'output from SPEAD accumulation')
            bls_ordering = test_dump['bls_ordering'].value
            input_labels = sorted(tuple(test_dump['input_labelling'].value[:, 0]))
            baselines_lookup = get_baselines_lookup(test_dump)
            present_baselines = sorted(baselines_lookup.keys())

            possible_baselines = set()
            [possible_baselines.add((li, lj)) for li in input_labels for lj in input_labels]

            test_bl = sorted(list(possible_baselines))
            Aqf.step('[CBF-REQ-0087] Check that each baseline (or its reverse-order '
                     'counterpart) is present in the correlator output')
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
                        '[CBF-REQ-0087] Check that all baselines are present in '
                        'correlator output.')
            test_data = test_dump['xeng_raw'].value
            Aqf.step('[CBF-REQ-0213] Expect all baselines and all channels to be '
                     'non-zero with Digitiser Simulator set to output AWGN.')
            Aqf.is_false(zero_baselines(test_data),
                         '[CBF-REQ-0213] Confirm that no baselines have all-zero visibilities.')

            msg = ('[CBF-REQ-0213] Confirm that all baseline visibilities are '
                   'non-zero across all channels')
            Aqf.equals(nonzero_baselines(test_data),
                       all_nonzero_baselines(test_data), msg)
            Aqf.step('Save initial f-engine equalisations, and ensure they are '
                     'restored at the end of the test')

            initial_equalisations = get_and_restore_initial_eqs(self, self.correlator)

            Aqf.step('Set all inputs gains to \'Zero\', and confirm that output product '
                     'is all-zero')

            try:
                for inp in input_labels:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(inp, 0)
                    if not reply.reply_ok():
                        raise Exception
            except Exception:
                Aqf.failed('Failed to set equalisations on all F-engines')
            else:
                Aqf.passed('All the inputs equalisations have been set to Zero.')

            try:
                eq_values = []
                for inp in input_labels:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(inp)
                    if not reply.reply_ok():
                        raise Exception
                    eq_values.append(reply.arguments[-1])
            except Exception:
                Aqf.failed('Failed to get equalisations on {}'.format(inp))
            else:
                all_eqs = list(set(eq_values))
                msg = 'Confirm that all the inputs equalisations have been set to \'Zero\'.'
                Aqf.equals(all_eqs, ['0j'], msg)

            def _retrieve_clean_dump(self):
                """Recursive SPEAD dump retrieval"""
                for i in xrange(5):
                    try:
                        test_data = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    except Exception:
                        pass
                    else:
                        return test_data['xeng_raw'].value

            test_data = _retrieve_clean_dump(self)
            msg = 'Confirm that all baseline visibilities are \'Zero\'.\n'
            Aqf.is_false(nonzero_baselines(test_data), msg)
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

            bls_msg = ('Stepping through input combinations, verifying for each that '
                       'the correct output appears in the correct baseline product')
            Aqf.step(bls_msg)
            dataFrame = pandas.DataFrame(index=sorted(input_labels),
                                         columns=list(sorted(present_baselines)))
            for count, inp in enumerate(input_labels, start=1):
                old_eq = complex(initial_equalisations[inp][0])
                reply, informs = self.corr_fix.katcp_rct.req.gain(inp, old_eq)
                msg = ('[CBF-REQ-0071] Gain correction on input {} set to {}.'.format(inp, old_eq))

                if reply.reply_ok():
                    Aqf.passed(msg)
                    zero_inputs.remove(inp)
                    nonzero_inputs.add(inp)
                    expected_z_bls, expected_nz_bls = (calc_zero_and_nonzero_baselines(
                        nonzero_inputs))
                    test_data = _retrieve_clean_dump(self)
                    # plot baseline channel response
                    plot_data = [normalised_magnitude(test_data[:, i, :])
                                 # plot_data = [loggerise(test_data[:, i, :])
                                 for i in plot_baseline_inds]
                    plot_filename = '{}_channel_resp_{}.png'.format(self._testMethodName, inp)

                    plot_title = ('Baseline Correlation Products on input: {}\n'
                                  'Bls channel response \'Non-Zero\' inputs:\n {}\n'
                                  '\'Zero\' inputs:\n {}'.format(inp,
                                                                 ' \n'.join(textwrap.wrap(', \n'.join(sorted(
                                                                     nonzero_inputs)))),
                                                                 ' \n'.join(textwrap.wrap(', \n'.join(sorted(
                                                                     zero_inputs))))))

                    caption = ('Baseline channel response on input:{}'
                               ' {} with the following non-zero inputs:\n {} \n and\n'
                               'zero inputs:\n {}'.format(inp, bls_msg, sorted(nonzero_inputs),
                                                          sorted(zero_inputs)))

                    aqf_plot_channels(zip(plot_data, plot_baseline_legends), plot_filename,
                                      plot_title, log_dynamic_range=90,
                                      log_normalise_to=1, caption=caption, ylimits=(-100, 0))

                    actual_nz_bls_indices = all_nonzero_baselines(test_data)
                    actual_nz_bls = set(tuple(bls_ordering[i]) for i in actual_nz_bls_indices)

                    actual_z_bls_indices = zero_baselines(test_data)
                    actual_z_bls = set(tuple(bls_ordering[i]) for i in actual_z_bls_indices)
                    msg = ('Check that expected baseline visibilities are nonzero with '
                           'non-zero inputs {} and,'.format(sorted(nonzero_inputs)))
                    Aqf.equals(actual_nz_bls, expected_nz_bls, msg)

                    msg = ('Confirm that expected baselines visibilities are \'Zeros\'.\n')
                    Aqf.equals(actual_z_bls, expected_z_bls, msg)

                    # Sum of all baselines powers expected to be non zeros
                    sum_of_bl_powers = (
                        [normalised_magnitude(test_data[:, expected_bl, :])
                         for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                                             for expected_nz_bl_ind in sorted(expected_nz_bls)]])

                    dataFrame.loc[inp][
                        sorted([i for i in expected_nz_bls])[-1]] = np.sum(
                        sum_of_bl_powers)
                else:
                    Aqf.failed(msg + str(reply))
            dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')
            who_ran_test()

    def _test_back2back_consistency(self):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
                identical/bit-perfect.
        """
        test_chan = randrange(self.corr_freqs.n_chans)
        test_baseline = 0  # auto-corr
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step(Style.Bold(self._test_back2back_consistency.__doc__))
        Aqf.addLine('-', 100)
        source_period_in_samples = self.corr_freqs.n_chans * 2
        cw_scale = 0.675
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=cw_scale,
                                          repeatN=source_period_in_samples)
        Aqf.step('Digitiser simulator configured to generate periodic wave '
                 '({0:.3f}Hz with FFT-length {1}) in order for each FFT to be '
                 'identical.'.format(expected_fc / 1e6, source_period_in_samples))

        def retrieve_clean_dump(self, spead_failure_counter=0):
            try:
                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                spead_failure_counter += 1
                errmsg = ('Could not retrieve clean SPEAD accumulation, as # %s '
                          'Queue is Empty.' % (spead_failure_counter))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if spead_failure_counter > 5:
                    spead_failure_counter = 0
                    Aqf.failed('Bailed: Kept receiving empty SPEAD accumulations')
                    return False
            else:
                return this_freq_dump

        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:

            for i, freq in enumerate(requested_test_freqs):
                Aqf.hop('Getting channel response for freq {0}/{1} @ {2:.3f} MHz.'.format(
                    i + 1, len(requested_test_freqs), freq / 1e6))
                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                  repeatN=source_period_in_samples)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency
                dumps_data = []
                chan_responses = []
                for dump_no in xrange(3):
                    if dump_no == 0:
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                            return False
                        else:
                            initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                    else:
                        this_freq_dump = retrieve_clean_dump(self)

                    this_freq_data = this_freq_dump['xeng_raw'].value
                    dumps_data.append(this_freq_data)
                    this_freq_response = normalised_magnitude(
                        this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)

                    # diff_dumps = []
                    # for comparison in xrange(1, len(dumps_data)):
                    # d2 = dumps_data[0]
                    # d1 = dumps_data[comparison]
                    ## Maximum difference between dump2 and dump1
                    # diff_dumps.append(np.max(d2 - d1))

                # dumps_comp = np.max(np.array(diff_dumps) / initial_max_freq)

                # Maximum difference between the initial max frequency and the last max freq
                dumps_comp = np.max(dumps_data[-1]) - initial_max_freq
                # msg = (
                # 'Check that back-to-back accumulations({0:.3f}/{1:.3f}dB) with the '
                # 'same frequency input differ by no more than {2} dB threshold.'.format(
                # dumps_comp, 10 * np.log10(dumps_comp), 10 * np.log10(threshold)))

                msg = ('Check that the maximum difference between the subsequent SPEAD accumulations'
                       ' with the same frequency input ({}Hz) is \'Zero\'.\n'.format(
                            this_source_freq))

                # if not Aqf.equal(dumps_comp, 1, msg):
                if not Aqf.equals(dumps_comp, 0, msg):
                    legends = ['dump #{}'.format(x) for x in xrange(len(chan_responses))]
                    plot_filename = ('{}_chan_resp_{}.png'.format(self._testMethodName,
                                                                  i + 1))
                    plot_title = 'Frequency Response {0} @ {1:.3f}MHz'.format(test_chan,
                                                                              this_source_freq / 1e6)
                    caption = (
                        'Comparison of back-to-back SPEAD accumulations with digitiser simulator '
                        'configured to generate periodic wave ({0:.3f}Hz with FFT-length {1}) '
                        'in order for each FFT to be identical'.format(this_source_freq,
                                                                       source_period_in_samples))
                    aqf_plot_channels(zip(chan_responses, legends), plot_filename, plot_title,
                                      log_dynamic_range=90, log_normalise_to=1, normalise=False,
                                      caption=caption)
        who_ran_test()

    def _test_freq_scan_consistency(self, threshold=1e-1):
        """This test confirms if the identical frequency scans produce equal results."""
        Aqf.step(Style.Bold(self._test_freq_scan_consistency.__doc__))
        Aqf.addLine('-', 100)
        test_chan = randrange(self.corr_freqs.n_chans)
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        source_period_in_samples = self.corr_freqs.n_chans * 2

        try:
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            cw_scale = 0.675
            Aqf.step('Digitiser simulator configured to generate continuous wave')
            Aqf.step('Randomly selected Frequency channel {0} @ {1:.3f}MHz for testing.'.format(
                test_chan, expected_fc / 1e6))
            Aqf.step('Sweeping the digitiser simulator over the center frequencies of at '
                     'least all channels that fall within the complete L-band: {} Hz'.format(
                expected_fc))

            for scan_i in xrange(3):
                scan_dumps = []
                frequencies = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        Aqf.hop('Getting channel response for freq {} @ {}: {} MHz.'
                                .format(i + 1, len(requested_test_freqs), freq / 1e6))
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                          repeatN=source_period_in_samples)
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                            this_freq_data = this_freq_dump['xeng_raw'].value
                            initial_max_freq_list.append(initial_max_freq)
                    else:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                          repeatN=source_period_in_samples)
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            this_freq_data = this_freq_dump['xeng_raw'].value

                    this_freq_response = normalised_magnitude(
                        this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)
                    scan_dumps.append(this_freq_data)
                    frequencies.append(freq_val)

            for scan_i in xrange(1, len(scans)):
                for freq_i, freq_x in zip(xrange(len(scans[0])), frequencies):
                    s0 = scans[0][freq_i]
                    s1 = scans[scan_i][freq_i]
                    norm_fac = initial_max_freq_list[freq_i]
                    # TODO Convert to a less-verbose comparison for Aqf.
                    # E.g. test all the frequencies and only save the error cases,
                    # then have a final Aqf-check so that there is only one step
                    # (not n_chan) in the report.
                    max_freq_scan = np.max(np.abs(s1 - s0)) / norm_fac

                    msg = ('Confirm that identical frequency ({:.3f} MHz) scans between subsequent '
                           'SPEAD accumulations produce equal results.\n'.format(freq_x / 1e6))

                    if not Aqf.less(np.abs(max_freq_scan), np.abs(np.log10(threshold)), msg):
                        legends = ['Freq scan #{}'.format(x) for x in xrange(len(chan_responses))]
                        caption = ('A comparison of frequency sweeping from {0:.3f}Mhz to {1:.3f}Mhz '
                                   'scan channelisation and also, {2}'.format(
                            requested_test_freqs[0] / 1e6,
                            requested_test_freqs[-1] / 1e6, expected_fc, msg))

                        aqf_plot_channels(zip(chan_responses, legends),
                                          plot_filename='{}_chan_resp.png'.format(
                                              self._testMethodName), caption=caption)
        who_ran_test()

    def _test_restart_consistency(self, instrument, no_channels):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect on CBF restart.
        """
        Aqf.step(Style.Bold(self._testMethodDoc))
        threshold = 1.0e1  #
        test_baseline = 0

        test_chan = randrange(self.corr_freqs.n_chans)
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Sweeping the digitiser simulator over {0:.3f}MHz of the channels that '
                 'fall within {1} complete L-band'.format(np.max(requested_test_freqs) / 1e6,
                                                          test_chan))

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                 'with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}'
                 .format(cw_scale, awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=cw_scale, freq=expected_fc,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            # Plot an overall frequency response at the centre frequency just as
            # a sanity check
            init_source_freq = normalised_magnitude(
                this_freq_dump['xeng_raw'].value[:, test_baseline, :])
            filename = '{}_channel_response.png'.format(self._testMethodName)
            title = ('Frequency response at {0} @ {1:.3f} MHz.\n'.format(test_chan,
                                                                         expected_fc / 1e6))
            caption = ('An overrall frequency response at the center frequency.')
            aqf_plot_channels(init_source_freq, filename, title, caption=caption)
            restart_retries = 5

            def _restart_instrument(retries=restart_retries):
                if not self.corr_fix.stop_x_data():
                    Aqf.failed('Could not stop x data from capturing.')
                with ignored(Exception):
                    deprogram_hosts(self)

                corr_init = False
                _empty = True
                with ignored(Queue.Empty):
                    self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    _empty = False

                Aqf.is_true(_empty,
                            'Confirm that the SPEAD accumulations have stopped being produced.')

                self.corr_fix.halt_array()
                xhosts = self.correlator.xhosts
                fhosts = self.correlator.fhosts

                while retries and not corr_init:
                    Aqf.step('Re-initialising the {} instrument'.format(instrument))
                    with ignored(Exception):
                        corr_init = self.set_instrument(instrument)

                    retries -= 1
                    if retries == 0:
                        errmsg = ('Could not restart the correlator after %s tries.' % (retries))
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)

                if corr_init.keys()[0] is not True and retries == 0:
                    msg = ('Could not restart {} after {} tries.'.format(instrument, retries))
                    Aqf.end(passed=False, message=msg)
                else:
                    startx = self.corr_fix.start_x_data()
                    if not startx:
                        Aqf.failed('Failed to enable/start output product capturing.')
                    host = (xhosts + fhosts)[randrange(len(xhosts + fhosts))]
                    msg = ('Confirm that the instrument is initialised by checking if a '
                           'random host: {} is programmed and running.'.format(host.host))
                    Aqf.is_true(host, msg)

                    try:
                        self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                        self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    except AssertionError:
                        errmsg = 'Correlator Receiver could not be instantiated.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    else:
                        msg = ('Check that data product has the number of frequency '
                               'channels {no_channels} corresponding to the {instrument} '
                               'instrument product'.format(**locals()))
                        try:
                            spead_chans = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            Aqf.equals(spead_chans['xeng_raw'].value.shape[0], no_channels, msg)
                            return True

            initial_max_freq_list = []
            scans = []
            channel_responses = []
            for scan_i in xrange(3):
                if scan_i:
                    Aqf.step('#{scan_i}: Initialising {instrument} instrument'.format(**locals()))
                    intrument_success = _restart_instrument()
                    if not intrument_success:
                        msg = ('Failed to restart the correlator successfully.')
                        Aqf.failed(msg)
                        self.corr_fix.halt_array()
                        time.sleep(10)
                        self.set_instrument(instrument)
                        return False

                scan_dumps = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        if self.corr_fix.start_x_data():
                            Aqf.hop('Getting Frequency SPEAD accumulation #{0} with Digitiser simulator '
                                    'configured to generate cw at {1:.3f}MHz'.format(i, freq / 1e6))
                            try:
                                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                            except Queue.Empty:
                                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                                Aqf.failed(errmsg)
                                LOGGER.exception(errmsg)

                        initial_max_freq = np.max(this_freq_dump['xeng_raw'].value)
                        this_freq_data = this_freq_dump['xeng_raw'].value
                        initial_max_freq_list.append(initial_max_freq)
                        freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    else:
                        msg = ('Getting Frequency SPEAD accumulation #{0} with digitiser simulator '
                               'configured to generate cw at {1:.3f}MHz'.format(i, freq / 1e6))
                        Aqf.hop(msg)
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            this_freq_data = this_freq_dump['xeng_raw'].value
                            freq_response = normalised_magnitude(
                                this_freq_data[:, test_baseline, :])
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

            msg = ('Check that CBF restart SPEAD accumulations comparison results '
                   'with the same frequency input differ by no more than {0:.3f}dB '
                   'threshold.'.format(threshold))

            if not Aqf.less(diff_scans_comp, threshold, msg):
                legends = ['Channel Responce #{}'.format(x) for x in xrange(len(channel_responses))]
                plot_filename = '{}_chan_resp.png'.format(self._testMethodName)
                caption = ('Check that results are consistent on CBF restart')
                plot_title = ('CBF restart consistency channel response {}'.format(test_chan))
                aqf_plot_channels(zip(channel_responses, legends), plot_filename, plot_title,
                                  caption=caption)
        who_ran_test()

    def _test_delay_tracking(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay tracking"""

        setup_data = self._delays_setup()
        if setup_data:
            sampling_period = self.corr_freqs.sample_period
            no_chans = range(self.corr_freqs.n_chans)

            test_delays = [0, sampling_period, 1.5 * sampling_period,
                           2 * sampling_period]
            test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
            delays = [0] * setup_data['num_inputs']
            Aqf.step('[CBF-REQ-0185] Delays to be set (iteratively) {}\n'.format(test_delays))

            def get_expected_phases():
                expected_phases = []
                for delay in test_delays:
                    phases = self.corr_freqs.chan_freqs * 2 * np.pi * delay
                    phases -= np.max(phases) / 2.
                    expected_phases.append(phases)
                return zip(test_delays_ns, expected_phases)

            def get_actual_phases():
                actual_phases_list = []
                # chan_responses = []
                for delay in test_delays:
                    delays[setup_data['test_source_ind']] = delay
                    delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                    roundtrip = 300e-3
                    settling_time = 900e-3
                    cam_max_load_time = 1

                    try:
                        this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT, discard=0)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        try:
                            reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch()
                            sync_time = float(reply.arguments[-1])
                            if not reply.reply_ok():
                                raise Exception
                        except Exception:
                            Aqf.failed('Could not retrieve sync time via correlator object.')
                            return False

                        dump_timestamp = (roundtrip + sync_time +
                                          this_freq_dump['timestamp'].value /
                                          this_freq_dump['scale_factor_timestamp'].value)
                        t_apply = (dump_timestamp + 10 * this_freq_dump['int_time'].value)
                        try:
                            cmd_start_time = time.time()
                            reply, _informs = self.corr_fix.katcp_rct.req.delays(t_apply,
                                                                                 *delay_coefficients)
                            final_cmd_time = (time.time() - cmd_start_time - roundtrip)
                        except:
                            errmsg = ('Failed to set delays via CAM interface with loadtime: %s,'
                                      ' Delay coefficiencts: %s' % (t_apply, delay_coefficients))
                            LOGGER.exception(errmsg)
                            Aqf.failed(errmsg)

                        else:
                            msg = ('[CBF-REQ-0077, 0187]: Time it takes to load delays is less '
                                   'than {}s with integration time of {:.3f}s'.format(
                                cam_max_load_time, this_freq_dump['int_time'].value))

                            # Aqf.less(final_cmd_time, cam_max_load_time, msg)
                            Aqf.passed(msg)

                            msg = ('[CBF-REQ-0066, 0072] Delays Reply: {}'.format(
                                reply.arguments[1]))
                            Aqf.is_true(reply.reply_ok(), msg)

                            msg = ('Settling time in order to set delay in the SPEAD data:'
                                   ' {} ns.'.format(delay * 1e9))
                            Aqf.wait(settling_time, msg)

                            # time.sleep(settling_time)
                            try:
                                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                            except Queue.Empty:
                                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                                Aqf.failed(errmsg)
                                LOGGER.exception(errmsg)
                            else:
                                this_freq_data = this_freq_dump['xeng_raw'].value
                                # this_freq_response = normalised_magnitude(
                                #    this_freq_data[:, setup_data['test_source_ind'], :])
                                # chan_responses.append(this_freq_response)
                                data = complexise(dump['xeng_raw'].value
                                                  [:, setup_data['baseline_index'], :])

                                phases = np.angle(data)
                                actual_phases_list.append(phases)

                                # actual_channel_responses = zip(test_delays, chan_responses)
                                # return zip(actual_phases_list, actual_channel_responses)
                return actual_phases_list

            actual_phases = get_actual_phases()
            expected_phases = get_expected_phases()
            if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                Aqf.failed('Delays could not be applied at time_apply: {} '
                           'possibly in the past.\n'.format(setup_data['t_apply']))

            else:
                # actual_phases = [phases for phases, response in actual_data]
                # actual_response = [response for phases, response in actual_data]
                plot_title = 'CBF Delay Compensation'
                caption = ('Actual and expected Unwrapped Correlation Phase [Delay tracking].\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation.')
                plot_filename = '{}_phase_response.png'.format(self._testMethodName)
                plot_units = 'secs'

                aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

                expected_phases_ = [phase for _rads, phase in get_expected_phases()]

                degree = 1.0
                decimal = len(str(degree).split('.')[-1])

                for i, delay in enumerate(test_delays):
                    delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
                    delta_expected = np.max(expected_phases_[i]) - np.min(
                        expected_phases_[i])
                    abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    # abs_diff = np.abs(delta_expected - delta_actual)
                    msg = ('[CBF-REQ-0128, 0187] Check that if difference expected({0:.5f}) '
                           'and actual({1:.5f}) phases are equal at delay {2:.5f}ns within '
                           '{3} degree.'.format(delta_expected, delta_actual, delay * 1e9, degree))
                    Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                    Aqf.less(abs_diff, degree,
                             '[CBF-REQ-0187] Check that the maximum difference ({0:.3f} degree/'
                             ' {1:.3f} rad) between expected phase and actual phase between '
                             'integrations is less than {2} degree.\n'.format(
                                 abs_diff, np.deg2rad(abs_diff), degree))
                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                       decimal=decimal)
                    except AssertionError:
                        msg = (
                            '[CBF-REQ-0128, 01877] Difference expected({0:.5f}) phases'
                            ' and actual({1:.5f}) phases are \'Not almost equal\' '
                            'within {2} degree when delay of {3}ns is applied.'.format(
                                delta_expected, delta_actual, degree, delay * 1e9))
                        Aqf.step(msg)

                        caption = (
                            'The figure above shows, The difference between expected({0:.5f}) '
                            'phases and actual({1:.5f}) phases are \'Not almost equal\' within {2} '
                            'degree when a delay of {3:.5f}s is applied. Therefore CBF-REQ-0128 and'
                            ', CBF-REQ-0187 are not verified.'.format(delta_expected, delta_actual,
                                                                      degree, delay))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(no_chans, actual_phases_i, expected_phases_i,
                                               plot_filename='{}_{}_phase_resp.png'.format(self._testMethodName, i),
                                               plot_title='Delay offset:\nActual vs Expected Phase Response',
                                               plot_units=plot_units, caption=caption, )

                for delay, count in zip(test_delays, xrange(1, len(expected_phases))):
                    msg = ('[CBF-REQ-0128, 0187] Check that when a delay of {0} clock '
                           'cycle({1:.5f} ns) is introduced there is a phase change '
                           'of {2:.3f} degrees as expected to within {3} degree.'.format(
                        (count + 1) * .5, delay * 1e9, np.rad2deg(np.pi) * (count + 1) * .5,
                        degree))
                    Aqf.array_abs_error(actual_phases[count][1:],
                                        expected_phases_[count][1:], msg, degree)
        who_ran_test()

    def _test_sensor_values(self):
        """
        Report sensor values (AR1)
        Test Varifies:
            [CBF-REQ-0060]
            [CBF-REQ-0068]
            [CBF-REQ-0069]
            [CBF-REQ-0178]
            [CBF-REQ-0204]
            [CBF-REQ-0056]

        """

        def report_sensor_list(self):
            Aqf.step('Check that the number of sensors available on the primary '
                     'and subarray interface is consistent.')
            try:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_list()
            except:
                errmsg = 'CAM interface connection encountered errors.'
                Aqf.failed(errmsg)
            else:
                msg = ('[CBF-REQ-0068] Confirm that the number of sensors are equal '
                       'to the number of sensors listed on the running instrument.\n')
                Aqf.equals(int(reply.arguments[-1]), len(informs), msg)

        def report_time_sync(self):
            Aqf.step('Check that time synchronous is implemented on primary interface')
            try:
                reply, informs = self.corr_fix.rct.req.sensor_value('time.synchronised')
            except:
                Aqf.failed('CBF report time sync could not be retrieved from primary interface.')
            else:
                Aqf.is_true(reply.reply_ok(),
                            '[CBF-REQ-0069] CBF report time sync implemented in this release.')

            msg = ('[CBF-REQ-0069] Confirm that the CBF can report time sync status '
                   'via CAM interface. ')
            try:
                reply, informs = self.corr_fix.rct.req.sensor_value('time.synchronised')
            except:
                Aqf.failed(msg)
            else:
                msg = msg + 'CAM Reply: {}\n'.format(str(informs[0]))
                if not reply.reply_ok():
                    Aqf.failed(msg)
                Aqf.passed(msg)

        def report_small_buffer(self):
            Aqf.step('Check that Transient Buffer ready is implemented.')
            try:
                reply, _informs = self.corr_fix.katcp_rct.req.sensor_value(
                    'corr.transient-buffer-ready')
            except:
                Aqf.failed('[CBF-REQ-0056] CBF Transient buffer ready for triggering'
                           '\'Not\' implemented in this release.\n')
            else:
                Aqf.is_true(reply.reply_ok(),
                            '[CBF-REQ-0056] CBF Transient buffer ready for triggering'
                            ' implemented in this release.\n')

        def report_primary_sensors(self):
            Aqf.step('Check that all primary sensors are norminal.')
            for sensor in self.corr_fix.rct.sensor.values():
                msg = 'Primary sensor: {}, current status: {}'.format(sensor.name,
                                                                      sensor.get_status())
                Aqf.equals(sensor.get_status(), 'nominal', msg)

        # Confirm the CBF replies with "!sensor-list ok numSensors"
        # where numSensors is the number of sensor-list informs sent.
        report_sensor_list(self)
        # Request the time synchronisation status using CAM interface
        report_time_sync(self)
        # The CBF shall report the following transient search monitoring data
        report_small_buffer(self)
        # Check all sensors statuses if they are nominal
        report_primary_sensors(self)

    def _test_roach_qdr_sensors(self):

        def roach_qdr(corr_hosts, engine_type, sensor_timeout=60):
            try:
                array_sensors = self.corr_fix.katcp_rct.sensor
                self.assertIsInstance(array_sensors,
                                      katcp.resource_client.AttrMappingProxy)
            except:
                Aqf.failed('KATCP connection encountered errors.\n')
            else:
                hosts = corr_hosts
                # Random host selection, seems to be broken due to the instability on
                # hardware QDR's
                # host = hosts[randrange(len(hosts))]
                host = hosts[-1]

                Aqf.step('Randomly selected hardware {}-{} to test QDR failure '
                         'detection'.format(host.host.upper(), engine_type.capitalize()))
                clear_host_status(self)

                try:
                    host_sensor = getattr(array_sensors, '{}_{}_qdr_ok'.format(
                        host.host.lower(), engine_type))
                    assert isinstance(host_sensor,
                                      katcp.resource_client.ThreadSafeKATCPSensorWrapper)
                except Exception:
                    Aqf.failed('Could not retrieve hosts sensor attributes from katcp '
                               'client on {}'.format(host.host.upper()))
                else:
                    # Check if QDR is okay before test is ran
                    msg = ('Confirm that ({}) hardware sensor indicates QDR status is '
                           '\'Healthy\'. Current status on HW: {}.'.format(
                        host.host.upper(), host_sensor.get_status()))
                    Aqf.is_true(host_sensor.get_value(), msg)
                    host_sensor.set_strategy('auto')
                    self.addCleanup(host_sensor.set_sampling_strategy, 'none')

                    def blindwrite(host):
                        """Writes junk to memory"""
                        junk_msg = ('0x' + ''.join(x.encode('hex')
                                                   for x in 'oidhsdvwsfvbgrfbsdceijfp3ioejfg'))
                        try:
                            for i in xrange(100):
                                host.blindwrite('qdr0_memory', junk_msg)
                                host.blindwrite('qdr1_memory', junk_msg)
                            return True
                        except:
                            return False

                    msg = ('[CBF-REQ-0157] Writing random data to {} the '
                           'QDR memory.'.format(host.host.upper()))
                    Aqf.is_true(blindwrite(host), msg)

                    def status_change(host_sensor):
                        success = False
                        retries = 30
                        while retries and not success:
                            time.sleep(1)
                            retries -= 1
                            success = not host_sensor.get_value()
                            if retries == 0:
                                break
                        return success

                    try:
                        assert status_change(host_sensor)
                    except:
                        Aqf.failed('Failed to verify if the QDR memory is corrupted or unreadable')
                    else:
                        msg = ('[CBF-REQ-0157] Confirm that sensor indicates that the memory on {} '
                               'is unreadable/corrupted.'.format(host.host))
                        Aqf.is_false(host_sensor.get_value(), msg)

                    if engine_type == 'xeng':
                        current_errors = (
                            host.registers.vacc_errors1.read()['data']['parity'])
                    else:
                        ct_ctrs = host.registers.ct_ctrs.read()['data']
                        current_errors = (ct_ctrs['ct_parerr_cnt0'] +
                                          ct_ctrs['ct_parerr_cnt1'])
                    msg = ('Confirm that the error counters have incremented, showing '
                           'that the corner turner experienced faults.')
                    Aqf.is_not_equals(current_errors, 0, msg)

                    if engine_type == 'xeng':
                        host_vacc_errors = (
                            host.registers.vacc_errors1.read()['data']['parity'])
                        msg = ('Confirm that the error counters have stopped '
                               'incrementing: {} increments.'.format(current_errors))
                        if current_errors == host_vacc_errors:
                            Aqf.passed(msg)

                    else:
                        new_errors = (
                            host.registers.ct_ctrs.read()['data']['ct_parerr_cnt0'] +
                            host.registers.ct_ctrs.read()['data']['ct_parerr_cnt1'])
                        if current_errors == new_errors:
                            Aqf.passed('[CBF-REQ-0157] Confirm that the error counters '
                                       'have stopped incrementing with last known '
                                       'incremented #{}.'.format(current_errors))

                    clear_host_status(self)
                    if engine_type == 'xeng':
                        vacc_errors_final = (
                            host.registers.vacc_errors1.read()['data']['parity'])
                        final_errors = vacc_errors_final
                    else:
                        final_errors = (
                            host.registers.ct_ctrs.read()['data']['ct_parerr_cnt0'] +
                            host.registers.ct_ctrs.read()['data']['ct_parerr_cnt1'])
                    msg = ('Confirm that the error counters have been reset, '
                           'from {} to {}'.format(current_errors, final_errors))
                    Aqf.is_false(final_errors, msg)

                    if host_sensor.wait(True, timeout=sensor_timeout):
                        msg = ('Confirm that sensor indicates that the QDR memory '
                               'recovered. Status: {} on {}.\n'.format(host_sensor.status, host.host))
                        Aqf.is_true(host_sensor.get_value(), msg)
                    else:
                        Aqf.failed('[CBF-REQ-0157] QDR sensor failed to recover with'
                                   'CAM sensor status: {} on {}.\n'.format(host_sensor.status,
                                                                           host.host))

                clear_host_status(self)

        roach_qdr(self.correlator.fhosts, 'feng')
        roach_qdr(self.correlator.xhosts, 'xeng')

    def _test_roach_pfb_sensors(self):
        """Sensor PFB error"""

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=self.corr_freqs.bandwidth / 2,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False
        else:
            Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                     'with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}'
                     .format(cw_scale, awgn_scale, gain, fft_shift))

        sensor_poll_time = self.correlator.sensor_poll_time
        Aqf.step('Sensor poll time: {} seconds '.format(sensor_poll_time))

        def get_pfb_status(self):
            """Retrieve F-Engines PFB status
            :param: Object
            :rtype: Boolean or list
            """
            try:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_value()
            except Exception:
                return False
            else:
                hosts = [_i.host.lower() for _i in self.correlator.fhosts]
                try:
                    roach_dict = [getattr(self.corr_fix.katcp_rct.sensor, '{}_feng_pfb_ok'.format(host))
                                  for host in hosts]
                except AttributeError:
                    Aqf.failed('Failed to retrieve PFB status on F-hosts')
                    return False
                else:
                    pfb_status = [[' '.join(i.arguments[2:]) for i in informs
                                   if i.arguments[2] == '{}-feng-pfb-ok'.format(host)]
                                  for host in hosts]
                    return list(set([int(i[0].split()[-1]) for i in pfb_status]))[0]

        def confirm_pfb_status(self, get_pfb_status, fft_shift=0):

            fft_shift_val = self.correlator.fops.set_fft_shift_all(shift_value=fft_shift)
            if fft_shift_val is None:
                Aqf.failed('Could not set FFT shift for all F-Engine hosts')
            else:
                msg = ('An FFT Shift of {} was set on all F-Engines.'.format(fft_shift_val))
                Aqf.wait(self.correlator.sensor_poll_time * 2, msg)

                pfb_status = get_pfb_status(self)
                if pfb_status == 1:
                    msg = 'Confirm that the sensors indicate that F-Eng PFB status is \'Okay\'.\n'
                    Aqf.passed(msg)
                elif pfb_status == 0:
                    msg = ('Confirm that the sensors indicate that there is an \'Error\' on the '
                           'F-Eng PFB status.\n')
                    Aqf.passed(msg)
                else:
                    Aqf.failed('Could not retrieve PFB sensor status')

        confirm_pfb_status(self, get_pfb_status, fft_shift=fft_shift)
        confirm_pfb_status(self, get_pfb_status)
        Aqf.step('Restoring previous FFT Shift values')
        confirm_pfb_status(self, get_pfb_status, fft_shift=fft_shift)
        clear_host_status(self)

    def _test_link_error(self):

        def get_spead_data(self):
            try:
                self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                msg = ('Confirm that the SPEAD accumulation is being produced by '
                       'instrument but not verified.\n')
                Aqf.passed(msg)

        # Record the current multicast desitination of one of the F-engine data
        # ethernet ports,
        def get_host_ip(host):
            try:
                reply, _informs = host.katcprequest('wordread',
                                                    request_args=(['iptx_base']))
            except:
                Aqf.failed('Failed to retrieve multicast destination from {}'.format(
                    host.host.upper()))
            else:
                if reply.reply_ok() and len(reply.arguments) > 1:
                    hex_ip_ = reply.arguments[-1]
                    if hex_ip_.startswith('0x'):
                        return hex_ip_
                    else:
                        return None
                else:
                    return None

        def get_lru_status(self, host):

            if host in self.correlator.fhosts:
                engine_type = 'feng'
            else:
                engine_type = 'xeng'

            try:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_value(
                    '{}-{}-lru-ok'.format(host.host, engine_type))
            except:
                Aqf.failed('Could not get sensor attributes on {}'.format(
                    host.host.upper()))
            else:
                if reply.reply_ok() and (int(informs[0].arguments[-1]) == 1):
                    return 1
                elif reply.reply_ok() and (int(informs[0].arguments[-1]) == 0):
                    return 0
                else:
                    return False

        # configure the same port multicast destination to an unused address,
        # effectively dropping that data.
        def write_new_ip(host, ip_new, ip_old, get_host_ip, human_readable_ip):
            try:
                reply, informs = host.katcprequest('wordwrite',
                                                   request_args=(['iptx_base', '0',
                                                                  ip_new]))
            except:
                Aqf.failed('Failed to write new multicast destination on {}'.format(
                    host.host.upper()))
            else:
                if reply.reply_ok():
                    changed_ip = get_host_ip(host)
                    if not changed_ip:
                        Aqf.failed('Failed to retrieve multicast destination address '
                                   'of {}'.format(host.host.upper()))
                    else:
                        Aqf.passed(
                            'Confirm that the multicast destination address '
                            'for {} has been changed from {} to {}.'.format(
                                host.host.upper(), human_readable_ip(ip_old),
                                human_readable_ip(changed_ip)))
                else:
                    Aqf.failed('Failed to reconfigure multicast destination on '
                               '{}'.format(host.host.upper()))

        def report_lru_status(self, host, get_lru_status):
            Aqf.wait(self.correlator.sensor_poll_time,
                     'Wait until the sensors have been updated with new changes')
            if get_lru_status(self, host) == 1:
                Aqf.passed(
                    'Confirm that the X-engine {} LRU sensor is \'Okay\' and '
                    'that the X-eng is receiving feasible data.'.format(
                        host.host.upper()))
            elif get_lru_status(self, host) == 0:
                Aqf.passed('Confirm that the X-engine {} LRU sensor is reporting a '
                           'failure and that the X-eng is not receiving feasible '
                           'data.'.format(host.host.upper()))
            else:
                Aqf.failed('Failed to read {} sensor'.format(host.host.upper()))

        fhost = self.correlator.fhosts[randrange(len(self.correlator.fhosts))]
        xhost = self.correlator.xhosts[randrange(len(self.correlator.xhosts))]
        ip_new = '0xefefefef'

        Aqf.step('Randomly selected {} host that is being used to produce the test '
                 'data product on which to trigger the link error.'.format(
            fhost.host.upper()))
        current_ip = get_host_ip(fhost)
        if not current_ip:
            Aqf.failed('Failed to retrieve multicast destination address of {}'.format(
                fhost.host.upper()))
        elif current_ip != ip_new:
            Aqf.passed('Current multicast destination address for {}: {}.'.format(
                fhost.host.upper(), human_readable_ip(current_ip)))
        else:
            Aqf.failed('Multicast destination address of {} cannot be {}'.format(
                fhost.host.upper(), human_readable_ip(ip_new)))

        report_lru_status(self, xhost, get_lru_status)
        get_spead_data(self)

        write_new_ip(fhost, ip_new, current_ip, get_host_ip, human_readable_ip)
        time.sleep(self.correlator.sensor_poll_time / 2)
        report_lru_status(self, xhost, get_lru_status)
        get_spead_data(self)

        Aqf.step('Restoring the multicast destination from {} to the original {}'.format(
            human_readable_ip(ip_new), human_readable_ip(current_ip)))

        write_new_ip(fhost, current_ip, ip_new, get_host_ip, human_readable_ip)
        report_lru_status(self, xhost, get_lru_status)
        get_spead_data(self)
        clear_host_status(self)

    def _test_roach_sensors_status(self):
        Aqf.step('This test confirms that each ROACH sensor (Temp, Voltage, Current, '
                 'Fan) has not \'Failed\'.')
        for roach in (self.correlator.fhosts + self.correlator.xhosts):
            values_reply, sensors_values = roach.katcprequest('sensor-value')
            list_reply, sensors_list = roach.katcprequest('sensor-list')
            Aqf.step('Checking the sensors on host: {}'.format(roach.host.upper()))
            msg = ('[CBF-REQ-0068, 0178] {}: Verify that the number of hardware '
                   'sensors are consistent.'.format(roach.host.upper()))

            Aqf.is_true((values_reply.reply_ok() == list_reply.reply_ok()), msg)

            msg = (
                '[CBF-REQ-0068, 0178] {}: Confirm that the number of hardware '
                'sensors-list are equal to the sensor-values of specific '
                'hardware\n'.format(roach.host.upper()))

            Aqf.equals(len(sensors_list), int(values_reply.arguments[1]), msg)

            for sensor in sensors_values[1:]:
                sensor_name, sensor_status, sensor_value = (
                    sensor.arguments[2:])
                # Check if roach sensors are failing
                if sensor_status == 'fail':
                    msg = (
                        '[CBF-REQ-0068, 0178] Roach: {}, Sensor: {}, Status: {}'
                            .format(roach.host.upper(), sensor_name, sensor_status))
                    Aqf.failed(msg)

    def _test_vacc(self, test_chan, chan_index=None):
        """Test vector accumulator"""
        MAX_VACC_SYNCH_ATTEMPTS = corr2.fxcorrelator_xengops.MAX_VACC_SYNCH_ATTEMPTS

        # Choose a test freqency around the centre of the band.
        test_freq = self.corr_freqs.bandwidth / 2.
        test_input = sorted(self.correlator.fengine_sources.keys())[0]
        eq_scaling = 30
        acc_times = [0.89, 0.49]
        n_chans = self.corr_freqs.n_chans

        internal_accumulations = int(
            self.correlator.configd['xengine']['xeng_accumulation_len'])
        delta_acc_t = self.corr_freqs.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = np.argmin(
            np.abs(self.corr_freqs.chan_freqs[:chan_index] - test_freq))
        eqs = np.zeros(n_chans, dtype=np.complex)
        eqs[test_freq_channel] = eq_scaling
        get_and_restore_initial_eqs(self, self.correlator)
        try:
            reply, _informs = self.corr_fix.katcp_rct.req.gain(test_input, *list(eqs))
        except Exception:
            errmsg = 'Gains/Eq could not be set via CAM interface'
            Aqf.failed(errmsg)
            LOGGER.error(errmsg)
        else:
            if reply.reply_ok():
                Aqf.hop('[CBF-REQ-0119] Gain factors set successfully via CAM interface.')
        Aqf.step(
            'Configured Digitiser simulator output(cw0 @ {0:.3f}MHz) to be periodic in FFT-length: {1} '
            'in order for each FFT to be identical'.format(test_freq / 1e6, n_chans * 2))
        cw_scale = 0.125

        # Make dsim output periodic in FFT-length so that each FFT is identical
        self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=cw_scale, repeatN=n_chans * 2)

        # The re-quantiser outputs signed int (8bit), but the snapshot code
        # normalises it to floats between -1:1. Since we want to calculate the
        # output of the vacc which sums integers, denormalise the snapshot
        # output back to ints.
        q_denorm = 128
        quantiser_spectrum = get_quant_snapshot(
            self.correlator, test_input) * q_denorm
        Aqf.step('Test input: {0}, Test Channel :{1:.3f}'.format(test_input,
                                                                 test_freq_channel))
        # Check that the spectrum is not zero in the test channel
        # Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
        # 'Check that the spectrum is not zero in the test channel')
        # Check that the spectrum is zero except in the test channel
        Aqf.is_true(np.all(quantiser_spectrum[0:test_freq_channel] == 0),
                    'Check that the spectrum is zero except in the test channel:'
                    ' [0:test_freq_channel]')
        Aqf.is_true(np.all(quantiser_spectrum[test_freq_channel + 1:] == 0),
                    'Check that the spectrum is zero except in the test channel:'
                    ' [test_freq_channel+1:]')
        Aqf.step('FFT Window [{} samples] = {:.3f} micro seconds, Internal Accumulations = {}, '
                 'One VACC accumulation = {}s'.format(n_chans * 2,
                                                      self.corr_freqs.fft_period * 1e6, internal_accumulations,
                                                      delta_acc_t))

        chan_response = []
        # TODO MM 2016-10-07 Fix tests to use cam interface instead of corr object
        for vacc_accumulations, acc_time in zip(test_acc_lens, acc_times):
            try:
                # self.correlator.xops.set_acc_len(vacc_accumulations)
                reply = self.corr_fix.katcp_rct.req.accumulation_length(
                    acc_time, timeout=60)
                self.assertIsInstance(reply, katcp.resource.KATCPReply)

            except (TimeoutError, VaccSynchAttemptsMaxedOut):
                Aqf.failed('Failed to set accumulation length of {} after {} maximum vacc '
                           'sync attempts.'.format(vacc_accumulations,
                                                   MAX_VACC_SYNCH_ATTEMPTS))
            else:
                # TODO MM get acclen from CAM interface
                Aqf.almost_equals(vacc_accumulations,
                                  self.correlator.xops.get_acc_len(), 1e-2,
                                  'Confirm that vacc length was set successfully with'
                                  ' {}, which equates to an accumulation time of {:.6f}s'
                                  ''.format(vacc_accumulations,
                                            vacc_accumulations * delta_acc_t))
                no_accs = internal_accumulations * vacc_accumulations
                expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
                try:
                    d = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                except Queue.Empty:
                    errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                else:
                    actual_response = complexise(d['xeng_raw'].value[:, 0, :])
                    actual_response_ = loggerise(d['xeng_raw'].value[:, 0, :])
                    chan_response.append(normalised_magnitude(d['xeng_raw'].value[:, 0, :]))
                    # Check that the accumulator response is equal to the expected response

                    caption = (
                        'Accumulators actual response is equal to the expected response for {0} '
                        'accumulation length with a periodic cw tone every {1} samples'
                        ' at frequency of {2:.3f} MHz with scale {3}.'.format(test_acc_lens,
                                                                              n_chans * 2,
                                                                              test_freq / 1e6,
                                                                              cw_scale))

                    plot_filename = ('{}_chan_resp_{}_acc.png'.format(self._testMethodName,
                                                                      int(vacc_accumulations)))
                    plot_title = ('Vector Accumulation Length: channel {}'.format(test_freq_channel))
                    msg = ('Check that the accumulator actual response is equal to the '
                           'expected response for {} accumulation length'.format(vacc_accumulations))

                    if not Aqf.array_abs_error(expected_response[:chan_index].real,
                                               actual_response[:chan_index].real, msg):
                        aqf_plot_channels(actual_response_, plot_filename, plot_title,
                                          log_normalise_to=0, normalise=0, caption=caption)
        who_ran_test()

    def _test_product_switch(self, instrument, no_channels):

        Aqf.step('Confirm that SPEAD accumulations are being produced when Digitiser simulator is '
                 'configured to output correlated noise')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        with ignored(Queue.Empty):
            self.receiver.get_clean_dump(DUMP_TIMEOUT)

        self.corr_fix.stop_x_data()
        Aqf.step('[CBF-REQ-0064] Deprogramming xhosts first then fhosts avoid '
                 'reorder timeout errors')
        xhosts = self.correlator.xhosts
        fhosts = self.correlator.fhosts
        with ignored(Exception):
            deprogram_hosts(self)
        Aqf.step('Check that SPEAD accumulations are nolonger being produced.')
        with ignored(Queue.Empty):
            self.receiver.get_clean_dump(DUMP_TIMEOUT)
            Aqf.failed('SPEAD accumulations are still being produced.')

        self.corr_fix.halt_array()
        Aqf.step(
            '[CBF-REQ-0064] Re-initialising {instrument} instrument'.format(**locals()))
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
                    msg = 'CBF Re-Initialised Successfully'
                    Aqf.step(msg)
                    LOGGER.info(msg + 'after %s retries' % (retries))
            except:
                retries -= 1
                if retries == 0:
                    errmsg = 'Could not restart the correlator after %s tries.' % (retries)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

        if corr_init:
            end_time = time.time()
            host = xhosts[randrange(len(xhosts))]
            Aqf.is_true(host.is_running(), 'Confirm that the instrument is initialised by checking if '
                                           '{} is programmed.'.format(host.host))

            Aqf.hop('Capturing SPEAD Accumulation after re-initialisation to confirm '
                    'that the instrument activated is valid.')
            try:
                re_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            except AttributeError:
                errmsg = ('Could not retrieve clean SPEAD accumulation: Receiver could not '
                          'be instantiated.')
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            else:
                Aqf.is_true(re_dump,
                            'Confirm that SPEAD accumulations are being produced after instrument '
                            're-initialisation.')

                Aqf.equals(re_dump['xeng_raw'].value.shape[0], no_channels,
                           'Check that data product has the number of frequency '
                           'channels {no_channels} corresponding to the {instrument} '
                           'instrument product'.format(**locals()))

                final_time = end_time - start_time
                minute = 60.0
                Aqf.less(final_time, minute,
                         '[CBF-REQ-0013] Confirm that instrument switching to {instrument} '
                         'time is less than one minute'.format(**locals()))
        who_ran_test()

    def _test_adc_overflow_flag(self):
        """CBF flagging of data -- ADC overflow"""

        # TODO 2015-09-22 (NM): Test is currently failing since the noise diode flag is
        # also set when the overange occurs. Needs to check if the dsim is doing this or
        # if it is an error in the CBF. 2015-09-30 update: Nope, Digitiser simulator seems to be fine,
        # only the adc bit is set in the SPEAD header, checked many packets by network
        # packet capture.
        def enable_adc_overflow():
            self.dhost.registers.flag_setup.write(adc_flag=1, load_flags='pulse')

        def disable_adc_overflow():
            self.dhost.registers.flag_setup.write(adc_flag=0, load_flags='pulse')

        condition = 'ADC overflow flag on the digitiser simulator'
        try:
            dump1, dump2, dump3, = self.get_flag_dumps(enable_adc_overflow,
                                                       disable_adc_overflow, condition)
        except TypeError:
            Aqf.failed('Failed to retrieve adc overflow flags from spead accumulations')
        else:
            flag_bit = xeng_raw_bits_flags.overrange
            # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
            all_bits = set(xeng_raw_bits_flags)
            other_bits = all_bits - set([flag_bit])
            flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
            # flag_condition = 'ADC overrange'

            set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value, consider_bits=all_bits)
            Aqf.is_false(flag_bit in set_bits1,
                         'Check that {} is not set in SPEAD accumulation 1 before setting {}.'
                         .format(flag_descr, condition))
            # Bits that should not be set
            other_set_bits1 = set_bits1.intersection(other_bits)
            Aqf.equals(other_set_bits1, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

            set_bits2 = get_set_bits(dump2['flags_xeng_raw'].value, consider_bits=all_bits)
            other_set_bits2 = set_bits2.intersection(other_bits)
            Aqf.is_true(flag_bit in set_bits2,
                        'Check that {} is set in SPEAD accumulation 2 while toggeling {}.'
                        .format(flag_descr, condition))
            Aqf.equals(other_set_bits2, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

            set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value, consider_bits=all_bits)
            other_set_bits3 = set_bits3.intersection(other_bits)
            Aqf.is_false(flag_bit in set_bits3,
                         'Check that {} is not set in SPEAD accumulation 3 after clearing {}.'
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
        try:
            dump1, dump2, dump3, = self.get_flag_dumps(
                enable_noise_diode, disable_noise_diode, condition)
        except TypeError:
            Aqf.failed('Failed to retrieve  noise diode flags in the spead accumulations')
        else:
            flag_bit = xeng_raw_bits_flags.noise_diode
            # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
            all_bits = set(xeng_raw_bits_flags)
            other_bits = all_bits - set([flag_bit])
            flag_descr = 'noise diode fired, bit {},'.format(flag_bit)
            # flag_condition = 'digitiser noise diode fired flag'

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
                        'Check that {} is set in SPEAD accumulation 2 while toggeling {}.'
                        .format(flag_descr, condition))

            Aqf.equals(other_set_bits2, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

            set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                     consider_bits=all_bits)
            other_set_bits3 = set_bits3.intersection(other_bits)
            Aqf.is_false(flag_bit in set_bits3,
                         'Check that {} is not set in SPEAD accumulation 3 after clearing {}.'
                         .format(flag_descr, condition))

            Aqf.equals(other_set_bits3, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

    def _test_fft_overflow_flag(self):
        """CBF flagging of data -- FFT overflow"""
        freq = self.corr_freqs.bandwidth / 2.

        def enable_fft_overflow():
            # source that results in it producing all zeros... So using sin_0 and sin_1
            # instead
            # self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=1.)
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=1.)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=1.)
            # Set FFT to never shift, ensuring an FFT overflow with the large tone we are
            # putting in.
            try:
                self.correlator.fops.set_fft_shift_all(shift_value=0)
            except Exception:
                Aqf.failed('Failed to set FFT shift to all hosts')

        def disable_fft_overflow():
            # source that results in it producing all zeros... So using sin_0 and sin_1
            # instead
            # self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=0)
            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.)
            self.dhost.sine_sources.sin_1.set(frequency=freq, scale=0.)
            # Restore the default FFT shifts as per the correlator config.
            try:
                self.correlator.fops.set_fft_shift_all()
            except Exception:
                Aqf.failed('Failed to set FFT shift to all hosts')

        condition = ('FFT overflow by setting an aggressive FFT shift with '
                     'a pure tone input')
        try:
            dump1, dump2, dump3, = self.get_flag_dumps(enable_fft_overflow, disable_fft_overflow,
                                                       condition)
        except Exception:
            Aqf.failed('Failed to retrieve flagged SPEAD accumulations.')
        else:
            flag_bit = xeng_raw_bits_flags.overrange
            # All the non-debug bits, ie. all the bitfields listed in flags_xeng_raw_bit
            all_bits = set(xeng_raw_bits_flags)
            other_bits = all_bits - set([flag_bit])
            flag_descr = 'overrange in data path, bit {},'.format(flag_bit)
            # flag_condition = 'FFT overrange'

            set_bits1 = get_set_bits(dump1['flags_xeng_raw'].value,
                                     consider_bits=all_bits)
            Aqf.is_false(flag_bit in set_bits1,
                         'Check that {} is not set in SPEAD accumulation 1 before setting {}.'
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
                        'Check that {} is set in SPEAD accumulation 2 while toggling {}.'
                        .format(flag_descr, condition))
            Aqf.equals(other_set_bits2, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

            set_bits3 = get_set_bits(dump3['flags_xeng_raw'].value,
                                     consider_bits=all_bits)
            other_set_bits3 = set_bits3.intersection(other_bits)
            Aqf.is_false(flag_bit in set_bits3,
                         'Check that {} is not set in SPEAD accumulation 3 after clearing {}.'
                         .format(flag_descr, condition))

            Aqf.equals(other_set_bits3, set(),
                       'Check that no other flag bits (any of {}) are set.'
                       .format(sorted(other_bits)))

    def _test_delay_rate(self):
        """CBF Delay Compensation/LO Fringe stopping polynomial -- Delay Rate"""

        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            # delay_rate = ((setup_data['sample_period'] / setup_data['int_time']) *
            # np.random.rand() * (dump_counts - 3))
            # delay_rate = 3.98195128768e-09
            delay_rate = (0.7 * (setup_data['sample_period'] / setup_data['int_time']))
            delay_value = 0
            fringe_offset = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
            delay_rates = [0] * setup_data['num_inputs']
            delay_rates[setup_data['test_source_ind']] = delay_rate
            delay_coefficients = ['0,{}:0,0'.format(fr) for fr in delay_rates]
            Aqf.step('Setting Parameters')
            Aqf.step('Time apply: {}'.format(load_time))
            Aqf.step('[CBF-REQ-0185] Delay Rate: {}'.format(delay_rate))
            Aqf.step('[CBF-REQ-0185] Delay Value: {}'.format(delay_value))
            Aqf.step('[CBF-REQ-0112] Fringe Offset: {}'.format(fringe_offset))
            Aqf.step('[CBF-REQ-0112] Fringe Rate: {}'.format(fringe_rate))

            actual_data, _delay_coefficients = self._get_actual_data(
                setup_data, dump_counts, delay_coefficients)
            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]

            if _delay_coefficients is not None:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          _delay_coefficients, actual_phases)
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          delay_coefficients, actual_phases)

            no_chans = range(self.corr_freqs.n_chans)
            plot_units = 'ns/s'
            plot_title = 'Randomly generated delay rate {} {}'.format(delay_rate * 1e9,
                                                                      plot_units)
            plot_filename = '{}_phase_response.png'.format(self._testMethodName)
            caption = ('Actual vs Expected Unwrapped Correlation Phase [Delay Rate].\n'
                       'Note: Dashed line indicates expected value and solid line indicates '
                       'actual values received from SPEAD accumulation.')

            aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                   plot_filename, plot_title, plot_units, caption,
                                   dump_counts)

            if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                Aqf.failed('Delays could not be applied at time_apply: {} '
                           'is in the past'.format(setup_data['t_apply']))
            else:
                actual_phases_ = np.unwrap(actual_phases)
                degree = 1.0
                radians = (degree / 360) * np.pi * 2
                decimal = len(str(degree).split('.')[-1])
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                for i in xrange(0, len(expected_phases_) - 1):
                    delta_expected = np.abs(np.max(expected_phases_[i + 1] - expected_phases_[i]))
                    delta_actual = np.abs(np.max(actual_phases_[i + 1] - actual_phases_[i]))
                    # abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    abs_diff = np.abs(delta_expected - delta_actual)
                    msg = (
                        '[CBF-REQ-0187] Check if difference (radians) between expected({0:.3f}) '
                        'phases and actual({1:.3f}) phases are \'Almost Equal\' '
                        'within {2} degree when delay rate of {3} is applied.'.format(
                            delta_expected, delta_actual, degree, delay_rate))
                    Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                    msg = (
                        '[CBF-REQ-0187] Check that the maximum difference ({0:.3f} '
                        'degree/{1:.3f} rad) between expected phase and actual phase '
                        'between integrations is less than {2} degree.'.format(
                            np.rad2deg(abs_diff), abs_diff, degree))
                    Aqf.less(abs_diff, radians, msg)

                    abs_error = np.max(actual_phases_[i] - expected_phases_[i])
                    msg = (
                        '[CBF-REQ-0187] Check that the absolute maximum difference ({0:.3f} '
                        'degree/{1:.3f} rad) between expected phase and actual phase '
                        'is less than {2} degree.'.format(
                            np.rad2deg(abs_error), abs_error, degree))
                    Aqf.less(abs_error, radians, msg)

                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s,
                                                       delta_expected_s,
                                                       decimal=decimal)

                    except AssertionError:
                        Aqf.step(
                            '[CBF-REQ-0187] Difference  between expected({0:.3f}) '
                            'phases and actual({1:.3f}) phases are '
                            '\'Not almost equal\' within {2} degree when delay rate '
                            'of {3} is applied.'.format(delta_expected, delta_actual,
                                                        degree, delay_rate))
                        caption = ('[CBF-REQ-0128] Difference  expected({0:.3f}) and actual({1:.3f})'
                                   ' phases are not equal within {2} degree when delay rate of {3} '
                                   'is applied.'.format(delta_expected, delta_actual, degree,
                                                        delay_rate))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans, actual_phases_i, expected_phases_i,
                            plot_filename='{}_{}_phase_resp.png'.format(
                                self._testMethodName, i),
                            plot_title='Delay Rate:\nActual vs Expected Phase Response',
                            plot_units=plot_units, caption=caption, )
        who_ran_test()

    def _test_fringe_rate(self):
        """CBF per-antenna phase error -- Fringe rate"""

        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            fringe_rate = (((np.pi / 8.) / setup_data['int_time']) * np.random.rand() * dump_counts)
            delay_value = 0
            delay_rate = 0
            fringe_offset = 0
            load_time = setup_data['t_apply']
            fringe_rates = [0] * setup_data['num_inputs']
            fringe_rates[setup_data['test_source_ind']] = fringe_rate
            delay_coefficients = ['0,0:0,{}'.format(fr) for fr in fringe_rates]

            Aqf.step('Setting Parameters')
            Aqf.step('Time apply: {}'.format(load_time))
            Aqf.step('[CBF-REQ-0185] Delay Rate: {}'.format(delay_rate))
            Aqf.step('[CBF-REQ-0185] Delay Value: {}'.format(delay_value))
            Aqf.step('[CBF-REQ-0112] Fringe Offset: {}'.format(fringe_offset))
            Aqf.step('[CBF-REQ-0112] Fringe Rate: {}'.format(fringe_rate))

            actual_data, _delay_coefficients = self._get_actual_data(
                setup_data, dump_counts, delay_coefficients)

            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]

            if _delay_coefficients is not None:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          _delay_coefficients, actual_phases)
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          delay_coefficients, actual_phases)

            if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                Aqf.failed('Delays could not be applied at time_apply: {} '
                           'is in the past'.format(setup_data['t_apply']))
            else:
                no_chans = range(self.corr_freqs.n_chans)
                plot_units = 'rads/sec'
                plot_title = 'Randomly generated fringe rate {} {}'.format(fringe_rate,
                                                                           plot_units)
                plot_filename = '{}_phase_response.png'.format(self._testMethodName)
                caption = ('Actual vs Expected Unwrapped Correlation Phase [Fringe Rate].\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation.')

                aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

                degree = 1.0
                radians = (degree / 360) * np.pi * 2
                decimal = len(str(degree).split('.')[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])

                for i in xrange(0, len(expected_phases_) - 1):
                    delta_expected = np.max(expected_phases_[i + 1] - expected_phases_[i])
                    delta_actual = np.max(actual_phases_[i + 1] - actual_phases_[i])
                    abs_diff = np.abs(delta_expected - delta_actual)
                    # abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    msg = (
                        '[CBF-REQ-0128] Check if difference between expected({0:.3f}) '
                        'phases and actual({1:.3f}) phases are \'Almost Equal\' within '
                        '{2} degree when fringe rate of {3} is applied.'.format(
                            delta_expected, delta_actual, degree, fringe_rate))
                    Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                    msg = (
                        '[CBF-REQ-0128] Check that the maximum difference ({0:.3f} '
                        'deg / {1:.3f} rad) between expected phase and actual phase '
                        'between integrations is less than {2} degree\n'.format(
                            np.rad2deg(abs_diff), abs_diff, degree))
                    Aqf.less(abs_diff, radians, msg)

                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                       decimal=decimal)
                    except AssertionError:
                        Aqf.step(
                            '[CBF-REQ-0128] Difference between expected({0:.3f}) '
                            'phases and actual({1:.3f}) phases are '
                            '\'Not almost equal\' within {2} degree when fringe rate '
                            'of {3} is applied.'.format(delta_expected, delta_actual,
                                                        degree, fringe_rate))

                        caption = ('[CBF-REQ-0128] Difference expected({0:.3f}) and '
                                   'actual({1:.3f}) phases are not equal within {2} degree when '
                                   'fringe rate of {3} is applied.'.format(delta_expected,
                                                                           delta_actual, degree,
                                                                           fringe_rate))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])

                        aqf_plot_phase_results(
                            no_chans, actual_phases_i, expected_phases_i,
                            plot_filename='{}_{}_phase_resp.png'.format(
                                self._testMethodName, i),
                            plot_title='Fringe Rate: Actual vs Expected Phase Response',
                            plot_units=plot_units, caption=caption, )
        who_ran_test()

    def _test_fringe_offset(self):
        """CBF per-antenna phase error -- Fringe offset"""
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            # fringe_offset = (np.pi / 2.) * np.random.rand() * dump_counts
            fringe_offset = 1.22796022444
            delay_value = 0
            delay_rate = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
            fringe_offsets = [0] * setup_data['num_inputs']
            fringe_offsets[setup_data['test_source_ind']] = fringe_offset
            delay_coefficients = ['0,0:{},0'.format(fo) for fo in fringe_offsets]

            Aqf.step('Setting Parameters')
            Aqf.step('Time apply: {}'.format(load_time))
            Aqf.step('[CBF-REQ-0185] Delay Rate: {}'.format(delay_rate))
            Aqf.step('[CBF-REQ-0185] Delay Value: {}'.format(delay_value))
            Aqf.step('[CBF-REQ-0112] Fringe Offset: {}'.format(fringe_offset))
            Aqf.step('[CBF-REQ-0112] Fringe Rate: {}'.format(fringe_rate))

            actual_data, _delay_coefficients = self._get_actual_data(
                setup_data, dump_counts, delay_coefficients)
            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]
            if _delay_coefficients is not None:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          _delay_coefficients, actual_phases)
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          delay_coefficients, actual_phases)

            if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                Aqf.failed('Delays could not be applied at time_apply: {} '
                           'is in the past'.format(setup_data['t_apply']))
            else:
                no_chans = range(self.corr_freqs.n_chans)
                plot_units = 'rads'
                plot_title = 'Randomly generated fringe offset {0:.3f} {1}'.format(
                    fringe_offset, plot_units)
                plot_filename = '{}_phase_response.png'.format(self._testMethodName)
                caption = ('Actual vs Expected Unwrapped Correlation Phase [Fringe Offset].\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation. '
                           'Values are rounded off to 3 decimals places')

                aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

                # Ignoring first dump because the delays might not be set for full
                # integration.
                degree = 1.0
                decimal = len(str(degree).split('.')[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])

                for i in xrange(1, len(expected_phases) - 1):
                    delta_expected = np.abs(np.max(expected_phases_[i]))
                    delta_actual = np.abs(np.max(actual_phases_[i]))
                    # abs_diff = np.abs(delta_expected - delta_actual)
                    abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    msg = (
                        '[CBF-REQ-0128] Check if difference between expected({0:.3f})'
                        ' phases and actual({1:.3f}) phases are \'Almost Equal\' '
                        'within {2:.3f} degree when fringe offset of {3:.3f} is '
                        'applied.'.format(delta_expected, delta_actual, degree,
                                          fringe_offset))

                    Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                    Aqf.less(abs_diff, degree,
                             '[CBF-REQ-0128] Check that the maximum difference({0:.3f} '
                             'degrees/{1:.3f}rads) between expected phase and actual phase '
                             'between integrations is less than {2:.3f} degree\n'.format(
                                 abs_diff,
                                 np.deg2rad(abs_diff),
                                 degree))
                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s,
                                                       delta_expected_s,
                                                       decimal=decimal)

                    except AssertionError:
                        Aqf.step(
                            '[CBF-REQ-0128] Difference between expected({0:.5f}) phases '
                            'and actual({1:.5f}) phases are \'Not almost equal\' '
                            'within {2} degree when fringe offset of {3} is applied.'
                                .format(delta_expected, delta_actual, degree,
                                        fringe_offset))

                        caption = ('[CBF-REQ-0128] Difference expected({0:.3f}) and actual({1:.3f}) '
                                   'phases are not equal within {2:.3f} degree when fringe offset '
                                   'of {3:.3f} {4} is applied.'.format(delta_expected, delta_actual,
                                                                       degree, fringe_offset,
                                                                       plot_units))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans, actual_phases_i, expected_phases_i,
                            plot_filename='{}_{}_phase_resp.png'.format(
                                self._testMethodName, i),
                            plot_title=('Fringe Offset:\nActual vs Expected Phase Response'),
                            plot_units=plot_units, caption=caption, )
        who_ran_test()

    def _test_all_delays(self):
        """
        CBF per-antenna phase error --
        Delays, Delay Rate, Fringe Offset and Fringe Rate.
        """
        setup_data = self._delays_setup()
        if setup_data:
            # get_ranges = get_delay_bounds(self.correlator)
            dump_counts = 5
            delay_value = (self.corr_freqs.sample_period * np.random.rand() *
                           dump_counts)
            delay_rate = ((setup_data['sample_period'] / setup_data['int_time']) *
                          np.random.rand() * dump_counts)
            fringe_offset = (np.pi / 2.) * np.random.rand() * dump_counts
            fringe_rate = (((np.pi / 8.) / setup_data['int_time']) *
                           np.random.rand() * dump_counts)

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
                delay_coefficients.append('{},{}:{},{}'.format(delay_values[idx],
                                                               delay_rates[idx],
                                                               fringe_offsets[idx],
                                                               fringe_rates[idx]))

            Aqf.step('Setting Parameters')
            Aqf.step('Time apply: {}'.format(load_time))
            Aqf.step('[CBF-REQ-0185] Delay Rate: {}'.format(delay_rate))
            Aqf.step('[CBF-REQ-0185] Delay Value: {}'.format(delay_value))
            Aqf.step('[CBF-REQ-0112] Fringe Offset: {}'.format(fringe_offset))
            Aqf.step('[CBF-REQ-0112] Fringe Rate: {}'.format(fringe_rate))

            actual_data, delay_coefficients = self._get_actual_data(
                setup_data, dump_counts, delay_coefficients)
            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]

            expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                      delay_coefficients,
                                                      actual_phases)

            if set([float(0)]) in [set(i) for i in actual_phases]:
                Aqf.failed('Delays could not be applied at time_apply: {} '
                           'is in the past'.format(setup_data['t_apply']))
            else:
                no_chans = range(self.corr_freqs.n_chans)
                plot_units = ''
                plot_title = ('Randomly generated \n delay offset: {}, delay rate: {}, '
                              '\nfringe rate: {}, fringe offset: {}rads'.format(
                    delay_value, delay_rate, fringe_offset, fringe_rate))
                plot_filename = '{}_phase_response.png'.format(self._testMethodName)
                caption = ('Actual vs Expected Unwrapped Correlation Phase.\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation.')

                aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

                # Ignoring first dump because the delays might not be set for full
                # integration.
                degree = 1.0
                actual_phases = np.unwrap(actual_phases)
                expected_phases = np.unwrap([phase for label, phase in expected_phases])
        who_ran_test()
        # (MM) 14-07-2016
        # Still need more work in here

    def _test_config_report(self, verbose):
        """CBF Report configuration"""
        test_config = self.corr_fix._test_config_file

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
                Aqf.hop('Host {}: {}'.format(count, hostname.upper()))
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

                    uboot_ver = stdout.splitlines()[-6]
                    Aqf.hop('Current UBoot Version: {}'.format(uboot_ver))

                    romfs_ver = stdout.splitlines()[-4]
                    Aqf.hop('Current ROMFS Version: {}'.format(romfs_ver))

                    linux_ver = stdout.splitlines()[-2]
                    Aqf.hop('Linux Version: {}\n'.format(linux_ver))
                except:
                    errmsg = 'Could not connect to host: %s\n' % (hostname)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

        def get_src_dir():

            import corr2
            import casperfpga
            import katcp
            import spead2

            corr2_dir, _None = os.path.split(os.path.split(corr2.__file__)[0])
            corr2_name = corr2.__name__
            corr2_version = corr2.__version__

            casper_dir, _None = os.path.split(os.path.split(casperfpga.__file__)[0])
            casper_name = casperfpga.__name__
            casper_version = casperfpga.__version__

            katcp_dir, _None = os.path.split(os.path.split(katcp.__file__)[0])
            katcp_name = katcp.__name__
            katcp_version = katcp.__version__

            spead2_dir, _None = os.path.split(os.path.split(spead2.__file__)[0])
            spead2_name = spead2.__name__
            spead2_version = spead2.__version__

            try:
                bitstream_dir = self.correlator.configd['xengine']['bitstream']
            except AttributeError:
                mkat_name = None
                mkat_dir = None
                Aqf.failed('Failed to retrieve mkat_fpga info')
            else:
                mkat_dir, _None = os.path.split(os.path.split(os.path.dirname(
                    os.path.realpath(bitstream_dir)))[0])
                _None, mkat_name = os.path.split(mkat_dir)

            test_dir, test_name = os.path.split(os.path.dirname(
                os.path.realpath(__file__)))

            return {
                corr2_name: corr2_dir,
                casper_name: casper_dir,
                katcp_name: katcp_dir,
                spead2_name: spead2_dir,
                mkat_name: mkat_dir,
                test_name: test_dir
            }

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

                    Aqf.hop('Repo: {}, Git Dir: {}, Branch: {}, Last Hash: {}'
                            .format(name, repo_dir, git_branch, git_hash))

                    git_diff = subprocess.check_output(
                        ['git', '--git-dir={}/.git'.format(repo_dir),
                         '--work-tree={}'.format(repo_dir), 'diff', 'HEAD'])
                    if bool(git_diff):
                        if verbose:
                            Aqf.progress('Repo: {}: Contains changes not staged for commit.\n\n'
                                         'Difference: \n{}'.format(name, git_diff))
                        else:
                            Aqf.progress('Repo: {}: Contains changes not staged for'
                                         ' commit.\n'.format(name))
                    else:
                        Aqf.hop('Repo: {}: Up-to-date.\n'.format(name))
                except subprocess.CalledProcessError:
                    Aqf.hop('Repo: {}, Branch: Dirty, Last Hash: Dirty\n'.format(name))
                except AssertionError:
                    Aqf.failed('AssertionError occured while retrieving git repo: {}\n'.format(name))
                except OSError:
                    Aqf.failed('OS Error occured while retrieving gut repo: {}\n'.format(name))

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
                    errmsg = ('Could not connect to PDU host: %s \n' % (host_ip))
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
                    errmsg = (
                        'Failed to connect to Data switch %s on IP: %s\n' % (count, ip))
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

        def get_gateware_info(self):
            try:
                reply, informs = self.corr_fix.katcp_rct.req.version_list()
            except:
                errmsg = ('Could not retrieve version infomation.\n')
                Aqf.failed(errmsg)
            else:
                if not reply.reply_ok():
                    return False
                else:
                    version_lists = [str(i).split() for i in informs
                                     if str(i).startswith('#version-list')]
                    gateware_info = []
                    for version_list in version_lists:
                        for version in version_list:
                            if version.startswith('gateware.'):
                                gateware_info.append(version_list[1:])
                            else:
                                pass
                    if len(gateware_info) > 5:
                        return gateware_info[:4]
                    else:
                        return gateware_info

        def get_katcp_version(self):
            try:
                reply, informs = self.corr_fix.katcp_rct.req.version_list()
            except:
                errmsg = ('Could not retrieve version infomation.\n')
                Aqf.failed(errmsg)
            else:
                if not reply.reply_ok():
                    return False
                else:
                    version_lists = [str(i).split() for i in informs
                                     if str(i).startswith('#version-list')]
                    katcp_lib_lst = []
                    katcp_dev_lst = []
                    for version_list in version_lists:
                        for version in version_list:
                            if version == 'katcp-device':
                                katcp_dev_lst.append(version_list)
                            if version == 'katcp-library':
                                katcp_lib_lst.append(version_list)

                    katcp_dev = [i[-1] for i in katcp_dev_lst][-1]
                    katcp_lib = [i[-1] for i in katcp_lib_lst][-1]
                    return [katcp_dev, katcp_lib]

        Aqf.step('CMC CBF Package Software version information.')
        katcp_versions = get_katcp_version(self)
        if len(katcp_versions) == 2 and not False:
            Aqf.hop('Repo: katcp-device, Version info: {}'.format(katcp_versions[0]))
            Aqf.hop('Repo: katcp-library, Version info: {}\n'.format(katcp_versions[1]))
        else:
            msg = 'Failed to retrieve KATCP-Device and KATCP-Library version infomations.'
            Aqf.failed(msg)

        Aqf.step('CBF Gateware Software Version Information.')
        gateware_info = get_gateware_info(self)
        if gateware_info is not False:
            for i, v in gateware_info:
                Aqf.hop('Gateware: {}, Git Hash: {}'.format(
                    ' '.join(i.split('.')[1:]), v))

        Aqf.step('CBF Git Version Information.\n')
        get_package_versions()
        Aqf.step('CBF ROACH version information.\n')
        get_roach_config()

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
                    # Aqf.is_false(overtemp_ind,
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
                    # Aqf.is_true(undertemp_ind,
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
                    # Aqf.is_true(overtemp_ind,
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
                    # Aqf.is_false(undertemp_ind,
                    Aqf.passed('Confirm that the undertemp alarm is Triggered.')
                except ValueError:
                    Aqf.failed('Failed to read undertemp alarm on {}.'.format(hostname))

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
                    # Aqf.is_false(overtemp_ind,
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
                errmsg = ('Could not connect to host: %s' % (hostname))
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
        setup_data = self._delays_setup()
        if setup_data:
            sampling_period = self.corr_freqs.sample_period
            no_chans = range(len(self.corr_freqs.chan_freqs))
            test_delay = sampling_period
            expected_phases = self.corr_freqs.chan_freqs * 2 * np.pi * test_delay
            expected_phases -= np.max(expected_phases) / 2.

            test_source_idx = 2
            source_names = get_local_src_names(self)
            # (MM) 2016-07-12
            # Disabled source name randomisation due to the fact that some roach boards
            # are known to have QDR issues which results to test failures, hence
            # input1 has been statically assigned to be the testing input
            # delayed_input = source_names[randrange(len(source_names))]
            delayed_input = source_names[1]
            try:
                last_pfb_counts = get_pfb_counts(
                    get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
            except Exception:
                Aqf.failed('Failed to retrieve last PFB error counts')

            delays = [0] * setup_data['num_inputs']
            # Get index for input to delay
            test_source_idx = source_names.index(delayed_input)
            Aqf.step('Delayed selected input to test: {}'.format(delayed_input))
            delays[test_source_idx] = test_delay
            delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]

            settling_time = 600e-3
            try:
                reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch()
                if not reply.reply_ok():
                    raise Exception
            except Exception:
                Aqf.failed('Could not retrieve sync time via CAM Interface.')
                return False
            else:
                sync_time = float(reply.arguments[-1])

            try:
                this_freq_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT, discard=0)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                dump_timestamp = (sync_time +
                                  this_freq_dump['timestamp'].value /
                                  this_freq_dump['scale_factor_timestamp'].value)
                t_apply = (dump_timestamp + 10 * this_freq_dump['int_time'].value)

                reply = self.corr_fix.katcp_rct.req.delays(t_apply,
                                                           *delay_coefficients)
                msg = ('[CBF-REQ-0066, 0072]: Delays set to input: {} via CAM interface '
                       'and reply: {}'.format(delayed_input, reply.reply.arguments[1]))
                Aqf.is_true(reply.reply.reply_ok(), msg)

                Aqf.wait(settling_time, 'Settling time in order to set delay: {} ns.'.format(
                    test_delay * 1e9))

                Aqf.step('[CBF-REQ-0067] Check FFT overflow and QDR errors after channelisation.')

            try:
                QDR_error_roaches = check_fftoverflow_qdrstatus(self.correlator,
                                                                last_pfb_counts)
            except Exception:
                Aqf.failed('Failed to retrieve last PFB error counts')

            try:
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                baselines = get_baselines_lookup(this_freq_dump)
                sorted_bls = sorted(baselines.items(), key=operator.itemgetter(1))
                chan_response = []
                degree = 1.0
                for b_line in sorted_bls:
                    b_line_val = b_line[1]
                    b_line_dump = (dump['xeng_raw'].value[:, b_line_val, :])
                    b_line_cplx_data = complexise(b_line_dump)
                    b_line_phase = np.angle(b_line_cplx_data)
                    b_line_phase_max = np.max(b_line_phase)
                    if ((delayed_input in b_line[0]) and
                                b_line[0] != (delayed_input, delayed_input)):
                        msg = ('[CBF-REQ-0187] Confirm that baseline(s) {0} have '
                               'expected a delay within 1 degree.'.format(b_line[0]))
                        Aqf.array_abs_error(
                            np.abs(b_line_phase[1:-1]), np.abs(expected_phases[1:-1]),
                            msg, degree)

                    else:
                        if b_line_phase_max != 0:
                            desc = ('Checking baseline {0}, index = {1:02d}, '
                                    'phase offset found, maximum value = {0:0.8f}'.format(
                                b_line[0], b_line_val, b_line_phase_max))
                            Aqf.failed(desc)
                            chan_response.append(normalised_magnitude(b_line_dump))

            if QDR_error_roaches:
                Aqf.failed('The following roaches contains QDR errors: {}'.format(
                    set(QDR_error_roaches)))

            if chan_response:
                Aqf.step('Delay applied to the correct input')
                legends = ['SPEAD accumulations per Baseline #{}'.format(x)
                           for x in xrange(len(chan_response))]
                aqf_plot_channels(zip(chan_response, legends),
                                  plot_filename='{}_chan_resp.png'.format(
                                      self._testMethodName),
                                  plot_title='Channel Response Phase Offsets Found',
                                  log_dynamic_range=90, log_normalise_to=1,
                                  caption='Delay applied to the correct input')
        who_ran_test()

    def _test_data_product(self, instrument, no_channels):
        """CBF Imaging Data Product Set"""
        # Put some correlated noise on both outputs
        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise with scale: {}, '
                 'gain: {} and fft shift: {}.'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        Aqf.step('Retrieving initial SPEAD accumulation')
        try:
            test_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        except ValueError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)

        else:
            # Get baseline 0 data, i.e. auto-corr of m000h
            test_baseline = 0
            test_bls = tuple(test_dump['bls_ordering'].value[test_baseline])

            Aqf.equals(test_dump['xeng_raw'].value.shape[0], no_channels,
                       '[CBF-REQ-0104, 0213] Check that data product has the number of frequency '
                       'channels {no_channels} corresponding to the {instrument} '
                       'instrument product.\n'.format(**locals()))
            response = normalised_magnitude(test_dump['xeng_raw'].value[:, test_baseline, :])

            if response.shape[0] == no_channels:
                Aqf.passed('Confirm that imaging data product set has been '
                           'implemented for instrument: {}.'.format(instrument))
                plot_filename = '{}.png'.format(self._testMethodName)

                caption = ('An overrall frequency response at {} baseline, '
                           'when digitiser simulator is configured to generate gaussian noise, '
                           'with awgn scale: {}, eq gain: {} and fft shift: {}'.format(test_bls,
                                                                                       awgn_scale,
                                                                                       gain,
                                                                                       fft_shift))

                aqf_plot_channels(response, plot_filename, log_dynamic_range=90,
                                  caption=caption)
            else:
                Aqf.failed('Imaging data product set has not been implemented.')
        local_src_names = get_local_src_names(self)
        try:
            running_instrument = self.corr_fix.get_running_instrument().keys()[0]
        except Exception:
            Aqf.failed('Could not retrieve running instrument.')
        else:
            if running_instrument.endswith('4k'):

                local_src_names = get_local_src_names(self)

                try:
                    reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0x')
                    reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0y')
                    reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
                    if reply.reply_ok():
                        labels = reply.arguments[1:]
                    else:
                        raise Exception
                except Exception, e:
                    Aqf.failed(e)
                    return
                bw = self.corr_freqs.bandwidth
                nr_ch = self.corr_freqs.n_chans
                dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq

                # Start of test. Setting required partitions and center frequency
                partitions = 2
                part_size = bw / 16
                target_cfreq = bw + bw * 0.5
                target_pb = partitions * part_size
                ch_bw = bw / nr_ch
                beams = ('beam_0x', 'beam_0y')
                beam = beams[1]

                # Set beamformer quantiser gain for selected beam to 1
                self._set_beam_quant_gain(beam, 1)

                beam_dict = {}
                beam_pol = beam[-1]
                for label in labels:
                    if label.find(beam_pol) != -1:
                        beam_dict[label] = 0.0

                # Only one antenna gain is set to 1, this will be used as the reference
                # input level
                weight = 1.0
                beam_dict = self._populate_beam_dict(1, weight, beam_dict)
                bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = self._capture_beam_data(beam,
                                                                                 beam_dict, target_pb, target_cfreq)
                nc = 10000
                cap = [0] * nc
                for i in range(0, nc):
                    cap[i] = np.array(complexise(bf_raw[:, i, :]))
                cap_mag = np.abs(cap)
                cap_avg = cap_mag.sum(axis=0) / nc
                # Confirm that the beam channel bandwidth corresponds to the channel bandwidth
                # determined from the baseline capture
                baseline_ch_bw = bw * dsim_clk_factor / response.shape[0]
                beam_ch_bw = pb / len(cap_mag[0])
                Aqf.almost_equals(baseline_ch_bw, beam_ch_bw, 1e-3,
                                  '[CBF-REQ-0120] Confirm Baseline Correlation Product channel width {}Hz '
                                  'is the same as the Tied Array Beam channel width {}Hz'.format(
                                      baseline_ch_bw, beam_ch_bw))
                # Square the voltage data. This is a hack as aqf_plot expects squared
                # power data
                aqf_plot_channels(np.square(cap_avg),
                                  plot_filename='{}_beam_resp_{}.png'.format(self._testMethodName, beam),
                                  plot_title=('Beam = {}, Passband = {} MHz\nCenter Frequency = {} MHz'
                                              '\nIntegrated over {} captures'.format(beam, pb / 1e6, cf / 1e6, nc)),
                                  log_dynamic_range=90, log_normalise_to=1,
                                  caption=('Tied Array Beamformer data captured during Baseline Correlation '
                                           'Product test.'), plot_type='bf')
        who_ran_test()

    def _test_control_init(self):
        """
        CBF Control
        Test Varifies:
            [CBF-REQ-0071]
        """
        Aqf.passed('List of available commands\n{}'.format(
            self.corr_fix.katcp_rct.req.help()))
        Aqf.progress(
            '[CBF-REQ-011, 0114] Downconversion frequency has not been implemented '
            'in this release.')
        Aqf.progress(
            '[CBF-REQ-011, 0114] CBF Polarisation Correction has not been implemented '
            'in this release.')
        Aqf.is_true(self.corr_fix.katcp_rct.req.gain.is_active(),
                    '[CBF-REQ-0071] Re-quantiser settings (Gain) and Complex gain correction has '
                    'been implemented')
        Aqf.is_true(self.corr_fix.katcp_rct.req.accumulation_length.is_active(),
                    '[CBF-REQ-0071] Accumulation interval has been implemented')
        Aqf.is_true(self.corr_fix.katcp_rct.req.frequency_select.is_active(),
                    '[CBF-REQ-0071] Channelisation configuration has been implemented')
        reply, informs = self.corr_fix.katcp_rct.req.accumulation_length()

        msg = ('Readback of the last programmed value. Reply: {}'.format(str(reply)))
        Aqf.is_true(reply.reply_ok(), msg)

    def _test_time_sync(self):
        Aqf.step('CBF Absolute Timing Accuracy.')
        try:
            host_ip = '192.168.194.2'
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        except ntplib.NTPException:
            host_ip = '192.168.1.21'
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        req_sync_time = 5e-3
        msg = ('[CBF-REQ-0203] Confirm that the CBF synchronised time is within {}s of '
               'UTC time as provided via PTP (NTP server: {}) on the CBF-TRF '
               'interface.'.format(req_sync_time, host_ip))
        Aqf.less(ntp_offset, req_sync_time, msg)

    def _test_gain_correction(self):
        """CBF Gain Correction"""
        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.0645
            gain = 113
            gain_str = complex(gain)
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = 344
            gain_str = complex(gain)
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise, '
                 'with awgn scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, fft_shift=fft_shift, gain=gain_str)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitiser simulator levels')
            return False

        self.addCleanup(set_default_eq, self)
        source = randrange(len(self.correlator.fengine_sources))

        try:
            initial_dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        except ValueError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:

            test_input = [input_source[0]
                          for input_source in initial_dump['input_labelling'].value][source]
            Aqf.step('Randomly selected input {}'.format(test_input))

            # Get auto correlation index of the selected input
            bls_order = initial_dump['bls_ordering'].value
            for idx, val in enumerate(bls_order):
                if val[0] == test_input and val[1] == test_input:
                    auto_corr_idx = idx

            n_chans = self.corr_freqs.n_chans
            rand_ch = randrange(n_chans)
            gain_vector = [gain] * n_chans
            base_gain = gain
            initial_resp = np.abs(complexise(initial_dump['xeng_raw'].value[:, auto_corr_idx, :]))
            initial_resp = 10 * np.log10(initial_resp)
            chan_resp = []
            legends = []
            found = False
            fnd_less_one = False
            count = 0
            while not found:
                if not fnd_less_one:
                    target = 1
                    gain_inc = 5
                else:
                    target = 6
                    gain_inc = 150
                gain = gain + gain_inc
                gain_vector[rand_ch] = gain
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(test_input, *gain_vector)
                except TimeoutError:
                    msg = ('Could not set gains/eqs {} on input {} :CAM interface Timed-out, '.format(
                        gain, test_input))
                    Aqf.failed(msg)
                else:
                    msg = ('[CBF-REQ-0119] Gain correction on input {}, channel {} set to {}.'.format(
                        test_input, rand_ch, complex(gain)))
                    if reply.reply_ok():
                        Aqf.passed(msg)
                        try:
                            dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        except ValueError:
                            errmsg = ('Could not retrieve clean SPEAD accumulation, Item has too few '
                                      'elements for shape.')
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            response = np.abs(complexise(dump['xeng_raw'].value[:, auto_corr_idx, :]))
                            response = 10 * np.log10(response)
                            resp_diff = response[rand_ch] - initial_resp[rand_ch]
                            if resp_diff < target:
                                msg = ('[CBF-REQ-0119] Ouput power increased by less than 1 dB '
                                       '(actual = {:.2f} dB) with a gain '
                                       'increment of {}.'.format(resp_diff, complex(gain_inc)))
                                Aqf.passed(msg)
                                fnd_less_one = True
                                chan_resp.append(response)
                                legends.append('Gain set to {}'.format(complex(gain)))
                            elif fnd_less_one and (resp_diff > target):
                                msg = ('[CBF-REQ-0119] Ouput power increased by more than 6 dB '
                                       '(actual = {:.2f} dB) with a gain '
                                       'increment of {}.'.format(resp_diff, complex(gain_inc)))
                                Aqf.passed(msg)
                                found = True
                                chan_resp.append(response)
                                legends.append('Gain set to {}'.format(complex(gain)))
                    else:
                        Aqf.failed('Gain correction on {} could not be set to {}.: '
                                   'KATCP Reply: {}'.format(test_input, gain, reply))
                count += 1
                if count == 5:
                    Aqf.failed('Gains to change output power by less than 1 and more than 6 dB '
                               'could not be found.')
                    found = True

            if chan_resp != []:
                aqf_plot_channels(zip(chan_resp, legends),
                                  plot_filename='{}_chan_resp.png'.format(
                                      self._testMethodName),
                                  plot_title='Channel Response Gain Correction for channel {}'.format(
                                      rand_ch),
                                  log_dynamic_range=90, log_normalise_to=1,
                                  caption='Gain Correction channel response, gain varied for channel {}, '
                                          'all remaining channels are set to {}'.format(rand_ch,
                                                                                        complex(base_gain)))
            else:
                Aqf.failed('Could not retrieve channel response with gain/eq corrections.')
            who_ran_test()

    def _capture_beam_data(self, beam, beam_dict, target_pb, target_cfreq, capture_time=0.1):
        """ Capture beamformer data

        Parameters
        ----------
        beam (beam_0x, beam_0y):
            Polarisation to capture beam data
        beam_dict:
            Dictionary containing input:weight key pairs e.g.
            beam_dict = {'m000_x': 1.0, 'm000_y': 1.0}
        target_pb:
            Target passband in Hz
        target_cfreq:
            Target center frequency in Hz
        capture_time:
            Number of seconds to capture beam data

        Returns
        -------
            bf_raw:
                Raw beamformer data for the selected beam
            cap_ts:
                Captured timestamps, dropped packet timestamps will not be
                present
            bf_ts:
                Expected timestamps

        """
        dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq
        reply, informs = self.corr_fix.katcp_rct.req.beam_passband(
            beam, target_pb, target_cfreq)
        if reply.reply_ok():
            pb = float(reply.arguments[2]) * dsim_clk_factor
            cf = float(reply.arguments[3]) * dsim_clk_factor
            Aqf.step('Beam {0} passband set to {1} at center frequency {2}'
                     ''.format(reply.arguments[1], pb, cf))
        else:
            Aqf.failed('Beam passband not successfully set '
                       '(requested cf = {}, pb = {}): {}'
                       ''.format(target_cfreq, target_pb, reply.arguments))
        # Build new dictionary with only the requested beam keys:value pairs
        in_wgts = {}
        beam_pol = beam[-1]
        for key in beam_dict:
            if key.find(beam_pol) != -1:
                in_wgts[key] = beam_dict[key]

        for key in in_wgts:
            reply, informs = self.corr_fix.katcp_rct.req.beam_weights(
                beam, key, in_wgts[key])
            if reply.reply_ok():
                Aqf.step('Input {0} weight set to {1}'
                         ''.format(key, reply.arguments[1]))
            else:
                Aqf.failed('Beam weights not successfully set: {}'
                           ''.format(reply.arguments))

        ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node']
        ingst_nd_p = self.corr_fix._test_config_file['beamformer'] \
            ['ingest_node_port']
        p = Popen(
            ['kcpcmd', '-s', ingst_nd + ':' + ingst_nd_p, 'capture-init'],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            Aqf.failed(
                'Failure issuing capture-init to ingest process on ' + ingst_nd)
            Aqf.failed('Stdout: \n' + output)
            Aqf.failed('Stderr: \n' + err)
        else:
            Aqf.step('Capture-init issued on {}'.format(ingst_nd))
        reply, informs = self.corr_fix.katcp_rct.req.capture_meta(beam)
        if reply.reply_ok():
            Aqf.step('Meta data issued for beam {}'.format(beam))
        else:
            Aqf.failed('Meta data issue failed: {}'.format(reply.arguments))
        reply, informs = self.corr_fix.katcp_rct.req.capture_start(beam)
        if reply.reply_ok():
            Aqf.step('Data transmission for beam {} started'.format(beam))
        else:
            Aqf.failed(
                'Data transmission start failed: {}'.format(reply.arguments))
        time.sleep(capture_time)
        reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam)
        if reply.reply_ok():
            Aqf.step('Data transmission for beam {} stopped'.format(beam))
        else:
            Aqf.failed(
                'Data transmission stop failed: {}'.format(reply.arguments))
        p = Popen(
            ['kcpcmd', '-s', ingst_nd + ':' + ingst_nd_p, 'capture-done'],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            Aqf.failed('Failure issuing capture-done to ingest process on ' + ingst_nd)
            Aqf.failed('Stdout: \n' + output)
            Aqf.failed('Stderr: \n' + err)
            return False
        else:
            Aqf.step('Capture-done issued on {}.'.format(ingst_nd))

        p = Popen(['rsync', '-aPh', ingst_nd + ':/ramdisk/',
                   'mkat_fpga_tests/bf_data'],
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            Aqf.failed('rsync of beam data failed from' + ingst_nd)
            Aqf.failed('Stdout: \n' + output)
            Aqf.failed('Stderr: \n' + err)
        else:
            Aqf.step('Data transferred from ' + ingst_nd)
        newest_f = max(glob.iglob('mkat_fpga_tests/bf_data/*.h5'),
                       key=os.path.getctime)
        # Read data file
        fin = h5py.File(newest_f, 'r')
        data = fin['Data'].values()
        # Extract data
        bf_raw = np.array(data[0])
        cap_ts = np.array(data[1])
        bf_ts = np.array(data[2])
        fin.close()
        return bf_raw, cap_ts, bf_ts, in_wgts, pb, cf

    def _populate_beam_dict(self, num_wgts_to_set, value, beam_dict):
        """
            If num_wgts_to_set = -1 all inputs will be set
        """
        ctr = 0
        for key in beam_dict:
            if ctr < num_wgts_to_set or num_wgts_to_set == -1:
                beam_dict[key] = value
                ctr += 1
        return beam_dict

    def _set_beam_quant_gain(self, beam, gain):
        try:
            reply, informs = self.corr_fix.katcp_rct.req.beam_quant_gains(beam, gain)
            if reply.reply_ok():
                actual_beam_gain = float(reply.arguments[1])
                msg = ('[CBF-REQ-0117] Requested beamformer level adjust gain of {:.2f}, '
                       'actual gain set to {:.2f}.'.format(gain, actual_beam_gain))
                Aqf.almost_equals(actual_beam_gain, gain, 0.1, msg)
            else:
                raise Exception
        except Exception, e:
            Aqf.failed('Failed to set beamformer quantiser gain via CAM interface, {}'.format(str(e)))
            return 0
        return actual_beam_gain

    def _test_beamforming(self):
        """
        Apply weights and capture beamformer data, Verify that weights are correctly applied.
        """

        def get_beam_data(beam, beam_dict, target_pb, target_cfreq,
                          inp_ref_lvl=0, beam_quant_gain=1, num_caps=10000):
            bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = self._capture_beam_data(
                beam, beam_dict, target_pb, target_cfreq)
            data_type = bf_raw.dtype.name
            cap = [0] * num_caps
            cap_idx = 0
            for i in range(0, num_caps):
                if cap_ts[cap_idx] == bf_ts[i]:
                    cap[cap_idx] = np.array(complexise(bf_raw[:, i, :]))
                    cap_idx += 1
            cap = cap[:cap_idx]
            Aqf.equals(data_type, 'int8',
                       '[CBF-REQ-0118] Beamformer data type is {}, '
                       'example value for one channel: {}'.format(
                           data_type, cap[0][0]))
            cap_mag = np.abs(cap)
            cap_avg = cap_mag.sum(axis=0) / cap_idx
            cap_db = 20 * np.log10(cap_avg)
            cap_db_mean = np.mean(cap_db)
            lbls = self.correlator.get_labels()
            # NOT WORKING
            # labels = ''
            # for lbl in lbls:
            #    bm = beam[-1]
            #    if lbl.find(bm) != -1:
            #        wght = self.correlator.bops.get_beam_weights(beam, lbl)
            # print lbl, wght
            #        labels += (lbl+"={} ").format(wght)
            labels = ''
            for key in in_wgts:
                labels += (key + "={}\n").format(in_wgts[key])
            labels += 'Mean={0:0.2f}dB\n'.format(cap_db_mean)

            if inp_ref_lvl == 0:
                # Get the voltage level for one antenna. Gain for one input
                # should be set to 1, the rest should be 0
                inp_ref_lvl = np.mean(cap_avg)
            delta = 0.2
            expected = np.sum([inp_ref_lvl * in_wgts[key] for key in in_wgts]) * beam_quant_gain
            expected = 20 * np.log10(expected)
            msg = ('Check that the expected voltage level ({:.3f}dB) is within '
                   '{}dB of the measured mean value ({:.3f}dB)'.format(expected,
                                                                       delta, cap_db_mean))
            Aqf.almost_equals(cap_db_mean, expected, delta, msg)
            labels += 'Expected={:.2f}dB'.format(expected)

            return cap_avg, labels, inp_ref_lvl, pb, cf, expected, cap_idx

        ants = self.correlator.n_antennas
        local_src_names = get_local_src_names(self)
        try:
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0x')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0y')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('c856M4k')
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            if reply.reply_ok():
                labels = reply.arguments[1:]
            else:
                raise Exception
        except Exception, e:
            Aqf.failed(e)
            return
        bw = self.corr_freqs.bandwidth
        ch_list = self.corr_freqs.chan_freqs
        nr_ch = self.corr_freqs.n_chans

        # Start of test. Setting required partitions and center frequency
        target_cf = bw + bw * 0.5
        partitions = 4
        part_size = bw / 16
        target_pb = partitions * part_size
        ch_bw = bw / nr_ch
        beams = ('beam_0x', 'beam_0y')
        beam = beams[1]

        # Set beamformer quantiser gain for selected beam to 1
        self._set_beam_quant_gain(beam, 1)

        # dsim_set_success = set_input_levels(self.corr_fix, self.dhost, awgn_scale=0.05,
        # cw_scale=0.675, freq=target_cfreq-bw, fft_shift=8191, gain='11+0j')
        # TODO: Get dsim sample frequency from config file
        cw_freq = ch_list[int(nr_ch / 2)] + 400

        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise, '
                 'with awgn scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, freq=cw_freq, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        beam_data = []
        beam_lbls = []

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # Only one antenna gain is set to 1, this will be used as the reference
        # input level
        weight = 1.0
        beam_dict = self._populate_beam_dict(1, weight, beam_dict)
        rl = 0
        d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, target_pb, target_cf, rl)
        beam_data.append(d)
        beam_lbls.append(l)

        weight = 1.0 / ants
        beam_dict = self._populate_beam_dict(-1, weight, beam_dict)
        d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, target_pb, target_cf, rl)
        beam_data.append(d)
        beam_lbls.append(l)

        weight = 2.0 / ants
        beam_dict = self._populate_beam_dict(-1, weight, beam_dict)
        d, l, rl, pb, cf, exp1, nc = get_beam_data(beam, beam_dict, target_pb, target_cf, rl)
        beam_data.append(d)
        beam_lbls.append(l)
        # Square the voltage data. This is a hack as aqf_plot expects squared
        # power data
        aqf_plot_channels(zip(np.square(beam_data), beam_lbls),
                          plot_filename='{}_chan_resp_{}.png'.format(self._testMethodName, beam),
                          plot_title=('Beam = {}, Passband = {} MHz\nCenter Frequency = {} MHz'
                                      '\nIntegrated over {} captures'.format(beam, pb / 1e6, cf / 1e6, nc)),
                          log_dynamic_range=90, log_normalise_to=1,
                          caption='Captured beamformer data', hlines=[exp0, exp1],
                          plot_type='bf', hline_strt_idx=1)

        beam_data = []
        beam_lbls = []
        # Set beamformer quantiser gain for selected beam to 1/number inputs
        gain = 1.0 / ants
        gain = self._set_beam_quant_gain(beam, gain)
        weight = 1.0
        beam_dict = self._populate_beam_dict(-1, weight, beam_dict)
        rl = 0
        d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, target_pb, target_cf, rl, gain)
        beam_data.append(d)
        l += '\nLevel adjust gain={}'.format(gain)
        beam_lbls.append(l)

        gain = 2.0 / ants
        gain = self._set_beam_quant_gain(beam, gain)
        d, l, rl, pb, cf, exp1, nc = get_beam_data(beam, beam_dict, target_pb, target_cf, rl, gain)
        beam_data.append(d)
        l += '\nLevel adjust gain={}'.format(gain)
        beam_lbls.append(l)

        # Square the voltage data. This is a hack as aqf_plot expects squared
        # power data
        aqf_plot_channels(zip(np.square(beam_data), beam_lbls),
                          plot_filename='{}_level_adjust_after_bf_{}.png'.format(self._testMethodName, beam),
                          plot_title=('Beam = {}, Passband = {} MHz\nCenter Frequency = {} MHz'
                                      '\nIntegrated over {} captures'.format(beam, pb / 1e6, cf / 1e6, nc)),
                          log_dynamic_range=90, log_normalise_to=1,
                          caption='Captured beamformer data with level adjust after beamforming gain set.',
                          hlines=exp1, plot_type='bf', hline_strt_idx=1)
        who_ran_test()

    def _test_cap_beam(self, instrument='bc8n856M4k'):
        """Testing timestamp accuracy (bc8n856M4k)
        Confirm that the CBF subsystem do not modify and correctly interprets
        timestamps contained in each digitiser SPEAD accumulations (dump)
        """
        if self.set_instrument(instrument):
            Aqf.step('Checking timestamp accuracy: {}\n'.format(
                self.corr_fix.get_running_instrument()))
            main_offset = 2153064
            minor_offset = 0
            minor_offset = -6 * 4096 * 2
            manual_offset = main_offset + minor_offset

            ants = 4
            if ants == 4:
                local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                                   'm002_x', 'm002_y', 'm003_x', 'm003_y']
            elif ants == 8:
                local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                                   'm002_x', 'm002_y', 'm003_x', 'm003_y',
                                   'm004_x', 'm004_y', 'm005_x', 'm005_y',
                                   'm006_x', 'm006_y', 'm007_x', 'm007_y']

            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0x')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0y')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('c856M4k')
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq
            Aqf.hop('Dsim_clock_Factor = {}'.format(dsim_clk_factor))
            bw = self.corr_freqs.bandwidth  # * dsim_clk_factor

            target_cfreq = bw + bw * 0.5
            partitions = 1
            part_size = bw / 16
            target_pb = partitions * part_size
            ch_bw = bw / 4096
            target_pb = 100
            num_caps = 20000
            beam = 'beam_0y'
            if ants == 4:
                beam_dict = {'m000_x': 1.0, 'm001_x': 1.0, 'm002_x': 1.0,
                             'm003_x': 1.0,
                             'm000_y': 1.0, 'm001_y': 1.0, 'm002_y': 1.0,
                             'm003_y': 1.0, }
            elif ants == 8:
                beamx_dict = {'m000_x': 1.0, 'm001_x': 1.0, 'm002_x': 1.0,
                              'm003_x': 1.0,
                              'm004_x': 1.0, 'm005_x': 1.0, 'm006_x': 1.0,
                              'm007_x': 1.0}
                beamy_dict = {'m000_y': 1.0, 'm001_y': 1.0, 'm002_y': 1.0,
                              'm003_y': 1.0,
                              'm004_y': 1.0, 'm005_y': 1.0, 'm006_y': 1.0,
                              'm007_y': 1.0}

            self.dhost.sine_sources.sin_0.set(frequency=target_cfreq - bw,
                                              scale=0.1)
            self.dhost.sine_sources.sin_1.set(frequency=target_cfreq - bw,
                                              scale=0.1)
            this_source_freq0 = self.dhost.sine_sources.sin_0.frequency
            this_source_freq1 = self.dhost.sine_sources.sin_1.frequency
            Aqf.step('Sin0 set to {} Hz, Sin1 set to {} Hz'.format(
                this_source_freq0 + bw, this_source_freq1 + bw))

            bf_raw, cap_ts, bf_ts = self._capture_beam_data(beam,
                                                            beamy_dict, target_pb, target_cfreq)

            # cap_ts_diff = np.diff(cap_ts)
            # a = np.nonzero(np.diff(cap_ts)-8192)
            # cap_ts[a[0]+1]
            # cap_phase = numpy.angle(cap)
            # ts = [datetime.datetime.fromtimestamp(float(timestamp)/1000).strftime("%H:%M:%S") for timestamp in timestamps]

            # Average over timestamp show passband
            # for i in range(0,len(cap)):
            #    plt.plot(10*numpy.log(numpy.abs(cap[i])))

    def _test_bc8n856M4k_beamforming_ch(self, instrument='bc8n856M4k'):
        """CBF Beamformer channel accuracy (bc8n856M4k)

        Apply weights and capture beamformer data.
        Verify that weights are correctly applied.
        """
        instrument_success = self.set_instrument(instrument)
        if instrument_success.keys()[0] is not True:
            Aqf.end(passed=False, message=instrument_success.values()[0])
        else:
            _running_inst = self.corr_fix.get_running_instrument().keys()[0]
            msg = Style.Bold('CBF Beamformer channel accuracy: {}\n'.format(
                _running_inst.keys()[0]))
            Aqf.step(msg)
            self._systems_tests()
            self._test_beamforming_ch(ants=4)

    def _test_beamforming_ch(self, ants=4):
        # Set list for all the correlator input labels
        if ants == 4:
            local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                               'm002_x', 'm002_y', 'm003_x', 'm003_y']
        elif ants == 8:
            local_src_names = ['m000_x', 'm000_y', 'm001_x', 'm001_y',
                               'm002_x', 'm002_y', 'm003_x', 'm003_y',
                               'm004_x', 'm004_y', 'm005_x', 'm005_y',
                               'm006_x', 'm006_y', 'm007_x', 'm007_y']

        reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0x')
        reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0y')
        reply, informs = self.corr_fix.katcp_rct.req.capture_stop('c856M4k')
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
        bw = self.corr_freqs.bandwidth
        ch_list = self.corr_freqs.chan_freqs
        nr_ch = self.corr_freqs.n_chans

        # Start of test. Setting required partitions and center frequency
        partitions = 2
        part_size = bw / 16
        target_cfreq = bw + part_size  # + bw*0.5
        target_pb = partitions * part_size
        ch_bw = bw / nr_ch
        num_caps = 20000
        beams = ('beam_0x', 'beam_0y')
        offset = 74893  # ch_list[1]/2 # Offset in Hz to add to cw frequency
        beam = beams[1]

        # TODO: Get dsim sample frequency from config file
        cw_freq = ch_list[int(nr_ch / 2)]
        cw_freq = ch_list[128]

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0  # 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        freq = cw_freq + offset
        dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq
        eff_freq = (freq + bw) * dsim_clk_factor

        Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                 'at {}Hz with cw scale: {}, awgn scale: {}, eq gain: {}, fft '
                 'shift: {}'.format(freq * dsim_clk_factor, cw_scale, awgn_scale, gain,
                                    fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=cw_scale, freq=freq,
                                            fft_shift=fft_shift, gain=gain, cw_src=1)
        self.dhost.registers.scale_cwg0_const.write(scale=0.0)
        self.dhost.registers.scale_cwg1_const.write(scale=0.0)
        self.dhost.registers.cwg1_en.write(en=1)
        self.dhost.registers.cwg0_en.write(en=0)

        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        if ants == 4:
            beam_dict = {'m000_x': 1.0, 'm001_x': 0.0, 'm002_x': 0.0, 'm003_x': 0.0,
                         'm000_y': 1.0, 'm001_y': 0.0, 'm002_y': 0.0, 'm003_y': 0.0}
        elif ants == 8:
            beamx_dict = {'m000_x': 1.0, 'm001_x': 0.0, 'm002_x': 0.0, 'm003_x': 0.0,
                          'm004_x': 0.0, 'm005_x': 0.0, 'm006_x': 0.0, 'm007_x': 0.0,
                          'm000_y': 1.0, 'm001_y': 0.0, 'm002_y': 0.0, 'm003_y': 0.0,
                          'm004_y': 0.0, 'm005_y': 0.0, 'm006_y': 0.0, 'm007_y': 0.0}

        bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = self._capture_beam_data(beam,
                                                                         beam_dict, target_pb, target_cfreq)
        fft_length = 1024
        strt_idx = 0
        num_caps = np.shape(bf_raw)[1]
        cap = [0] * num_caps
        cap_half = [0] * int(num_caps / 2)
        for i in range(0, num_caps):
            cap[i] = np.array(complexise(bf_raw[:, i, :]))
            if i % 2 != 0:
                cap_half[int(i / 2)] = cap[i]
        cap = np.asarray(cap[strt_idx:strt_idx + fft_length])
        cap_half = np.asarray(cap_half[strt_idx:strt_idx + fft_length])
        cap_mag = np.abs(cap)
        max_ch = np.argmax(np.sum((cap_mag), axis=0))
        Aqf.step('CW found in relative channel {}'.format(max_ch))
        plt.plot(np.log10(np.abs(np.fft.fft(cap[:, max_ch]))))
        plt.plot(np.log10(np.abs(np.fft.fft(cap_half[:, max_ch]))))
        plt.show()

    def _bf_efficiency(self):

        local_src_names = get_local_src_names(self)
        try:
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0x')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('beam_0y')
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop('c856M4k')
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            if reply.reply_ok():
                labels = reply.arguments[1:]
            else:
                raise Exception
        except Exception, e:
            Aqf.failed(e)
            return
        bw = self.corr_freqs.bandwidth
        ch_list = self.corr_freqs.chan_freqs
        nr_ch = self.corr_freqs.n_chans

        # Start of test. Setting required partitions and center frequency
        partitions = 1
        part_size = bw / 16
        target_cfreq = bw + bw * 0.5
        target_pb = partitions * part_size
        ch_bw = bw / nr_ch
        beams = ('beam_0x', 'beam_0y')
        beam = beams[1]

        # Set beamformer quantiser gain for selected beam to 1
        self._set_beam_quant_gain(beam, 1)

        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.032
            gain = '226+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise, '
                 'with awgn scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                inp = label
                break
        try:
            reply, informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(inp)
        except Exception:
            Aqf.failed('Failed to grab quantiser snapshot.')
        quant_snap = [eval(v) for v in (reply.arguments[1:][1:])]
        try:
            reply, informs = self.corr_fix.katcp_rct.req.adc_snapshot(inp)
        except Exception:
            Aqf.failed('Failed to grab adc snapshot.')
        fpga = self.correlator.fhosts[0]
        adc_data = fpga.get_adc_snapshots()['p0'].data
        p_std = np.std(adc_data)
        p_levels = p_std * 512
        aqf_plot_histogram(adc_data,
                           plot_filename='{}_adc_hist_{}.png'.format(self._testMethodName, inp),
                           plot_title=(
                               'ADC Histogram for input {}\nNoise Profile: '
                               'Std Dev: {:.3f} equates to {:.1f} levels '
                               'toggling.'.format(inp, p_std, p_levels)),
                           caption='ADC input histogram for beamformer efficiency test, '
                                   'with the digitiser simulator noise scale at {}, '
                                   'quantiser gain at {} and fft shift at {}.'.format(
                               awgn_scale, gain, fft_shift),
                           bins=256, ranges=(-1, 1))
        p_std = np.std(quant_snap)
        aqf_plot_histogram(np.abs(quant_snap),
                           plot_filename='{}_quant_hist_{}.png'.format(self._testMethodName, inp),
                           plot_title=('Quantiser Histogram for input {}\n '
                                       'Standard Deviation: {:.3f}'.format(inp, p_std)),
                           caption='Quantiser histogram for beamformer efficiency test, '
                                   'with the digitiser simulator noise scale at {}, '
                                   'quantiser gain at {} and fft shift at {}.'.format(
                               awgn_scale, gain, fft_shift),
                           bins=64, ranges=(0, 1.5))

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # Only one antenna gain is set to 1, this will be used as the reference
        # input level
        weight = 1.0
        beam_dict = self._populate_beam_dict(1, weight, beam_dict)
        bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = self._capture_beam_data(beam,
                                                                         beam_dict, target_pb, target_cfreq,
                                                                         capture_time=0.3)

        Aqf.hop('Packaging beamformer data.')
        num_caps = np.shape(bf_raw)[1]
        cap = [0] * num_caps
        cap_idx = 0
        for i in range(0, num_caps):
            if cap_ts[cap_idx] == bf_ts[i]:
                cap[cap_idx] = complexise(bf_raw[:, i, :])
                cap_idx += 1
        del bf_raw
        cap = np.asarray(cap[:cap_idx])
        # Output of beamformer is a voltage, get the power
        cap = np.power(np.abs(cap), 2)
        nr_ch = len(cap)
        Aqf.step('Calculating time series mean.')
        ch_mean = cap.mean(axis=0)
        Aqf.step('Calculating time series standard deviation')
        ch_std = cap.std(axis=0, ddof=1)
        ch_bw = self.corr_freqs.delta_f
        acc_time = self.corr_freqs.fft_period
        sqrt_bw_at = np.sqrt(ch_bw * acc_time)
        Aqf.step('Calculating channel efficiency.')
        eff = 1 / ((ch_std / ch_mean) * sqrt_bw_at)
        Aqf.step('Beamformer mean efficiency for {} channels = {:.2f}%'
                 ''.format(nr_ch, 100 * eff.mean()))
        plt_filename = '{}_beamformer_efficiency.png'.format(self._testMethodName)
        plt_title = ('Beamformer Efficiency per Channel\n '
                     'Mean Efficiency = {:.2f}%'.format(100 * eff.mean()))
        caption = ('Beamformer efficiency per channel calculated over {} samples '
                   'with a channel bandwidth of {:.2f}Hz and a FFT window length '
                   'of {:.3f} micro seconds per sample.'.format(
            cap_idx, ch_bw, acc_time * 1000000.))
        aqf_plot_channels(eff * 100, plt_filename, plt_title, caption=caption,
                          log_dynamic_range=None, hlines=95, ylimits=(90, 105), plot_type='eff')

        who_ran_test()

    def _timestamp_accuracy(self, manual=False, manual_offset=0,
                            future_dump=3):
        """

        Parameters
        ----------
        manual : Manually set the offset from the future_dump point.
        manual_offset : Offset in adc sample clocks.
        future_dump : Dump in which impulse is expected

        Returns
        -------

        """

        def load_dsim_impulse(load_timestamp, offset=0):
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=2)
            self.dhost.registers.src_sel_cntrl.write(src_sel_1=0)
            self.dhost.registers.impulse_delay_correction.write(reg=16)
            load_timestamp = load_timestamp + offset
            lt_abs_t = datetime.datetime.fromtimestamp(
                sync_time + load_timestamp / scale_factor_timestamp)
            print 'Impulse load time = {}:{}.{}'.format(lt_abs_t.minute,
                                                        lt_abs_t.second,
                                                        lt_abs_t.microsecond)
            print 'Number of dumps in future = {:.10f}'.format(
                (load_timestamp - dump_ts) / dump_ticks)
            # Digitiser simulator local clock factor of 8 slower
            # (FPGA clock = sample clock / 8).
            load_timestamp = load_timestamp / 8
            if not load_timestamp.is_integer():
                Aqf.failed('Timestamp received in accumulation not divisible' \
                           ' by 8: {:.15f}'.format(load_timestamp))
            load_timestamp = int(load_timestamp)
            reg_size = 32
            load_ts_lsw = load_timestamp & (pow(2, reg_size) - 1)
            load_ts_msw = load_timestamp >> reg_size

            # dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()['data']['reg']
            # dsim_loc_msw = self.dhost.registers.local_time_msw.read()['data']['reg']
            # dsim_loc_time = dsim_loc_msw * pow(2,reg_size) + dsim_loc_lsw
            # print 'timestamp difference: {}'.format((load_timestamp - dsim_loc_time)*8/dump['scale_factor_timestamp'].value)
            self.dhost.registers.impulse_load_time_lsw.write(reg=load_ts_lsw)
            self.dhost.registers.impulse_load_time_msw.write(reg=load_ts_msw)

        # try:
        #     reply, informs = self.corr_fix.katcp_rct.req.accumulation_length(1, timeout=60)
        #     if not reply.reply_ok():
        #         raise Exception
        # except:
        #     errmsg = 'Failed to set accumulation time withing {}s'.format(
        #         reply)
        #     LOGGER.exception(errmsg)
        #     Aqf.failed(errmsg)
        #     return False

        dsim_set_success = set_input_levels(self.corr_fix, self.dhost,
                                            awgn_scale=0.0,
                                            cw_scale=0.0, freq=100000000,
                                            fft_shift=0, gain='32767+0j')
        self.dhost.outputs.out_1.scale_output(0)
        dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        baseline_lookup = get_baselines_lookup(dump)
        sync_time = dump['sync_time'].value
        scale_factor_timestamp = dump['scale_factor_timestamp'].value
        inp = dump['input_labelling'].value[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = dump['n_chans'].value * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = dump['int_time'].value * dump['adc_sample_rate'].value
        print dump_ticks
        dump_ticks = dump['n_accs'].value * dump['n_chans'].value * 2
        print dump_ticks
        print dump['adc_sample_rate'].value
        print dump['timestamp'].value
        if not (dump_ticks / 8.0).is_integer():
            Aqf.failed('Number of ticks per dump is not divisible' \
                       ' by 8: {:.3f}'.format(dump_ticks))
        # Create a linear array spaced by 8 for finding dump timestamp edge
        tick_array = np.linspace(-dump_ticks / 2, dump_ticks / 2,
                                 num=(dump_ticks / 8) + 1)
        # num=fft_sliding_window+1)
        # Offset into tick array to step impulse.
        tckar_idx = len(tick_array) / 2
        tckar_upper_idx = len(tick_array) - 1
        tckar_lower_idx = 0
        future_ticks = dump_ticks * future_dump
        found = False
        prev_imp_loc = 0
        first_run = True
        split_found = False
        single_step = False
        while not found:
            if manual:
                found = True
                offset = manual_offset
            else:
                offset = tick_array[int(tckar_idx)]
            dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            print dump['timestamp'].value
            dump_ts = dump['timestamp'].value
            dump_abs_t = datetime.datetime.fromtimestamp(
                sync_time + dump_ts / scale_factor_timestamp)
            print 'Start dump time = {}:{}.{}'.format(dump_abs_t.minute,
                                                      dump_abs_t.second,
                                                      dump_abs_t.microsecond)
            load_timestamp = dump_ts + future_ticks
            load_dsim_impulse(load_timestamp, offset)
            dump_list = []
            cnt = 0
            for i in range(future_dump):
                cnt += 1
                dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
                print dump['timestamp'].value
                dval = dump['xeng_raw'].value
                auto_corr = dval[:, inp_autocorr_idx, :]
                curr_ts = dump['timestamp'].value
                delta_ts = curr_ts - dump_ts
                dump_ts = curr_ts
                if delta_ts != dump_ticks:
                    Aqf.failed('Accumulation dropped, Expected timestamp = {}, ' \
                               'received timestamp = {}' \
                               ''.format(dump_ts + dump_ticks, curr_ts))
                print 'Maximum value found in dump {} = {}, average = {}' \
                      ''.format(cnt, np.max(auto_corr), np.average(auto_corr))
                dump_list.append(dval)
            # Find dump containing impulse, check that other dumps are empty.
            val_found = 0
            auto_corr = []
            auto_corr_avg = []
            dumps_nzero = 0
            for idx in range(len(dump_list)):
                dmp = dump_list[idx][:, inp_autocorr_idx, :]
                auto_corr.append(dmp)
                auto_corr_std = np.std(dmp)
                auto_corr_avg_val = np.average(dmp)
                if auto_corr_avg_val > 0:
                    print auto_corr_avg_val
                    print auto_corr_std
                    print auto_corr_avg_val - auto_corr_std
                    if abs(auto_corr_avg_val - auto_corr_std) < (auto_corr_avg_val * 0.4):
                        dumps_nzero += 1
                        auto_corr_avg.append(auto_corr_avg_val)
                    else:
                        dumps_nzero = 3
                        auto_corr_avg.append(0)
                else:
                    auto_corr_avg.append(0)
            # imp_loc = np.argmax(auto_corr_avg) + 1
            imp_loc = next((i for i, x in enumerate(auto_corr_avg) if x), None) + 1
            # if (dumps_nzero == 1) and split_found:
            #    single_step = True
            if (dumps_nzero == 2):
                Aqf.step('Two dumps found containing impulse.')
                # Only start stepping by one once the split is close
                split_found = True
            elif dumps_nzero > 2:
                Aqf.failed('Invalid data found in dumps.')
                # for dmp in auto_corr:
                #    plt.plot(dmp)
                # plt.show()
            # Set the index into the time stamp offset array
            print
            print
            if first_run:
                tckar_idx_prev = tckar_idx
                first_run = False
                if imp_loc == future_dump - 1:
                    tckar_idx = tckar_upper_idx
                elif imp_loc == future_dump:
                    tckar_idx = tckar_lower_idx
                else:
                    Aqf.failed('Impulse not where expected.')
                    found = True
            else:
                idx_diff = abs(tckar_idx_prev - tckar_idx)
                tckar_idx_prev = tckar_idx
                if single_step and (dumps_nzero == 1):
                    found = True
                    print 'Edge of dump found at offset {} (ticks)'.format(
                        offset)
                elif ((idx_diff < 10) and (dumps_nzero == 2)) or single_step:
                    single_step = True
                    tckar_idx += 1
                elif (imp_loc == future_dump - 1):
                    tckar_lower_idx = tckar_idx
                    tckar_idx = tckar_idx + (tckar_upper_idx - tckar_idx) / 2
                elif (imp_loc == future_dump):
                    tckar_upper_idx = tckar_idx
                    tckar_idx = tckar_idx - (tckar_idx - tckar_lower_idx) / 2
                else:
                    Aqf.failed('Impulse not where expected.')
                    found = True
            print 'Tick array index = {}, Diff = {}'.format(tckar_idx,
                                                            tckar_idx - tckar_idx_prev)


            # for idx in range(len(auto_corr_list)):
            #     #plt.plot(auto_corr_list[idx][:,inp_autocorr_idx,:])
            #     if idx != future_dump-2:
            #         for i in range(4096):
            #             for j in range(40):
            #                 if auto_corr_list[idx][i, j, 0] > 0:
            #                     print i, j

            # plt.show()

    def _test_timestamp_shift(self, instrument='bc8n856M4k'):
        """Testing timestamp accuracy (bc8n856M4k)
        Confirm that the CBF subsystem do not modify and correctly interprets
        timestamps contained in each digitiser SPEAD accumulations (dump)
        """
        if self.set_instrument(instrument):
            Aqf.step('Checking timestamp accuracy: {}\n'.format(
                self.corr_fix.get_running_instrument()))
            main_offset = 2153064
            minor_offset = 0
            minor_offset = -10 * 4096 * 2
            manual_offset = main_offset + minor_offset
            self._timestamp_shift(offset=manual_offset)

    def _timestamp_shift(self, shift_nr=12, offset=0, future_dump=3):
        """

        Parameters
        ----------
        shift_nr : Number of shifts to perform during shift test
        future_dump : Dump in which impulse is expected

        Returns
        -------

        """

        def load_dsim_impulse(load_timestamp, offset=0):
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=2)
            self.dhost.registers.src_sel_cntrl.write(src_sel_1=0)
            self.dhost.registers.impulse_delay_correction.write(reg=16)
            load_timestamp = load_timestamp + offset
            lt_abs_t = datetime.datetime.fromtimestamp(
                sync_time + load_timestamp / scale_factor_timestamp)
            print 'Impulse load time = {}:{}.{}'.format(lt_abs_t.minute,
                                                        lt_abs_t.second,
                                                        lt_abs_t.microsecond)
            print 'Number of dumps in future = {:.10f}'.format(
                (load_timestamp - dump_ts) / dump_ticks)
            # Digitiser simulator local clock factor of 8 slower
            # (FPGA clock = sample clock / 8).
            load_timestamp = load_timestamp / 8
            if not load_timestamp.is_integer():
                Aqf.failed('Timestamp received in accumulation not divisible' \
                           ' by 8: {:.15f}'.format(load_timestamp))
            load_timestamp = int(load_timestamp)
            reg_size = 32
            load_ts_lsw = load_timestamp & (pow(2, reg_size) - 1)
            load_ts_msw = load_timestamp >> reg_size
            self.dhost.registers.impulse_load_time_lsw.write(reg=load_ts_lsw)
            self.dhost.registers.impulse_load_time_msw.write(reg=load_ts_msw)

        dsim_set_success = set_input_levels(self.corr_fix, self.dhost,
                                            awgn_scale=0.0,
                                            cw_scale=0.0, freq=100000000,
                                            fft_shift=0, gain='32767+0j')
        self.dhost.outputs.out_1.scale_output(0)
        dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        baseline_lookup = get_baselines_lookup(dump)
        sync_time = dump['sync_time'].value
        scale_factor_timestamp = dump['scale_factor_timestamp'].value
        inp = dump['input_labelling'].value[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = dump['n_chans'].value * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = dump['int_time'].value * dump['adc_sample_rate'].value
        dump_ticks = dump['n_accs'].value * dump['n_chans'].value * 2
        input_spec_ticks = dump['n_chans'].value * 2
        if not (dump_ticks / 8.0).is_integer():
            Aqf.failed('Number of ticks per dump is not divisible' \
                       ' by 8: {:.3f}'.format(dump_ticks))
        future_ticks = dump_ticks * future_dump
        shift_set = [[[], []] for x in range(5)]
        for shift in range(len(shift_set)):
            set_offset = offset + 1024 * shift
            list0 = []
            list1 = []
            for step in range(shift_nr):
                set_offset = set_offset + input_spec_ticks
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                dump_ts = dump['timestamp'].value
                dump_abs_t = datetime.datetime.fromtimestamp(
                    sync_time + dump_ts / scale_factor_timestamp)
                print 'Start dump time = {}:{}.{}'.format(dump_abs_t.minute,
                                                          dump_abs_t.second,
                                                          dump_abs_t.microsecond)
                load_timestamp = dump_ts + future_ticks
                load_dsim_impulse(load_timestamp, set_offset)
                dump_list = []
                cnt = 0
                for i in range(future_dump):
                    cnt += 1
                    dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
                    print dump['timestamp'].value
                    dval = dump['xeng_raw'].value
                    auto_corr = dval[:, inp_autocorr_idx, :]
                    curr_ts = dump['timestamp'].value
                    delta_ts = curr_ts - dump_ts
                    dump_ts = curr_ts
                    if delta_ts != dump_ticks:
                        Aqf.failed('Accumulation dropped, Expected timestamp = {}, ' \
                                   'received timestamp = {}' \
                                   ''.format(dump_ts + dump_ticks, curr_ts))
                    print 'Maximum value found in dump {} = {}, average = {}' \
                          ''.format(cnt, np.max(auto_corr), np.average(auto_corr))
                    dump_list.append(auto_corr)
                list0.append(np.std(dump_list[future_dump - 1]))
                list1.append(np.std(dump_list[future_dump - 2]))
            shift_set[shift][0] = list0
            shift_set[shift][1] = list1

        # shift_output0 = [np.log(x) if x > 0 else 0 for x in shift_output0]
        # shift_output1 = [np.log(x) if x > 0 else 0 for x in shift_output1]
        for std_dev_set in shift_set:
            plt.plot(std_dev_set[0])
            plt.plot(std_dev_set[1])
        plt.show()

    def _test_input_levels(self, instrument='bc8n856M4k'):
        """Testing Digitiser simulator input levels (bc8n856M4k)
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        if self.set_instrument(instrument):
            Aqf.step('Setting and checking Digitiser simulator input levels: {}\n'.format(
                self.corr_fix.get_running_instrument()))
            self._set_input_levels_and_gain(profile='noise', cw_freq=100000, cw_margin=0.3,
                                            trgt_bits=4, trgt_q_std=0.30, fft_shift=511)

    def _set_input_levels_and_gain(self, profile='noise', cw_freq=0, cw_src=0,
                                   cw_margin=0.05, trgt_bits=3.5,
                                   trgt_q_std=0.30, fft_shift=511):
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
            # if cw_src == 0:
            self.dhost.sine_sources.sin_0.set(frequency=cw_freq,
                                              scale=round(scale, 3))
            #    return self.dhost.sine_sources.sin_0.frequency
            # else:
            self.dhost.sine_sources.sin_1.set(frequency=cw_freq,
                                              scale=round(scale, 3))
            return self.dhost.sine_sources.sin_1.frequency

        # main code
        sources = {}
        Aqf.step('Requesting input labels.')
        for i in range(2):
            self.corr_fix.issue_metadata()
            self.corr_fix.start_x_data()

        # Build dictionary with inputs and
        # which fhosts they are associated with.
        dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
        inp_labelling = dump['input_labelling'].value
        for inp_lables in inp_labelling:
            inp = inp_lables[0]
            pol = inp_lables[3]
            fpga_name = inp_lables[2]
            for fpga in self.correlator.fhosts:
                if fpga_name == fpga.host:
                    sources[inp] = ('p' + pol, fpga)
        # try:
        #    # Build dictionary with inputs and
        #    # which fhosts they are associated with.
        #    reply, informs = self.corr_fix.katcp_rct.req.input_labels()
        #    if not reply.reply_ok():
        #        raise Exception
        #    for key in reply.arguments[1:]:
        #        for fpga in self.correlator.fhosts:
        #            for pol in [0,1]:
        #                if str(fpga.data_sources[pol]).find(key) != -1:
        #                    sources[key] = ('p'+str(pol), fpga)
        # except:
        #    Aqf.failed('Failed to get input lables. KATCP Reply: {}'.format(reply))
        #    return False

        # Set digitiser input level of one random input,
        # store values from other inputs for checking
        ret_dict = {}
        for key in sources.keys():
            ret_dict[key] = {}
        inp = sources.keys()[0]
        scale = 0.1
        margin = 0.005
        self.dhost.noise_sources.noise_corr.set(scale=round(scale, 3))
        # Get target standard deviation. ADC is represented by Q10.9
        # signed fixed point.
        target_std = pow(2.0, trgt_bits) / 512
        found = False
        count = 0
        pol = sources[inp][0]
        fpga = sources[inp][1]
        Aqf.step('Setting input noise level to toggle {} bits at ' \
                 'standard deviation.'.format(trgt_bits))
        Aqf.step('Capturing ADC Snapshots.')
        while not found:
            for i in range(5):
                adc_data = fpga.get_adc_snapshots()[pol].data
            cur_std = np.std(adc_data)
            cur_diff = target_std - cur_std
            if (abs(cur_diff) < margin) or count > 5:
                found = True
            else:
                count += 1
                perc_change = target_std / cur_std
                scale = scale * perc_change
                # Maximum noise power
                if scale > 1:
                    scale = 1
                    found = True
                self.dhost.noise_sources.noise_corr.set(scale=round(scale, 3))
        noise_scale = scale
        p_std = np.std(adc_data)
        p_bits = np.log2(p_std * 512)
        Aqf.step('Digitiser simulator noise scale set to {:.3f}, toggling {:.2f} bits at ' \
                 'standard deviation.'.format(noise_scale, p_bits))

        if profile == 'cw':
            Aqf.step('Setting CW scale to {} below saturation point.' \
                     ''.format(cw_margin))
            # Find closest center frequency to requested value to ensure
            # correct quantiser gain is set. Requested frequency will be set
            # at the end.

            # reply, informs = self.corr_fix.katcp_rct. \
            #    req.quantiser_snapshot(inp)
            # data = [eval(v) for v in (reply.arguments[2:])]
            # nr_ch = len(data)
            # ch_bw = bw / nr_ch
            # ch_list = np.linspace(0, bw, nr_ch, endpoint=False)

            bw = self.corr_freqs.bandwidth
            nr_ch = self.corr_freqs.n_chans
            ch_bw = self.corr_freqs.chan_freqs[1]
            ch_list = self.corr_freqs.chan_freqs
            freq_ch = int(round(cw_freq / ch_bw))
            scale = 1.0
            step = 0.005
            count = 0
            found = False
            while not found:
                set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                adc_data = fpga.get_adc_snapshots()[pol].data
                if (count < 4) and (np.abs(np.max(adc_data) or
                                               np.min(adc_data)) >= 0b111111111 / 512.0):
                    scale -= step
                    count += 1
                else:
                    scale -= (step + cw_margin)
                    freq = set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                    adc_data = fpga.get_adc_snapshots()[pol].data
                    found = True
            Aqf.step('Digitiser simulator CW scale set to {:.3f}.'.format(scale))
            aqf_plot_histogram(adc_data,
                               plot_filename='adc_hist_{}.png'.format(inp),
                               plot_title=(
                                   'ADC Histogram for input {}\nAdded Noise Profile: '
                                   'Std Dev: {:.3f} equates to {:.1f} bits '
                                   'toggling.'.format(inp, p_std, p_bits)),
                               caption='ADC Input Histogram',
                               bins=256, ranges=(-1, 1))

        else:
            aqf_plot_histogram(adc_data,
                               plot_filename='adc_hist_{}.png'.format(inp),
                               plot_title=(
                                   'ADC Histogram for input {}\n Standard Deviation: {:.3f} equates '
                                   'to {:.1f} bits toggling'.format(inp, p_std, p_bits)),
                               caption='ADC Input Histogram',
                               bins=256, ranges=(-1, 1))

        for key in sources.keys():
            pol = sources[key][0]
            fpga = sources[key][1]
            adc_data = fpga.get_adc_snapshots()[pol].data
            if profile != 'cw':  # use standard deviation of noise before CW
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

        Aqf.step('Setting FFT Shift to {}.'.format(fft_shift))
        try:
            reply, informs = self.corr_fix.katcp_rct.req.fft_shift(fft_shift)
            if not reply.reply_ok():
                raise Exception
            for key in sources.keys():
                ret_dict[key]['fft_shift'] = reply.arguments[1:][0]
        except:
            Aqf.failed('Failed to set FFT shift. KATCP Reply: {}'.format(reply))
            return False

        if profile == 'cw':
            Aqf.step('Setting quantiser gain for CW input.')
            gain = 1
            gain_str = '{}'.format(int(gain)) + '+0j'
            try:
                reply, informs = self.corr_fix.katcp_rct.req.gain(inp, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed(
                    'Failed to set quantiser gain. KATCP Reply: {}'.format(reply))
                return False
            try:
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            except ValueError:
                errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)

            else:
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
                        reply, informs = self.corr_fix.katcp_rct.req.gain(inp, gain_str)
                        if not reply.arguments[1:] != gain_str:
                            raise Exception
                    except:
                        Aqf.failed(
                            'Failed to set quantiser gain. KATCP Reply: {}'.format(reply))
                        return False
                    try:
                        dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    except ValueError:
                        errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)

                    else:
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
            x = [x[1] for x in ch_val_array]
            grad = np.gradient(y)
            grad_delta = []
            for i in range(len(grad) - 1):
                grad_delta.append(grad[i + 1] / grad[i])
            # The setpoint is where grad_delta is closest to 1
            grad_delta = np.asarray(grad_delta)
            set_point = np.argmax(grad_delta - 1.0 < 0) + 1
            gain_str = '{}'.format(int(x[set_point])) + '+0j'
            plt.plot(x, y, label='Channel Response')
            plt.plot(x[set_point], y[set_point], 'ro', label='Gain Set Point = ' \
                                                             '{}'.format(x[set_point]))
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
            Aqf.step('Setting quantiser gain for noise input with a target '
                     'standard deviation of {}.'.format(trgt_q_std))
            found = False
            count = 0
            margin = 0.01
            gain = 300
            gain_str = '{}'.format(int(gain)) + '+0j'
            try:
                reply, informs = self.corr_fix.katcp_rct.req.gain(inp, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed('Failed to set quantiser gain. KATCP Reply: {}'.format(reply))
                return False
            while (not found):
                Aqf.step('Capturing quantiser snapshot for gain of ' + gain_str)
                reply, informs = self.corr_fix.katcp_rct. \
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
                        reply, informs = self.corr_fix.katcp_rct.req.gain(inp, gain_str)
                        if not reply.arguments[1:] != gain_str:
                            raise Exception
                    except:
                        Aqf.failed(
                            'Failed to set quantiser gain. KATCP Reply: {}'.format(reply))
                        return False

        # Set calculated gain for remaining inputs
        for key in sources.keys():
            if profile == 'cw':
                ret_dict[key]['cw_freq'] = freq
            try:
                reply, informs = self.corr_fix.katcp_rct.req.gain(key, gain_str)
                if not reply.arguments[1:] != gain_str:
                    raise Exception
            except:
                Aqf.failed(
                    'Failed to set quantiser gain. KATCP Reply: {}'.format(reply))
                return False
            pol = sources[key][0]
            fpga = sources[key][1]
            reply, informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(key)
            data = [eval(v) for v in (reply.arguments[1:][1:])]
            p_std = np.std(data)
            ret_dict[key]['q_gain'] = gain_str
            ret_dict[key]['q_std_dev'] = p_std
            ret_dict[key]['q_satr'] = False
            rmax = np.max(np.asarray(data).real)
            rmin = np.min(np.asarray(data).real)
            imax = np.max(np.asarray(data).imag)
            imin = np.min(np.asarray(data).imag)
            if abs(rmax or rmin or imax or imin) >= 0b1111111 / 128.0:
                ret_dict[key]['q_satr'] = True
                count = 0
                for val in data:
                    if abs(val) >= 0b1111111 / 128.0:
                        count += 1
                ret_dict[key]['num_sat'] = count

        if profile == 'cw':
            try:
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            except ValueError:
                errmsg = 'Could not retrieve clean SPEAD accumulation, Item has too few elements for shape.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                dval = dump['xeng_raw'].value
                auto_corr = dval[:, inp_autocorr_idx, :]
                plot_filename = 'spectrum_plot_{}.png'.format(key)
                plot_title = ('Spectrum for Input {}\n'
                              'Quantiser Gain: {}'.format(key, gain_str))
                caption = 'Spectrum for CW input'
                aqf_plot_channels(10 * np.log10(auto_corr[:, 0]),
                                  plot_filename=plot_filename,
                                  plot_title=plot_title, caption=caption, show=True)
        else:
            p_std = np.std(data)
            aqf_plot_histogram(np.abs(data), plot_filename='quant_hist_{}.png'.format(key),
                               plot_title=('Quantiser Histogram for input {}\n '
                                           'Standard Deviation: {:.3f},'
                                           'Quantiser Gain: {}'.format(key, p_std, gain_str)),
                               caption='Quantiser Histogram',
                               bins=64, range=(0, 1.5))

        key = ret_dict.keys()[0]
        if profile == 'cw':
            Aqf.step('Digitiser simulator Sine Wave scaled at {:0.3f}'.format(ret_dict[key]['scale']))
        Aqf.step('Digitiser simulator Noise scaled at {:0.3f}'.format(ret_dict[key]['noise_scale']))
        Aqf.step('FFT Shift set to {}'.format(ret_dict[key]['fft_shift']))
        for key in sources.keys():
            Aqf.step('{} ADC standard deviation: {:0.3f} toggling {:0.2f} bits'.format(
                key, ret_dict[key]['std_dev'], ret_dict[key]['bits_t']))
            Aqf.step('{} quantiser standard deviation: {:0.3f} at a gain of {}'.format(
                key, ret_dict[key]['q_std_dev'], ret_dict[key]['q_gain']))
            if ret_dict[key]['adc_satr']:
                Aqf.failed('ADC snapshot for {} contains saturated samples.'.format(key))
            if ret_dict[key]['q_satr']:
                Aqf.failed('Quantiser snapshot for {} contains saturated samples.'.format(key))
                Aqf.failed('{} saturated samples found'.format(ret_dict[key]['num_sat']))
        return ret_dict

    def _corr_efficiency(self, n_accs=8000):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if self.corr_freqs.n_chans == 4096:
            # 4K
            awgn_scale = 0.032
            gain = '226+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.032
            gain = '600+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate gaussian noise, '
                 'with awgn scale: {}, eq gain: {}, fft shift: {}'.format(awgn_scale, gain,
                                                                          fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=0.0,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            inp = get_input_labels(self)
            assert inp != False
        except AssertionError:
            Aqf.failed('Failed to retrieve input labels via CAM interface')

        try:
            reply, informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(inp)
        except Exception:
            Aqf.failed('Failed to grab quantiser snapshot.')
        quant_snap = [eval(v) for v in (reply.arguments[1:][1:])]
        try:
            reply, informs = self.corr_fix.katcp_rct.req.adc_snapshot(inp)
        except Exception:
            Aqf.failed('Failed to grab adc snapshot.')
        print_counts = 3
        fpga = self.correlator.fhosts[0]
        adc_data = fpga.get_adc_snapshots()['p0'].data
        p_std = np.std(adc_data)
        p_levels = p_std * 512
        aqf_plot_histogram(adc_data,
                           plot_filename='{}_adc_hist_{}.png'.format(self._testMethodName, inp),
                           plot_title=('ADC Histogram for input {}\nNoise Profile: '
                                       'Std Dev: {:.3f} equates to {:.1f} levels '
                                       'toggling.'.format(inp, p_std, p_levels)),
                           caption=('ADC input histogram for correlator efficiency test, '
                                    'with the digitiser simulator noise scale at {}, '
                                    'quantiser gain at {} and fft shift at {}.'.format(awgn_scale,
                                                                                       gain,
                                                                                       fft_shift)),
                           bins=256, ranges=(-1, 1))
        p_std = np.std(quant_snap)
        aqf_plot_histogram(np.abs(quant_snap),
                           plot_filename='{}_quant_hist_{}.png'.format(self._testMethodName, inp),
                           plot_title=('Quantiser Histogram for input {}\n '
                                       'Standard Deviation: {:.3f}'.format(inp, p_std)),
                           caption=('Quantiser histogram for correlator efficiency test, '
                                    'with the digitiser simulator noise scale at {}, '
                                    'quantiser gain at {} and fft shift at {}.'.format(awgn_scale,
                                                                                       gain,
                                                                                       fft_shift)),
                           bins=64, ranges=(0, 1.5))

        csvfile = 'ch_time_series_{}.csv'.format(self._testMethodName)
        with open(csvfile, 'wb') as file:
            writer = csv.writer(file)
            Aqf.step('Getting initial Spead Heap')
            try:
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                baseline_lookup = get_baselines_lookup(dump)
                inp_autocorr_idx = baseline_lookup[(inp, inp)]
                acc_time = dump['int_time'].value
                ch_bw = self.corr_freqs.delta_f
                dval = dump['xeng_raw'].value
                auto_corr = dval[:, inp_autocorr_idx, :][:, 0]
                writer.writerow(auto_corr)
                # ch_time_series = auto_corr
                for i in range(n_accs - 1):
                    if i < print_counts:
                        Aqf.hop('Getting Spead Heap #{}'.format(i + 1))
                    elif i == print_counts:
                        Aqf.hop('.' * print_counts)
                    elif i > (n_accs - print_counts):
                        Aqf.hop('Getting Spead Heap #{}'.format(i + 1))
                    else:
                        LOGGER.info('Getting Spead Heap #{}'.format(i + 1))
                    try:
                        dump = self.receiver.data_queue.get(DUMP_TIMEOUT)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        dval = dump['xeng_raw'].value
                        auto_corr = dval[:, inp_autocorr_idx, :][:, 0]
                        writer.writerow(auto_corr)
                        # ch_time_series = np.vstack((ch_time_series, auto_corr))

        Aqf.hop('Reading data from file.')
        file = open(csvfile, 'rb')
        read = csv.reader(file)
        ch_time_series = []
        for row in read:
            ch_time_series.append([int(y) for y in row])
            time.sleep(1)
        file.close()
        ch_time_series = np.asarray(ch_time_series)
        Aqf.step('Calculating time series mean.')
        ch_mean = ch_time_series.mean(axis=0)
        Aqf.step('Calculating time series standard deviation')
        ch_std = ch_time_series.std(axis=0, ddof=1)
        sqrt_bw_at = np.sqrt(ch_bw * acc_time)

        Aqf.step('Calculating channel efficiency.')
        eff = 1 / ((ch_std / ch_mean) * sqrt_bw_at)

        expected_eff = 0.98  # [CBF-REQ-0127]
        Aqf.more(eff.mean(), expected_eff, 'Mean channel efficiency is {:.2f}%'.format(
            100 * eff.mean()))

        plt_filename = '{}_correlator_efficiency.png'.format(self._testMethodName)
        plt_title = ('Correlator Efficiency per Channel\n '
                     'Mean Efficiency is {:.2f}%'.format(100 * eff.mean()))
        caption = ('Correlator efficiency per channel calculated over {} samples '
                   'with a channel bandwidth of {:.2f}Hz and an accumulation time '
                   'of {:.4f} seconds per sample.'.format(n_accs, ch_bw, acc_time))
        aqf_plot_channels(eff * 100, plt_filename, plt_title, caption=caption,
                          log_dynamic_range=None, hlines=98, plot_type='eff')

    def _small_voltage_buffer(self):

        ch_list = self.corr_freqs.chan_freqs
        # Choose a frequency 3 quarters through the band
        cw_chan_set = int(self.corr_freqs.n_chans * 3 / 4)
        cw_freq = ch_list[cw_chan_set]
        dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq
        bw = self.corr_freqs.bandwidth
        eff_freq = (cw_freq + bw) * dsim_clk_factor
        ch_bw = self.corr_freqs.delta_f

        if self.corr_freqs.n_chans == 4096:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = '11+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave '
                 'at {} Hz (channel={}), with cw scale: {}, awgn scale: {}, '
                 'eq gain: {}, fft shift: {}'.format(cw_freq, cw_chan_set, cw_scale,
                                                     awgn_scale, gain, fft_shift))
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=cw_freq, fft_shift=fft_shift, gain=gain, cw_src=0)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False
        dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=cw_freq, fft_shift=fft_shift, gain=gain, cw_src=1)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            labels = get_input_labels(self)
            assert labels != False
        except AssertionError:
            Aqf.failed('Failed to retrieve input labels via CAM interface')

        try:
            reply, informs = self.corr_fix.katcp_rct.req.transient_buffer_trigger()
            Aqf.passed('Transient buffer trigger present.')
        except Exception:
            Aqf.failed('Transient buffer trigger failed.')

        label = labels[0]
        try:
            reply, informs = self.corr_fix.katcp_rct.req.adc_snapshot(label)
        except Exception:
            Aqf.failed('Failed to grab adc snapshot.')
        fpga = self.correlator.fhosts[0]
        adc_data = fpga.get_adc_snapshots()['p0'].data
        fft_len = len(adc_data)
        Aqf.step('ADC capture length: {}'.format(fft_len))
        fft_real = np.abs(np.fft.fft(adc_data))
        fft_pos = fft_real[0:fft_len / 2]
        cw_chan = np.argmax(fft_pos)
        cw_freq_found = cw_chan / (fft_len / 2) * bw
        msg = ('Check that the expected frequency: {}Hz and measured frequency: '
               '{}Hz matches to within a channel bandwidth: {:.3f}Hz'.format(
            cw_freq_found, cw_freq, ch_bw))
        Aqf.almost_equals(cw_freq_found, cw_freq, ch_bw, msg)
        aqf_plot_channels(np.log10(fft_pos),
                          plot_filename='{}_fft_{}.png'.format(self._testMethodName, label),
                          plot_title=('Input Frequency = {} Hz\nMeasured Frequency at FFT bin {} '
                                      '= {}Hz'.format(cw_freq, cw_chan, cw_freq_found)), log_dynamic_range=None,
                          caption=('FFT of captured small voltage buffer. {} voltage points captured '
                                   'on input {}. Input bandwidth = {}Hz'.format(fft_len, label, bw)),
                          xlabel='FFT bins')
