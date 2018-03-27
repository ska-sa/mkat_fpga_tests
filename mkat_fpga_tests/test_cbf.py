#!/usr/bin/env python
# https://stackoverflow.com/a/44077346
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cbf@ska.ac.za                                                       #
# Maintainer: mmphego@ska.ac.za, alec@ska.ac.za                               #
# Copyright @ 2016 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from __future__ import division

import casperfpga
import corr2
import csv
import gc
import katcp
import katcp
import logging
import os
import Queue
import random
import scipy.interpolate
import scipy.signal
import socket
import spead2
import struct
import subprocess
import sys
import textwrap
import threading
import time
import unittest

import matplotlib.pyplot as plt
import ntplib
import numpy as np
import pandas as pd

from corr2.corr_rx import CorrRx
from corr2.fxcorrelator_xengops import VaccSynchAttemptsMaxedOut
from katcp.testutils import start_thread_with_cleanup

# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
# from memory_profiler import profile as DetectMemLeaks
from datetime import datetime

from mkat_fpga_tests import correlator_fixture
from mkat_fpga_tests import add_cleanup

from mkat_fpga_tests.aqf_utils import *
from mkat_fpga_tests.utils import *
from nosekatreport import *
from descriptions import TestProcedure
from power_logger import PowerLogger

logger_name = 'mkat_fpga_tests'
LOGGER = logging.getLogger(logger_name)
# How long to wait for a correlator dump to arrive in tests
DUMP_TIMEOUT = 10
# ToDo MM (2017-07-21) Improve the logging for debugging
set_dsim_epoch = False
dsim_timeout = 60

@cls_end_aqf
@system('all')
class test_CBF(unittest.TestCase):
    """ Unit-testing class for mkat_fpga_tests"""
    # Hard-coded, perhaps fix this later
    cur_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    _csv_filename = os.path.join(cur_path, 'docs/Manual_Tests.csv')
    _images_dir = os.path.join(cur_path, 'docs/manual_tests_images')

    if os.path.exists(_csv_filename):
        csv_manual_tests = CSV_Reader(_csv_filename, set_index="Verification Event Number")

    def setUp(self):
        global set_dsim_epoch
        super(test_CBF, self).setUp()
        self.receiver = None
        self._dsim_set = False
        self.corr_fix = correlator_fixture
        self.logs_path = None
        try:
            self.logs_path = create_logs_directory(self)
            self.conf_file = self.corr_fix.test_config
            self.corr_fix.katcp_client  = self.conf_file['instrument_params']['katcp_client']
            msg = 'Connecting to katcp client on %s' % self.corr_fix.katcp_client
            Aqf.note(msg)
            LOGGER.info(msg)
        except Exception:
            errmsg = 'Failed to read test config file.'
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        try:
            self.dhost = self.corr_fix.dhost
            errmsg = 'Failed to instantiate the dsim, investigate'
            assert isinstance(self.dhost, corr2.dsimhost_fpga.FpgaDsimHost), errmsg
        except Exception:
            LOGGER.exception(errmsg)
        else:

            # See: https://docs.python.org/2/library/functions.html#super
            if set_dsim_epoch is False:
                try:
                    LOGGER.info('This should only run once...')
                    if not self.dhost.is_running():
                        errmsg = 'Dsim is not running, ensure dsim is running before test commences'
                        Aqf.end(message=errmsg)
                        sys.exit(errmsg)
                    self.dhost.get_system_information(filename=self.dhost.config.get('bitstream'))
                    errmsg = 'Issues with the defined instrument, figure it out'
                    assert isinstance(self.corr_fix.instrument, str), errmsg
                    # cbf_title_report(self.corr_fix.instrument)
                    # Disable warning messages(logs) once
                    disable_warnings_messages()
                    errmsg = 'katcp connection could not be established, investigate!!!'
                    self.assertIsInstance(self.corr_fix.katcp_rct,
                        katcp.resource_client.ThreadSafeKATCPClientResourceWrapper), errmsg
                    errmsg = 'Failed to set Digitiser sync epoch via CAM interface.'
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value('synchronisation-epoch')
                    assert reply.reply_ok(), errmsg
                    sync_time = float(informs[0].arguments[-1])
                    errmsg = 'Issues with reading Sync epoch'
                    assert isinstance(sync_time, float), errmsg
                    reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch(sync_time)
                    errmsg = 'Failed to set digitiser sync epoch'
                    assert reply.reply_ok(), errmsg
                    LOGGER.info('Digitiser sync epoch set successfully')
                    set_dsim_epoch = True
                    self._dsim_set = True
                except Exception:
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

    # This needs proper testing
    def tearDown(self):
        try:
            assert not self.receiver
        except AssertionError:
            LOGGER.info('Cleaning up the receiver!!!!')
            add_cleanup(self.receiver.stop)
            self.receiver = None
            del self.receiver


    def set_instrument(self, acc_time=None, **kwargs):
        self.receiver = None
        acc_timeout = 60
        self.errmsg = None
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)
        self.cam_sensors = GetSensors(self.corr_fix)
        self.addCleanup(init_dsim_sources, self.dhost)

        try:
            Aqf.step('Confirm running instrument, else start a new instrument')
            self.instrument = self.cam_sensors.get_value('instrument_state').split('_')[0]
            Aqf.progress('Currently running instrument %s-%s as per /etc/corr' % (
                self.corr_fix.array_name, self.instrument))
            LOGGER.info('Yeyyy!!! Got running instrument from sensors: %s' % self.instrument)
        except Exception:
            errmsg = ('No running instrument on array: %s, Exiting....' % self.corr_fix.array_name)
            LOGGER.exception(errmsg)
            Aqf.end(message=errmsg)
            sys.exit(errmsg)

        if self._dsim_set:
            Aqf.step('Configure a digitiser simulator to be used as input source to F-Engines.')
            msg = 'Digitiser Simulator running on host: %s' % self.dhost.host
            Aqf.progress(msg)
            LOGGER.info(msg)

        try:
            n_ants = self.cam_sensors.get_value('n_ants')
            if acc_time:
                pass
            elif n_ants == 4:
                acc_time = 0.5
            else:
                acc_time = n_ants / 32.
            reply, informs = self.corr_fix.katcp_rct.req.accumulation_length(acc_time,
                timeout=acc_timeout)
            assert reply.reply_ok()
            acc_time = float(reply.arguments[-1])
            Aqf.step('Set and confirm accumulation period via CAM interface.')
            Aqf.progress('Accumulation time set to {:.3f} seconds'.format(acc_time))
        except Exception as e:
            self.errmsg = ('Failed to set accumulation time due to :%s' % str(e))
            Aqf.failed(self.errmsg)
            LOGGER.exception(self.errmsg)

        try:
            self.correlator = self.corr_fix.correlator
            self.errmsg = 'Failed to instantiate a correlator object'
            self.assertIsInstance(self.correlator, corr2.fxcorrelator.FxCorrelator), self.errmsg
        except Exception:
            Aqf.failed(self.errmsg)
            LOGGER.exception(self.errmsg)
            return False
        try:
            output_product = self.conf_file.get('output_product', 'baseline-correlation-products')
            data_output_ip, data_output_port = self.cam_sensors.get_value(
                    output_product.replace('-','_') + '_destination').split(':')
            Aqf.step('Starting SPEAD receiver listening on %s:%s, CBF output product: %s' % (
                data_output_ip, data_output_port, output_product))
            katcp_ip = self.corr_fix.katcp_client
            katcp_port = int(self.corr_fix.katcp_rct.port)
            LOGGER.info('Connecting to katcp on %s' % katcp_ip)
            start_channels = int(self.conf_file.get('start_channels', 0))
            stop_channels = int(self.conf_file.get('stop_channels', 2047))
            LOGGER.info('Starting receiver on port %s, will only capture channels between %s-%s' %(
                data_output_port, start_channels, stop_channels))
            if n_ants >= 16:
                Aqf.note('Due to performance related issues, only 2048 channels will be captured in '
                     'the SPEAD accumulation')
                self.receiver = CorrRx(product_name=output_product, katcp_ip=katcp_ip,
                    katcp_port=katcp_port, port=data_output_port, channels=(start_channels,
                                                                            stop_channels))
            else:
                self.receiver = CorrRx(product_name=output_product, katcp_ip=katcp_ip,
                    katcp_port=katcp_port, port=data_output_port)

            self.receiver.setName('CorrRx Thread')
            self.errmsg = 'Failed to create SPEAD data receiver'
            self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx), self.errmsg
            start_thread_with_cleanup(self, self.receiver, timeout=10, start_timeout=1)
            self.errmsg = 'Spead Receiver not Running, possible '
            assert self.receiver.isAlive(), self.errmsg
            self.corr_fix.start_x_data
            LOGGER.info('Getting a test dump to confirm number of channels else, test fails '
                        'if cannot retrieve dump')
            _test_dump = self.receiver.get_clean_dump()
            self.assertIsInstance(_test_dump, dict)
            self.n_chans_selected = int(_test_dump.get('n_chans_selected',
                self.cam_sensors.get_value('n_chans')))
            LOGGER.info('Confirmed number of channels %s, from initial dump' % self.n_chans_selected)
        except Exception as e:
            Aqf.failed('%s' % str(e))
            LOGGER.exception('%s' % str(e))
            return False
        else:
            # Run system tests before each test is ran
            self._systems_tests()
            self.addCleanup(self.corr_fix.stop_x_data)
            self.addCleanup(self.receiver.stop)
            self.addCleanup(executed_by)
            self.addCleanup(self._systems_tests)
            self.addCleanup(gc.collect)
            return True

    @instrument_4k
    @aqf_vr('CBF.V.3.30')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_course(self):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                n_chans = self.cam_sensors.get_value('n_chans')
                test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
                test_heading("CBF Channelisation Wideband Coarse L-band")
                self._test_channelisation(test_chan, no_channels=n_chans, req_chan_spacing=250e3)
            else:
                Aqf.failed(self.errmsg)

    @instrument_32k
    @aqf_vr('CBF.V.3.30')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine(self):
        # Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                n_chans = self.cam_sensors.get_value('n_chans')
                test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
                test_heading("CBF Channelisation Wideband Fine L-band")
                self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
            else:
                Aqf.failed(self.errmsg)

    @slow
    @instrument_4k
    @aqf_vr('CBF.V.3.30')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_course_sfdr_peaks(self):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                test_heading("CBF Channelisation Wideband Coarse SFDR L-band")
                n_channels = self.cam_sensors.get_value('n_chans')
                self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=n_channels)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @instrument_32k
    @aqf_vr('CBF.V.3.30')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine_sfdr_peaks(self):
        # Aqf.procedure(TestProcedure.ChannelisationSFDR)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                test_heading("CBF Channelisation Wideband Fine SFDR L-band")
                n_channels = self.cam_sensors.get_value('n_chans')
                self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=n_channels)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.46')
    @aqf_requirements("CBF-REQ-0164", "CBF-REQ-0191")
    def test_power_consumption(self):
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            Aqf.step("Test is being qualified by CBF.V.3.30")

    # @generic_test
    # @aqf_vr('CBF.V.3.35')
    # @aqf_requirements("CBF-REQ-0124")
    # def test_beamformer_efficiency(self):
    #     Aqf.procedure(TestProcedure.BeamformerEfficiency)
    #     try:
    #         assert eval(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument()
    #         if instrument_success:
    #             # self._test_efficiency()
    #             pass
    #         else:
    #             Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.4.10')
    @aqf_requirements("CBF-REQ-0127")
    def test_lband_efficiency(self):
        Aqf.procedure(TestProcedure.LBandEfficiency)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_efficiency()
            else:
                Aqf.failed(self.errmsg)

    @instrument_4k
    @aqf_vr('CBF.V.3.34')
    @aqf_requirements("CBF-REQ-0094", "CBF-REQ-0117", "CBF-REQ-0118", "CBF-REQ-0123", "CBF-REQ-0183")
    def test_beamforming(self):
        Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_beamforming()
            else:
                Aqf.failed(self.errmsg)

    # Test still under development, Alec will put it under test_informal
    # @instrument_4k
    # def test_beamforming_timeseries(self):
    #     #Aqf.procedure(TestProcedure.Beamformer)
    #     try:
    #         assert eval(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument()
    #         if instrument_success:
    #             self._test_beamforming_timeseries()
    #         else:
    #             Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.4.4')
    @aqf_requirements("CBF-REQ-0087", "CBF-REQ-0225", "CBF-REQ-0104")
    def test_baseline_correlation_product(self):
        Aqf.procedure(TestProcedure.BaselineCorrelation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_product_baselines()
                self._test_back2back_consistency()
                self._test_freq_scan_consistency()
                self._test_spead_verify()
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.62')
    @aqf_requirements("CBF-REQ-0238")
    def test_imaging_data_product_set(self):
        Aqf.procedure(TestProcedure.ImagingDataProductSet)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_baseline=True)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr("CBF.V.3.67")
    @aqf_requirements("CBF-REQ-0120")
    def test_tied_array_aux_baseline_correlation_products(self):
        Aqf.procedure(TestProcedure.TiedArrayAuxBaselineCorrelationProducts)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_baseline=True, _tiedarray=True)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr("CBF.V.3.64")
    @aqf_requirements("CBF-REQ-0242")
    def test_tied_array_voltage_data_product_set(self):
        Aqf.procedure(TestProcedure.TiedArrayVoltageDataProductSet)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_tiedarray=True)
            else:
                Aqf.failed(self.errmsg)


    @generic_test
    @aqf_vr('CBF.V.4.7')
    @aqf_requirements("CBF-REQ-0096")
    def test_accumulation_length(self):
        # The CBF shall set the Baseline Correlation Products accumulation interval to a fixed time
        # in the range $$500 +0 -20ms$$.
        Aqf.procedure(TestProcedure.VectorAcc)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                if '32k' in self.instrument:
                    Aqf.step('Testing maximum channels to %s due to quantiser snap-block and '
                             'system performance limitations.' % self.n_chans_selected)
                chan_index = self.n_chans_selected
                n_chans = self.cam_sensors.get_value('n_chans')
                test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
                self._test_vacc(test_chan, chan_index)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.4.9')
    @aqf_requirements("CBF-REQ-0119")
    def test_gain_correction(self):
        # The CBF shall apply gain correction per antenna, per polarisation, per frequency channel
        # with a range of at least $$\pm 6 \; dB$$ and a resolution of $$\le 1 \; db$$.
        Aqf.procedure(TestProcedure.GainCorr)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_gain_correction()
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.4.23')
    @aqf_requirements("CBF-REQ-0013")
    def test_product_switch(self):
        # The CBF shall, on request via the CAM interface, switch between Sub-Array data product
        #  combinations, using the same combination of Receptors, in less than 60 seconds.
        Aqf.procedure(TestProcedure.ProductSwitching)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            Aqf.failed("This requirement is currently not being tested in this release.")
            # _running_inst = which_instrument(self, instrument)
            # instrument_success = self.set_instrument()
            # if instrument_success:
            #     with RunTestWithTimeout(300):
            #         self._test_product_switch(instrument)
            # else:
            #     Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.31')
    @aqf_requirements("CBF-REQ-0066", "CBF-REQ-0072", "CBF-REQ-0077", "CBF-REQ-0110", "CBF-REQ-0200")
    def test_delay_phase_compensation_control(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation_Control)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_delays_control()
                clear_all_delays(self)
                restore_src_names(self)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.32')
    @aqf_requirements("CBF-REQ-0112", "CBF-REQ-0128", "CBF-REQ-0185", "CBF-REQ-0187", "CBF-REQ-0188")
    def test_delay_phase_compensation_functional(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(acc_time=1)
            if instrument_success:
                self._test_delay_tracking()
                self._test_delay_rate()
                self._test_fringe_rate()
                self._test_fringe_offset()
                # self._test_delay_inputs()
                clear_all_delays(self)
                restore_src_names(self)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.27')
    @aqf_requirements("CBF-REQ-0178")
    def test_report_configuration(self):
        # The CBF shall, on request via the CAM interface, report sensors that identify the installed
        # configuration of the CBF unambiguously, including hardware, software and firmware part
        # numbers and versions.

        Aqf.procedure(TestProcedure.ReportConfiguration)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_report_config()
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.29')
    @aqf_requirements("CBF-REQ-0067")
    def test_systematic_error_reporting(self):
        # The CBF shall detect and flag data where the signal integrity has been compromised due to:
        #     a. Digitiser data acquisition and/or signal processing (e.g. ADC saturation),
        #     b. Signal processing and/or data manipulation performed in the CBF (e.g. FFT overflow).
        Aqf.procedure(TestProcedure.PFBFaultDetection)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_fft_overflow()
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.28')
    @aqf_requirements("CBF-REQ-0157")
    def test_fault_detection(self):
        # The CBF shall monitor functions which are in use, and report detected failures of those
        # functions including but not limited to:
        #     a) processing pipeline failures
        #     b) memory errors (SKARAB uses HMC instead of QDR might be tricky to test)
        #     c) network link errors
        # Detected failures shall be reported over the CBF-CAM interface.
        Aqf.procedure(TestProcedure.LinkFaultDetection)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                # self._test_network_link_error()
                # self._test_memory_error()
                test_heading('Processing Pipeline Failures')
                Aqf.note("Test is being qualified by CBF.V.3.29")
                test_heading('HMC Memory errors')
                Aqf.note("See waiver")
                test_heading('Network Link errors')
                Aqf.note("See waiver")
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('CBF.V.3.26')
    @aqf_requirements("CBF-REQ-0056", "CBF-REQ-0068", "CBF-REQ-0069")
    def test_monitor_sensors(self):
        # The CBF shall report the following transient search monitoring data:
        #     a) Transient buffer ready for triggering
        # The CBF shall, on request via the CAM interface, report sensor values.
        # The CBF shall, on request via the CAM interface, report time synchronisation status.
        Aqf.procedure(TestProcedure.MonitorSensors)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_sensor_values()
                self._test_host_sensors_status()
            else:
                Aqf.failed(self.errmsg)


    @generic_test
    @aqf_vr('CBF.V.3.38')
    @aqf_requirements("CBF-REQ-0203")
    def test_time_synchronisation(self):
        Aqf.procedure(TestProcedure.TimeSync)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            self._test_time_sync()

    @generic_test
    @aqf_vr('CBF.V.4.26')
    @aqf_requirements("CBF-REQ-0083", "CBF-REQ-0084", "CBF-REQ-0085", "CBF-REQ-0086", "CBF-REQ-0221")
    def test_antenna_voltage_buffer(self):
        Aqf.procedure(TestProcedure.VoltageBuffer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._small_voltage_buffer()
            else:
                Aqf.failed(self.errmsg)


#---------------------------------------------------------------------------------------------------
#----------------------------------------------MANUAL TESTS-----------------------------------------
#---------------------------------------------------------------------------------------------------

# Perhaps, enlist all manual tests here with VE & REQ

    @manual_test
    @aqf_vr("CBF.V.3.56")
    @aqf_requirements("CBF-REQ-0228")
    def test__subarray(self):
        self._test_global_manual("CBF.V.3.56")

    @manual_test
    @generic_test
    @aqf_vr('CBF.V.3.37')
    @aqf_requirements("CBF-REQ-0071", "CBF-REQ-0204")
    def test__control(self):
        self._test_global_manual("CBF.V.3.37")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.37*'))
        Report_Images(image_files)


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.1.11")
    @aqf_requirements("CBF-REQ-0137")
    def test__procured_items_emc_certification(self):
        self._test_global_manual("CBF.V.1.11")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.3")
    @aqf_requirements("CBF-REQ-0018", "CBF-REQ-0019", "CBF-REQ-0022", "CBF-REQ-0024")
    @aqf_requirements("CBF-REQ-0011", "CBF-REQ-0012", "CBF-REQ-0014", "CBF-REQ-0016", "CBF-REQ-0017")
    @aqf_requirements("CBF-REQ-0027", "CBF-REQ-0064")
    def test__states_and_modes_ve(self):
        self._test_global_manual("CBF.V.3.3")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.77")
    @aqf_requirements("CBF-REQ-0021")
    def test__full_functional_mode_ve(self):
        self._test_global_manual("CBF.V.3.77")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.15")
    @aqf_requirements("CBF-REQ-0131", "CBF-REQ-0132", "CBF-REQ-0133")
    def test__power_supply_ve(self):
        self._test_global_manual("CBF.V.3.15")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.16")
    @aqf_requirements("CBF-REQ-0199")
    def test__safe_design_ve(self):
        self._test_global_manual("CBF.V.3.16")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.17")
    @aqf_requirements("CBF-REQ-0061")
    def test__lru_status_and_display_ve(self):
        self._test_global_manual("CBF.V.3.17")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.18")
    @aqf_requirements("CBF-REQ-0197")
    def test__cots_lru_status_and_display_ve(self):
        self._test_global_manual("CBF.V.3.18")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.18*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.19")
    @aqf_requirements("CBF-REQ-0182")
    def test__interchangeability_ve(self):
        self._test_global_manual("CBF.V.3.19")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.20")
    @aqf_requirements("CBF-REQ-0168", "CBF-REQ-0171" )
    def test__periodic_maintenance_lru_storage_ve(self):
        self._test_global_manual("CBF.V.3.20")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.21")
    @aqf_requirements("CBF-REQ-0169", "CBF-REQ-0170", "CBF-REQ-0172", "CBF-REQ-0173")
    def test__lru_storage_ve(self):
        self._test_global_manual("CBF.V.3.21")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.22")
    @aqf_requirements("CBF-REQ-0147"," CBF-REQ-0148" )
    def test__item_handling_ve(self):
        self._test_global_manual("CBF.V.3.22")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.23")
    @aqf_requirements("CBF-REQ-0152", "CBF-REQ-0153", "CBF-REQ-0154", "CBF-REQ-0155", "CBF-REQ-0184")
    def test__item_marking_and_labelling_ve(self):
        self._test_global_manual("CBF.V.3.23")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.23*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.24")
    @aqf_requirements("CBF-REQ-0162")
    def test__use_of_cots_equipment_ve(self):
        self._test_global_manual("CBF.V.3.24")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.25")
    @aqf_requirements("CBF-REQ-0060", "CBF-REQ-0177", "CBF-REQ-0196")
    def test__logging_ve(self):
        self._test_global_manual("CBF.V.3.25")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.25*'))
        Report_Images(image_files)


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.33")
    @aqf_requirements("CBF-REQ-0103")
    def test__accumulator_dynamic_range_ve(self):
        self._test_global_manual("CBF.V.3.33")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.36")
    @aqf_requirements("CBF-REQ-0001")
    def test__data_products_available_for_all_receivers_ve(self):
        self._test_global_manual("CBF.V.3.36")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.39")
    @aqf_requirements("CBF-REQ-0140")
    def test__cooling_method_ve(self):
        self._test_global_manual("CBF.V.3.39")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.39*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.40")
    @aqf_requirements("CBF-REQ-0142", "CBF-REQ-0143")
    def test__humidity_ve(self):
        self._test_global_manual("CBF.V.3.40")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.41")
    @aqf_requirements("CBF-REQ-0145")
    def test__storage_environment_ve(self):
        self._test_global_manual("CBF.V.3.41")


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.42")
    @aqf_requirements("CBF-REQ-0141")
    def test__temperature_range_ve(self):
        self._test_global_manual("CBF.V.3.42")


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.43")
    @aqf_requirements("CBF-REQ-0146")
    def test__transportation_of_components_ve(self):
        self._test_global_manual("CBF.V.3.43")


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.44")
    @aqf_requirements("CBF-REQ-0156")
    def test__product_marking_environmentals_ve(self):
        self._test_global_manual("CBF.V.3.44")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.44*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.45")
    @aqf_requirements("CBF-REQ-0158", "CBF-REQ-0160")
    def test__fail_safe_ve(self):
        self._test_global_manual("CBF.V.3.45")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.47")
    @aqf_requirements("CBF-REQ-0161", "CBF-REQ-0186")
    def test__safe_physical_design_ve(self):
        self._test_global_manual("CBF.V.3.47")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.48")
    @aqf_requirements("CBF-REQ-0107")
    def test__digitiser_cam_data_ve(self):
        self._test_global_manual("CBF.V.3.48")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.50")
    @aqf_requirements("CBF-REQ-0149")
    def test__mtbf_ve(self):
        self._test_global_manual("CBF.V.3.50")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.52")
    @aqf_requirements("CBF-REQ-0179", "CBF-REQ-0180", "CBF-REQ-0190"," CBF-REQ-0194")
    @aqf_requirements("CBF-REQ-0201", "CBF-REQ-0202")
    def test__internal_interfaces_ve(self):
        self._test_global_manual("CBF.V.3.52")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.53")
    @aqf_requirements("CBF-REQ-0136", "CBF-REQ-0166")
    def test__external_interfaces_ve(self):
        self._test_global_manual("CBF.V.3.53")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.54")
    @aqf_requirements("CBF-REQ-0150", "CBF-REQ-0151")
    def test__lru_replacement_ve(self):
        self._test_global_manual("CBF.V.3.54")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.3.54*'))
        Report_Images(image_files)

    @untested
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.57")
    @aqf_requirements("CBF-REQ-0193")
    # @aqf_requirements("CBF-REQ-0195", "CBF-REQ-0230", "CBF-REQ-0231", "CBF-REQ-0232",)
    # @aqf_requirements("CBF-REQ-0233", "CBF-REQ-0235")
    def test__data_subscribers_link_ve(self):
        self._test_global_manual("CBF.V.3.57")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.6.9")
    @aqf_requirements("CBF-REQ-0138")
    def test__design_to_emc_sans_standard_ve(self):
        self._test_global_manual("CBF.V.6.9")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.6.9*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.6.10")
    @aqf_requirements("CBF-REQ-0139")
    def test__design_standards_ve(self):
        self._test_global_manual("CBF.V.6.10")
        image_files = sorted(glob.glob(self._images_dir + '/CBF.V.6.10*'))
        Report_Images(image_files)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.66")
    @aqf_requirements("CBF-REQ-0223")
    def test__channelised_voltage_data_transfer_ve(self):
        self._test_global_manual("CBF.V.3.66")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.49")
    @aqf_requirements("CBF-REQ-0224")
    def test__route_basic_spectrometer_data_ve(self):
        self._test_global_manual("CBF.V.3.49")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.58")
    @aqf_requirements("CBF-REQ-0237")
    def test__subarray_data_product_set_ve(self):
        self._test_global_manual("CBF.V.3.58")


#----------------------------------------------NOT TESTED-----------------------------------------
#---------------------------------------------------------------------------------------------------

    # @untested
    # @generic_test
    # @aqf_vr('CBF.V.3.61')
    # @aqf_requirements("CBF-REQ-0007")
    # def test_1_vlbi_data_product(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be qualified on AR3.")

    # @untested
    # @generic_test
    # @aqf_vr('CBF.V.3.68')
    # @aqf_requirements("CBF-REQ-0025.1")
    # def test_1_data_product_xband(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be qualified on AR3.")

    # @untested
    # @generic_test
    # @aqf_vr('CBF.V.3.30')
    # @aqf_requirements("CBF-REQ-0035", "CBF-REQ-0036", "CBF-REQ-0039")
    # @aqf_requirements("CBF-REQ-0041", "CBF-REQ-0044", "CBF-REQ-0050")
    # @aqf_requirements("CBF-REQ-0051", "CBF-REQ-0052", "CBF-REQ-0054", "CBF-REQ-0198", "CBF-REQ-0226")
    # @aqf_requirements("CBF-REQ-0227", "CBF-REQ-0236", "CBF-REQ-0243")
    # def test__channelisation(self):
    #     # Aqf.procedure(TestProcedure.Channelisation)
    #     Aqf.not_tested("This requirement will not be qualified on AR3.")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.71")
    # @aqf_requirements("CBF-REQ-0076")
    # def test__tied_array_repoint_time(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.76")
    # @aqf_requirements("CBF-REQ-0081", "CBF-REQ-0082")
    # def test__incoherent_beam_total_power_ve(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.7.11")
    # @aqf_requirements("CBF-REQ-0088", "CBF-REQ-0089", "CBF-REQ-0090", "CBF-REQ-0091")
    # def test__antenna_correlation_products(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.69")
    # @aqf_requirements("CBF-REQ-0093", "CBF-REQ-0042")
    # def test__vlbi_channelisation(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.63")
    # @aqf_requirements("CBF-REQ-0095", "CBF-REQ-0030")
    # def test__pulsar_timing_data_product_set(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.75")
    # @aqf_requirements("CBF-REQ-0114", "CBF-REQ-0115")
    # def test__polarisation_correction_ve(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.72")
    # @aqf_requirements("CBF-REQ-0121")
    # def test__ta_antenna_delay_correction(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.73")
    # @aqf_requirements("CBF-REQ-0122")
    # def test__ta_beam_pointing(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.70")
    # @aqf_requirements("CBF-REQ-0220")
    # def test__beam_pointing_polynomial(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.74")
    # @aqf_requirements("CBF-REQ-0229")
    # def test__incoherent_summation(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.65")
    # @aqf_requirements("CBF-REQ-0239")
    # def test__transient_search_data_product_set(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.59")
    # @aqf_requirements("CBF-REQ-0240")
    # def test__flys_eye_data_product_set(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    # @untested
    # @generic_test
    # @aqf_vr("CBF.V.3.60")
    # @aqf_requirements("CBF-REQ-0241")
    # def test__generic_tiedarray_data_product_set(self):
    #     Aqf.procedure("TBD")
    #     Aqf.not_tested("This requirement will not be tested on AR3")

    @generic_test
    @aqf_vr("CBF.V.A.IF")
    def test__informal(self):
        Aqf.procedure("This verification event pertains to tests that are executed, "
                      "but do not verify any formal requirements."
                      "The procedures and results shall be available in the Qualification Test Report.")
        self._test_informal()

#-----------------------------------------------------------------------------------------------------

    def _systems_tests(self):
        """Checking system stability before and after use"""
        try:
            FNULL = open(os.devnull, 'w')
            subprocess.check_call(['pgrep', '-fol', 'corr2_sensor_servlet.py'], stdout=FNULL,
                stderr=FNULL)
        except Exception:
            LOGGER.exception('Sensor_Servlet PID could not be discovered, might not be running.')

        if not confirm_out_dest_ip(self):
            Aqf.failed('Output destination IP is not the same as the one stored in the register, '
                       'i.e. data is being spewed elsewhere.')
        set_default_eq(self)
        # ---------------------------------------------------------------
        try:
            msg = ('Checking system sensors stability')
            Aqf.step(msg)
            LOGGER.info(msg)
            for i in xrange(1):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=30)
                except:
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value(timeout=30)
                time.sleep(10)

            _errored_sensors_ = ', '.join(sorted(list(set([i.arguments[2] for i in informs
                                                           if 'error' in i.arguments[-2]]))))
            _warning_sensors_ = ', '.join(sorted(list(set([i.arguments[2] for i in informs
                                                           if 'warn' in i.arguments[-2]]))))
        except Exception:
            Aqf.note("Could not retrieve sensors via CAM interface.")
        else:
            LOGGER.info('Done' + msg)
            if _errored_sensors_:
                Aqf.note('Following sensors have ERRORS: %s' % _errored_sensors_)
                # print('Following sensors have ERRORS: %s' % _errored_sensors_)
            if _warning_sensors_:
                Aqf.note('Following sensors have WARNINGS: %s' % _warning_sensors_)
                # print('Following sensors have WARNINGS: %s' % _warning_sensors_)


    def _delays_setup(self, test_source_idx=2):
        # Put some correlated noise on both outputs
        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Configure digitiser simulator to generate Gaussian noise.')
        Aqf.progress('Digitiser simulator configured to generate Gaussian noise with scale: {}, '
                 'gain: {} and fft shift: {}.'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, fft_shift=fft_shift,
                                            gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False


        local_src_names = self.cam_sensors.custom_input_labels
        network_latency = float(self.conf_file['instrument_params']['network_latency'])
        cam_max_load_time = int(self.conf_file['instrument_params']['cam_max_load_time'])
        self.corr_fix.issue_metadata
        source_names = self.cam_sensors.input_labels
        # Get name for test_source_idx
        test_source = source_names[test_source_idx]
        ref_source = source_names[0]
        num_inputs = len(source_names)
        # Number of integrations to load delays in the future
        num_int = int(self.conf_file['instrument_params']['num_int_delay_load'])
        Aqf.step('Clear all coarse and fine delays for all inputs before test commences.')
        delays_cleared = clear_all_delays(self)
        if not delays_cleared:
            Aqf.failed('Delays were not completely cleared, data might be corrupted.')
        else:
            Aqf.passed('Cleared all previously applied delays prior to test.')

        Aqf.step('Retrieve initial SPEAD accumulation, in-order to calculate all '
                 'relevant parameters.')
        try:
            initial_dump = get_clean_dump(self)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.progress('Successfully retrieved initial spead accumulation')
            int_time = self.cam_sensors.get_value('int_time')
            synch_epoch = self.cam_sensors.get_value('synch_epoch')
            # n_accs = self.cam_sensors.get_value('n_accs')]
            no_chans = range(self.n_chans_selected)
            time_stamp = initial_dump['timestamp']
            # ticks_between_spectra = initial_dump['ticks_between_spectra'].value
            # int_time_ticks = n_accs * ticks_between_spectra
            t_apply = (initial_dump['dump_timestamp'] + num_int * int_time)
            t_apply_readable =   datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
            curr_time = time.time()
            curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
            try:
                baseline_lookup = get_baselines_lookup(self)
                # Choose baseline for phase comparison
                baseline_index = baseline_lookup[(ref_source, test_source)]
                Aqf.step('Get list of all the baselines present in the correlator output')
                Aqf.progress('Selected input and baseline for testing respectively: %s, %s.'%(
                    test_source, baseline_index))
                Aqf.progress('Time to apply delays: %s (%s), Current cmc time: %s (%s), Delays will be '
                             'applied %s integrations/accumulations in the future.' % (t_apply, 
                             t_apply_readable, curr_time, curr_time_readable, num_int))
            except KeyError:
                Aqf.failed('Initial SPEAD accumulation does not contain correct baseline '
                           'ordering format.')
                return False
            else:
                return {
                        'baseline_index': baseline_index,
                        'baseline_lookup': baseline_lookup,
                        'initial_dump': initial_dump,
                        'int_time': int_time,
                        'network_latency': network_latency,
                        'num_inputs': num_inputs,
                        'sample_period': self.cam_sensors.sample_period,
                        't_apply': t_apply,
                        'test_source': test_source,
                        'test_source_ind': test_source_idx,
                        'time_stamp': time_stamp,
                        'synch_epoch': synch_epoch,
                        'num_int': num_int,
                        'cam_max_load_time': cam_max_load_time,
                       }


    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients, max_wait_dumps=50):
        try:
            Aqf.step('Request Fringe/Delay(s) Corrections via CAM interface.')
            load_strt_time = time.time()
            reply, _informs = self.corr_fix.katcp_rct.req.delays(setup_data['t_apply'],
                                                                 *delay_coefficients, timeout=30)
            load_done_time = time.time()
            errmsg = ('%s: Failed to set delays via CAM interface with load-time: %s, '
                      'Delay coefficients: %s' % (str(reply).replace('\_', ' '), setup_data['t_apply'],
                        delay_coefficients))
            assert reply.reply_ok(), errmsg
            actual_delay_coef = reply.arguments[1:]
            cmd_load_time = round(load_done_time - load_strt_time, 3)
            Aqf.step('Fringe/Delay load command took {} seconds'.format(cmd_load_time))
            _give_up = int(setup_data['num_int'] * setup_data['int_time'] * 2)
            while True:
                _give_up -= 1
                try:
                    LOGGER.info('Waiting for the delays to be updated on sensors: %s retry' % _give_up)
                    try:
                        reply_, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                    except:
                        reply_, informs = self.corr_fix.katcp_rct.req.sensor_value()
                    assert reply_.reply_ok()
                except Exception:
                    LOGGER.exception('Weirdly I could not get the sensor values')
                else:
                    delays_updated = list(set([int(i.arguments[-1]) for i in informs
                                            if '.cd.delay' in i.arguments[2]]))[0]
                    if delays_updated:
                        LOGGER.info('Delays have been successfully set')
                        msg = ('Delays set successfully via CAM interface: reply %s' % str(reply))
                        Aqf.passed(msg)
                        break
                if _give_up == 0:
                    msg = ("Could not confirm the delays in the time stipulated, exiting")
                    LOGGER.error(msg)
                    Aqf.failed(msg)
                    break
                time.sleep(1)

            cam_max_load_time = setup_data['cam_max_load_time']
            msg = 'Time it took to load delay/fringe(s) %s is less than %ss' % (cmd_load_time,
                    cam_max_load_time)
            Aqf.less(cmd_load_time, cam_max_load_time, msg)
        except Exception:
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        last_discard = setup_data['t_apply'] - setup_data['int_time']
        num_discards = 0
        fringe_dumps = []
        Aqf.step('Getting SPEAD accumulation containing the change in fringes(s) on input: %s '
                 'baseline: %s, and discard all irrelevant accumulations.' % (
                  setup_data['test_source'], setup_data['baseline_index']))
        while True:
            num_discards += 1
            try:
                dump = self.receiver.data_queue.get()
                self.assertIsInstance(dump, dict)
            except Exception:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                time_diff = np.abs(dump['dump_timestamp'] - last_discard)
                if time_diff  < 0.1 * setup_data['int_time']:
                    fringe_dumps.append(dump)
                    msg = ('Received final accumulation before fringe '
                           'application with dump timestamp: %s, relevant to time apply: %s '
                           '(Difference %s)' % (dump['dump_timestamp'], setup_data['t_apply'],
                                time_diff))
                    Aqf.passed(msg)
                    LOGGER.info(msg)
                    break

                if num_discards > max_wait_dumps:
                    Aqf.failed('Could not get accumulation with correct timestamp within %s '
                               'accumulation periods.' % max_wait_dumps)
                    # break
                    return
                else:
                    msg = ("Discarding (#%d) Spead accumulation with dump timestamp: %s"
                           ", relevant to time to apply: %s"
                           "(Difference %.2f), Current cmc time: %s." % (num_discards,
                            dump['dump_timestamp'], setup_data['t_apply'], time_diff, time.time()))
                    LOGGER.info(msg)
                    if num_discards <= 2:
                        Aqf.progress(msg)
                    elif num_discards == 3:
                        Aqf.progress('...')
                    elif time_diff < 3:
                        Aqf.progress(msg)

        for i in xrange(dump_counts - 1):
            Aqf.progress('Getting subsequent SPEAD accumulation {}.'.format(i + 1))
            try:
                dump = self.receiver.data_queue.get()
                self.assertIsInstance(dump, dict)
            except Exception:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                fringe_dumps.append(dump)

        chan_resp = []
        phases = []
        for acc in fringe_dumps:
            dval = acc['xeng_raw']
            freq_response = normalised_magnitude(dval[:, setup_data['baseline_index'], :])
            chan_resp.append(freq_response)
            data = complexise(dval[:, setup_data['baseline_index'], :])
            phases.append(np.angle(data))
        return zip(phases, chan_resp), actual_delay_coef


    def _get_expected_data(self, setup_data, dump_counts, delay_coefficients, actual_phases):

        def calc_actual_delay(setup_data):
            no_ch = self.cam_sensors.get_value('n_chans')
            first_dump = np.unwrap(actual_phases[0])
            actual_slope = np.polyfit(xrange(0, no_ch), first_dump, 1)[0] * no_ch
            actual_delay = self.cam_sensors.sample_period * actual_slope / np.pi
            return actual_delay

        def gen_delay_vector(delay, setup_data):
            res = []
            no_ch = self.cam_sensors.get_value('n_chans')
            delay_slope = np.pi * (delay / self.cam_sensors.sample_period)
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
                tot_delay = (delay + avg_delay_rate * self.cam_sensors.get_value('int_time'))
                expected_phases.append(gen_delay_vector(tot_delay, setup_data))
            return expected_phases

        def calc_actual_offset(setup_data):
            no_ch = self.cam_sensors.get_value('n_chans')
            # mid_ch = no_ch / 2
            first_dump = actual_phases[0]
            # Determine average offset around 5 middle channels
            actual_offset = np.average(first_dump)  # [mid_ch-3:mid_ch+3])
            return actual_offset

        def gen_fringe_vector(offset, setup_data):
            return [offset] * self.cam_sensors.get_value('n_chans')

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
                offset = -(fringe_offset + avg_fringe_rate * self.cam_sensors.get_value('int_time'))
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

        delay_data = np.array((gen_delay_data(delay, delay_rate, dump_counts+1, setup_data)))[1:]
        fringe_data = np.array(gen_fringe_data(fringe_offset, fringe_rate, dump_counts+1, setup_data))[1:]
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


    def _process_power_log(self, start_timestamp, power_log_file):
        max_power_per_rack = 6.25
        max_power_diff_per_rack = 33
        max_power_cbf = 60
        time_gap = 60

        df = pd.read_csv(power_log_file, delimiter='\t')
        headers = list(df.keys())
        exp_headers = ['Sample Time', 'PDU Host', 'Phase Current', 'Phase Power']
        if headers != exp_headers:
            raise IOError(power_log_file)
        pdus = list(set(list(df[headers[1]])))
        # Slice out requested time block
        end_ts = df['Sample Time'].iloc[-1]
        try:
            strt_idx = df[df['Sample Time'] >= int(start_timestamp)].index
        except TypeError:
            msg = ''
            Aqf.failed(msg)
            LOGGER.exception(msg)
        else:
            df = df.loc[strt_idx]
            end_idx = df[df['Sample Time'] <= end_ts].index
            df = df.loc[end_idx]
            # Check for gaps and warn
            time_stamps = df['Sample Time'].values
            ts_diff = np.diff(time_stamps)
            time_gaps = np.where(ts_diff > time_gap)
            for idx in time_gaps[0]:
                ts = time_stamps[idx]
                diff_time = datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d_%H:%M')
                diff = ts_diff[idx]
                Aqf.step('Time gap of {}s found at {} in PDU samples.'.format(diff, diff_time))
            # Convert power column to floats and build new array
            df_list = np.asarray(df.values.tolist())
            power_col = [x.split(',') for x in df_list[:, 3]]
            power_col = [[float(x) for x in y] for y in power_col]
            curr_col = [x.split(',') for x in df_list[:, 2]]
            curr_col = [[float(x) for x in y] for y in curr_col]
            cp_col = zip(curr_col, power_col)
            power_array = []
            for idx, val in enumerate(cp_col):
                power_array.append([df_list[idx, 0], df_list[idx, 1], val[0], val[1]])
            # Cut array into sets containing all pdus for a time slice
            num_pdus = len(pdus)
            rolled_up_samples = []
            time_slice = []
            name_found = []
            pdus = np.asarray(pdus)
            # Create dictionary with rack names
            pdu_samples = {x: [] for x in pdus}
            for _time, name, cur, power in power_array:
                try:
                    name_found.index(name)
                except ValueError:
                    # Remove NANs
                    if not (np.isnan(cur[0]) or np.isnan(power[0])):
                        time_slice.append([_time, name, cur, power])
                    name_found.append(name)
                else:
                    # Only add time slices with samples from all PDUS
                    if len(time_slice) == num_pdus:
                        rolled_up = np.zeros(3)
                        for sample in time_slice:
                            rolled_up += np.asarray(sample[3])
                        # add first timestamp from slice
                        rolled_up = np.insert(rolled_up, 0, int(time_slice[0][0]))
                        rolled_up_samples.append(rolled_up)
                        # Populate samples per pdu
                        for name in pdus:
                            sample = next(x for x in time_slice if x[1] == name)
                            sample = (sample[2], sample[3])
                            smple = np.asarray(sample)
                            pdu_samples[name].append(smple)

                    time_slice = []
                    name_found = []
            if rolled_up_samples:
                start_time = datetime.fromtimestamp(rolled_up_samples[0][0]).strftime('%Y-%m-%d %H:%M:%S')
                end_time = datetime.fromtimestamp(rolled_up_samples[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
                ru_smpls = np.asarray(rolled_up_samples)
                tot_power = ru_smpls[:, 1:4].sum(axis=1)
                Aqf.step("Compile Power consumption report while running SFDR test.")
                Aqf.progress('Power report from {} to {}'.format(start_time, end_time))
                Aqf.progress('Average sample time: {}s'.format(int(np.diff(ru_smpls[:, 0]).mean())))
                # Add samples for pdus in same rack
                rack_samples = {x[:x.find('-')]: [] for x in pdus}
                for name in pdu_samples:
                    rack_name = name[:name.find('-')]
                    if rack_samples[rack_name] != []:
                        sample = np.add(rack_samples[rack_name], pdu_samples[name])
                    else:
                        sample = pdu_samples[name]
                    rack_samples[rack_name] = sample
                for rack in rack_samples:
                    val = np.asarray(rack_samples[rack])
                    curr = val[:, 0]
                    power = val[:, 1]
                    watts = power.sum(axis=1).mean()
                    Aqf.step("Measure CBF Power rack and confirm power consumption is less than 6.25kW")
                    msg = ('Measured power for rack {} ({:.2f}kW) is less than {}kW'.format(
                            rack, watts, max_power_per_rack))
                    Aqf.less(watts, max_power_per_rack, msg)
                    phase = np.zeros(3)
                    for i, x in enumerate(phase):
                        phase[i] = curr[:, i].mean()
                    Aqf.step("Measure CBF Power and confirm power consumption is less than 60kW")
                    Aqf.progress('Average current per phase for rack {}: P1={:.2f}A, P2={:.2f}A, '
                            'P3={:.2f}A'.format(rack, phase[0], phase[1], phase[2]))
                    ph_m = np.max(phase)
                    max_diff = np.max([100 * (x / ph_m) for x in ph_m - phase])
                    max_diff = float('{:.1f}'.format(max_diff))
                    Aqf.step("Measure CBF Peak Power and confirm power consumption is less than 60kW")
                    msg = ('Maximum difference in current per phase for rack {} ({:.1f}%) is '
                           'less than {}%'.format(rack, max_diff, max_power_diff_per_rack))
                    # Aqf.less(max_diff,max_power_diff_per_rack,msg)
                    # Aqf.waived(msg)
                watts = tot_power.mean()
                msg = 'Measured power for CBF ({:.2f}kW) is less than {}kW'.format(watts,
                    max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)
                watts = tot_power.max()
                msg = 'Measured peak power for CBF ({:.2f}kW) is less than {}kW'.format(watts,
                    max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)

    #################################################################
    #                       Test Methods                            #
    #################################################################

    def _test_channelisation(self, test_chan=1500, no_channels=None, req_chan_spacing=None):
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=101,
                                                                 chans_around=2)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
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

        print_counts = 3

        if '4k' in self.instrument:
            cw_scale = 0.7
            awgn_scale = 0.085
            gain = '7+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave (cwg0), '
                 'with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}'.format(cw_scale,
                                                                                        awgn_scale,
                                                                                        gain,
                                                                                        fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=expected_fc, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False
        try:
            Aqf.step('Randomly select a frequency channel to test. Capture an initial correlator '
                     'SPEAD accumulation, determine the number of frequency channels')
            initial_dump = self.receiver.get_clean_dump(discard=30)
            self.assertIsInstance(initial_dump, dict)
        except Exception:
            errmsg = 'Could not retrieve initial clean SPEAD accumulation: Queue is Empty.'
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return
        else:
            bls_to_test = eval(self.cam_sensors.get_value('bls_ordering'))[test_baseline]
            Aqf.progress('Randomly selected frequency channel to test: {} and '
                         'selected baseline {} / {} to test.'.format(test_chan, test_baseline,
                            bls_to_test))
            Aqf.equals(4096, no_channels,
                       'Confirm that the number of channels in the SPEAD accumulation, is equal '
                       'to the number of frequency channels as calculated: {}'.format(
                          no_channels))
            Aqf.step('The CBF, when configured to produce the Imaging data product set and Wideband '
                    'Fine resolution channelisation, shall channelise a total bandwidth of >= %s' %(
                        min_bandwithd_req))
            Aqf.is_true(self.cam_sensors.get_value('bandwidth') >= min_bandwithd_req,
                        'Channelise total bandwidth {}Hz shall be >= {}Hz.'.format(
                            self.cam_sensors.get_value('bandwidth'), min_bandwithd_req))
            # TODO (MM) 2016-10-27, As per JM
            # Channel spacing is reported as 209.266kHz. This is probably spot-on, considering we're
            # using a dsim that's not actually sampling at 1712MHz. But this is problematic for the
            # test report. We would be getting 1712MHz/8192=208.984375kHz on site.
            # Maybe we should be reporting this as a fraction of total sampling rate rather than
            # an absolute value? ie 1/4096=2.44140625e-4 I will speak to TA about how to handle this.
            # chan_spacing = 856e6 / np.shape(initial_dump['xeng_raw'])[0]
            chan_spacing = self.cam_sensors.get_value('bandwidth') / self.cam_sensors.get_value('n_chans')
            chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100),
                                chan_spacing + (chan_spacing * 1 / 100)]
            Aqf.step('CBF-REQ-0043 and CBF-REQ-0053 Confirm that the number of calculated channel '
                     'frequency step is within requirement.')
            msg = ('Verify that the calculated channel frequency (%s Hz)step size is between %s and '
                   '%s Hz' % (chan_spacing, req_chan_spacing / 2, req_chan_spacing))
            Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

            Aqf.step('CBF-REQ-0046 and CBF-REQ-0047 Confirm that the channelisation spacing and '
                     'confirm that it is within the maximum tolerance.')
            msg = ('Channelisation spacing is within maximum tolerance of 1% of the '
                   'channel spacing.')
            Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

            initial_freq_response = normalised_magnitude(initial_dump['xeng_raw'][:, test_baseline, :])
            where_is_the_tone = np.argmax(initial_freq_response)
            max_tone_val = np.max(initial_freq_response)
            Aqf.note("Single peak found at channel %s, with max power of %.5f(%.5fdB)" % (
                where_is_the_tone, max_tone_val, 10*np.log10(max_tone_val)))

            plt_filename = '{}/{}_overall_channel_resolution_Initial_capture.png'.format(self.logs_path,
                self._testMethodName)
            plt_title = 'Initial Overall frequency response at %s'% test_chan
            caption = ('An overall frequency response at the centre frequency %s,'
                       'and selected baseline %s to test. Digitiser simulator is configured to '
                       'generate a continuous wave, with cw scale: %s, awgn scale: %s, Eq gain: %s '
                       'and FFT shift: %s' % (test_chan, test_baseline, cw_scale, awgn_scale, gain,
                        fft_shift))
            aqf_plot_channels(initial_freq_response, plt_filename, plt_title, caption=caption,
                ylimits=(-100, 1))

        Aqf.step('Sweep the digitiser simulator over the centre frequencies of at '
                 'least all the channels that fall within the complete L-band')
        failure_count = 0
        for i, freq in enumerate(requested_test_freqs):
            _msg = ('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(i + 1,
                    len(requested_test_freqs), freq / 1e6))
            if i < print_counts:
                Aqf.progress(_msg)
            elif i == print_counts:
                Aqf.progress('.' * print_counts)
            elif i >= (len(requested_test_freqs) - print_counts):
                Aqf.progress(_msg)
            # else:
            #     LOGGER.debug(_msg)

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
            # self.dhost.sine_sources.sin_1.set(frequency=freq, scale=cw_scale)
            this_source_freq = self.dhost.sine_sources.sin_0.frequency

            if this_source_freq == last_source_freq:
                LOGGER.debug('Skipping channel response for freq %s @ %s: %s MHz.\n'
                            'Digitiser frequency is same as previous.' % (
                                i + 1, len(requested_test_freqs), freq / 1e6))
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq

            try:
                this_freq_dump = self.receiver.get_clean_dump()
                self.assertIsInstance(this_freq_dump, dict)
            except AssertionError:
                failure_count += 1
                errmsg = ('Could not retrieve clean accumulation for freq (%s @ %s: %sMHz).'%(
                            i + 1, len(requested_test_freqs), freq/1e6))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                if failure_count >= 5:
                    _errmsg = 'Cannot continue running the test, Not receiving clean accumulations.'
                    LOGGER.error(_errmsg)
                    Aqf.failed(_errmsg)
                    return False
            else:
                # No of spead heap discards relevant to vacc
                discards = 0
                max_wait_dumps = 50
                deng_timestamp = self.dhost.registers.sys_clkcounter.read().get('timestamp')
                while True:
                    try:
                        queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                        self.assertIsInstance(queued_dump, dict)
                    except Exception:
                        errmsg = ('Could not retrieve clean queued accumulation for freq(%s @ %s: '
                                  '%s MHz).'%(i + 1, len(requested_test_freqs), freq/1e6))
                        LOGGER.exception(errmsg)
                        Aqf.failed(errmsg)
                        break
                    else:
                        timestamp_diff = np.abs(queued_dump['dump_timestamp'] - deng_timestamp)
                        if (timestamp_diff < 1):
                            msg = ('Received correct accumulation timestamp: %s, relevant to '
                                   'DEngine timestamp: %s (Difference %.2f)' % (
                                    queued_dump['dump_timestamp'], deng_timestamp, timestamp_diff))
                            LOGGER.info(_msg)
                            LOGGER.info(msg)
                            break

                        if discards > max_wait_dumps:
                            errmsg = ('Could not get accumulation with correct timestamp within %s '
                                      'accumulation periods.' % max_wait_dumps)
                            Aqf.failed(errmsg)
                            LOGGER.error(errmsg)
                            if discards > 10:
                                return
                            break
                        else:
                            msg = ('Discarding subsequent dumps (%s) with dump timestamp (%s) '
                                   'and DEngine timestamp (%s) with difference of %s.' %(discards,
                                    queued_dump['dump_timestamp'], deng_timestamp, timestamp_diff))
                            LOGGER.info(msg)
                    discards += 1

                this_freq_data = queued_dump['xeng_raw']
                this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            # Plot an overall frequency response at the centre frequency just as
            # a sanity check

            if np.abs(freq - expected_fc) < 0.1:
                plt_filename = '{}/{}_overall_channel_resolution.png'.format(self.logs_path,
                    self._testMethodName)
                plt_title = 'Overall frequency response at {} at {:.3f}MHz.'.format(
                    test_chan, this_source_freq / 1e6)
                max_peak = np.max(loggerise(this_freq_response))
                Aqf.note("Single peak found at channel %s, with max power of %s (%fdB) midway "
                         "channelisation, to confirm if there is no offset." % (
                    np.argmax(this_freq_response), np.max(this_freq_response), max_peak))
                new_cutoff = max_peak - cutoff
                y_axis_limits = (-100, 1)
                caption = ('An overall frequency response at the centre frequency, and ({:.3f}dB) '
                           'and selected baseline {} / {} to test. CBF channel isolation [max channel'
                           ' peak ({:.3f}dB) - ({}dB) cut-off] when '
                           'digitiser simulator is configured to generate a continuous wave, with '
                           'cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(
                                new_cutoff, test_baseline, bls_to_test, max_peak, cutoff, cw_scale,
                                awgn_scale, gain, fft_shift))
                aqf_plot_channels(this_freq_response, plt_filename, plt_title, caption=caption,
                                  ylimits=y_axis_limits, cutoff=new_cutoff)

        if not where_is_the_tone == test_chan:
            Aqf.note("We expect the channel response at %s, but in essence it is in channel %s, ie "
                     "There's a channel offset of %s" % (test_chan, where_is_the_tone,
                        np.abs(test_chan - where_is_the_tone)))
            test_chan += np.abs(test_chan - where_is_the_tone)


        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        df = self.cam_sensors.delta_f
        try:
            rand_chan_response = len(chan_responses[random.randrange(len(chan_responses))])
            # assert rand_chan_response == self.n_chans_selected
        except AssertionError:
            errmsg = ('Number of channels (%s) found on the spead data is inconsistent with the '
                      'number of channels (%s) expected.' %(rand_chan_response,
                        self.n_chans_selected))
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        else:
            np.savetxt("CBF_Efficiency_Data.csv", zip(chan_responses[:, test_chan],
                requested_test_freqs), delimiter=",")
            plt_filename = '{}/{}_Channel_Response.png'.format(self.logs_path,
                self._testMethodName)
            plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90,
                normalise=True)
            plt_caption = ('Frequency channel {} @ {}MHz response vs source frequency and '
                           'selected baseline {} / {} to test.'.format(test_chan, expected_fc / 1e6,
                            test_baseline, bls_to_test))
            plt_title = 'Channel {} @ {:.3f}MHz response.'.format(test_chan, expected_fc / 1e6)
            # Plot channel response with -53dB cutoff horizontal line
            aqf_plot_and_save(freqs=actual_test_freqs[1:-1], data=plot_data[1:-1], df=df,
                              expected_fc=expected_fc, plot_filename=plt_filename,
                              plt_title=plt_title, caption=plt_caption, cutoff=-cutoff)
            try:
                # CBF-REQ-0126
                pass_bw_min_max = np.argwhere((np.abs(plot_data) >= 3.0) & (np.abs(plot_data) <= 3.3))
                pass_bw = float(np.abs(
                        actual_test_freqs[pass_bw_min_max[0]] - actual_test_freqs[pass_bw_min_max[-1]]))

                att_bw_min_max = [np.argwhere(plot_data==i)[0][0] for i in plot_data
                         if (abs(i) >= (cutoff-1)) and (abs(i) <= (cutoff+1))]
                att_bw = actual_test_freqs[att_bw_min_max[-1]] - actual_test_freqs[att_bw_min_max[0]]

            except Exception:
                msg = ('Could not compute if, CBF performs channelisation such that the 53dB '
                       'attenuation bandwidth is less/equal to 2x the pass bandwidth')
                Aqf.failed(msg)
                LOGGER.exception(msg)
            else:
                msg = ('The CBF shall perform channelisation such that the 53dB attenuation bandwidth(%s)'
                       'is less/equal to 2x the pass bandwidth(%s)' %(att_bw, pass_bw))
                Aqf.is_true(att_bw >= pass_bw, msg)

            # Get responses for central 80% of channel
            df = self.cam_sensors.delta_f
            central_indices = ((actual_test_freqs <= expected_fc + 0.4 * df) &
                               (actual_test_freqs >= expected_fc - 0.4 * df))
            central_chan_responses = chan_responses[central_indices]
            central_chan_test_freqs = actual_test_freqs[central_indices]

            # Plot channel response for central 80% of channel
            graph_name_central = '{}/{}_central.png'.format(self.logs_path, self._testMethodName)
            plot_data_central = loggerise(central_chan_responses[:, test_chan], dynamic_range=90,
                                          normalise=True)

            n_chans = self.n_chans_selected
            caption = ('Channel {} central response vs source frequency on max channels {} and '
                       'selected baseline {} / {} to test.'.format(test_chan, n_chans, test_baseline,
                                                                   bls_to_test))
            plt_title = 'Channel {} @ {:.3f} MHz response @ 80%'.format(test_chan, expected_fc / 1e6)

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
                Aqf.failed('The following input frequencies (first and last): {!r} '
                           'respectively had peak channeliser responses in channels '
                           '{!r}\n, and not test channel {} as expected.'.format(
                                fault_freqs[1::-1], set(sorted(fault_channels)), test_chan))

                LOGGER.error('The following input frequencies: %s respectively had '
                             'peak channeliser responses in channels %s, not '
                             'channel %s as expected.' % (fault_freqs, set(sorted(fault_channels)),
                                                          test_chan))

            Aqf.less(np.max(np.abs(central_chan_responses[:, test_chan])), 0.99,
                     'Confirm that the VACC output is at < 99% of maximum value, if fails '
                     'then it is probably over-ranging.')

            max_central_chan_response = np.max(10 * np.log10(central_chan_responses[:, test_chan]))
            min_central_chan_response = np.min(10 * np.log10(central_chan_responses[:, test_chan]))
            chan_ripple = max_central_chan_response - min_central_chan_response
            acceptable_ripple_lt = 1.5
            Aqf.less(chan_ripple, acceptable_ripple_lt,
                     'Confirm that the ripple within 80% of cut-off '
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
            Aqf.step('Confirm that the response at channel-edges are -3 dB '
                     'relative to the channel centre at {:.3f} Hz, actual source freq '
                     '{:.3f} Hz'.format(expected_fc, fc_src_freq))

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

            no_of_responses = 3
            center_bin = [150, 250, 350]
            y_axis_limits = (-90, 1)
            legends = ['Channel {} / Sample {} \n@ {:.3f} MHz'.format(((test_chan + i) - 1), v,
                            self.cam_sensors.ch_center_freqs[test_chan + i] / 1e6)
                       for i, v in zip(range(no_of_responses), center_bin)]
            #center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.n_chans_selected / 1e3))
            center_bin.append('Channel spacing: {:.3f}kHz'.format(chan_spacing / 1e3))

            channel_response_list = [chan_responses[:, test_chan + i - 1]
                                     for i in range(no_of_responses)]
            plot_title = 'PFB Channel Response'
            plot_filename = '{}/{}_adjacent_channels.png'.format(self.logs_path, self._testMethodName)

            caption = ('Sample PFB central channel response between channel {} and selected baseline '
                       '{}/{},with channelisation spacing of {:.3f}kHz within tolerance of 1%, with '
                       'the digitiser simulator configured to generate a continuous wave, with cw scale:'
                       ' {}, awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(test_chan,
                       test_baseline, bls_to_test, chan_spacing / 1e3, cw_scale, awgn_scale, gain,
                       fft_shift))

            np.savetxt("CBF_Efficiency_Data.csv", zip(chan_responses[:, test_chan],
                requested_test_freqs), delimiter=",")
            aqf_plot_channels(zip(channel_response_list, legends), plot_filename, plot_title,
                              normalise=True, caption=caption, cutoff=-cutoff_edge, vlines=center_bin,
                              xlabel='Sample Steps', ylimits=y_axis_limits)

            Aqf.step("Measure the power difference between the middle of the center and the middle of "
                     "the next adjacent bins and confirm that is > -%sdB" % cutoff)
            for bin_num, chan_resp in enumerate(channel_response_list, 1):
                power_diff = np.max(loggerise(chan_resp)) - cutoff
                msg = "Confirm that the power difference (%.2fdB) in bin %s is more than %sdB" %(
                    power_diff, bin_num, -cutoff)
                Aqf.less(power_diff, -cutoff, msg)

            # Plot Central PFB channel response with ylimit 0 to -6dB
            y_axis_limits = (-7, 1)
            plot_filename = '{}/{}_central_adjacent_channels.png'.format(self.logs_path,
                self._testMethodName)
            plot_title = 'PFB Central Channel Response'
            caption = ('Sample PFB central channel response between channel {} and selected baseline '
                       '{}/{}, with the digitiser simulator configured to generate a continuous wave, '
                       'with cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}'.format(
                            test_chan, test_baseline, bls_to_test, cw_scale, awgn_scale, gain, fft_shift))

            aqf_plot_channels(zip(channel_response_list, legends), plot_filename, plot_title,
                              normalise=True, caption=caption, cutoff=-1.5,
                              xlabel='Sample Steps', ylimits=y_axis_limits)

            Aqf.is_true(low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
                        'Confirm that the relative response at the low band-edge '
                        '(-{co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq '
                        '{co_low_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                        'relative to channel centre response.'.format(**locals()))

            Aqf.is_true(low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
                        'Confirm that the relative response at the high band-edge '
                        '(-{co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq '
                        '{co_high_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                        'relative to channel centre response.'.format(**locals()))

    def _test_sfdr_peaks(self, required_chan_spacing, no_channels, cutoff=53, plots_debug=False,
                        log_power=True):
        """Test channel spacing and out-of-channel response

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

        Will loop over all the channels, placing the source frequency as close to the
        centre frequency of that channel as possible.

        Parameters
        ----------
        required_chan_spacing: float
        no_channels: int
        cutoff : float
            Responses in other channels must be at least `-cutoff` dB below the response
            of the channel with centre frequency corresponding to the source frequency

        """
        # Start a power logger in a thread
        if log_power:
            try:
                power_logger = PowerLogger(self.corr_fix._test_config_file)
                power_logger.start()
                power_logger.setName('CBF Power Consumption')
                self.addCleanup(power_logger.stop)
            except Exception:
                errmsg = 'Failed to start power usage logging.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)

        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel no with max response for each frequency
        max_channels = []
        # Channel responses higher than -cutoff dB relative to expected channel
        extra_peaks = []

        # Checking for all channels.
        n_chans = self.n_chans_selected
        msg = ('This tests confirms that the correct channels have the peak response to each'
               ' frequency and that no other channels have significant relative power, while logging '
               'the power usage of the CBF in the background.')
        Aqf.step(msg)
        if log_power:
            Aqf.progress('Logging power usage in the background.')

        if '4k' in self.instrument:
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
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=self.cam_sensors.get_value('bandwidth') / 2.0,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        Aqf.step('Capture an initial correlator SPEAD accumulation, determine the '
                 'number of frequency channels.')
        try:
            initial_dump = get_clean_dump(self)
            self.assertIsInstance(initial_dump, dict)
        except AssertionError:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            Aqf.equals(np.shape(initial_dump['xeng_raw'])[0], no_channels,
                       'Captured an initial correlator SPEAD accumulation, '
                       'determine the number of channels and processing bandwidth: '
                       '{}Hz.'.format(self.cam_sensors.get_value('bandwidth')))
            chan_spacing = (self.cam_sensors.get_value('bandwidth') / np.shape(initial_dump['xeng_raw'])[0])
            # [CBF-REQ-0043]
            calc_channel = ((required_chan_spacing / 2) <= chan_spacing <= required_chan_spacing)
            Aqf.step('Confirm that the number of calculated channel '
                     'frequency step is within requirement.')
            msg = ('Confirm that the calculated channel frequency step size is between {} and '
                   '{} Hz'.format(required_chan_spacing / 2, required_chan_spacing))
            Aqf.is_true(calc_channel, msg)

        Aqf.step('Sweep a digitiser simulator tone over the all channels that fall within the '
                 'complete L-band.')
        channel_response_lst = []
        print_counts = 4
        start_chan = 1  # skip DC channel since dsim puts out zeros for freq=0
        failure_count = 0
        if self.n_chans_selected != self.cam_sensors.get_value('n_chans'):
            _msg = 'Due to system performance the test will sweep a limited number (ie %s) of channels' % (
                self.n_chans_selected)
            Aqf.note(_msg)
            channel_freqs = self.cam_sensors.ch_center_freqs[start_chan:self.n_chans_selected]
        else:
            channel_freqs = self.cam_sensors.ch_center_freqs[start_chan:]

        for channel, channel_f0 in enumerate(channel_freqs, start_chan):
            if channel < print_counts:
                Aqf.progress('Getting channel response for freq %s @ %s: %.3f MHz.' % (channel,
                    len(channel_freqs), channel_f0 / 1e6))
            elif channel == print_counts:
                Aqf.progress('...')
            elif channel > (len(channel_freqs) - print_counts):
                Aqf.progress('Getting channel response for freq %s @ %s: %.3f MHz.' % (channel,
                    len(channel_freqs), channel_f0 / 1e6))
            else:
                LOGGER.info('Getting channel response for freq %s @ %s: %s MHz.' % (channel,
                        len(channel_freqs), channel_f0 / 1e6))

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)
            self.dhost.sine_sources.sin_1.set(frequency=0, scale=0)
            # self.dhost.sine_sources.sin_corr.set(frequency=0, scale=0)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            try:
                this_freq_dump = get_clean_dump(self)
                self.assertIsInstance(this_freq_dump, dict)
            except AssertionError:
                errmsg = ('Could not retrieve clean SPEAD accumulation')
                Aqf.failed(errmsg)
                LOGGER.info(errmsg)
                if failure_count >= 5:
                    _errmsg = 'Giving up the test, failed to capture accumulations after 5 tries.'
                    LOGGER.error(_errmsg)
                    Aqf.failed(_errmsg)
                    failure_count = 0
                    return False
                failure_count += 1
            else:
                this_freq_data = this_freq_dump['xeng_raw']
                this_freq_response = (normalised_magnitude(this_freq_data[:, test_baseline, :]))
                # List of channels to be plotted
                chans_to_plot = (n_chans // 10, n_chans // 2, 9 * n_chans // 10)
                if channel in chans_to_plot:
                    channel_response_lst.append(this_freq_response)

                max_chan = np.argmax(this_freq_response)
                max_channels.append(max_chan)
                # Find responses that are more than -cutoff relative to max
                new_cutoff = np.max(loggerise(this_freq_response)) + cutoff
                unwanted_cutoff = this_freq_response[max_chan] / 10 ** (new_cutoff / 100.)
                extra_responses = [i for i, resp in enumerate(loggerise(this_freq_response))
                                   if i != max_chan and resp >= unwanted_cutoff]

                plt_title = 'Frequency response at {}'.format(channel)
                plt_filename = '{}/{}_channel_{}_resp.png'.format(self.logs_path,
                    self._testMethodName, channel)
                if extra_responses:
                    msg = ('Weirdly found an extra responses on channel %s'% (channel))
                    LOGGER.error(msg)
                    Aqf.note(msg)
                    plt_title = 'Extra responses found around {}'.format(channel)
                    plt_filename = '{}_extra_responses.png'.format(self._testMethodName)
                    plots_debug = True

                extra_peaks.append(extra_responses)
                if plots_debug:
                    plots_debug = False
                    new_cutoff = np.max(loggerise(this_freq_response)) - cutoff
                    aqf_plot_channels(this_freq_response, plt_filename, plt_title,
                                    log_dynamic_range=90, hlines=new_cutoff)

        for channel, channel_resp in zip(chans_to_plot, channel_response_lst):
            plt_filename = '{}/{}_channel_{}_resp.png'.format(self.logs_path,
                self._testMethodName, channel)
            test_freq_mega = channel_freqs[channel] / 1e6
            plt_title = 'Frequency response at {} @ {:.3f} MHz'.format(channel, test_freq_mega)
            caption = ('An overall frequency response at channel {} @ {:.3f}MHz, '
                       'when digitiser simulator is configured to generate a continuous wave, '
                       'with cw scale: {}. awgn scale: {}, eq gain: {}, fft shift: {}'.format(
                            channel, test_freq_mega, cw_scale, awgn_scale, gain, fft_shift))

            new_cutoff = np.max(loggerise(channel_resp)) - cutoff
            aqf_plot_channels(channel_resp, plt_filename, plt_title, log_dynamic_range=90,
                              caption=caption, hlines=new_cutoff)

        channel_range = range(start_chan, len(max_channels) + start_chan)
        Aqf.step('Check that the correct channels have the peak response to each frequency')
        msg = ('Confirm that the correct channel(s) (eg expected channel %s vs actual channel %s) '
               'have the peak response to each frequency' % (max_channels[1], channel_range[1]))

        if max_channels == channel_range:
            Aqf.passed(msg)
        else:
            Aqf.array_almost_equal(max_channels[1:], channel_range[1:], msg)

        msg = ("Confirm that no other channels response more than -%s dB.\n"% cutoff)
        if extra_peaks == [[]] * len(max_channels):
            Aqf.passed(msg)
        else:
            LOGGER.debug('Expected: %s\n\nGot: %s' % (extra_peaks, [[]] * len(max_channels)))
            Aqf.failed(msg)
        if power_logger:
            power_logger.stop()
            start_timestamp = power_logger.start_timestamp
            power_log_file = power_logger.log_file_name
            power_logger.join()
            try:
                test_heading("CBF Power Consumption")
                self._process_power_log(start_timestamp, power_log_file)
            except Exception:
                msg = 'Failed to read/decode the PDU log.'
                Aqf.failed(msg)
                LOGGER.exception(msg)


    def _test_spead_verify(self):
        """This test verifies if a cw tone is only applied to a single input 0,
            figure out if VACC is rooted by 1
        """
        test_heading("SPEAD Accumulation Verification")
        cw_scale = 0.035
        freq = 300e6
        Aqf.step('Digitiser simulator configured to generate cw tone with frequency: {}MHz, '
                 'scale:{} on input 0'.format(freq / 1e6, cw_scale))
        init_dsim_sources(self.dhost)
        LOGGER.info('Set cw tone on pole 0')
        self.dhost.sine_sources.sin_0.set(scale=cw_scale, frequency=freq)
        try:
            Aqf.step('Capture a correlator SPEAD accumulation and, ')
            dump = self.receiver.get_clean_dump(discard=50)
            self.assertIsInstance(dump, dict)
        except AssertionError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            vacc_offset = get_vacc_offset(dump['xeng_raw'])
            msg = ('Confirm that the auto-correlation in baseline 0 contains Non-Zeros, '
                   'and baseline 1 is Zeros, when cw tone is only outputted on input 0.')
            Aqf.equals(vacc_offset, 0, msg)

            # TODO Plot baseline
            Aqf.step('Digitiser simulator reset to Zeros, before next test')
            Aqf.step('Digitiser simulator configured to generate cw tone with frequency: {}Mhz, '
                     'scale:{} on input 1'.format(freq / 1e6, cw_scale))
            init_dsim_sources(self.dhost)
            LOGGER.info('Set cw tone on pole 1')
            self.dhost.sine_sources.sin_1.set(scale=cw_scale, frequency=freq)
            Aqf.step('Capture a correlator SPEAD accumulation and,')
            dump = self.receiver.get_clean_dump(discard=50)
            vacc_offset = get_vacc_offset(dump['xeng_raw'])
            msg = ('Confirm that the auto-correlation in baseline 1 contains non-Zeros, '
                   'and baseline 0 is Zeros, when cw tone is only outputted on input 1.')
            Aqf.equals(vacc_offset, 1, msg)
            init_dsim_sources(self.dhost)


    def _test_product_baselines(self):
        test_heading("CBF Baseline Correlation Products")
        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate Gaussian noise, '
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale,
                                            freq=self.cam_sensors.ch_center_freqs[1500],
                                            fft_shift=fft_shift, gain=gain)

        try:
            Aqf.step('Change CBF input labels and confirm via CAM interface.')
            reply_, _informs = self.corr_fix.katcp_rct.req.input_labels(timeout=60)
            assert reply_.reply_ok()
            ori_source_name = reply_.arguments[1:]
            Aqf.progress('Original source names: {}'.format(', '.join(ori_source_name)))
        except Exception:
            Aqf.failed('Failed to retrieve input labels via CAM interface')
        try:
            for i in xrange(2):
                self.corr_fix.issue_metadata
                time.sleep(1)
            local_src_names = self.cam_sensors.custom_input_labels
            reply, _informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names, timeout=60)
            assert reply.reply_ok()
        except Exception:
            Aqf.failed('Could not retrieve new source names via CAM interface:\n %s' % (str(reply)))
        else:
            source_names = reply.arguments[1:]
            msg = ('Source names changed to: {}'.format(', '.join(source_names)))
            Aqf.passed(msg)

        Aqf.step('Capture an initial correlator SPEAD accumulation, and retrieve list '
                 'of all the correlator input labels via Cam interface.')
        try:
            test_dump = get_clean_dump(self)
            self.assertIsInstance(test_dump, dict)
        except AssertionError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            # Get bls ordering from get dump
            Aqf.step('Get list of all possible baselines (including redundant baselines) present '
                     'in the correlator output from SPEAD accumulation')

            bls_ordering = eval(self.cam_sensors.get_value('bls_ordering'))
            input_labels = sorted(self.cam_sensors.input_labels)
            inputs_to_plot = random.shuffle(input_labels)
            inputs_to_plot = input_labels[:8]
            bls_to_plot = [0, 2, 4, 8, 11, 14, 23, 33]
            baselines_lookup = get_baselines_lookup(self)
            present_baselines = sorted(baselines_lookup.keys())
            possible_baselines = set()
            _ = [possible_baselines.add((li, lj)) for li in input_labels for lj in input_labels]

            test_bl = sorted(list(possible_baselines))
            Aqf.step('Confirm that each baseline (or its reverse-order counterpart) is present in '
                     'the correlator output')

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
            plot_baseline_inds = []
            for bl in plot_baselines:
                if bl in baselines_lookup:
                    plot_baseline_inds.append(baselines_lookup[bl])
                else:
                    plot_baseline_inds.append(baselines_lookup[bl[::-1]])

            plot_baseline_legends = tuple('{bl[0]}, {bl[1]}: {ind}'.format(bl=bl, ind=ind)
                for bl, ind in zip(plot_baselines, plot_baseline_inds))

            msg = 'Confirm that all baselines are present in correlator output.'
            Aqf.is_true(all(baseline_is_present.values()), msg)
            test_data = test_dump['xeng_raw']
            Aqf.step('Expect all baselines and all channels to be '
                     'non-zero with Digitiser Simulator set to output AWGN.')
            msg = 'Confirm that no baselines have all-zero visibilities.'
            Aqf.is_false(zero_baselines(test_data), msg)

            msg = 'Confirm that all baseline visibilities are non-zero across all channels'
            Aqf.equals(nonzero_baselines(test_data), all_nonzero_baselines(test_data), msg)

            Aqf.step('Save initial f-engine equalisations, and ensure they are '
                     'restored at the end of the test')

            initial_equalisations = get_and_restore_initial_eqs(self)
            Aqf.passed('Stored initial F-engine equalisations: %s' % initial_equalisations)

            def set_zero_gains():
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain_all(0, timeout=60)
                    assert reply.reply_ok()
                except Exception as e:
                    Aqf.failed('Failed to set equalisations on all F-engines: due to %s'%str(e))
                else:
                    Aqf.passed('%s: All the inputs equalisations have been set to Zero.'%str(reply))

            def read_zero_gains():
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain_all()
                    assert reply.reply_ok()
                    eq_values = reply.arguments[-1]
                except Exception:
                    Aqf.failed('{}: Failed to retrieve gains/equalisations'.format(str(reply)))
                else:
                    msg = 'Confirm that all the inputs equalisations have been set to \'Zero\'.'
                    Aqf.equals(eq_values, '0j', msg)


            Aqf.step('Set all inputs gains to \'Zero\', and confirm that output product '
                     'is all-zero')

            set_zero_gains()
            read_zero_gains()

            test_data = self.receiver.get_clean_dump(discard=50)

            Aqf.is_false(nonzero_baselines(test_data['xeng_raw']),
                'Confirm that all baseline visibilities are \'Zero\'.\n')
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

            bls_msg = ('Iterate through input combinations, verifying for each that '
                       'the correct output appears in the correct baseline product.\n')
            Aqf.step(bls_msg)
            # dataFrame = pd.DataFrame(index=sorted(input_labels),
            #                          columns=list(sorted(present_baselines)))

            for count, inp in enumerate(input_labels, start=1):
                old_eq = complex(initial_equalisations[inp])
                Aqf.step('Iteratively set gain/equalisation correction on relevant '
                         'input %s set to %s.' % (inp, old_eq))
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(inp, old_eq)
                    assert reply.reply_ok()
                except Exception:
                    errmsg = '%s: Failed to set gain/eq of %s for input %s' % (str(reply), old_eq,
                                                                                inp)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                else:
                    msg = 'Gain/Equalisation correction on input %s set to %s.' % (inp, old_eq)
                    Aqf.passed(msg)
                    zero_inputs.remove(inp)
                    nonzero_inputs.add(inp)
                    expected_z_bls, expected_nz_bls = (calc_zero_and_nonzero_baselines(nonzero_inputs))
                    try:
                        Aqf.step('Retrieving SPEAD accumulation and confirm if gain/equalisation '
                                 'correction has been applied.')
                        test_dump = get_clean_dump(self)
                        self.assertIsInstance(test_dump, dict)
                    except Exception:
                        errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        test_data = test_dump['xeng_raw']
                        # plot baseline channel response
                        if inp in inputs_to_plot:
                            plot_data = [normalised_magnitude(test_data[:, i, :])
                                         # plot_data = [loggerise(test_data[:, i, :])
                                         for i in plot_baseline_inds]
                            plot_filename = '{}/{}_channel_resp_{}.png'.format(self.logs_path,
                                            self._testMethodName.replace(' ', '_'), inp)

                            plot_title = ('Baseline Correlation Products on input: %s' % inp)

                            _caption = ('Baseline Correlation Products on input:{} {} with the '
                                        'following non-zero inputs:\n {} \n '
                                        'and\nzero inputs:\n {}'.format(inp, bls_msg,
                                        ', '.join(sorted(nonzero_inputs)),
                                        ', '.join(sorted(zero_inputs))))

                            aqf_plot_channels(zip(plot_data, plot_baseline_legends), plot_filename,
                                              plot_title, log_dynamic_range=None, log_normalise_to=1,
                                              caption=_caption, ylimits=(-0.1, np.max(plot_data) + 0.1))
                        actual_nz_bls_indices = all_nonzero_baselines(test_data)
                        actual_nz_bls = set([tuple(bls_ordering[i]) for i in actual_nz_bls_indices])

                        actual_z_bls_indices = zero_baselines(test_data)
                        actual_z_bls = set([tuple(bls_ordering[i]) for i in actual_z_bls_indices])
                        msg = ('Confirm that the expected baseline visibilities are non-zero with '
                               'non-zero inputs')
                        Aqf.step(msg)
                        msg = msg + ' (%s) and,' % (sorted(nonzero_inputs))
                        Aqf.equals(actual_nz_bls, expected_nz_bls, msg)


                        msg = ('Confirm that the expected baselines visibilities are \'Zeros\'.\n')
                        Aqf.step(msg)
                        Aqf.equals(actual_z_bls, expected_z_bls, msg)

                        # Sum of all baselines powers expected to be non zeros
                        sum_of_bl_powers = (
                            [normalised_magnitude(test_data[:, expected_bl, :])
                             for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                                                 for expected_nz_bl_ind in sorted(expected_nz_bls)]])
                        test_data = None
                        # dataFrame.loc[inp][sorted(
                        #     [i for i in expected_nz_bls])[-1]] = np.sum(sum_of_bl_powers)

            # dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')


    def _test_back2back_consistency(self):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect.
        """
        test_heading("Spead Accumulation Back-to-Back Consistency")
        Aqf.step('Randomly select a channel to test.')
        n_chans = self.cam_sensors.get_value('n_chans')
        test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
        test_baseline = 0  # auto-corr
        Aqf.progress('Randomly selected test channel %s and bls %s'%(test_chan, test_baseline))
        Aqf.step('Calculate a list of frequencies to test')
        requested_test_freqs = self.cam_sensors.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        source_period_in_samples = self.n_chans_selected * 2
        cw_scale = 0.675
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=cw_scale,
                                          repeat_n=source_period_in_samples)
        Aqf.step('Digitiser simulator configured to generate periodic wave '
                 '({:.3f}Hz with FFT-length {}) in order for each FFT to be '
                 'identical.'.format(expected_fc / 1e6, source_period_in_samples))

        try:
            this_freq_dump = get_clean_dump(self)
            assert isinstance(this_freq_dump, dict)
        except AssertionError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        else:
            Aqf.step('Sweep the digitiser simulator over the selected/requested frequencies fall '
                     'within the complete L-band')
            for i, freq in enumerate(requested_test_freqs):
                Aqf.hop('Getting channel response for freq {}/{} @ {:.3f} MHz.'.format(
                    i + 1, len(requested_test_freqs), freq / 1e6))
                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                  repeat_n=source_period_in_samples)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency
                dumps_data = []
                chan_responses = []
                Aqf.step('Getting SPEAD accumulation and confirm that the difference between'
                         ' subsequent accumulation is Zero.')
                for dump_no in xrange(3):
                    if dump_no == 0:
                        try:
                            this_freq_dump = get_clean_dump(self)
                            assert isinstance(this_freq_dump, dict)
                        except AssertionError:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                            return False
                        else:
                            initial_max_freq = np.max(this_freq_dump['xeng_raw'])
                    else:
                        try:
                            this_freq_dump = get_clean_dump(self)
                            assert isinstance(this_freq_dump, dict)
                        except AssertionError:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)

                    this_freq_data = this_freq_dump['xeng_raw']
                    dumps_data.append(this_freq_data)
                    this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
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
                # 'Check that back-to-back accumulations({:.3f}/{:.3f}dB) with the '
                # 'same frequency input differ by no more than {} dB threshold.'.format(
                # dumps_comp, 10 * np.log10(dumps_comp), 10 * np.log10(threshold)))

                msg = ('Confirm that the maximum difference between the subsequent SPEAD accumulations'
                       ' with the same frequency input ({}Hz) is \'Zero\' on baseline {}.'.format(
                            this_source_freq, test_baseline))

                # if not Aqf.equal(dumps_comp, 1, msg):
                if not Aqf.equals(dumps_comp, 0, msg):
                    legends = ['dump #{}'.format(x) for x in xrange(len(chan_responses))]
                    plot_filename = ('{}/{}_chan_resp_{}.png'.format(self.logs_path,
                                     self._testMethodName, i + 1))
                    plot_title = 'Frequency Response {} @ {:.3f}MHz'.format(test_chan,
                                                                            this_source_freq / 1e6)
                    caption = (
                        'Comparison of back-to-back SPEAD accumulations with digitiser simulator '
                        'configured to generate periodic wave ({:.3f}Hz with FFT-length {}) '
                        'in order for each FFT to be identical'.format(this_source_freq,
                                                                       source_period_in_samples))
                    aqf_plot_channels(zip(chan_responses, legends), plot_filename, plot_title,
                                      log_dynamic_range=90, log_normalise_to=1, normalise=False,
                                      caption=caption)


    def _test_freq_scan_consistency(self, threshold=1e-1):
        """This test confirms if the identical frequency scans produce equal results."""
        test_heading("Spead Accumulation Frequency Consistency")
        Aqf.step('Randomly select a channel to test.')
        n_chans = self.cam_sensors.get_value('n_chans')
        test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        Aqf.step('Randomly selected Frequency channel {} @ {:.3f}MHz for testing, and calculate a '
                 'list of frequencies to test'.format(test_chan, expected_fc / 1e6))
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=3,
                                                                chans_around=1)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        source_period_in_samples = self.n_chans_selected * 2

        try:
            test_dump = self.receiver.get_clean_dump()
            assert isinstance(test_dump, dict)
        except Exception:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            cw_scale = 0.675
            Aqf.step('Digitiser simulator configured to generate continuous wave')
            Aqf.step('Sweeping the digitiser simulator over the centre frequencies of at '
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
                                                          repeat_n=source_period_in_samples)
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            this_freq_dump = get_clean_dump(self)
                            assert isinstance(this_freq_dump, dict)
                        except Exception:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            initial_max_freq = np.max(this_freq_dump['xeng_raw'])
                            this_freq_data = this_freq_dump['xeng_raw']
                            initial_max_freq_list.append(initial_max_freq)
                    else:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale,
                                                          repeat_n=source_period_in_samples)
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            this_freq_dump = get_clean_dump(self)
                            assert isinstance(this_freq_dump, dict)
                        except Exception:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            this_freq_data = this_freq_dump['xeng_raw']

                    this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
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
                           'SPEAD accumulations produce equal results.'.format(freq_x / 1e6))

                    if not Aqf.less(np.abs(max_freq_scan), np.abs(np.log10(threshold)), msg):
                        legends = ['Freq scan #{}'.format(x) for x in xrange(len(chan_responses))]
                        caption = ('A comparison of frequency sweeping from {:.3f}Mhz to {:.3f}Mhz '
                                   'scan channelisation and also, {}'.format(
                                        requested_test_freqs[0] / 1e6,
                                        requested_test_freqs[-1] / 1e6, expected_fc, msg))

                        aqf_plot_channels(zip(chan_responses, legends),
                                          plot_filename='{}/{}_chan_resp.png'.format(self.logs_path,
                                              self._testMethodName), caption=caption)

    def _test_restart_consistency(self, instrument, no_channels):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect on CBF restart.
        """
        Aqf.step(self._testMethodDoc)
        threshold = 1.0e1  #
        test_baseline = 0
        n_chans = self.cam_sensors.get_value('n_chans')
        test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan,
            samples_per_chan=3, chans_around=1)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        Aqf.step('Sweeping the digitiser simulator over {:.3f}MHz of the channels that '
                 'fall within {} complete L-band'.format(np.max(requested_test_freqs) / 1e6,
                                                          test_chan))

        if '4k' in self.instrument:
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
                    awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=cw_scale, freq=expected_fc,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            this_freq_dump = get_clean_dump(self)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            # Plot an overall frequency response at the centre frequency just as
            # a sanity check
            init_source_freq = normalised_magnitude(
                this_freq_dump['xeng_raw'][:, test_baseline, :])
            filename = '{}/{}_channel_response.png'.format(self.logs_path, self._testMethodName)
            title = ('Frequency response at {} @ {:.3f} MHz.\n'.format(test_chan,
                                                                         expected_fc / 1e6))
            caption = ('An overall frequency response at the centre frequency.')
            aqf_plot_channels(init_source_freq, filename, title, caption=caption)
            restart_retries = 5

            def _restart_instrument(retries=restart_retries):
                if not self.corr_fix.stop_x_data():
                    Aqf.failed('Could not stop x data from capturing.')
                with ignored(Exception):
                    #deprogram_hosts(self)
                    Aqf.failed('Fix deprogram Hosts')

                corr_init = False
                _empty = True
                with ignored(Queue.Empty):
                    get_clean_dump(self)
                    _empty = False

                Aqf.is_true(_empty,
                            'Confirm that the SPEAD accumulations have stopped being produced.')

                self.corr_fix.halt_array
                xhosts = self.correlator.xhosts
                fhosts = self.correlator.fhosts

                while retries and not corr_init:
                    Aqf.step('Re-initialising the {} instrument'.format(instrument))
                    with ignored(Exception):
                        corr_init = self.set_instrument()

                    retries -= 1
                    if retries == 0:
                        errmsg = ('Could not restart the correlator after %s tries.' % (retries))
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)

                if corr_init.keys()[0] is not True and retries == 0:
                    msg = ('Could not restart {} after {} tries.'.format(instrument, retries))
                    Aqf.end(passed=False, message=msg)
                else:
                    startx = self.corr_fix.start_x_data
                    if not startx:
                        Aqf.failed('Failed to enable/start output product capturing.')
                    host = (xhosts + fhosts)[random.randrange(len(xhosts + fhosts))]
                    msg = ('Confirm that the instrument is initialised by checking if a '
                           'random host: {} is programmed and running.'.format(host.host))
                    Aqf.is_true(host, msg)

                    try:
                        self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                        freq_dump = get_clean_dump(self)
                        assert np.shape(freq_dump['xeng_raw'])[0] == self.n_chans_selected
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    except AssertionError:
                        errmsg = ('Correlator Receiver could not be instantiated or No of channels '
                                  '(%s) in the spead data is inconsistent with the no of'
                                  ' channels (%s) expected' %(np.shape(freq_dump['xeng_raw'])[0],
                                     self.n_chans_selected))
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    else:
                        msg = ('Confirm that the data product has the same number of frequency '
                               'channels {no_channels} corresponding to the {instrument} '
                               'instrument product'.format(**locals()))
                        try:
                            spead_chans = get_clean_dump(self)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            Aqf.equals(4096, no_channels, msg)
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
                        self.corr_fix.halt_array
                        time.sleep(10)
                        self.set_instrument()
                        return False

                scan_dumps = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        if self.corr_fix.start_x_data:
                            Aqf.hop('Getting Frequency SPEAD accumulation #{} with Digitiser simulator '
                                    'configured to generate cw at {:.3f}MHz'.format(i, freq / 1e6))
                            try:
                                this_freq_dump = get_clean_dump(self)
                            except Queue.Empty:
                                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                                Aqf.failed(errmsg)
                                LOGGER.exception(errmsg)
                        initial_max_freq = np.max(this_freq_dump['xeng_raw'])
                        this_freq_data = this_freq_dump['xeng_raw']
                        initial_max_freq_list.append(initial_max_freq)
                        freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    else:
                        msg = ('Getting Frequency SPEAD accumulation #{} with digitiser simulator '
                               'configured to generate cw at {:.3f}MHz'.format(i, freq / 1e6))
                        Aqf.hop(msg)
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        try:
                            this_freq_dump = get_clean_dump(self)
                        except Queue.Empty:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)
                        else:
                            this_freq_data = this_freq_dump['xeng_raw']
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

            msg = ('Confirm that CBF restart SPEAD accumulations comparison results '
                   'with the same frequency input differ by no more than {:.3f}dB '
                   'threshold.'.format(threshold))

            if not Aqf.less(diff_scans_comp, threshold, msg):
                legends = ['Channel Response #{}'.format(x) for x in xrange(len(channel_responses))]
                plot_filename = '{}/{}_chan_resp.png'.format(self.logs_path, self._testMethodName)
                caption = ('Confirm that results are consistent on CBF restart')
                plot_title = ('CBF restart consistency channel response {}'.format(test_chan))
                aqf_plot_channels(zip(channel_responses, legends), plot_filename, plot_title,
                                  caption=caption)

    def _test_delay_tracking(self):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Delay tracking"
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            num_int = setup_data['num_int']
            int_time = self.cam_sensors.get_value('int_time')
            katcp_port = self.cam_sensors.get_value('katcp_port')
            no_chans = range(self.n_chans_selected)
            sampling_period = self.cam_sensors.sample_period
            test_delays = [0, sampling_period, 1.5 * sampling_period, 2 * sampling_period]
            test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
            num_inputs = len(self.cam_sensors.input_labels)
            delays = [0] * setup_data['num_inputs']
            Aqf.step('Delays to be set (iteratively) %s for testing purposes\n' % (test_delays))

            def get_expected_phases():
                expected_phases = []
                for delay in test_delays:
                    # phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                    phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                    phases -= np.max(phases) / 2.
                    expected_phases.append(phases)
                return zip(test_delays_ns, expected_phases)

            def get_actual_phases():
                actual_phases_list = []
                # chan_responses = []
                for count, delay in enumerate(test_delays, 1):
                    delays[setup_data['test_source_ind']] = delay
                    delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                    try:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        this_freq_dump = self.receiver.get_clean_dump(discard=0)
                        self.assertIsInstance(this_freq_dump, dict), errmsg
                        t_apply = this_freq_dump['dump_timestamp'] + (num_int * int_time)
                        t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
                        curr_time = time.time()
                        curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
                        Aqf.step('Delay #%s will be applied with the following parameters:' %count)
                        msg = ('On baseline %s and input %s, Current cmc time: %s (%s)'
                              ', Current Dump timestamp: %s (%s), '
                              'Delay(s) will be applied @ %s (%s), Delay to be applied: %s' %(
                            setup_data['baseline_index'], setup_data['test_source'],
                            curr_time, curr_time_readable, this_freq_dump['dump_timestamp'],
                            this_freq_dump['dump_timestamp_readable'], t_apply, t_apply_readable,
                            delay))
                        Aqf.progress(msg)
                        Aqf.step('Execute delays via CAM interface and calculate the amount of time '
                                 'it takes to load the delays')
                        LOGGER.info('Setting a delay of %s via cam interface' % delay)
                        load_strt_time = time.time()
                        reply, _informs = self.corr_fix.katcp_rct.req.delays(t_apply,
                                                                             *delay_coefficients)
                        load_done_time = time.time()
                        formated_reply = str(reply).replace('\_', ' ')
                        errmsg = ('CAM Reply: %s: Failed to set delays via CAM interface with '
                                  'load-time: %s vs Current cmc time: %s' % (formated_reply,
                                    t_apply, time.time()))
                        assert reply.reply_ok(), errmsg
                        cmd_load_time = round(load_done_time - load_strt_time, 3)
                        Aqf.step('Delay load command took {} seconds'.format(cmd_load_time))
                        _give_up = int(num_int * int_time * 2)
                        while True:
                            _give_up -= 1
                            try:
                                LOGGER.info('Waiting for the delays to be updated on sensors: '
                                            '%s retry' % _give_up)
                                try:
                                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                                except:
                                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value()
                                assert reply.reply_ok()
                            except Exception:
                                LOGGER.exception('Weirdly I couldnt get the sensor values')
                            else:
                                delays_updated = list(set([int(i.arguments[-1]) for i in informs
                                                        if '.cd.delay' in i.arguments[2]]))[0]
                                if delays_updated:
                                    LOGGER.info('%s delay(s) have been successfully set' % delay)
                                    msg = ('Delays set successfully via CAM interface: Reply: %s' %
                                            formated_reply)
                                    Aqf.passed(msg)
                                    break
                            if _give_up == 0:
                                msg = ("Could not confirm the delays in the time stipulated, exiting")
                                LOGGER.error(msg)
                                Aqf.failed(msg)
                                break
                            time.sleep(1)

                        cam_max_load_time = setup_data['cam_max_load_time']
                        msg = ('Time it took to load delays {}s is less than {}s with an '
                              'integration time of {:.3f}s'
                              .format(cmd_load_time, cam_max_load_time, int_time))
                        Aqf.less(cmd_load_time, cam_max_load_time, msg)
                    except Exception as e:
                        Aqf.failed(errmsg + ' Exception: {}'.format(e))
                        LOGGER.exception(errmsg)

                    try:
                        _num_discards = num_int + 5
                        Aqf.step('Getting SPEAD accumulation(while discarding %s dumps) containing '
                                 'the change in delay(s) on input: %s baseline: %s.'%(_num_discards,
                                    setup_data['test_source'], setup_data['baseline_index']))
                        LOGGER.info('Getting dump...')
                        dump = self.receiver.get_clean_dump(discard=_num_discards)
                        LOGGER.info('Done...')
                        assert isinstance(dump, dict)
                        Aqf.progress('Readable time stamp received on SPEAD accumulation: %s '
                                     'after %s number of discards \n'%(
                                        dump['dump_timestamp_readable'], _num_discards))
                    except Exception:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        # # this_freq_data = this_freq_dump['xeng_raw']
                        # # this_freq_response = normalised_magnitude(
                        # #    this_freq_data[:, setup_data['test_source_ind'], :])
                        # # chan_responses.append(this_freq_response)
                        data = complexise(dump['xeng_raw'][:, setup_data['baseline_index'], :])
                        phases = np.angle(data)
                        # # actual_channel_responses = zip(test_delays, chan_responses)
                        # # return zip(actual_phases_list, actual_channel_responses)
                        actual_phases_list.append(phases)
                return actual_phases_list

            expected_phases = get_expected_phases()
            actual_phases = get_actual_phases()

            try:
                if set([float(0)]) in [set(i) for i in actual_phases[1:]] or not actual_phases:
                    Aqf.failed('Delays could not be applied at time_apply: {} '
                               'possibly in the past.\n'.format(setup_data['t_apply']))
                else:
                    # actual_phases = [phases for phases, response in actual_data]
                    # actual_response = [response for phases, response in actual_data]
                    plot_title = 'CBF Delay Compensation'
                    caption = ('Actual and expected Unwrapped Correlation Phase [Delay tracking].\n'
                               'Note: Dashed line indicates expected value and solid line '
                               'indicates actual values received from SPEAD accumulation.')
                    plot_filename = '{}/{}_test_delay_tracking.png'.format(self.logs_path,
                        self._testMethodName)
                    plot_units = 'secs'

                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                           plot_filename, plot_title, plot_units, caption)

                    expected_phases_ = [phase for _rads, phase in expected_phases][:2047]

                    degree = 1.0
                    decimal = len(str(degree).split('.')[-1])
                    try:
                        for i, delay in enumerate(test_delays):
                            delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
                            delta_expected = np.max(expected_phases_[i]) - np.min(
                                expected_phases_[i])
                            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                            # abs_diff = np.abs(delta_expected - delta_actual)
                            msg = ('Confirm that if difference expected({:.5f}) '
                                   'and actual({:.5f}) phases are equal at delay {:.5f}ns within '
                                   '{} degree.'.format(delta_expected, delta_actual, delay * 1e9, degree))
                            Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                            Aqf.less(abs_diff, degree,
                                     'Confirm that the maximum difference ({:.3f} degree/'
                                     ' {:.3f} rad) between expected phase and actual phase between '
                                     'integrations is less than {} degree.\n'.format(abs_diff,
                                        np.deg2rad(abs_diff), degree))
                            try:
                                delta_actual_s = delta_actual - (delta_actual % degree)
                                delta_expected_s = delta_expected - (delta_expected % degree)
                                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                               decimal=decimal)
                            except AssertionError:
                                msg = ('Difference expected({:.5f}) phases'
                                       ' and actual({:.5f}) phases are \'Not almost equal\' '
                                       'within {} degree when delay of {}ns is applied.'.format(
                                            delta_expected, delta_actual, degree, delay * 1e9))
                                Aqf.step(msg)

                                caption = (
                                    'The figure above shows, The difference between expected({:.5f}) '
                                    'phases and actual({:.5f}) phases are \'Not almost equal\' within {} '
                                    'degree when a delay of {:.5f}s is applied. Therefore CBF-REQ-0128 and'
                                    ', CBF-REQ-0187 are not verified.'.format(delta_expected, delta_actual,
                                                                              degree, delay))

                                actual_phases_i = (delta_actual, actual_phases[i])
                                if len(expected_phases[i]) == 2:
                                    expected_phases_i = (delta_expected, expected_phases[i][-1])
                                else:
                                    expected_phases_i = (delta_expected, expected_phases[i])
                                aqf_plot_phase_results(no_chans, actual_phases_i, expected_phases_i,
                                   plot_filename='{}/{}_{}_delay_tracking.png'.format(self.logs_path,
                                        self._testMethodName, i),
                                   plot_title=('Delay offset:\n'
                                               'Actual vs Expected Phase Response'),
                                   plot_units=plot_units, caption=caption)

                        for delay, count in zip(test_delays, xrange(1, len(expected_phases))):
                            msg = ('Confirm that when a delay of {} clock '
                                   'cycle({:.5f} ns) is introduced there is a phase change '
                                   'of {:.3f} degrees as expected to within {} degree.'.format(
                                        (count + 1) * .5, delay * 1e9,
                                        np.rad2deg(np.pi) * (count + 1) * .5, degree))
                            try:
                                Aqf.array_abs_error(actual_phases[count][5:-5],
                                                expected_phases_[count][5:-5], msg, degree)
                            except Exception:
                                Aqf.array_abs_error(actual_phases[count][5:-5],
                                    expected_phases_[count][5:-5+len(actual_phases[count])], msg, degree)
                    except Exception as e:
                        Aqf.failed(e.message)
                        LOGGER.exception(e.message)
                        return
            except Exception as e:
                        Aqf.failed(e.message)
                        LOGGER.exception(e.message)
                        return


    def _test_sensor_values(self):
        """
        Report sensor values
        """
        test_heading("Monitor Sensors: Report Sensor Values")
        def report_sensor_list(self):
            Aqf.step('Confirm that the number of sensors available on the primary '
                     'and sub array interface is consistent.')
            try:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_list(timeout=60)
            except:
                errmsg = 'CAM interface connection encountered errors.'
                Aqf.failed(errmsg)
            else:
                msg = ('Confirm that the number of sensors are equal '
                       'to the number of sensors listed on the running instrument.\n')
                Aqf.equals(int(reply.arguments[-1]), len(informs), msg)

        def report_time_sync(self):
            Aqf.step('Confirm that the time synchronous is implemented on primary interface')
            try:
                reply, informs = self.corr_fix.rct.req.sensor_value('time.synchronised')
            except:
                Aqf.failed('CBF report time sync could not be retrieved from primary interface.')
            else:
                Aqf.is_true(reply.reply_ok(),
                            'CBF report time sync implemented in this release.')

            msg = ('Confirm that the CBF can report time sync status via CAM interface. ')
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
            Aqf.step('Confirm that the Transient Buffer ready is implemented.')
            try:
                assert self.corr_fix.katcp_rct.req.transient_buffer_trigger.is_active()
            except Exception:
                errmsg = ('CBF Transient buffer ready for triggering'
                           '\'Not\' implemented in this release.\n')
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                Aqf.passed('CBF Transient buffer ready for triggering'
                            ' implemented in this release.\n')

        def report_primary_sensors(self):
            Aqf.step('Confirm that all primary sensors are nominal.')
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

    def _test_fft_overflow(self):
        """Sensor PFB error"""
        test_heading("Systematic Errors Reporting: FFT Overflow")
        sensor_poll_time = self.correlator.sensor_poll_time
        # TODO MM, Simplify the test
        ch_list = self.cam_sensors.ch_center_freqs
        cw_freq = ch_list[int(self.n_chans_selected/2)]

        if '4k' in self.instrument:
            cw_scale = 0.7
            awgn_scale = 0.085
            gain = '7+0j'
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = '11+0j'
            fft_shift = 32767

        Aqf.step('Digitiser simulator configured to generate a continuous wave (cwg0), '
                'with cw scale: {}, cw frequency: {}, awgn scale: {}, eq gain: {}, '
                'fft shift: {}'.format(cw_scale, cw_freq, awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=cw_freq, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False
        try:
            Aqf.step('Get the current FFT Shift before manipulation.')
            reply, informs = self.corr_fix.katcp_rct.req.fft_shift()
            assert reply.reply_ok()
            fft_shift = int(reply.arguments[-1])
            Aqf.progress('Current system FFT Shift: %s' % fft_shift)
        except Exception:
            LOGGER.exception()
            Aqf.failed('Could not get the F-Engine FFT Shift value')
            return

        try:
            Aqf.step('Confirm all F-engines do not contain PFB errors/warnings')
            for i in range(3):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=60)
                except:
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value(timeout=60)
                assert reply.reply_ok()
        except Exception:
            msg = 'Failed to retrieve sensor values via CAM interface'
            LOGGER.exception(msg)
            Aqf.failed(msg)
            return
        else:
            pfb_status = list(set([i.arguments[-2] for i in informs
                                    if 'pfb.or0-err-cnt' in i.arguments[2]]))[0]
            Aqf.equals(pfb_status, 'nominal', 'Confirm that all F-Engines report nominal PFB status')

        try:
            Aqf.step('Set an FFT shift of 0 on all f-engines, and confirm if system integrity is affected')
            reply, informs = self.corr_fix.katcp_rct.req.fft_shift(0)
            assert reply.reply_ok()
        except AssertionError:
            msg = 'Could not set FFT shift for all F-Engine hosts'
            Aqf.failed(msg)
            LOGGER.exception(msg)
            return

        try:
            msg = ('Waiting for sensors to trigger.')
            Aqf.wait(self.correlator.sensor_poll_time * 3, msg)

            Aqf.step('Check if all F-engines contain(s) PFB errors/warnings')
            for i in range(3):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=60)
                except:
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value(timeout=60)
                assert reply.reply_ok()
        except Exception:
            msg = 'Failed to retrieve sensor values via CAM interface'
            LOGGER.exception(msg)
            Aqf.failed(msg)
            return
        else:
            pfb_status = list(set([i.arguments[-2] for i in informs
                                    if 'pfb.or0-err-cnt' in i.arguments[2]]))[0]
            Aqf.equals(pfb_status, 'warn',
                'Confirm that all F-Engines report warnings/errors PFB status')

        try:
            Aqf.step('Restore original FFT Shift values')
            reply, informs = self.corr_fix.katcp_rct.req.fft_shift(fft_shift)
            assert reply.reply_ok()
            Aqf.passed('FFT Shift: %s restored.' % fft_shift)
        except Exception:
            LOGGER.exception()
            Aqf.failed('Could not set the F-Engine FFT Shift value')
            return

    def _test_memory_error(self):
        pass

    def _test_network_link_error(self):
        test_heading("Fault Detection: Network Link Errors")
        int2ip = lambda n: socket.inet_ntoa(struct.pack('!I', n))
        ip2int = lambda ipstr: struct.unpack('!I', socket.inet_aton(ipstr))[0]

        def get_spead_data():
            try:
                dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                msg = ('Confirm that the SPEAD accumulation is being produced by '
                       'instrument but not verified.\n')
                Aqf.passed(msg)

        # Record the current multicast destination of one of the F-engine data
        # ethernet ports,
        def get_host_ip(host):
            try:
                int_ip = host.registers.iptx_base.read()['data'].get('reg')
                assert isinstance(int_ip, int)
                return int2ip(int_ip)
            except:
                Aqf.failed('Failed to retrieve multicast destination from %s'.format(
                    host.host.upper()))
                return

        def get_lru_status(self, host):

            if host in self.correlator.fhosts:
                engine_type = 'feng'
            else:
                engine_type = 'xeng'

            try:
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(
                        '%s-%s-lru-ok'.format(host.host, engine_type))
                except:
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value(
                        '%s-%s-lru-ok'.format(host.host, engine_type))
            except:
                Aqf.failed('Could not get sensor attributes on %s'.format(host.host))
            else:
                if reply.reply_ok() and (int(informs[0].arguments[-1]) == 1):
                    return 1
                elif reply.reply_ok() and (int(informs[0].arguments[-1]) == 0):
                    return 0
                else:
                    return False

        # configure the same port multicast destination to an unused address,
        # effectively dropping that data.

        def write_new_ip(host, ip_new, ip_old):
            try:
                ip_new = ip2int(ip_new)
                host.registers.iptx_base.write(**{'reg':int(ip_new)})
                changed_ip = host.registers.iptx_base.read()['data'].get('reg')
                assert isinstance(changed_ip, int)
                changed_ip = int2ip(changed_ip)
            except:
                Aqf.failed('Failed to write new multicast destination on %s' % host.host)
            else:
                Aqf.passed('Confirm that the multicast destination address for %s has been changed '
                           'from %s to %s.' % (host.host, ip_old, changed_ip))

        def report_lru_status(self, host, get_lru_status):
            Aqf.wait(self.correlator.sensor_poll_time,
                     'Wait until the sensors have been updated with new changes')
            if get_lru_status(self, host) == 1:
                Aqf.passed('Confirm that the X-engine %s LRU sensor is OKAY and '
                           'that the X-eng is receiving feasible data.' % (host.host))
            elif get_lru_status(self, host) == 0:
                Aqf.passed('Confirm that the X-engine %s LRU sensor is reporting a '
                           'failure and that the X-eng is not receiving feasible '
                           'data.' % (host.host))
            else:
                Aqf.failed('Failed to read %s sensor' % (host.host))

        fhost = self.correlator.fhosts[random.randrange(len(self.correlator.fhosts))]
        xhost = self.correlator.xhosts[random.randrange(len(self.correlator.xhosts))]
        ip_new = '239.101.2.250'

        Aqf.step('Randomly selected %s host that is being used to produce the test '
                 'data product on which to trigger the link error.' % (fhost.host))
        current_ip = get_host_ip(fhost)
        if not current_ip:
            Aqf.failed('Failed to retrieve multicast destination address of %s' % fhost.host)
        elif current_ip != ip_new:
            Aqf.passed('Current multicast destination address for %s: %s.'%(fhost.host, current_ip))
        else:
            Aqf.failed('Multicast destination address of %s'% (fhost.host))

        Aqf.note('Debug code')
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)

        # write_new_ip(fhost, ip_new, current_ip)
        # time.sleep(self.correlator.sensor_poll_time / 2)
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)

        # Aqf.step('Restoring the multicast destination from %s to the original %s' % (
        #     human_readable_ip(ip_new), human_readable_ip(current_ip)))

        # write_new_ip(fhost, current_ip, ip_new, get_host_ip, human_readable_ip)
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)
        #

    def _test_host_sensors_status(self):
        test_heading('Monitor Sensors: Processing Node\'s Sensor Status')

        Aqf.step('This test confirms that each processing node\'s sensor (Temp, Voltage, Current, '
                 'Fan) has not FAILED, Reports only errors.')
        try:
            try:
                reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
            except:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_value()
            assert reply.reply_ok()
        except AssertionError:
            errmsg = 'Failed to retrieve sensors via CAM interface'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            for i in informs:
                if i.arguments[2].startswith('xhost') or i.arguments[2].startswith('fhost'):
                    if i.arguments[-2].lower() != 'nominal':
                        Aqf.note(' contains a '.join(i.arguments[2:-1]))

    def _test_vacc(self, test_chan, chan_index=None, acc_time=0.998):
        """Test vector accumulator"""
        # Choose a test frequency around the centre of the band.
        test_freq = self.cam_sensors.get_value('bandwidth') / 2.

        test_input = self.cam_sensors.input_labels[0]
        eq_scaling = 30
        acc_times = [acc_time / 2, acc_time]
        #acc_times = [acc_time/2, acc_time, acc_time*2]
        n_chans = self.cam_sensors.get_value("n_chans")
        try:
            internal_accumulations = int(self.cam_sensors.get_value('xeng_acc_len'))
        except Exception as e:
            errmsg = 'Failed to retrieve X-engine accumulation length: %s.' %str(e)
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        try:
            initial_dump = get_clean_dump(self)
            assert isinstance(initial_dump, dict)
        except Exception:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return

        delta_acc_t = self.cam_sensors.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = abs(np.argmin(
                    np.abs(self.cam_sensors.ch_center_freqs[:chan_index] - test_freq)) - test_chan)
        Aqf.step('Selected test input {} and test frequency channel {}'.format(test_input,
            test_freq_channel))
        eqs = np.zeros(n_chans, dtype=np.complex)
        eqs[test_freq_channel] = eq_scaling
        get_and_restore_initial_eqs(self)
        try:
            reply, _informs = self.corr_fix.katcp_rct.req.gain(test_input, *list(eqs))
            assert reply.reply_ok()
            Aqf.hop('Gain successfully set on input %s via CAM interface.' %test_input)
        except Exception:
            errmsg = 'Gains/Eq could not be set on input %s via CAM interface' %test_input
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)

        Aqf.step('Configured Digitiser simulator output(cw0 @ {:.3f}MHz) to be periodic in '
                 'FFT-length: {} in order for each FFT to be identical'.format(test_freq / 1e6,
                        n_chans * 2))

        cw_scale = 0.125
        # The re-quantiser outputs signed int (8bit), but the snapshot code
        # normalises it to floats between -1:1. Since we want to calculate the
        # output of the vacc which sums integers, denormalise the snapshot
        # output back to ints.
        # q_denorm = 128
        # quantiser_spectrum = get_quant_snapshot(self, test_input) * q_denorm
        try:
            # Make dsim output periodic in FFT-length so that each FFT is identical
            self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=cw_scale,
                repeat_n=n_chans * 2)
            self.dhost.sine_sources.sin_1.set(frequency=test_freq, scale=cw_scale,
                repeat_n=n_chans * 2)
            assert self.dhost.sine_sources.sin_0.repeat == n_chans * 2
        except AssertionError:
            errmsg = 'Failed to make the DEng output periodic in FFT-length so that each FFT is identical'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        try:
            reply, informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(test_input)
            assert reply.reply_ok()
            informs = informs[0]
        except Exception:
            errmsg = ('Failed to retrieve quantiser snapshot of input %s via '
                      'CAM Interface: \nReply %s' %(test_input, str(reply).replace('_',' ')))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            quantiser_spectrum = np.array(eval(informs.arguments[-1]))
            if chan_index:
                quantiser_spectrum = quantiser_spectrum[:chan_index]
            # Check that the spectrum is not zero in the test channel
            # Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
            # 'Check that the spectrum is not zero in the test channel')
            # Check that the spectrum is zero except in the test channel
            Aqf.is_true(np.all(quantiser_spectrum[0:test_freq_channel] == 0),
                        'Confirm that the spectrum is zero except in the test channel:'
                        ' [0:test_freq_channel]')
            Aqf.is_true(np.all(quantiser_spectrum[test_freq_channel + 1:] == 0),
                        'Confirm that the spectrum is zero except in the test channel:'
                        ' [test_freq_channel+1:]')
            Aqf.step('FFT Window [{} samples] = {:.3f} micro seconds, Internal Accumulations = {}, '
                     'One VACC accumulation = {}s'.format(n_chans * 2,
                                                          self.cam_sensors.fft_period * 1e6,
                                                          internal_accumulations, delta_acc_t))

            chan_response = []
            for vacc_accumulations, acc_time in zip(test_acc_lens, acc_times):
                try:
                    reply = self.corr_fix.katcp_rct.req.accumulation_length(acc_time, timeout=60)
                    assert reply.succeeded
                except Exception:
                    Aqf.failed('Failed to set accumulation length of {} after maximum vacc '
                               'sync attempts.'.format(vacc_accumulations))
                else:
                    internal_acc = (2 * internal_accumulations * n_chans)
                    accum_len = int(np.ceil((acc_time * self.cam_sensors.get_value('sample')) / internal_acc))
                    Aqf.almost_equals(vacc_accumulations, accum_len, 1,
                                      'Confirm that vacc length was set successfully with'
                                      ' {}, which equates to an accumulation time of {:.6f}s'.format(
                                            vacc_accumulations, vacc_accumulations * delta_acc_t))
                    no_accs = internal_accumulations * vacc_accumulations
                    expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
                    try:
                        dump = get_clean_dump(self)
                        assert isinstance(dump, dict)
                    except Exception:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        actual_response = complexise(dump['xeng_raw'][:, 0, :])
                        actual_response_ = loggerise(dump['xeng_raw'][:, 0, :])
                        actual_response_mag = normalised_magnitude(dump['xeng_raw'][:, 0, :])
                        chan_response.append(actual_response_mag)
                        # Check that the accumulator response is equal to the expected response
                        caption = (
                            'Accumulators actual response is equal to the expected response for {} '
                            'accumulation length with a periodic cw tone every {} samples'
                            ' at frequency of {:.3f} MHz with scale {}.'.format(test_acc_lens,
                                                                                  n_chans * 2,
                                                                                  test_freq / 1e6,
                                                                                  cw_scale))

                        plot_filename = ('{}/{}_chan_resp_{}_vacc.png'.format(self.logs_path,
                                                                             self._testMethodName,
                                                                             int(vacc_accumulations)))
                        plot_title = ('Vector Accumulation Length: channel %s' % test_freq_channel)
                        msg = ('Confirm that the accumulator actual response is '
                               'equal to the expected response for {} accumulation length'.format(
                                    vacc_accumulations))

                        if not Aqf.array_abs_error(expected_response[:chan_index],
                                                   actual_response_mag[:chan_index], msg):
                            aqf_plot_channels(actual_response_mag, plot_filename, plot_title,
                                              log_normalise_to=0, normalise=0, caption=caption)

    def _test_product_switch(self, instrument):
        Aqf.step('Confirm that the SPEAD accumulations are being produced when Digitiser simulator is '
                 'configured to output correlated noise')
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        with ignored(Queue.Empty):
            get_clean_dump(self)
        xhosts = self.correlator.xhosts
        fhosts = self.correlator.fhosts

        Aqf.step('Capture stopped, deprogramming hosts by halting the katcp connection.')
        self.corr_fix.stop_x_data()
        self.corr_fix.halt_array

        no_channels = self.n_chans_selected
        Aqf.step('Re-initialising {instrument} instrument'.format(**locals()))
        corr_init = False
        retries = 5
        start_time = time.time()
        Aqf.step('Correlator initialisation timer-started: %s' %start_time)
        while retries and not corr_init:
            try:
                self.set_instrument()
                self.corr_fix.start_x_data
                corr_init = True
                retries -= 1
                if corr_init:
                    end_time = time.time()
                    msg = ('Correlator initialisation (%s) timer end: %s' %(instrument, end_time))
                    Aqf.step(msg)
                    LOGGER.info(msg + ' within %s retries' % (retries))
            except:
                retries -= 1
                if retries == 0:
                    errmsg = 'Could not restart the correlator after %s tries.' % (retries)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)

        if corr_init:
            host = xhosts[random.randrange(len(xhosts))]
            Aqf.is_true(host.is_running(), 'Confirm that the instrument is initialised by checking if '
                                           '%s is programmed.' % host.host)
            self.set_instrument()
            try:
                Aqf.hop('Capturing SPEAD Accumulation after re-initialisation to confirm '
                    'that the instrument activated is valid.')
                self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                re_dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            except AttributeError:
                errmsg = ('Could not retrieve clean SPEAD accumulation: Receiver could not '
                          'be instantiated')
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            else:
                msg = ('Confirm that the SPEAD accumulations are being produced after instrument '
                       're-initialisation.')
                Aqf.is_true(re_dump, msg)

                msg = ('Confirm that the data product has the number of frequency channels %s '
                       'corresponding to the %s instrument product' %(no_channels, instrument))
                Aqf.equals(4096, self.cam_sensors.get_value('no_chans'), msg)

                final_time = end_time - start_time - float(self.corr_fix.halt_wait_time)
                minute = 60.0
                msg = ('Confirm that instrument switching to %s '
                       'time is less than one minute' % instrument)
                Aqf.less(final_time, minute, msg)

    def _test_delay_rate(self, plot_diagram=True):
        msg = ("CBF Delay and Phase Compensation Functional VR: -- Delay Rate")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            # delay_rate = ((setup_data['sample_period'] / self.cam_sensors.get_value('int_time']) *
            # np.random.rand() * (dump_counts - 3))
            # delay_rate = 3.98195128768e-09
            delay_rate = (0.7 * (
                          self.cam_sensors.sample_period / self.cam_sensors.get_value('int_time')))
            delay_value = 0
            fringe_offset = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
            delay_rates = [0] * setup_data['num_inputs']
            delay_rates[setup_data['test_source_ind']] = delay_rate
            delay_coefficients = ['0,{}:0,0'.format(fr) for fr in delay_rates]
            Aqf.step('Calculate the parameters to be used for setting Fringe(s)/Delay(s).')
            Aqf.progress('Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s ' % (
                delay_rate, delay_value, fringe_offset, fringe_rate))

            try:
                actual_data, _delay_coefficients = self._get_actual_data(
                    setup_data, dump_counts, delay_coefficients)
            except TypeError as e:
                errmsg = ('Could not retrieve actual delay rate data. Aborting test: Exception: {}'
                          .format(e))
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return
            actual_phases = [phases for phases, response in actual_data]
            actual_response = [response for phases, response in actual_data]

            if _delay_coefficients is not None:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          _delay_coefficients, actual_phases)
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts,
                                                          delay_coefficients, actual_phases)

            no_chans = range(self.n_chans_selected)
            plot_units = 'ns/s'
            plot_title = 'Randomly generated delay rate {} {}'.format(delay_rate * 1e9, plot_units)
            plot_filename = '{}/{}_delay_rate.png'.format(self.logs_path, self._testMethodName)
            caption = ('Actual vs Expected Unwrapped Correlation Phase [Delay Rate].\n'
                       'Note: Dashed line indicates expected value and solid line indicates '
                       'actual values received from SPEAD accumulation.')

            msg = ('Observe the change in the phase slope, and confirm the phase change is as '
                       'expected.')
            Aqf.step(msg)
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
                    msg = ('Confirm that if difference (radians) between expected({:.3f}) '
                            'phases and actual({:.3f}) phases are \'Almost Equal\' '
                            'within {} degree when delay rate of {} is applied.'.format(
                                delta_expected, delta_actual, degree, delay_rate))
                    Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                    msg = ('Confirm that the maximum difference ({:.3f} '
                           'degree/{:.3f} rad) between expected phase and actual phase '
                           'between integrations is less than {} degree.'.format(
                                np.rad2deg(abs_diff), abs_diff, degree))
                    Aqf.less(abs_diff, radians, msg)

                    try:
                        abs_error = np.max(actual_phases_[i] - expected_phases_[i])
                    except ValueError:
                        abs_error = np.max(actual_phases_[i] - expected_phases_[
                                                                i][:len(actual_phases_[i])])
                    msg = ('Confirm that the absolute maximum difference ({:.3f} '
                           'degree/{:.3f} rad) between expected phase and actual phase '
                           'is less than {} degree.'.format(
                                np.rad2deg(abs_error), abs_error, degree))
                    Aqf.less(abs_error, radians, msg)

                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                       decimal=decimal)

                    except AssertionError:
                        Aqf.step('Difference  between expected({:.3f}) '
                                 'phases and actual({:.3f}) phases are '
                                 '\'Not almost equal\' within {} degree when delay rate '
                                 'of {} is applied.'.format(delta_expected, delta_actual,
                                                        degree, delay_rate))
                        caption = ('Difference  expected({:.3f}) and actual({:.3f})'
                                   ' phases are not equal within {} degree when delay rate of {} '
                                   'is applied.'.format(delta_expected, delta_actual, degree,
                                                        delay_rate))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans, actual_phases_i, expected_phases_i,
                            plot_filename='{}/{}_{}_delay_rate.png'.format(self.logs_path,
                                self._testMethodName, i),
                            plot_title='Delay Rate:\nActual vs Expected Phase Response',
                            plot_units=plot_units, caption=caption)
                if plot_diagram:
                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases, plot_filename,
                                   plot_title, plot_units, caption, dump_counts)

    def _test_fringe_rate(self, plot_diagram=True):
        msg = ("CBF Delay and Phase Compensation Functional VR: -- Fringe rate")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            _rand_gen = self.cam_sensors.get_value('int_time') * np.random.rand() * dump_counts
            fringe_rate = ((np.pi / 8.) / _rand_gen)
            delay_value = 0
            delay_rate = 0
            fringe_offset = 0
            load_time = setup_data['t_apply']
            fringe_rates = [0] * setup_data['num_inputs']
            fringe_rates[setup_data['test_source_ind']] = fringe_rate
            delay_coefficients = ['0,0:0,{}'.format(fr) for fr in fringe_rates]

            Aqf.step('Calculate the parameters to be used for setting Fringe(s)/Delay(s).')
            Aqf.progress('Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s ' % (
                delay_rate, delay_value, fringe_offset, fringe_rate))
            try:
                actual_data, _delay_coefficients = self._get_actual_data(setup_data, dump_counts,
                    delay_coefficients)
            except TypeError as e:
                errmsg = ('Could not retrieve actual delay rate data. Aborting test: Exception: {}'
                          .format(e))
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return

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
                no_chans = range(self.n_chans_selected)
                plot_units = 'rads/sec'
                plot_title = 'Randomly generated fringe rate {} {}'.format(fringe_rate,
                                                                           plot_units)
                plot_filename = '{}/{}_fringe_rate.png'.format(self.logs_path, self._testMethodName)
                caption = ('Actual vs Expected Unwrapped Correlation Phase [Fringe Rate].\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation.')

                degree = 1.0
                radians = (degree / 360) * np.pi * 2
                decimal = len(str(degree).split('.')[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                msg = ('Observe the change in the phase slope, and confirm the phase change is as '
                       'expected.')
                Aqf.step(msg)
                for i in xrange(0, len(expected_phases_) - 1):
                    try:
                        delta_expected = np.max(expected_phases_[i + 1] - expected_phases_[i])
                        delta_actual = np.max(actual_phases_[i + 1] - actual_phases_[i])
                    except IndexError:
                        errmsg = 'Failed: Index is out of bounds'
                        LOGGER.exception(errmsg)
                        Aqf.failed(errmsg)
                    else:
                        abs_diff = np.abs(delta_expected - delta_actual)
                        # abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                        msg = ('Confirm that the difference between expected({:.3f}) '
                               'phases and actual({:.3f}) phases are \'Almost Equal\' within '
                               '{} degree when fringe rate of {} is applied.'.format(
                                    delta_expected, delta_actual, degree, fringe_rate))
                        Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                        msg = ('Confirm that the maximum difference ({:.3f} '
                               'deg / {:.3f} rad) between expected phase and actual phase '
                               'between integrations is less than {} degree\n'.format(
                                    np.rad2deg(abs_diff), abs_diff, degree))
                        Aqf.less(abs_diff, radians, msg)

                        try:
                            delta_actual_s = delta_actual - (delta_actual % degree)
                            delta_expected_s = delta_expected - (delta_expected % degree)
                            np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                           decimal=decimal)
                        except AssertionError:
                            Aqf.step('Difference between expected({:.3f}) phases and actual({:.3f}) '
                                     'phases are \'Not almost equal\' within {} degree when fringe rate '
                                     'of {} is applied.'.format(delta_expected, delta_actual,
                                                                degree, fringe_rate))

                            caption = ('Difference expected({:.3f}) and '
                                       'actual({:.3f}) phases are not equal within {} degree when '
                                       'fringe rate of {} is applied.'.format(delta_expected,
                                            delta_actual, degree, fringe_rate))

                            actual_phases_i = (delta_actual, actual_phases[i])
                            if len(expected_phases[i]) == 2:
                                expected_phases_i = (delta_expected, expected_phases[i][-1])
                            else:
                                expected_phases_i = (delta_expected, expected_phases[i])

                            aqf_plot_phase_results(
                                no_chans, actual_phases_i, expected_phases_i,
                                plot_filename='{}/{}_fringe_rate_{}.png'.format(self.logs_path,
                                    self._testMethodName, i),
                                plot_title='Fringe Rate: Actual vs Expected Phase Response',
                                plot_units=plot_units, caption=caption)

                if plot_diagram:
                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

    def _test_fringe_offset(self, plot_diagram=True):
        msg = ("CBF Delay and Phase Compensation Functional VR: Fringe offset")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            fringe_offset = (np.pi / 2.) * np.random.rand() * dump_counts
            #fringe_offset = 1.22796022444
            delay_value = 0
            delay_rate = 0
            fringe_rate = 0
            load_time = setup_data['t_apply']
            fringe_offsets = [0] * setup_data['num_inputs']
            fringe_offsets[setup_data['test_source_ind']] = fringe_offset
            delay_coefficients = ['0,0:{},0'.format(fo) for fo in fringe_offsets]

            Aqf.step('Calculate the parameters to be used for setting Fringe(s)/Delay(s).')
            Aqf.progress('Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s ' % (
                delay_rate, delay_value, fringe_offset, fringe_rate))

            try:
                actual_data, _delay_coefficients = self._get_actual_data(
                    setup_data, dump_counts, delay_coefficients)
            except TypeError as e:
                errmsg = ('Could not retrieve actual delay rate data. Aborting test: Exception: {}'
                          .format(e))
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return
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
                no_chans = range(self.n_chans_selected)
                plot_units = 'rads'
                plot_title = 'Randomly generated fringe offset {:.3f} {}'.format(
                    fringe_offset, plot_units)
                plot_filename = '{}/{}_fringe_offset.png'.format(self.logs_path, self._testMethodName)
                caption = ('Actual vs Expected Unwrapped Correlation Phase [Fringe Offset].\n'
                           'Note: Dashed line indicates expected value and solid line '
                           'indicates actual values received from SPEAD accumulation. '
                           'Values are rounded off to 3 decimals places')

                # Ignoring first dump because the delays might not be set for full
                # integration.
                degree = 1.0
                decimal = len(str(degree).split('.')[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                msg = ('Observe the change in the phase slope, and confirm the phase change is as '
                       'expected.')
                Aqf.step(msg)
                for i in xrange(1, len(expected_phases) - 1):
                    delta_expected = np.abs(np.max(expected_phases_[i]))
                    delta_actual = np.abs(np.max(actual_phases_[i]))
                    # abs_diff = np.abs(delta_expected - delta_actual)
                    abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    msg = (
                        'Confirm that the difference between expected({:.3f})'
                        ' phases and actual({:.3f}) phases are \'Almost Equal\' '
                        'within {:.3f} degree when fringe offset of {:.3f} is '
                        'applied.'.format(delta_expected, delta_actual, degree,
                                          fringe_offset))

                    Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                    Aqf.less(abs_diff, degree,
                             'Confirm that the maximum difference({:.3f} '
                             'degrees/{:.3f}rads) between expected phase and actual phase '
                             'between integrations is less than {:.3f} degree\n'.format(
                                 abs_diff, np.deg2rad(abs_diff), degree))
                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s,
                                                       decimal=decimal)

                    except AssertionError:
                        Aqf.step('Difference between expected({:.5f}) phases '
                                 'and actual({:.5f}) phases are \'Not almost equal\' '
                                 'within {} degree when fringe offset of {} is applied.'
                                    .format(delta_expected, delta_actual, degree, fringe_offset))

                        caption = ('Difference expected({:.3f}) and actual({:.3f}) '
                                   'phases are not equal within {:.3f} degree when fringe offset '
                                   'of {:.3f} {} is applied.'.format(delta_expected, delta_actual,
                                        degree, fringe_offset, plot_units))

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans, actual_phases_i, expected_phases_i,
                            plot_filename='{}/{}_{}_fringe_offset.png'.format(self.logs_path,
                                self._testMethodName, i),
                            plot_title=('Fringe Offset:\nActual vs Expected Phase Response'),
                            plot_units=plot_units, caption=caption)
                if plot_diagram:
                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

    def _test_delay_inputs(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial:
        Delay applied to the correct input
        """
        msg = ("CBF Delay and Phase Compensation Functional VR: Delays applied to the correct input")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            Aqf.step("The test will sweep through four(4) randomly selected baselines, select and "
                    "set a delay value, Confirm if the delay set is as expected.")
            input_labels = self.cam_sensors.input_labels
            random.shuffle(input_labels)
            input_labels = input_labels[4:]
            for delayed_input in input_labels:
                test_delay_val = random.randrange(self.cam_sensors.sample_period, step=.83e-10,
                    int=float)
                # test_delay_val = self.cam_sensors.sample_period  # Pi
                expected_phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * test_delay_val
                expected_phases -= np.max(expected_phases) / 2.
                Aqf.step('Clear all coarse and fine delays for all inputs before testing input %s.'
                    % delayed_input)
                delays_cleared = clear_all_delays(self)
                if not delays_cleared:
                    Aqf.failed('Delays were not completely cleared, data might be corrupted.\n')
                else:
                    Aqf.passed('Cleared all previously applied delays prior to test.\n')
                    delays = [0] * setup_data['num_inputs']
                    # Get index for input to delay
                    test_source_idx = input_labels.index()
                    Aqf.step('Selected input to test: {}'.format(delayed_input))
                    delays[test_source_idx] = test_delay_val
                    Aqf.step('Randomly selected delay value ({}) relevant to sampling period'.format(
                        test_delay_val))
                    delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                    int_time = setup_data['int_time']
                    num_int = setup_data['num_int']
                    try:
                        this_freq_dump = get_clean_dump(self)
                        t_apply = this_freq_dump['dump_timestamp'] + (num_int * int_time)
                        t_apply_readable = this_freq_dump['dump_timestamp_readable']
                        Aqf.step('Delays will be applied with the following parameters:')
                        Aqf.progress('Current cmc time: %s (%s)' %(time.time(), time.strftime("%H:%M:%S")))
                        Aqf.progress('Current Dump timestamp: %s (%s)'%(this_freq_dump['dump_timestamp'],
                            this_freq_dump['dump_timestamp_readable']))
                        Aqf.progress('Time delays will be applied: %s (%s)' %(t_apply, t_apply_readable))
                        Aqf.progress('Delay coefficients: %s' %delay_coefficients)
                        reply, _informs = self.corr_fix.katcp_rct.req.delays(t_apply, *delay_coefficients)
                        assert reply.reply_ok()
                    except Exception:
                        errmsg = '%s'%str(reply).replace('\_',' ')
                        Aqf.failed(errmsg)
                        LOGGER.error(errmsg)
                        return
                    else:
                        Aqf.is_true(reply.reply_ok(), 'CAM Reply: {}'.format(str(reply)))
                        Aqf.passed('Delays where successfully applied on input: {}'.format(delayed_input))
                    try:
                        Aqf.step('Getting SPEAD accumulation (while discarding subsequent dumps) containing '
                                 'the change in delay(s) on input: %s.'%(test_source_idx))
                        dump = self.receiver.get_clean_dump(discard=35)
                    except Exception:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        sorted_bls = get_baselines_lookup(self, this_freq_dump, sorted_lookup=True)
                        degree = 1.0
                        Aqf.step('Maximum expected delay: %s' % np.max(expected_phases))
                        for b_line in sorted_bls:
                            b_line_val = b_line[1]
                            b_line_dump = (dump['xeng_raw'][:, b_line_val, :])
                            b_line_phase = np.angle(complexise(b_line_dump))
                            # np.deg2rad(1) = 0.017 ie error should be withing 2 decimals
                            b_line_phase_max = round(np.max(b_line_phase), 2)
                            if ((delayed_input in b_line[0]) and
                                        b_line[0] != (delayed_input, delayed_input)):
                                msg = ('Confirm that the baseline(s) {} expected delay is within 1 '
                                       'degree.'.format(b_line[0]))
                                Aqf.array_abs_error(np.abs(b_line_phase[5:-5]),
                                                    np.abs(expected_phases[5:-5]), msg, degree)
                            else:
                                # TODO Readdress this failure and calculate
                                if b_line_phase_max != 0.0:
                                    desc = ('Checking baseline {}, index: {}, phase offset found, '
                                            'maximum error value = {} rads'.format(b_line[0], b_line_val,
                                                b_line_phase_max))
                                    Aqf.failed(desc)

    def _test_min_max_delays(self):
        delays_cleared = clear_all_delays(self)
        setup_data = self._delays_setup()

        num_int = setup_data['num_int']
        int_time = self.cam_sensors.get_value('int_time')
        if setup_data:
            Aqf.step('Clear all coarse and fine delays for all inputs before test commences.')
            if not delays_cleared:
                Aqf.failed('Delays were not completely cleared, data might be corrupted.\n')
            else:
                dump_counts = 5
                delay_bounds = get_delay_bounds(self.correlator)
                for _name, _values in sorted(delay_bounds.iteritems()):
                    _new_name = _name.title().replace('_', ' ')
                    Aqf.step('Calculate the parameters to be used for setting %s.' % _new_name)
                    delay_coefficients = 0
                    dump = get_clean_dump(self)
                    t_apply = (dump['dump_timestamp'] + num_int * int_time)
                    setup_data['t_apply'] = t_apply
                    no_inputs =  [0] * setup_data['num_inputs']
                    input_source = setup_data['test_source']
                    no_inputs[setup_data['test_source_ind']] = _values * dump_counts
                    if 'delay_value' in _name:
                        delay_coefficients = ['{},0:0,0'.format(dv) for dv in no_inputs]
                    if 'delay_rate' in _name:
                        delay_coefficients = ['0,{}:0,0'.format(dr) for dr in no_inputs]
                    if 'phase_offset' in _name:
                        delay_coefficients = ['0,0:{},0'.format(fo) for fo in no_inputs]
                    else:
                        delay_coefficients = ['0,0:0,{}'.format(fr) for fr in no_inputs]

                    Aqf.progress('%s of %s will be set on input %s. Note: All other parameters '
                                 'will be set to zero' % (_name.title(),  _values, input_source))
                    try:
                        actual_data, _delay_coefficients = self._get_actual_data(
                            setup_data, dump_counts, delay_coefficients)
                    except TypeError:
                        errmsg = 'Failed to set the delays/fringes'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        Aqf.step('Confirm that the %s where successfully set' % _new_name)
                        reply, informs = self.corr_fix.katcp_rct.req.delays()
                        msg = ('%s where successfully set via CAM interface.'
                               '\n\t\t\t    Reply: %s\n\n' % (_new_name, reply))
                        Aqf.is_true(reply.reply_ok(), msg)

    def _test_delays_control(self):
        delays_cleared = clear_all_delays(self)
        setup_data = self._delays_setup()

        num_int = setup_data['num_int']
        int_time = self.cam_sensors.get_value('int_time')
        Aqf.step('Disable Delays and/or Phases for all inputs.')
        if not delays_cleared:
            Aqf.failed('Delays were not completely cleared, data might be corrupted.\n')
        else:
            Aqf.passed("Confirm that the user can disable Delays and/or Phase changes via CAM interface.")
        dump = get_clean_dump(self)
        t_apply = (dump['dump_timestamp'] + num_int * int_time)
        no_inputs =  [0] * setup_data['num_inputs']
        input_source = setup_data['test_source']
        no_inputs[setup_data['test_source_ind']] = self.cam_sensors.sample_period * 2
        delay_coefficients = ['{},0:0,0'.format(dv) for dv in no_inputs]
        try:
            Aqf.step('Request and enable Delays and/or Phases Corrections on input (%s) '
                     'via CAM interface.'% input_source)
            load_strt_time = time.time()
            reply_, _informs = self.corr_fix.katcp_rct.req.delays(t_apply, *delay_coefficients,
                timeout=30)
            load_done_time = time.time()
            msg = ('Delay/Fringe(s) set via CAM interface reply : %s' % str(reply_))
            assert reply_.reply_ok()
            cmd_load_time = round(load_done_time - load_strt_time, 3)
            Aqf.step('Fringe/Delay load command took {} seconds'.format(cmd_load_time))
            Aqf.is_true(reply_.reply_ok(), msg)
            _give_up = int(num_int * int_time * 2)
            while True:
                _give_up -= 1
                try:
                    LOGGER.info('Waiting for the delays to be updated')
                    try:
                        reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                    except:
                        reply, informs = self.corr_fix.katcp_rct.req.sensor_value()
                    assert reply.reply_ok()
                except Exception:
                    LOGGER.exception('Weirdly I couldnt get the sensor values')
                else:
                    delays_updated = list(set([int(i.arguments[-1]) for i in informs
                                            if '.cd.delay' in i.arguments[2]]))[0]
                    if delays_updated:
                        LOGGER.info('Delays have been successfully set')
                        break
                if _give_up == 0:
                    LOGGER.error("Could not confirm the delays in the time stipulated, exiting")
                    break
                time.sleep(1)

        except Exception:
            errmsg = ('%s: Failed to set delays via CAM interface with load-time: %s, '
                      'Delay coefficients: %s' % (str(reply), setup_data['t_apply'],
                        delay_coefficients))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            cam_max_load_time = setup_data['cam_max_load_time']
            msg = 'Time it took to load delay/fringe(s) %s is less than %ss' % (cmd_load_time,
                    cam_max_load_time)
            Aqf.less(cmd_load_time, cam_max_load_time, msg)

    def _test_report_config(self):
        """CBF Report configuration"""
        test_config = self.corr_fix._test_config_file


        def git_revision_short_hash(mod_name=None, dir_name=None):
            return subprocess.check_output(['git',
                                            '--git-dir=%s/.git' % dir_name,
                                            '--work-tree=%s' % mod_name,
                                            'rev-parse',
                                            '--short',
                                            'HEAD']).strip() \
                                            if mod_name and dir_name else None

        def get_skarab_config(_timeout=30):
            from casperfpga import utils as fpgautils
            Aqf.step('List of all processing nodes')
            Aqf.progress('D-Engine :{}'.format(self.dhost.host))
            fhosts = [fhost.host for fhost in self.correlator.fhosts]
            Aqf.progress('List of F-Engines :{}'.format(', '.join(fhosts)))
            xhosts = [xhost.host for xhost in self.correlator.xhosts]
            Aqf.progress('List of X-Engines :{}\n'.format(', '.join(xhosts)))
            skarabs = FPGA_Connect(self._hosts)
            if skarabs:
                version_info = fpgautils.threaded_fpga_operation(
                    skarabs, timeout=_timeout, target_function=(lambda fpga:
                        fpga.transport.get_skarab_version_info(), [], {}))
            # ToDo (MM) Get a list of all skarabs available including ip's and
            # leaf the host is connected to.
            # subprocess.check_output(['bash', 'scripts/find-skarabs-arp.sh'])
            for _host, _versions in version_info.iteritems():
                Aqf.step('%s [R3000-0000] Software/Hardware Version Information' % _host)
                Aqf.progress('IP Address: %s' % (socket.gethostbyname(_host)))
                for _name, _version in _versions.iteritems():
                    try:
                        assert isinstance(_version, str)
                        _name =  _name.title().replace('_',' ')
                        if _name.startswith('Microblaze Hardware'):
                            Aqf.progress('%s [M1200-0070]: %s\n' % (_name, _version))
                        elif _name.startswith('Microblaze Software'):
                            Aqf.progress('%s [M1200-0071]: %s' % (_name, _version))
                        elif _name.startswith('Spartan'):
                            Aqf.progress( '%s [M1200-0069]: %s' % (_name, _version))
                    except:
                        pass

        def get_package_versions():
            corr2_name = corr2.__name__
            corr2_version = corr2.__version__
            corr2_pn = "M1200-0046"
            try:
                assert 'devel' in corr2_version
                corr2_version = ''.join([i for i in corr2_version.split('.') if len(i) == 7])
                corr2_link = ("https://github.com/ska-sa/%s/commit/%s" % (corr2_name, corr2_version))
            except Exception:
                corr2_link = "Not Version Controlled at this time."

            casper_name = casperfpga.__name__
            casper_version = casperfpga.__version__
            casper_pn = "M1200-0055"
            try:
                assert 'dev' in casper_version
                casper_version = ''.join([i for i in casper_version.split('.') if len(i) == 7])
                casper_link = ("https://github.com/ska-sa/%s/commit/%s" % (casper_name, casper_version))
            except Exception:
                casper_link = "Not Version Controlled at this time."

            katcp_name = katcp.__name__
            katcp_version = katcp.__version__
            katcp_pn = "M1200-0053"
            try:
                assert 'dev' in katcp_version
                katcp_version = ''.join([i for i in katcp_version.split('.') if len(i) == 7])
                assert len(katcp_version) == 7
                katcp_link = ("https://github.com/ska-sa/%s-python/commit/%s" % (katcp_name,
                    katcp_version))
            except Exception:
                katcp_link = ("https://github.com/ska-sa/%s/releases/tag/v%s" % (katcp_name,
                    katcp_version))

            spead2_name = spead2.__name__
            spead2_version = spead2.__version__
            spead2_pn = "M1200-0056"
            try:
                assert 'dev' in spead2_version
                assert len(spead2_version) == 7
                spead2_version = ''.join([i for i in spead2_version.split('.') if len(i) == 7])
                spead2_link = ("https://github.com/ska-sa/%s/commit/%s" % (spead2_name, spead2_version))
            except Exception:
                spead2_link = ("https://github.com/ska-sa/%s/releases/tag/v%s" % (spead2_name,
                    spead2_version))

            try:
                bitstream_dir = self.correlator.configd['xengine']['bitstream']
                mkat_dir, _ = os.path.split(os.path.split(os.path.dirname(
                    os.path.realpath(bitstream_dir)))[0])
                _, mkat_name = os.path.split(mkat_dir)
                assert mkat_name
                mkat_version = git_revision_short_hash(dir_name=mkat_dir,mod_name=mkat_name)
                assert len(mkat_version) == 7
                mkat_link = ("https://github.com/ska-sa/%s/commit/%s" % (mkat_name, mkat_version))
            except Exception:
                mkat_name = 'mkat_fpga'
                mkat_link = "Not Version Controlled at this time."
                mkat_version = "Not Version Controlled at this time."

            try:
                test_dir, test_name = os.path.split(os.path.dirname(os.path.realpath(__file__)))
                testing_version = git_revision_short_hash(dir_name=test_dir, mod_name=test_name)
                assert len(testing_version) == 7
                testing_link = ("https://github.com/ska-sa/%s/commit/%s" % (test_name,
                        testing_version))
            except AssertionError:
                testing_link = "Not Version Controlled at this time."

            try:
                with open('/etc/cmc.conf') as f:
                    cmc_conf =  f.readlines()
                templates_loc  = [i.strip().split('=') for i in cmc_conf
                                  if i.startswith('CORR_TEMPLATE')][0][-1]
                # template_name = template_name.replace('_', ' ').title()
                config_dir = os.path.split(templates_loc)[0]
                config_dir_name = os.path.split(config_dir)[-1]
                config_version = git_revision_short_hash(dir_name=config_dir,
                    mod_name=config_dir_name)
                config_pn = "M1200-0063"
                assert len(config_version) == 7
                config_link = ("https://github.com/ska-sa/%s/commit/%s" % (config_dir_name,
                    config_version))
            except Exception:
                config_dir_name = "mkat_config_templates"
                config_version = 'Not Version Controlled'
                config_link = 'Not Version Controlled'

            return {
                    corr2_name: [corr2_version, corr2_link, corr2_pn],
                    casper_name: [casper_version, casper_link, casper_pn],
                    katcp_name: [katcp_version, katcp_link, katcp_pn],
                    spead2_name: [spead2_version,spead2_link, spead2_pn],
                    mkat_name: [mkat_version, mkat_link, "None"],
                    test_name: [testing_version, testing_link, "None"],
                    config_dir_name: [config_version, config_link, "None"]
                    }


        def get_gateware_info():
            try:
                reply, informs = self.corr_fix.katcp_rct.req.version_list()
                assert reply.reply_ok()
            except AssertionError:
                Aqf.failed('Could not retrieve CBF Gate-ware Version Information')
            else:
                for inform in informs:
                    if [s for s in inform.arguments if 'xengine-firmware' in s]:
                        _hash = inform.arguments[-1].split(' ')
                        _hash = ''.join([i.replace('[','').replace(']', '')
                                        for i in _hash if 40 < len(i) < 42])
                        Aqf.progress('%s: %s' %(inform.arguments[0], _hash))
                        Aqf.progress("X/B-ENGINE (CBF) : M1200-0067")
                    elif [s for s in inform.arguments if 'fengine-firmware' in s]:
                        _hash = inform.arguments[-1].split(' ')
                        _hash = ''.join([i.replace('[','').replace(']', '')
                                        for i in _hash if 40 < len(i) < 42])
                        Aqf.progress('%s: %s' %(inform.arguments[0], _hash))
                        Aqf.progress("F-ENGINE (CBF) : M1200-0064")
                    else:
                        Aqf.progress(': '.join(inform.arguments))
                Aqf.progress("CMC KATCP_C : M1200-0047")
                Aqf.progress("CMC CBF SCRIPTS : M1200-0048")
                Aqf.progress("CORRELATOR MASTER CONTROLLER (CMC) : M1200-0012")

        test_heading('CBF CMC Operating System.')
        Aqf.progress("CBF OS: %s | CMC OS P/N: M1200-0045" % ' '.join(os.uname()))

        test_heading('CBF Software Packages Version Information.')
        Aqf.progress("CORRELATOR BEAMFORMER GATEWARE (CBF) : M1200-0041")
        get_gateware_info()

        test_heading('CBF Git Version Information.')
        Aqf.progress("CORRELATOR BEAMFORMER SOFTWARE : M1200-0036")
        packages_info = get_package_versions()
        for name, repo_dir in packages_info.iteritems():
            try:
                if name and (len(repo_dir[0]) == 7):
                    Aqf.progress('Repo: %s | Part Number: %s | Git Commit: %s | GitHub: %s'%(name,
                                 repo_dir[2],repo_dir[0], repo_dir[1]))
                else:
                    Aqf.progress('Repo: %s | Git Tag: %s | GitHub: %s'%(name,
                                 repo_dir[0], repo_dir[1]))
            except Exception:
                pass

        test_heading('CBF Processing Node Version Information')
        self._hosts = list(np.concatenate(
                        [i.get('hosts', None).split(',') for i in self.corr_fix.corr_config.values()
                        if i.get('hosts')]))

        get_skarab_config()

    def _test_data_product(self, _baseline=False, _tiedarray=False):
        """CBF Imaging Data Product Set"""
        # Put some correlated noise on both outputs
        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = '113+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Configure a digitiser simulator to generate correlated noise.')
        Aqf.progress('Digitiser simulator configured to generate Gaussian noise with scale: {}, '
                 'gain: {} and fft shift: {}.'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, fft_shift=fft_shift,
                                                gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False


        if _baseline:
            Aqf.step('Configure the CBF to generate Baseline-Correlation-Products(If available).')
            try:
                Aqf.progress('Retrieving initial SPEAD accumulation, in-order to confirm the number of '
                             'channels in the SPEAD data.')
                test_dump = get_clean_dump(self)
                assert isinstance(test_dump, dict)
            except Exception:
                errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return
            else:
                # TODO MM, get number of channels from dump[n_chans_selected] instead of sensors
                # no_channels = self.n_chans_selected

                no_channels = self.cam_sensors.get_value('n_chans')
                # Get baseline 0 data, i.e. auto-corr of m000h
                test_baseline = 0
                test_bls = eval(self.cam_sensors.get_value('bls_ordering'))[test_baseline]
                Aqf.equals(4096, no_channels,
                           'Confirm that the baseline-correlation-products has the same number of '
                           'frequency channels ({}) corresponding to the {} '
                           'instrument currently running,'.format(no_channels, self.instrument))
                Aqf.passed('and confirm that imaging data product set has been '
                           'implemented for instrument: {}.'.format(self.instrument))

                response = normalised_magnitude(test_dump['xeng_raw'][:, test_baseline, :])
                plot_filename = '{}/{}.png'.format(self.logs_path, self._testMethodName)

                caption = ('An overall frequency response at {} baseline, '
                           'when digitiser simulator is configured to generate Gaussian noise, '
                           'with scale: {}, eq gain: {} and fft shift: {}'.format(test_bls, awgn_scale,
                            gain, fft_shift))
                aqf_plot_channels(response, plot_filename, log_dynamic_range=90, caption=caption)

        if _tiedarray:
            try:
                LOGGER.info('Checking if Docker is running!!!')
                output = subprocess.check_output(['docker', 'run', 'hello-world'])
                LOGGER.info(output)
            except subprocess.CalledProcessError:
                errmsg = 'Cannot connect to the Docker daemon. Is the docker daemon running on this host?'
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return False

            try:
                #Set custom source names
                local_src_names = self.cam_sensors.custom_input_labels
                reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
                assert reply.reply_ok()
                labels = reply.arguments[1:]
                beams = ['tied-array-channelised-voltage.0x','tied-array-channelised-voltage.0y']
                running_instrument = self.instrument
                assert running_instrument is not False
                msg = 'Running instrument currently does not have beamforming capabilities.'
                assert running_instrument.endswith('4k'), msg
                Aqf.step('Discontinue any capturing of %s and %s, if active.' %(beams[0],beams[1]))
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[0])
                assert reply.reply_ok(), str(reply)
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[1])
                assert reply.reply_ok(), str(reply)

                # Get instrument parameters
                bw = self.cam_sensors.get_value('bandwidth')
                nr_ch = self.cam_sensors.get_value('n_chans')
                ants = self.cam_sensors.get_value('n_ants')
                ch_list = self.cam_sensors.ch_center_freqs
                ch_bw = ch_list[1]
                dsim_factor = (float(self.conf_file['instrument_params']['sample_freq'])/
                               self.cam_sensors.get_value('scale_factor_timestamp'))
                substreams = self.cam_sensors.get_value('n_xengs')
            except AssertionError:
                errmsg = '%s'%str(reply).replace('\_', ' ')
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return False
            except Exception as e:
                errmsg = 'Exception: {}'.format(str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

            Aqf.progress('Bandwidth = {}Hz'.format(bw*dsim_factor))
            Aqf.progress('Number of channels = {}'.format(nr_ch))
            Aqf.progress('Channel spacing = {}Hz'.format(ch_bw*dsim_factor))

            beam = beams[0]
            try:
                beam_name = beam.replace('-','_').replace('.','_')
                beam_ip, beam_port = self.cam_sensors.get_value(
                        beam_name+'_destination').split(':')
                beam_ip = beam_ip.split('+')[0]
                start_beam_ip = beam_ip
                n_substrms_to_cap_m = int(self.conf_file['beamformer']['substreams_to_capture'])
                start_substream     = int(self.conf_file['beamformer']['start_substream_idx'])
                if start_substream+n_substrms_to_cap_m > substreams:
                    errmsg = ('Substream start + substreams to process '
                              'is more than substreams available: {}. '
                              'Fix in test configuration file'.format(substeams))
                    LOGGER.error(errmsg)
                    Aqf.failed(errmsg)
                    return False
                ticks_between_spectra = self.cam_sensors.get_value(
                        'antenna_channelised_voltage_n_samples_between_spectra')
                assert isinstance(ticks_between_spectra,int)
                spectra_per_heap = self.cam_sensors.get_value(beam_name+'_spectra_per_heap')
                assert isinstance(spectra_per_heap,int)
                ch_per_substream = self.cam_sensors.get_value(beam_name+'_n_chans_per_substream')
                assert isinstance(ch_per_substream, int)
            except AssertionError:
                errmsg = '%s'%str(reply).replace('\_', ' ')
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return False
            except Exception as e:
                errmsg = 'Exception: {}'.format(str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

            # Compute the start IP address according to substream start index
            beam_ip = int2ip(ip2int(beam_ip) + start_substream)
            # Compute spectrum parameters
            strt_ch_idx = start_substream * ch_per_substream
            strt_freq = ch_list[strt_ch_idx]*dsim_factor
            Aqf.step('Start a KAT SDP docker ingest node for beam captures')
            docker_status = start_katsdpingest_docker(self, beam_ip, beam_port,
                                                      n_substrms_to_cap_m, nr_ch,
                                                      ticks_between_spectra,
                                                      ch_per_substream, spectra_per_heap)
            if docker_status:
                Aqf.progress('KAT SDP Ingest Node started. Capturing {} substream/s '
                             'starting at {}'.format(n_substrms_to_cap_m, beam_ip))
            else:
                Aqf.failed('KAT SDP Ingest Node failed to start')


            Aqf.step("Set beamformer quantiser gain for selected beam to 1")
            #set_beam_quant_gain(self, beam, 1)
            bq_gain = set_beam_quant_gain(self, beams[1 - beams.index(beam)], 1)

            beam_dict = {}
            beam_pol = beam[-1]
            for label in labels:
                if label.find(beam_pol) != -1:
                    beam_dict[label] = 0.0

            Aqf.progress("Only one antenna gain is set to 1, the reset are set to zero")
            weight = 1.0
            beam_dict = populate_beam_dict(self, 1, weight, beam_dict)
            try:
                bf_raw, cap_ts, bf_ts, in_wgts = capture_beam_data(self, beam, beam_dict)
            except TypeError, e:
                errmsg = ('Failed to capture beam data: %s\n\n Confirm that Docker container is '
                         'running and also confirm the igmp version = 2 ' % str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

            try:
                nc = 10000
                cap = [0] * nc
                for i in range(0, nc):
                    cap[i] = np.array(complexise(bf_raw[:, i, :]))
                cap_mag = np.abs(cap)
                cap_avg = cap_mag.sum(axis=0) / nc
                # Confirm that the beam channel bandwidth corresponds to the channel bandwidth
                # determined from the baseline capture
                # baseline_ch_bw = bw * dsim_clk_factor / response.shape[0]

                # hardcoded the bandwidth value due to a custom dsim frequency used in the config file
                # Square the voltage data. This is a hack as aqf_plot expects squared
                # power data
                aqf_plot_channels(
                    np.square(cap_avg),
                    plot_filename='{}/{}_beam_resp_{}.png'.format(self.logs_path,
                                                                  self._testMethodName, beam),
                    plot_title=('Beam = {}, Spectrum Start Frequency = {} MHz\n'
                                'Number of Channels Captured = {}\n'
                                'Integrated over {} captures'.format(beam, strt_freq / 1e6,
                                                n_substrms_to_cap_m*ch_per_substream, nc)),
                    log_dynamic_range=90,
                    log_normalise_to=1,
                    caption=('Tied Array Beamformer data captured during Baseline Correlation '
                             'Product test.'),
                    plot_type='bf')
            except Exception as e:
                Aqf.failed(str(e))

        if  _baseline and _tiedarray:
                nominal_bw = float(self.conf_file['instrument_params']['sample_freq'])/2.0
                baseline_ch_bw = nominal_bw / test_dump['xeng_raw'].shape[0]
                beam_ch_bw = nominal_bw / len(cap_mag[0])
                msg = ('Confirm that the baseline-correlation-product channel width'
                       ' {}Hz is the same as the tied-array-channelised-voltage channel width '
                       '{}Hz'.format(baseline_ch_bw, beam_ch_bw))
                Aqf.almost_equals(baseline_ch_bw, beam_ch_bw, 1e-3, msg)

    def _test_time_sync(self):
        Aqf.step('Request NTP pool address used.')
        try:
            host_ip = '192.168.194.2'
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        except ntplib.NTPException:
            host_ip = '192.168.1.21'
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        req_sync_time = 5e-3
        msg = ('Confirm that the CBF synchronised time is within {}s of '
               'UTC time as provided via PTP (NTP server: {}) on the CBF-TRF '
               'interface.'.format(req_sync_time, host_ip))
        Aqf.less(ntp_offset, req_sync_time, msg)

    def _test_gain_correction(self):
        """CBF Gain Correction"""
        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = 113
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = 344
            fft_shift = 4095

        Aqf.step('Configure a digitiser simulator to generate correlated noise.')
        Aqf.progress('Digitiser simulator configured to generate Gaussian noise, '
                 'with scale: %s, eq gain: %s, fft shift: %s' % (awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitiser simulator levels')
            return False

        self.addCleanup(set_default_eq, self)
        source = random.randrange(len(self.cam_sensors.input_labels))
        _discards = 50
        try:
            initial_dump = self.receiver.get_clean_dump(discard=_discards)
            self.assertIsInstance(initial_dump, dict)
            assert np.any(initial_dump['xeng_raw'])
        except Exception:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            test_input = random.choice(self.cam_sensors.input_labels)
            Aqf.step('Randomly selected input to test: %s' % (test_input))
            # Get auto correlation index of the selected input
            bls_order = eval(self.cam_sensors.get_value('bls_ordering'))
            for idx, val in enumerate(bls_order):
                if val[0] == test_input and val[1] == test_input:
                    auto_corr_idx = idx

            n_chans = self.cam_sensors.get_value('n_chans')
            rand_ch = random.choice(range(n_chans)[:self.n_chans_selected])
            gain_vector = [gain] * n_chans
            base_gain = gain
            initial_resp = np.abs(complexise(initial_dump['xeng_raw'][:, auto_corr_idx, :]))
            initial_resp = 10 * np.log10(initial_resp)
            chan_resp = []
            legends = []
            found = False
            fnd_less_one = False
            count = 0
            Aqf.step('Note: Gains are relative to reference channels, and are increased '
                     'iteratively until output power is increased by more than 6dB.')
            while not found:
                if not fnd_less_one:
                    target = 1
                    gain_inc = 5
                else:
                    target = 6
                    gain_inc = 200
                gain = gain + gain_inc
                gain_vector[rand_ch] = gain
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(test_input, *gain_vector,
                        timeout=60)
                    assert reply.reply_ok()
                except Exception as e:
                        Aqf.failed('Gain correction on %s could not be set to %s.: '
                                   'KATCP Reply: %s' % (test_input, gain, reply))
                        return
                else:
                    msg = ('Gain correction on input %s, channel %s set to %s.' % (test_input,
                        rand_ch, complex(gain)))
                    Aqf.passed(msg)
                    try:
                        dump = self.receiver.get_clean_dump(discard=_discards)
                        self.assertIsInstance(dump, dict)
                    except AssertionError:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        response = np.abs(complexise(dump['xeng_raw'][:, auto_corr_idx, :]))
                        response = 10 * np.log10(response)
                        resp_diff = response[rand_ch] - initial_resp[rand_ch]
                        if resp_diff < target:
                            msg = ('Output power increased by less than 1 dB '
                                   '(actual = {:.2f} dB) with a gain '
                                   'increment of {}.'.format(resp_diff, complex(gain_inc)))
                            Aqf.passed(msg)
                            fnd_less_one = True
                            chan_resp.append(response)
                            legends.append('Gain set to %s' % (complex(gain)))
                        elif fnd_less_one and (resp_diff > target):
                            msg = ('Output power increased by more than 6 dB '
                                   '(actual = {:.2f} dB) with a gain '
                                   'increment of {}.'.format(resp_diff, complex(gain_inc)))
                            Aqf.passed(msg)
                            found = True
                            chan_resp.append(response)
                            legends.append('Gain set to %s' % (complex(gain)))
                        else:
                            pass
                count += 1
                if count == 7:
                    Aqf.failed('Gains to change output power by less than 1 and more than 6 dB '
                               'could not be found.')
                    found = True

            if chan_resp != []:
                zipped_data = zip(chan_resp, legends)
                zipped_data.reverse()
                aqf_plot_channels(zipped_data,
                  plot_filename='{}/{}_chan_resp.png'.format(self.logs_path, self._testMethodName),
                  plot_title='Channel Response Gain Correction for channel %s' % (rand_ch),
                  log_dynamic_range=90, log_normalise_to=1,
                  caption='Gain Correction channel response, gain varied for channel %s, '
                          'all remaining channels are set to %s' % (rand_ch, complex(base_gain)))
            else:
                Aqf.failed('Could not retrieve channel response with gain/eq corrections.')

    def _test_beamforming(self):
        """
        Apply weights and capture beamformer data, Verify that weights are correctly applied.
        """
        # Main test code
        # TODO AR
        # Neccessarry to compare output products with capture-list output products?

        try:
            output = subprocess.check_output(['docker', 'run', 'hello-world'])
            LOGGER.info(output)
        except subprocess.CalledProcessError:
            errmsg = 'Cannot connect to the Docker daemon. Is the docker daemon running on this host?'
            LOGGER.error(errmsg)
            Aqf.failed(errmsg)
            return False

        try:
            #Set custom source names
            local_src_names = self.cam_sensors.custom_input_labels
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            assert reply.reply_ok()
            labels = reply.arguments[1:]
            beams = ['tied-array-channelised-voltage.0x','tied-array-channelised-voltage.0y']
            # running_instrument = self.corr_fix.get_running_instrument()
            # assert running_instrument is not False
            # msg = 'Running instrument currently does not have beamforming capabilities.'
            # assert running_instrument.endswith('4k'), msg
            Aqf.step('Discontinue any capturing of %s and %s, if active.' %(beams[0],beams[1]))
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[0])
            assert reply.reply_ok(), str(reply)
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[1])
            assert reply.reply_ok(), str(reply)

            # Get instrument parameters
            bw =      self.cam_sensors.get_value('bandwidth')
            nr_ch =   self.cam_sensors.get_value('n_chans')
            ants =    self.cam_sensors.get_value('n_ants')
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]
            dsim_factor = (float(self.conf_file['instrument_params']['sample_freq'])/
                           self.cam_sensors.get_value('scale_factor_timestamp'))
            substreams = self.cam_sensors.get_value('n_xengs')
            # For substream alignment test only print out 5 results
            align_print_modulo = int(substreams/4)
        except AssertionError:
            errmsg = '%s'%str(reply).replace('\_', ' ')
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        Aqf.progress('Bandwidth = {}Hz'.format(bw*dsim_factor))
        Aqf.progress('Number of channels = {}'.format(nr_ch))
        Aqf.progress('Channel spacing = {}Hz'.format(ch_bw*dsim_factor))


        beam = beams[0]
        try:
            beam_name = beam.replace('-','_').replace('.','_')
            beam_ip, beam_port = self.cam_sensors.get_value(
                    beam_name+'_destination').split(':')
            beam_ip = beam_ip.split('+')[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap_m = int(self.conf_file['beamformer']['substreams_to_capture'])
            start_substream     = int(self.conf_file['beamformer']['start_substream_idx'])
            if start_substream+n_substrms_to_cap_m > substreams:
                errmsg = ('Substream start + substreams to process '
                          'is more than substreams available: {}. '
                          'Fix in test configuration file'.format(substeams))
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return False
            ticks_between_spectra = self.cam_sensors.get_value(
                    'antenna_channelised_voltage_n_samples_between_spectra')
            assert isinstance(ticks_between_spectra,int)
            spectra_per_heap = self.cam_sensors.get_value(beam_name+'_spectra_per_heap')
            assert isinstance(spectra_per_heap,int)
            ch_per_substream = self.cam_sensors.get_value(beam_name+'_n_chans_per_substream')
            assert isinstance(ch_per_substream, int)
        except AssertionError:
            errmsg = '%s'%str(reply).replace('\_', ' ')
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_freq = ch_list[strt_ch_idx]*dsim_factor
        Aqf.step('Start a KAT SDP docker ingest node for beam captures')
        docker_status = start_katsdpingest_docker(self, beam_ip, beam_port,
                                                  n_substrms_to_cap_m, nr_ch,
                                                  ticks_between_spectra,
                                                  ch_per_substream, spectra_per_heap)
        if docker_status:
            Aqf.progress('KAT SDP Ingest Node started. Capturing {} substream/s '
                         'starting at {}'.format(n_substrms_to_cap_m, beam_ip))
        else:
            Aqf.failed('KAT SDP Ingest Node failed to start')

        # Create a katcp client to connect to katcpingest
        if os.uname()[1] == 'cmc2':
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node_cmc2']
        elif os.uname()[1] == 'cmc3':
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node_cmc3']
        else:
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node']
        ingst_nd_p = self.corr_fix._test_config_file['beamformer']['ingest_node_port']
        _timeout = 10
        try:
            import katcp
            ingest_kcp_client = katcp.BlockingClient(ingst_nd, ingst_nd_p)
            ingest_kcp_client.setDaemon(True)
            ingest_kcp_client.start()
            self.addCleanup(ingest_kcp_client.stop)
            is_connected = ingest_kcp_client.wait_connected(_timeout)
            if not is_connected:
                errmsg = 'Could not connect to %s:%s, timed out.' %(ingst_nd, ingst_nd_p)
                ingest_kcp_client.stop()
                raise RuntimeError(errmsg)
        except Exception as e:
            LOGGER.exception(str(e))
            Aqf.failed(str(e))
        def substreams_to_capture(lbeam, lbeam_ip, lsubstrms_to_cap, lbeam_port):
            """ Set ingest node capture substreams """
            try:
                LOGGER.info('Setting ingest node to capture beam, substreams: {}, {}+{}:{}'
                            .format(lbeam,lbeam_ip,lsubstrms_to_cap-1,lbeam_port))
                reply, informs = ingest_kcp_client.blocking_request(
                    katcp.Message.request(
                    'substreams-to-capture',
                    '{}+{}:{}'.format(lbeam_ip,lsubstrms_to_cap-1,lbeam_port)),
                    timeout=_timeout)
                assert reply.reply_ok()
            except Exception as e:
                print e
                errmsg = 'Failed to issues ingest node capture-init: {}'.format(str(reply))
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)

        for beam in beams:
            beam_name = beam.replace('-','_').replace('.','_')
            beam_ip, beam_port = self.cam_sensors.get_value(
                    beam_name+'_destination').split(':')
            beam_ip = beam_ip.split('+')[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap = n_substrms_to_cap_m
            # Compute the start IP address according to substream start index
            beam_ip = int2ip(ip2int(beam_ip) + start_substream)
            substreams_to_capture(beam, beam_ip, n_substrms_to_cap, beam_port)
            Aqf.hop('')
            Aqf.hop('')
            Aqf.step('Testing beam: {}'.format(beam))

            def get_beam_data(beam, beam_dict=None, inp_ref_lvl=0, beam_quant_gain=1,
                act_wgts = None,
                exp_cw_ch=-1, s_ch_idx=0,
                s_substream= start_substream,
                subs_to_cap = n_substrms_to_cap,
                max_cap_retries=5, conf_data_type=False, avg_only=False, data_only=False):
                """
                    beam - beam name
                    beam_dict - Required beam weights dict. If this is none weights will not
                                be set, but act_wgts must be supplied for calculations
                    inp_ref_lvl - Input reference level for calculations, will be obtained
                                  if value is 0
                    beam_quant_gain - beam quant gain (level adjust after beamforming)
                    act_wgts = Dict containing actual set weights for beam, needed if beam_dict
                               not supplied as they will not be returned if this is the case.
                    exp_cw_ch - Expect a cw in this channel, ignore if -1
                    s_ch_idx - start channel of the captured substream, used to calculate real
                               frequencies.
                    s_substream = Start substream
                    subs_to_cap = Number of stubstreams in the capture
                    max_cap_retries = max number of retries if data cap failed
                    conf_data_type= If true print the beam data type
                    avg_only = only return a list of averaged beam power. list lenght is the number
                                of captured channels.
                    data_only = only return a matrix of beam power.
                                of captured channels.
                """


                # Determine slice of valid data in bf_raw
                bf_raw_str = s_substream*ch_per_substream
                bf_raw_end = bf_raw_str + ch_per_substream*subs_to_cap

                # Capture beam data, retry if more than 20% of heaps dropped or empty data
                retries = 0
                while retries < max_cap_retries:
                    if retries == max_cap_retries - 1:
                        Aqf.failed('Error capturing beam data.')
                        return False
                    retries += 1
                    try:
                        bf_raw, bf_flags, bf_ts, in_wgts = capture_beam_data(self, beam,
                            beam_dict, ingest_kcp_client)
                        # Set beamdict to None in case the capture needs to be retried.
                        # The beam weights have already been set.
                        beam_dict = None
                        if (len(in_wgts) == 0) and (type(act_wgts)==dict):
                            in_wgts = act_wgts.copy()
                    except Exception as e:
                        Aqf.failed('Confirm that the Docker container is running and also confirm the '
                            'igmp version = 2')
                        errmsg = 'Failed to capture beam data due to error: %s' % str(e)
                        Aqf.failed(errmsg)
                        LOGGER.error(errmsg)
                        return False

                    data_type = bf_raw.dtype.name
                    # Cut selected partitions out of bf_flags
                    flags = bf_flags[s_substream:s_substream+subs_to_cap]
                    #Aqf.step('Finding missed heaps for all partitions.')
                    if flags.size == 0:
                        LOGGER.warning('Beam data empty. Capture failed. Retrying...')
                        Aqf.failed('Beam data empty. Capture failed. Retrying...')
                    else:
                        missed_err = False
                        for part in flags:
                            missed_heaps = np.where(part>0)[0]
                            missed_perc = missed_heaps.size/part.size
                            perc = 0.50
                            if missed_perc > perc:
                                Aqf.progress('Missed heap percentage = {}%%'.format(missed_perc*100))
                                Aqf.progress('Missed heaps = {}'.format(missed_heaps))
                                LOGGER.warning('Beam captured missed more than %s%% heaps. Retrying...'%(perc*100))
                                Aqf.failed('Beam captured missed more than %s%% heaps. Retrying...'%(perc*100))
                                #missed_err = True
                                break
                        # Good capture, break out of loop
                        if not missed_err:
                            break

                # Print missed heaps
                idx = s_substream
                for part in flags:
                    missed_heaps = np.where(part>0)[0]
                    if missed_heaps.size > 0:
                        LOGGER.info('Missed heaps for substream {} at heap indexes {}'.format(idx,
                            missed_heaps))
                    idx += 1
                # Combine all missed heap flags. These heaps will be discarded
                flags = np.sum(flags,axis=0)
                #cap = [0] * num_caps
                #cap = [0] * len(bf_raw.shape[1])
                cap = []
                cap_idx = 0
                raw_idx = 0
                try:
                    for heap_flag in flags:
                        if heap_flag == 0:
                            for raw_idx in range(raw_idx, raw_idx+spectra_per_heap):
                                cap.append(np.array(complexise(bf_raw[bf_raw_str:bf_raw_end, raw_idx, :])))
                                cap_idx += 1
                            raw_idx += 1
                        else:
                            if raw_idx == 0:
                                raw_idx = spectra_per_heap
                            else:
                                raw_idx = raw_idx + spectra_per_heap
                except Exception, e:
                    errmsg = 'Failed to capture beam data due to error: %s' % str(e)
                    LOGGER.exception(errmsg)
                    Aqf.failed(errmsg)

                if conf_data_type:
                    Aqf.step('Confirm that the data type of the beamforming data for one channel.')
                    msg = ('Beamformer data type is {}, example value for one channel: {}'.format(
                        data_type, cap[0][0]))
                    Aqf.equals(data_type, 'int8', msg)


                cap_mag = np.abs(cap)
                if data_only:
                    return cap_mag, in_wgts
                cap_avg = cap_mag.sum(axis=0) / cap_idx
                cap_db = 20 * np.log10(cap_avg)
                cap_db_mean = np.mean(cap_db)
                if avg_only:
                    return cap_avg, in_wgts
                # NOT WORKING
                # labels = ''
                # lbls = self.parameters(self)
                # for lbl in lbls:
                #    bm = beam[-1]
                #    if lbl.find(bm) != -1:
                #        wght = self.correlator.bops.get_beam_weights(beam, lbl)
                # print lbl, wght
                #        labels += (lbl+"={} ").format(wght)
                labels = ''
                label_values = in_wgts.values()
                if label_values[1:] == label_values[:-1]:
                    labels += ("All inputs = {}\n".format(label_values[0]))
                else:
                    tmp = {}
                    for key,val in in_wgts.items():
                        if val not in tmp.values():
                            tmp[key] = val
                        else:
                            for k,v in tmp.items():
                                if val == v:
                                    tmp.pop(k)
                            tmp['Multiple Inputs'] = val
                    for key in tmp:
                        labels += (key + " = {}\n").format(tmp[key])
                labels += 'Mean = {:0.2f}dB\n'.format(cap_db_mean)

                failed = False
                if inp_ref_lvl == 0:
                    # Get the voltage level for one antenna. Gain for one input
                    # should be set to 1, the rest should be 0
                    inp_ref_lvl = np.mean(cap_avg)
                    Aqf.step('Input ref level: {}'.format(inp_ref_lvl))
                    Aqf.step('Reference level measured by setting the '
                             'gain for one antenna to 1 and the rest to 0. '
                             'Reference level = {:.3f}dB'.format(20*np.log10(inp_ref_lvl)))
                    Aqf.step('Reference level averaged over {} channels. '
                             'Channel averages determined over {} '
                             'samples.'.format(n_substrms_to_cap*ch_per_substream, cap_idx))
                    expected = 0
                else:
                    delta = 0.2
                    expected = np.sum([inp_ref_lvl * in_wgts[key] for key in in_wgts]) * beam_quant_gain
                    expected = 20 * np.log10(expected)

                    if exp_cw_ch != -1:
                        local_substream = s_ch_idx/ch_per_substream
                        # Find cw in expected channel, all other channels must be at expected level
                        max_val_ch = np.argmax(cap_db)
                        max_val = np.max(cap_db)
                        if max_val_ch == (exp_cw_ch-s_ch_idx):
                            msg = ('CW at {:.3f}MHz found in channel {}, magnitude = {:.1f}dB, '
                                   'spectrum mean = {:.1f}dB'.format(
                                    ch_list[exp_cw_ch]/1e6,
                                    exp_cw_ch,
                                    max_val,
                                    cap_db_mean))
                            LOGGER.info(msg)
                            if local_substream % align_print_modulo == 0:
                                Aqf.passed(msg)
                        else:
                            failed = True
                            Aqf.failed('CW at {:.3f}MHz not found in channel {}. '
                                       'Maximum value of {}dB found in channel {}. '
                                       'Mean spectrum value = {}dB'.format(
                                           ch_list[exp_cw_ch]/1e6,
                                           exp_cw_ch,
                                           max_val,
                                           max_val_ch+s_ch_idx,
                                           cap_db_mean))

                        spikes = np.where(cap_db > expected + delta)[0]
                        if len(spikes == 1):
                            msg = ('No spikes found in sub spectrum.')
                            LOGGER.info(msg)
                            if local_substream % align_print_modulo == 0:
                                Aqf.passed(msg)
                        else:
                            failed = True
                            Aqf.failed('Spikes found at: {}'.format(spikes))
                    else:
                        Aqf.step('Expected value is calculated by taking the reference input level '
                                 'and multiplying by the channel weights and quantiser gain.')
                        labels += 'Expected = {:.2f}dB\n'.format(expected)
                        msg = ('Confirm that the expected voltage level ({:.3f}dB) is within '
                            '{}dB of the measured mean value ({:.3f}dB)'.format(expected,delta, cap_db_mean))
                        Aqf.almost_equals(cap_db_mean, expected, delta, msg)
                return cap_avg, labels, inp_ref_lvl, expected, cap_idx, in_wgts, failed


            # Setting DSIM to generate noise
            if '4k' in self.instrument:
                # 4K
                awgn_scale = 0.0645
                cw_scale = 0.0
                gain = '113+0j'
                fft_shift = 511
            else:
                # 32K
                awgn_scale = 0.063
                cw_scale = 0.0
                gain = '344+0j'
                fft_shift = 4095

            Aqf.progress('Digitiser simulator configured to generate Gaussian noise: '
                         'Noise scale: {}, eq gain: {}, fft shift: {}'.format(
                         awgn_scale, gain, fft_shift))
            dsim_set_success = False
            with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
                dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                               cw_scale=cw_scale, freq=0, fft_shift=fft_shift, gain=gain)
            if not dsim_set_success:
                Aqf.failed('Failed to configure digitise simulator levels')
                return False

            # Only one antenna gain is set to 1, this will be used as the reference
            # input level
            # Set beamformer quantiser gain for selected beam to 1 quant gain reversed TODO: Fix
            bq_gain = set_beam_quant_gain(self, beams[1 - beams.index(beam)], 1)
            # Generating a dictionary to contain beam weights
            beam_dict = {}
            act_wgts = {}
            beam_pol = beam[-1]
            for label in labels:
                if label.find(beam_pol) != -1:
                    beam_dict[label] = 0.0
            if len(beam_dict) == 0:
                Aqf.failed('Beam dictionary not created, beam labels or beam name incorrect')
                return False
            ants = self.cam_sensors.get_value('n_ants')
            ref_input = np.random.randint(ants)
            # Find reference input label
            for key in beam_dict:
                if int(filter(str.isdigit,key)) == ref_input:
                    ref_input_label = key
                    break
            weight = 1.0
            beam_dict = populate_beam_dict_idx(self, ref_input, weight, beam_dict)
            beam_data = []
            beam_lbls = []
            Aqf.step('Testing individual beam weights.')
            try:
                # Calculate reference level by not specifying ref level
                # Use weights from previous test
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(
                        beam, beam_dict=beam_dict, conf_data_type=True)
            except TypeError, e:
                errmsg = 'Failed to retrieve beamformer data'
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return False
            beam_data.append(d)
            beam_lbls.append(l)

            # Characterise beam weight application:
            Aqf.step('Characterising beam weight application.')
            Aqf.step('Step weight for one input and plot the mean value for all channels against expected value.')
            Aqf.step('Expected value calculated by multiplying reference value with weight.')
            weight = 0.1
            mean_vals = []
            exp_mean_vals = []
            weight_lbls = []

            retry_cnt = 0
            while weight <= 4:
                # Set weight for reference input, the rest are all zero
                LOGGER.info('Confirm that antenna input ({}) weight has been set to the desired weight.'.format(
                    ref_input_label))
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.beam_weights(
                        beam, ref_input_label, round(weight,1))
                    assert reply.reply_ok()
                    actual_weight = float(reply.arguments[1])
                    retry_cnt = 0
                except AssertionError:
                    retry_cnt += 1
                    Aqf.failed('Beam weight not successfully set: {}'.format(reply))
                    if retry_cnt == 5:
                        Aqf.failed('Beam weight could not be set after 5 retries... Exiting test.')
                        return False
                    continue
                except Exception as e:
                    retry_cnt += 1
                    errmsg = 'Test failed due to %s'%str(e)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                    if retry_cnt == 5:
                        Aqf.failed('Beam weight could not be set after 5 retries... Exiting test.')
                        return False
                    continue
                else:
                    Aqf.passed('Antenna input {} weight set to {}'.format(key, actual_weight))

                # Get mean beam data
                try:
                    cap_data, act_wgts = get_beam_data(beam, avg_only=True)
                    cap_mean = np.mean(cap_data)
                    exp_mean = rl * actual_weight
                    mean_vals.append(cap_mean)
                    exp_mean_vals.append(exp_mean)
                    weight_lbls.append(weight)
                    Aqf.progress('Captured mean value = {:.2f}, Calculated mean value '
                            '(using reference value) = {:.2f}'.format(cap_mean,exp_mean))
                except TypeError, e:
                    errmsg = 'Failed to retrieve beamformer data'
                    Aqf.failed(errmsg)
                    LOGGER.error(errmsg)
                    return
                if round(weight,1) < 1:
                    weight += 0.1
                else:
                    weight += 0.5
            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(((mean_vals,
                'Captured mean beam output.\nStepping one input weight,\nwith remaining weights set to 0.'),
                (exp_mean_vals, 'Value calculated from reference,\nwhere reference measured at\nan input weight of 1.')),
                              plot_filename='{}/{}_weight_application_{}.png'.format(self.logs_path,
                                self._testMethodName, beam),
                              plot_title=('Beam = {}\n'
                                'Expected vs Actual Mean Beam Output for Input Weight.'.format(beam)),
                              log_dynamic_range=None, #90, log_normalise_to=1,
                              ylabel='Mean Beam Output',
                              xlabel='{} Weight'.format(ref_input_label), xvals=weight_lbls)

            # Test weight application across all antennas
            Aqf.step('Testing weight application across all antennas.')
            weight = 0.4 / ants
            beam_dict = populate_beam_dict(self, -1, weight, beam_dict)
            try:
                d, l, rl, exp1, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)
            weight = 1.0 / ants
            beam_dict = populate_beam_dict(self, -1, weight, beam_dict)
            try:
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except IndexError, e:
                errmsg = 'Failed to retrieve beamformer data'
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)
            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(zip(np.square(beam_data), beam_lbls),
                              plot_filename='{}/{}_chan_resp_{}.png'.format(self.logs_path,
                                self._testMethodName, beam),
                              plot_title=('Beam = {}\nSpectrum Start Frequency = {} MHz\n'
                                'Number of Channels Captured = {}'
                                '\nIntegrated over {} captures'.format(beam,
                                    strt_freq / 1e6, n_substrms_to_cap*ch_per_substream, nc)),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='Captured beamformer data', hlines=[exp0, exp1],
                              plot_type='bf', hline_strt_idx=1)

            Aqf.step('Testing quantiser gain adjustment.')
            # Level adjust after beamforming gain has already been set to 1
            beam_data = []
            beam_lbls = []
            try:
                # Recalculate reference level by not specifying ref level
                # Use weights from previous test
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(
                        beam, beam_quant_gain=bq_gain, act_wgts=act_wgts)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            l += 'Level adjust gain={}'.format(bq_gain)
            beam_lbls.append(l)

            # Set level adjust after beamforming gain to 0.5
            bq_gain = set_beam_quant_gain(self, beams[1 - beams.index(beam)], 0.5)
            #bq_gain = set_beam_quant_gain(self, beam, 0.5)
            try:
                 d, l, rl, exp1, nc, act_wgts, dummy = get_beam_data(
                    beam, inp_ref_lvl=rl, beam_quant_gain=bq_gain, act_wgts=act_wgts)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            l += 'Level adjust gain={}'.format(bq_gain)
            beam_lbls.append(l)

            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(zip(np.square(beam_data), beam_lbls),
                              plot_filename='{}/{}_level_adjust_after_bf_{}.png'.format(self.logs_path,
                                self._testMethodName, beam),
                              plot_title=('Beam = {}\nSpectrum Start Frequency = {} MHz\n'
                                'Number of Channels Captured = {}'
                                '\nIntegrated over {} captures'.format(beam,
                                    strt_freq / 1e6, n_substrms_to_cap*ch_per_substream, nc)),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='Captured beamformer data with level adjust after beam-forming gain set.',
                              hlines=exp1, plot_type='bf', hline_strt_idx=1)

            Aqf.step('Checking beamformer substream alignment by injecting a CW in each substream.')
            Aqf.step('Stepping through {} substreams and checking that the CW is in the correct '
                     'position.'.format(substreams))
            # Reset quantiser gain
            #bq_gain = set_beam_quant_gain(self, beam, 1)
            bq_gain = set_beam_quant_gain(self, beams[1 - beams.index(beam)], 1)
            if '4k' in self.instrument:
                # 4K
                awgn_scale = 0.0645
                cw_scale = 0.01
                gain = '113+0j'
                fft_shift = 511
            else:
                # 32K
                awgn_scale = 0.063
                cw_scale = 0.01
                gain = '344+0j'
                fft_shift = 4095

            Aqf.progress('Digitiser simulator configured to generate a stepping '
                         'Constant Wave and Gaussian noise, '
                         'CW scale: {}, Noise scale: {}, eq gain: {}, fft shift: {}'.format(
                         cw_scale, awgn_scale, gain, fft_shift))
            Aqf.step('This test will take a long time... check log for progress.')
            Aqf.step('Only 5 results will be printed, all {} substreams will be tested. '
                     'All errors will be displayed'.format(substreams))
            aligned_failed = False
            for substream in range(substreams):
                # Get substream start channel index
                strt_ch_idx = substream*ch_per_substream
                n_substrms_to_cap = 1
                # Compute the start IP address according to substream
                beam_ip = int2ip(ip2int(start_beam_ip) + substream)
                substreams_to_capture(beam, beam_ip, n_substrms_to_cap, beam_port)
                msg = ('Capturing 1 substream at {}'.format(beam_ip))
                LOGGER.info(msg)
                if substream % align_print_modulo == 0:
                    Aqf.passed(msg)

                # Step dsim CW
                dsim_set_success = False
                with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
                    cw_ch = strt_ch_idx + int(ch_per_substream/4)
                    freq = ch_list[cw_ch]
                    dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                                   cw_scale=cw_scale, freq=freq, fft_shift=fft_shift, gain=gain)
                if not dsim_set_success:
                    Aqf.failed('Failed to configure digitise simulator levels')
                    return False

                try:
                    d, l, rl, exp0, nc, act_wgts, failed = get_beam_data(
                            beam, inp_ref_lvl=rl, act_wgts=act_wgts,
                            exp_cw_ch=cw_ch, s_ch_idx = strt_ch_idx,
                            s_substream = substream,
                            subs_to_cap = n_substrms_to_cap)
                    if failed:
                        aligned_failed = True
                except IndexError, e:
                    errmsg = 'Failed to retrieve beamformer data'
                    Aqf.failed(errmsg)
                    LOGGER.error(errmsg)
                    return False
            if aligned_failed:
                Aqf.failed('Beamformer substream alignment test failed.')
            else:
                Aqf.passed('All beamformer substreams correctly aligned.')

        # Close any KAT SDP ingest nodes
        stop_katsdpingest_docker(self)


    def _test_beamforming_timeseries(self):
        """
        Perform a time series analysis of the beamforming data
        """
        # Main test code

        try:
            output = subprocess.check_output(['docker', 'run', 'hello-world'])
            LOGGER.info(output)
        except subprocess.CalledProcessError:
            errmsg = 'Cannot connect to the Docker daemon. Is the docker daemon running on this host?'
            LOGGER.error(errmsg)
            Aqf.failed(errmsg)
            return False

        try:
            #Set custom source names
            local_src_names = self.cam_sensors.custom_input_labels
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            assert reply.reply_ok()
            labels = reply.arguments[1:]
            beams = ['tied-array-channelised-voltage.0x','tied-array-channelised-voltage.0y']
            running_instrument = self.instrument
            assert running_instrument is not False
            msg = 'Running instrument currently does not have beamforming capabilities.'
            assert running_instrument.endswith('4k'), msg
            Aqf.step('Discontinue any capturing of %s and %s, if active.' %(beams[0],beams[1]))
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[0])
            assert reply.reply_ok(), str(reply)
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[1])
            assert reply.reply_ok(), str(reply)

            # Get instrument parameters
            bw = self.cam_sensors.get_value('bandwidth')
            nr_ch = self.cam_sensors.get_value('n_chans')
            ants = self.cam_sensors.get_value('n_ants')
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]
            dsim_factor = (float(self.conf_file['instrument_params']['sample_freq'])/
                           self.cam_sensors.get_value('scale_factor_timestamp'))
            print self.cam_sensors.get_value('scale_factor_timestamp')
            substreams = self.cam_sensors.get_value('n_xengs')
        except AssertionError:
            errmsg = '%s'%str(reply).replace('\_', ' ')
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        Aqf.progress('Bandwidth = {}Hz'.format(bw*dsim_factor))
        Aqf.progress('Number of channels = {}'.format(nr_ch))
        Aqf.progress('Channel spacing = {}Hz'.format(ch_bw*dsim_factor))

        beam = beams[0]
        try:
            beam_name = beam.replace('-','_').replace('.','_')
            beam_ip, beam_port = self.cam_sensors.get_value(
                    beam_name+'_destination').split(':')
            beam_ip = beam_ip.split('+')[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap_m = int(self.conf_file['beamformer']['substreams_to_capture'])
            start_substream     = int(self.conf_file['beamformer']['start_substream_idx'])
            if start_substream+n_substrms_to_cap_m > substreams:
                errmsg = ('Substream start + substreams to process '
                          'is more than substreams available: {}. '
                          'Fix in test configuration file'.format(substeams))
                LOGGER.error(errmsg)
                Aqf.failed(errmsg)
                return False
            ticks_between_spectra = self.cam_sensors.get_value(
                    'antenna_channelised_voltage_n_samples_between_spectra')
            assert isinstance(ticks_between_spectra,int)
            spectra_per_heap = self.cam_sensors.get_value(beam_name+'_spectra_per_heap')
            assert isinstance(spectra_per_heap,int)
            ch_per_substream = self.cam_sensors.get_value(beam_name+'_n_chans_per_substream')
            assert isinstance(ch_per_substream, int)
        except AssertionError:
            errmsg = '%s'%str(reply).replace('\_', ' ')
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_freq = ch_list[strt_ch_idx]*dsim_factor
        Aqf.step('Start a KAT SDP docker ingest node for beam captures')
        docker_status = start_katsdpingest_docker(self, beam_ip, beam_port,
                                                  n_substrms_to_cap_m, nr_ch,
                                                  ticks_between_spectra,
                                                  ch_per_substream, spectra_per_heap)
        if docker_status:
            Aqf.progress('KAT SDP Ingest Node started. Capturing {} substream/s '
                         'starting at {}'.format(n_substrms_to_cap_m, beam_ip))
        else:
            Aqf.failed('KAT SDP Ingest Node failed to start')

        # Setting DSIM to generate off center bin CW time sequence
        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.5
            cw_scale = 0.675
            #gain = '113+0j'
            gain = 11
            fft_shift = 8191
        else:
            # 32K
            awgn_scale = 0.063
            cw_scale = 0.01
            gain = '344+0j'
            fft_shift = 4095

        # Determine CW frequency
        center_bin_offset = float(self.conf_file['beamformer']['center_bin_offset'])
        center_bin_offset = 0.02
        center_bin_offset_freq = ch_bw * center_bin_offset
        cw_ch = strt_ch_idx + int(ch_per_substream/4)
        #cw_ch = 262
        freq = ch_list[cw_ch] + center_bin_offset_freq

        Aqf.step('Generating time analysis plots of beam for channel {} containing a '
                 'CW offset from center of a bin.'.format(cw_ch))
        Aqf.progress('Digitiser simulator configured to generate a '
                     'Constant Wave at {} Hz offset from the center '
                     'of a bin by {} Hz.'.format(freq, center_bin_offset_freq))
        Aqf.progress('CW scale: {}, Noise scale: {}, eq gain: {}, fft shift: {}'.format(
                     cw_scale, awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                           cw_scale=cw_scale, freq=freq, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        Aqf.step("Set beamformer quantiser gain for selected beam to 1")
        #set_beam_quant_gain(self, beam, 1)
        bq_gain = set_beam_quant_gain(self, beams[1 - beams.index(beam)], 1)

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        Aqf.progress("Only one antenna gain is set to 1, the reset are set to zero")
        ref_input = np.random.randint(ants)
        ref_input = 1
        # Find reference input label
        for key in beam_dict:
            if int(filter(str.isdigit,key)) == ref_input:
                ref_input_label = key
                break
        weight = 1.0
        beam_dict = populate_beam_dict_idx(self, ref_input, weight, beam_dict)
        try:
            bf_raw, bf_flags, bf_ts, in_wgts = capture_beam_data(self, beam, beam_dict)
            # Close any KAT SDP ingest nodes
            stop_katsdpingest_docker(self)
        except TypeError, e:
            errmsg = ('Failed to capture beam data: %s\n\n Confirm that Docker container is '
                     'running and also confirm the igmp version = 2 ' % str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        flags = bf_flags[start_substream:start_substream+n_substrms_to_cap_m]
        #Aqf.step('Finding missed heaps for all partitions.')
        if flags.size == 0:
            LOGGER.warning('Beam data empty. Capture failed. Retrying...')
            Aqf.failed('Beam data empty. Capture failed. Retrying...')
        else:
            missed_err = False
            for part in flags:
                missed_heaps = np.where(part>0)[0]
                missed_perc = missed_heaps.size/part.size
                perc = 0.50
                if missed_perc > perc:
                    Aqf.progress('Missed heap percentage = {}%%'.format(missed_perc*100))
                    Aqf.progress('Missed heaps = {}'.format(missed_heaps))
                    LOGGER.warning('Beam captured missed more than %s%% heaps. Retrying...'%(perc*100))
                    Aqf.failed('Beam captured missed more than %s%% heaps. Retrying...'%(perc*100))
        # Print missed heaps
        idx = start_substream
        for part in flags:
            missed_heaps = np.where(part>0)[0]
            if missed_heaps.size > 0:
                LOGGER.info('Missed heaps for substream {} at heap indexes {}'.format(idx,
                    missed_heaps))
            idx += 1
        # Combine all missed heap flags. These heaps will be discarded
        flags = np.sum(flags,axis=0)
        # Find longest run of uninterrupted data
        # Create an array that is 1 where flags is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(flags, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        # Find max run
        max_run = ranges[np.argmax(np.diff(ranges))]
        bf_raw_strt = max_run[0]*spectra_per_heap
        bf_raw_stop = max_run[1]*spectra_per_heap
        bf_raw = bf_raw[:,bf_raw_strt:bf_raw_stop,:]

        #np.save('skarab_bf_data_plus.np', bf_raw)
        #return True
        from skarab_bf_analysis import analyse_beam_data
        analyse_beam_data(bf_raw, dsim_settings = [freq, cw_scale, awgn_scale],
                cbf_settings = [fft_shift, gain],
                do_save = True,
                spectra_use = 'all',
                chans_to_use = n_substrms_to_cap_m*ch_per_substream,
                xlim = [20,21],
                dsim_factor = 1.0,
                ref_input_label = ref_input_label,
                bandwidth = bw)

        #aqf_plot_channels(beam_data[0:50, cw_ch-strt_ch_idx],
        #                  plot_filename='{}/{}_beam_cw_offset_from_centerbin_{}.png'.format(self.logs_path,
        #                    self._testMethodName, beam),
        #                  plot_title=('Beam = {}\n'
        #                    'Input = CW offset by {} Hz from the center of bin {}'
        #                    .format(beam, center_bin_offset_freq, cw_ch)),
        #                  log_dynamic_range=None, #90, log_normalise_to=1,
        #                  ylabel='Beam Output',
        #                  xlabel='Samples')

    def _test_cap_beam(self):
        """Testing timestamp accuracy
        Confirm that the CBF subsystem do not modify and correctly interprets
        timestamps contained in each digitiser SPEAD accumulations (dump)
        """
        if self.set_instrument():
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
            dsim_clk_factor = 1.712e9 / self.cam_sensors.sample_period
            Aqf.hop('Dsim_clock_Factor = {}'.format(dsim_clk_factor))
            bw = self.cam_sensors.get_value('bandwidth')  # * dsim_clk_factor

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

            try:
                bf_raw, cap_ts, bf_ts = capture_beam_data(self, beam, beamy_dict, target_pb,
                    target_cfreq)
            except TypeError, e:
                errmsg = 'Failed to capture beam data: %s' % str(e)
                Aqf.failed(errmsg)
                LOGGER.info(errmsg)
                return

                # cap_ts_diff = np.diff(cap_ts)
                # a = np.nonzero(np.diff(cap_ts)-8192)
                # cap_ts[a[0]+1]
                # cap_phase = numpy.angle(cap)
                # ts = [datetime.datetime.fromtimestamp(float(timestamp)/1000).strftime("%H:%M:%S") for timestamp in timestamps]

                # Average over timestamp show passband
                # for i in range(0,len(cap)):
                #    plt.plot(10*numpy.log(numpy.abs(cap[i])))

    # def _test_bc8n856M4k_beamforming_ch(self):
    #     """CBF Beamformer channel accuracy

    #     Apply weights and capture beamformer data.
    #     Verify that weights are correctly applied.
    #     """
    #     instrument_success = self.set_instrument()
    #     if instrument_success:
    #         msg = ('CBF Beamformer channel accuracy: {}\n'.format(_running_inst.keys()[0]))
    #         Aqf.step(msg)
    #         self._test_beamforming_ch(ants=4)

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
        bw = self.cam_sensors.get_value('bandwidth')
        ch_list = self.cam_sensors.ch_center_freqs
        nr_ch = self.n_chans_selected

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

        if '4k' in self.instrument:
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
        dsim_clk_factor = 1.712e9 / self.cam_sensors.sample_period
        eff_freq = (freq + bw) * dsim_clk_factor

        Aqf.step('Digitiser simulator configured to generate a continuous wave, '
                 'at {}Hz with cw scale: {}, awgn scale: {}, eq gain: {}, fft '
                 'shift: {}'.format(freq * dsim_clk_factor, cw_scale, awgn_scale, gain,
                                    fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
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

        try:
            bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = capture_beam_data(self, beam, beam_dict,
                target_pb, target_cfreq)
        except TypeError, e:
            errmsg = 'Failed to capture beam data: %s' % str(e)
            Aqf.failed(errmsg)
            LOGGER.info(errmsg)
            return
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

        local_src_names = self.cam_sensors.custom_input_labels
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
        bw = self.cam_sensors.get_value('bandwidth')
        ch_list = self.cam_sensors.ch_center_freqs
        nr_ch = self.n_chans_selected

        # Start of test. Setting required partitions and center frequency
        partitions = 1
        part_size = bw / 16
        target_cfreq = bw + bw * 0.5
        target_pb = partitions * part_size
        ch_bw = bw / nr_ch
        beams = ('beam_0x', 'beam_0y')
        beam = beams[1]

        # Set beamformer quantiser gain for selected beam to 1
        set_beam_quant_gain(self, beam, 1)

        if '4k' in self.instrument:
            # 4K
            awgn_scale = 0.032
            gain = '226+0j'
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = '344+0j'
            fft_shift = 4095

        Aqf.step('Digitiser simulator configured to generate Gaussian noise, '
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
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
                           plot_filename='{}/{}_adc_hist_{}.png'.format(self.logs_path,
                                self._testMethodName, inp),
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
                           plot_filename='{}/{}_quant_hist_{}.png'.format(self.logs_path,
                                self._testMethodName, inp),
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
        beam_dict = populate_beam_dict(self, 1, weight, beam_dict)
        try:
            bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = capture_beam_data(self, beam, beam_dict,
                target_pb, target_cfreq, capture_time=0.3)
        except TypeError, e:
            errmsg = 'Failed to capture beam data: %s' % str(e)
            Aqf.failed(errmsg)
            LOGGER.info(errmsg)
            return
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
        ch_bw = self.cam_sensors.delta_f
        acc_time = self.cam_sensors.fft_period
        sqrt_bw_at = np.sqrt(ch_bw * acc_time)
        Aqf.step('Calculating channel efficiency.')
        eff = 1 / ((ch_std / ch_mean) * sqrt_bw_at)
        Aqf.step('Beamformer mean efficiency for {} channels = {:.2f}%'.format(nr_ch, 100 * eff.mean()))
        plt_filename = '{}/{}_beamformer_efficiency.png'.format(self.logs_path, self._testMethodName)
        plt_title = ('Beamformer Efficiency per Channel\n '
                     'Mean Efficiency = {:.2f}%'.format(100 * eff.mean()))
        caption = ('Beamformer efficiency per channel calculated over {} samples '
                   'with a channel bandwidth of {:.2f}Hz and a FFT window length '
                   'of {:.3f} micro seconds per sample.'.format(
            cap_idx, ch_bw, acc_time * 1000000.))
        aqf_plot_channels(eff * 100, plt_filename, plt_title, caption=caption,
                          log_dynamic_range=None, hlines=95, ylimits=(90, 105), plot_type='eff')

    def _timestamp_accuracy(self, manual=False, manual_offset=0, future_dump=3):
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
                Aqf.failed('Timestamp received in accumulation not divisible'
                           ' by 8: {:.15f}'.format(load_timestamp))
            load_timestamp = int(load_timestamp)
            reg_size = 32
            load_ts_lsw = load_timestamp & (pow(2, reg_size) - 1)
            load_ts_msw = load_timestamp >> reg_size

            # dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()['data']['reg']
            # dsim_loc_msw = self.dhost.registers.local_time_msw.read()['data']['reg']
            # dsim_loc_time = dsim_loc_msw * pow(2,reg_size) + dsim_loc_lsw
            # print 'timestamp difference: {}'.format((load_timestamp - dsim_loc_time)*8/dump['scale_factor_timestamp'])
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

        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self.corr_fix, self.dhost,
                                            awgn_scale=0.0,
                                            cw_scale=0.0, freq=100000000,
                                            fft_shift=0, gain='32767+0j')
        self.dhost.outputs.out_1.scale_output(0)
        dump = get_clean_dump(self)
        baseline_lookup = get_baselines_lookup(self, dump)
        sync_time = self.cam_sensors.get_values('synch_epoch')
        scale_factor_timestamp = self.cam_sensors.get_values('scale_factor_timestamp')
        inp = self.cam_sensors.get_values('input_labels')[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = dump['n_chans'].value * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = self.cam_sensors.get_values('int_time') * self.cam_sensors.get_values('adc_sample_rate')
        # print dump_ticks
        dump_ticks = self.cam_sensors.get_values('n_accs') * self.cam_sensors.get_values('n_chans') * 2
        # print dump_ticks
        # print ['adc_sample_rate'].value
        # print dump['timestamp']
        if not (dump_ticks / 8.0).is_integer():
            Aqf.failed('Number of ticks per dump is not divisible'
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
            dump = get_clean_dump(self)
            print dump['timestamp']
            dump_ts = dump['timestamp']
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
                dump = self.receiver.data_queue.get()
                print dump['timestamp']
                dval = dump['xeng_raw']
                auto_corr = dval[:, inp_autocorr_idx, :]
                curr_ts = dump['timestamp']
                delta_ts = curr_ts - dump_ts
                dump_ts = curr_ts
                if delta_ts != dump_ticks:
                    Aqf.failed('Accumulation dropped, Expected timestamp = {}, '
                        'received timestamp = {}'.format(dump_ts + dump_ticks, curr_ts))
                print 'Maximum value found in dump {} = {}, average = {}'.format(cnt,
                    np.max(auto_corr), np.average(auto_corr))
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

    def _test_timestamp_shift(self):
        """Testing timestamp accuracy
        Confirm that the CBF subsystem do not modify and correctly interprets
        timestamps contained in each digitiser SPEAD accumulations (dump)
        """
        if self.set_instrument():
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

        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self.corr_fix, self.dhost,
                                            awgn_scale=0.0,
                                            cw_scale=0.0, freq=100000000,
                                            fft_shift=0, gain='32767+0j')
        self.dhost.outputs.out_1.scale_output(0)
        dump = get_clean_dump(self)
        baseline_lookup = get_baselines_lookup(self, dump)
        sync_time = self.cam_sensors.get_value('synch_epoch')
        scale_factor_timestamp = self.cam_sensors.get_value('scale_factor_timestamp')
        inp = self.cam_sensors.input_labels[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = self.n_chans_selected * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = self.cam_sensors.get_value('int_time') * self.cam_sensors.get_value('adc_sample_rate')
        dump_ticks = self.cam_sensors.get_value('n_accs') * self.n_chans_selected * 2
        input_spec_ticks = self.n_chans_selected * 2
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
                dump = get_clean_dump(self)
                dump_ts = dump['timestamp']
                sync_time = self.cam_sensors.get_value('synch_epoch')
                scale_factor_timestamp = self.cam_sensors.get_value('scale_factor_timestamp')
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
                    dump = self.receiver.data_queue.get()
                    print dump['timestamp']
                    dval = dump['xeng_raw']
                    auto_corr = dval[:, inp_autocorr_idx, :]
                    curr_ts = dump['timestamp']
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

    # def _test_bc8n856M32k_input_levels(self):
    #     """
    #     Testing Digitiser simulator input levels
    #     Set input levels to requested values and check that the ADC and the
    #     quantiser block do not see saturated samples.
    #     """
    #     Aqf.procedure(TestProcedure.Channelisation)
    #     try:
    #         assert eval(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument()
    #         _running_inst = self.corr_fix.get_running_instrument()
    #         if instrument_success:
    #             fft_shift = pow(2, 15) - 1
    #             self._set_input_levels_and_gain(profile='cw', cw_freq=200000000, cw_margin=0.6,
    #                                             trgt_bits=5, trgt_q_std=0.30, fft_shift=fft_shift)
    #         else:
    #             Aqf.failed(self.errmsg)

    def _test_input_levels(self):
        """Testing Digitiser simulator input levels
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        if self.set_instrument():
            Aqf.step('Setting and checking Digitiser simulator input levels')
            self._set_input_levels_and_gain(profile='noise', cw_freq=100000, cw_margin=0.3,
                                            trgt_bits=4, trgt_q_std=0.30, fft_shift=511)


    def _set_input_levels_and_gain(self, profile='noise', cw_freq=0, cw_src=0, cw_margin=0.05,
        trgt_bits=3.5, trgt_q_std=0.30, fft_shift=511):
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

        def adc_snapshot(source):
            try:
                reply,informs = self.corr_fix.katcp_rct.req.adc_snapshot(source)
                assert reply.reply_ok()
                adc_data = eval(informs[0].arguments[1])
                assert len(adc_data)==8192
                return adc_data
            except AssertionError as e:
                errmsg = 'Failed to get adc snapshot for input {}, reply = {}. {}'.format(source,reply,str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False
            except Exception as e:
                errmsg = 'Exception: {}'.format(str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

        def quant_snapshot(source):
            try:
                reply,informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(source)
                assert reply.reply_ok()
                quant_data = eval(informs[0].arguments[1])
                assert len(quant_data)==4096
                return quant_data
            except AssertionError as e:
                errmsg = 'Failed to get quantiser snapshot for input {}, reply = {}. {}'.format(source,reply,str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False
            except Exception as e:
                errmsg = 'Exception: {}'.format(str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

        def set_gain(source,gain_str):
            try:
                reply, informs = self.corr_fix.katcp_rct.req.gain(source, gain_str)
                assert reply.reply_ok()
                assert reply.arguments[1:][0] == gain_str
            except AssertionError as e:
                errmsg = 'Failed to set gain for input {}, reply = {}. {}'.format(source,reply,str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False
            except Exception as e:
                errmsg = 'Exception: {}'.format(str(e))
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

        # main code
        Aqf.step('Requesting input labels.')
        try:
            katcp_rct = self.corr_fix.katcp_rct.sensors
            input_labels = eval(katcp_rct.input_labelling.get_value())
            assert isinstance(input_labels,list)
            inp_labels = [x[0] for x in input_labels]
        except AssertionError as e:
            errmsg = 'Failed to get input labels. {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False

        # Set digitiser input level of one random input,
        # store values from other inputs for checking
        inp = random.choice(inp_labels)
        ret_dict = dict.fromkeys(inp_labels,{})
        scale = 0.1
        margin = 0.005
        self.dhost.noise_sources.noise_corr.set(scale=round(scale, 3))
        # Get target standard deviation. ADC is represented by Q10.9
        # signed fixed point.
        target_std = pow(2.0, trgt_bits) / 512
        found = False
        count = 1
        Aqf.step('Setting input noise level to toggle {} bits at ' \
                 'standard deviation.'.format(trgt_bits))
        while not found:
            Aqf.step('Capturing ADC Snapshot {} for input {}.'.format(count,inp))
            adc_data = adc_snapshot(inp)
            cur_std = np.std(adc_data)
            cur_diff = target_std - cur_std
            if (abs(cur_diff) < margin) or count > 6:
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

            bw = self.cam_sensors.get_value('bandwidth')
            nr_ch = self.n_chans_selected
            ch_bw = self.cam_sensors.ch_center_freqs[1]
            ch_list = self.cam_sensors.ch_center_freqs
            freq_ch = int(round(cw_freq / ch_bw))
            scale = 1.0
            step = 0.005
            count = 1
            found = False
            while not found:
                Aqf.step('Capturing ADC Snapshot {} for input {}.'.format(count,inp))
                set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                adc_data = adc_snapshot(inp)
                if (count < 5) and (np.abs(np.max(adc_data) or
                                               np.min(adc_data)) >= 0b111111111 / 512.0):
                    scale -= step
                    count += 1
                else:
                    scale -= (step + cw_margin)
                    freq = set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                    adc_data = adc_snapshot(inp)
                    found = True
            Aqf.step('Digitiser simulator CW scale set to {:.3f}.'.format(scale))
            aqf_plot_histogram(adc_data,
                               plot_filename='{}/adc_hist_{}.png'.format(self.logs_path, inp),
                               plot_title=(
                                   'ADC Histogram for input {}\nAdded Noise Profile: '
                                   'Std Dev: {:.3f} equates to {:.1f} bits '
                                   'toggling.'.format(inp, p_std, p_bits)),
                               caption='ADC Input Histogram',
                               bins=256, ranges=(-1, 1))

        else:
            aqf_plot_histogram(adc_data,
                               plot_filename='{}/adc_hist_{}.png'.format(self.logs_path, inp),
                               plot_title=(
                                   'ADC Histogram for input {}\n Standard Deviation: {:.3f} equates '
                                   'to {:.1f} bits toggling'.format(inp, p_std, p_bits)),
                               caption='ADC Input Histogram',
                               bins=256, ranges=(-1, 1))

        for key in ret_dict.keys():
            Aqf.step('Capturing ADC Snapshot for input {}.'.format(key))
            #adc_data = adc_snapshot(key)
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
            assert reply.reply_ok()
            for key in ret_dict.keys():
                ret_dict[key]['fft_shift'] = reply.arguments[1:][0]
        except AssertionError as e:
            errmsg = 'Failed to set FFT shift, reply = {}. {}'.format(reply,str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)

        if profile == 'cw':
            Aqf.step('Setting quantiser gain for CW input.')
            gain = 1
            gain_str = '{}'.format(int(gain)) + '+0j'
            set_gain(inp, gain_str)

            try:
                dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                baseline_lookup = get_baselines_lookup(self, dump)
                inp_autocorr_idx = baseline_lookup[(inp, inp)]
                dval = dump['xeng_raw']
                auto_corr = dval[:, inp_autocorr_idx, :]
                ch_val = auto_corr[freq_ch][0]
                next_ch_val = 0
                n_accs = self.cam_sensors.get_value('n_accs')
                ch_val_array = []
                ch_val_array.append([ch_val, gain])
                count = 0
                prev_ch_val_diff = 0
                found = False
                max_count = 100
                two_found=False
                while count < max_count:
                    count += 1
                    ch_val = next_ch_val
                    gain += 1
                    gain_str = '{}'.format(int(gain)) + '+0j'
                    Aqf.step('Setting quantiser gain of {} for input {}.'.format(gain_str,inp))
                    set_gain(inp, gain_str)
                    try:
                        dump = get_clean_dump(self)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    except AssertionError:
                        errmsg = ('No of channels (%s) in the spead data is inconsistent with the no of'
                                  ' channels (%s) expected' %(dump['xeng_raw'].shape[0],
                                    self.n_chans_selected))
                        Aqf.failed(errmsg)
                        LOGGER.error(errmsg)
                        return False
                    else:
                        dval = dump['xeng_raw']
                        auto_corr = dval[:, inp_autocorr_idx, :]
                        next_ch_val = auto_corr[freq_ch][0]
                        ch_val_diff = next_ch_val - ch_val
                        # When the gradient start decreasing the center of the linear
                        # section has been found. Grab the same number of points from
                        # this point. Find 2 decreasing differences in a row
                        if (not found) and (ch_val_diff < prev_ch_val_diff):
                            if two_found:
                                found = True
                                count = max_count - count - 1
                            else:
                                two_found = True
                        else:
                            two_found = False
                        ch_val_array.append([next_ch_val, gain])
                        prev_ch_val_diff = ch_val_diff

            y = [x[0] for x in ch_val_array]
            x = [x[1] for x in ch_val_array]
            grad = np.gradient(y)
            # This does not work relibably
            #grad_delta = []
            #for i in range(len(grad) - 1):
            #    grad_delta.append(grad[i + 1] / grad[i])
            ## The setpoint is where grad_delta is closest to 1
            #grad_delta = np.asarray(grad_delta)
            #set_point = np.argmax(grad_delta - 1.0 < 0) + 1
            set_point = np.argmax(grad)
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
            plot_filename = '{}/cw_ch_response_{}.png'.format(self.logs_path, inp)
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
            set_gain(inp, gain_str)
            while (not found):
                Aqf.step('Capturing quantiser snapshot for gain of ' + gain_str)
                data = quant_snapshot(inp)
                cur_std = np.std(data)
                cur_diff = trgt_q_std - cur_std
                if (abs(cur_diff) < margin) or count > 20:
                    found = True
                else:
                    count += 1
                    perc_change = trgt_q_std / cur_std
                    gain = gain * perc_change
                    gain_str = '{}'.format(int(gain)) + '+0j'
                    set_gain(inp, gain_str)

        # Set calculated gain for remaining inputs
        for key in ret_dict.keys():
            if profile == 'cw':
                ret_dict[key]['cw_freq'] = freq
            set_gain(key, gain_str)
            data = quant_snapshot(key)
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
                dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            except AssertionError:
                errmsg = ('No of channels (%s) in the spead data is inconsistent with the no of'
                          ' channels (%s) expected' %(dump['xeng_raw'].shape[0],
                            self.n_chans_selected))
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return False
            else:
                dval = dump['xeng_raw']
                auto_corr = dval[:, inp_autocorr_idx, :]
                plot_filename = '{}/spectrum_plot_{}.png'.format(self.logs_path, key)
                plot_title = ('Spectrum for Input {}\n'
                              'Quantiser Gain: {}'.format(key, gain_str))
                caption = 'Spectrum for CW input'
                aqf_plot_channels(10 * np.log10(auto_corr[:, 0]),
                                  plot_filename=plot_filename,
                                  plot_title=plot_title, caption=caption, show=True)
        else:
            p_std = np.std(data)
            aqf_plot_histogram(np.abs(data),
                               plot_filename='{}/quant_hist_{}.png'.format(self.logs_path, key),
                               plot_title=('Quantiser Histogram for input {}\n '
                                           'Standard Deviation: {:.3f},'
                                           'Quantiser Gain: {}'.format(key, p_std, gain_str)),
                               caption='Quantiser Histogram',
                               bins=64, ranges=(0, 1.5))

        key = ret_dict.keys()[0]
        if profile == 'cw':
            Aqf.step('Digitiser simulator Sine Wave scaled at {:0.3f}'.format(ret_dict[key]['scale']))
        Aqf.step('Digitiser simulator Noise scaled at {:0.3f}'.format(ret_dict[key]['noise_scale']))
        Aqf.step('FFT Shift set to {}'.format(ret_dict[key]['fft_shift']))
        for key in ret_dict.keys():
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


    def _small_voltage_buffer(self):
        channel_list = self.cam_sensors.ch_center_freqs
        # Choose a frequency 3 quarters through the band
        cw_chan_set = int(self.n_chans_selected * 3 / 4)
        cw_freq = channel_list[cw_chan_set]
        dsim_clk_factor = 1.712e9 / self.cam_sensors.sample_period
        bandwidth = self.cam_sensors.get_value("bandwidth")
        eff_freq = (cw_freq + bandwidth) * dsim_clk_factor
        channel_bandwidth = self.cam_sensors.delta_f
        input_labels = self.cam_sensors.input_labels

        if '4k' in self.instrument:
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

        Aqf.step('Digitiser simulator configured to generate a continuous wave at %s Hz (channel=%s),'
                 ' with cw scale: %s, awgn scale: %s, eq gain: %s, fft shift: %s' % (cw_freq,
                    cw_chan_set, cw_scale, awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=cw_freq, fft_shift=fft_shift, gain=gain, cw_src=0)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            Aqf.step('Confirm that the `Transient Buffer ready` is implemented.')
            reply, informs = self.corr_fix.katcp_rct.req.transient_buffer_trigger()
            assert reply.reply_ok()
            Aqf.passed('Transient buffer trigger present.')
        except Exception:
            Aqf.failed('Transient buffer trigger failed. \nReply: %s' % str(reply).replace('_', ' '))

        try:
            Aqf.step('Randomly select an input to capture ADC snapshot')
            input_label = random.choice(input_labels)
            Aqf.progress('Selected input %s to capture ADC snapshot from' % input_label)
            Aqf.step('Capture an ADC snapshot and confirm the fft length')
            reply, informs = self.corr_fix.katcp_rct.req.adc_snapshot(input_label,
                timeout=60)
            assert reply.reply_ok()
            informs = informs[0]
        except Exception:
            LOGGER.exception('Failed to capture ADC snapshot.')
            Aqf.failed('Failed to capture ADC snapshot. \nReply: %s' % str(reply).replace('_',' '))
            return
        else:
            adc_data = eval(informs.arguments[-1])
            fft_len = len(adc_data)
            Aqf.progress('ADC capture length: {}'.format(fft_len))
            fft_real = np.abs(np.fft.fft(adc_data))
            fft_pos = fft_real[0:int(fft_len / 2)]
            cw_chan = np.argmax(fft_pos)
            cw_freq_found = cw_chan / (fft_len / 2) * bandwidth
            msg = ('Confirm that the expected frequency: {}Hz and measured frequency: '
                   '{}Hz matches to within a channel bandwidth: {:.3f}Hz'.format(cw_freq_found,
                        cw_freq, channel_bandwidth))
            Aqf.almost_equals(cw_freq_found, cw_freq, channel_bandwidth, msg)
            aqf_plot_channels(np.log10(fft_pos),
                              plot_filename='{}/{}_fft_{}.png'.format(self.logs_path,
                                    self._testMethodName, input_label),
                              plot_title=('Input Frequency = %s Hz\nMeasured Frequency at FFT bin %s '
                                          '= %sHz' % (cw_freq, cw_chan, cw_freq_found)),
                              log_dynamic_range=None,
                              caption=('FFT of captured small voltage buffer. %s voltage points captured '
                                       'on input %s. Input bandwidth = %sHz' % (fft_len
                                            , input_label, bandwidth)),
                              xlabel='FFT bins')

    def _test_informal(self):
        pass

    def _test_global_manual(self, ve_num):
        """Manual Test Method, for executing all manual tests

        Parameters
        ----------
            ve_num: str, Verification Event Number
        Returns
        -------
            results:
                Pass or TBD
        """
        # Assumes dict is returned
        _results = self.csv_manual_tests.csv_to_dict(ve_num)
        ve_desc = _results.get('Verification Event Description', 'TBD')
        Aqf.procedure(r'%s' % ve_desc)
        try:
            assert (eval(os.getenv('MANUAL_TEST', 'False')) or eval(os.getenv('DRY_RUN', 'False')))
        except AssertionError:
            results = r'%s' % _results.get("Verification Event Results", "TBD")
            if results != "TBD":
                Aqf.step(r'%s' % _results.get("Verification Requirement Description", "TBD"))
                Aqf.passed(r'%s' % results)
                perf = _results.get("Verification Event Performed By", "TBD")
                _date = _results.get("Date of Verification Event", "TBD")
                if perf != 'TBD':
                    Aqf.hop(r"Test ran by: %s on %s" % (perf, _date))
            else:
                Aqf.tbd("This test results outstanding.")

    def _test_efficiency(self):


        csv_filename = r"CBF_Efficiency_Data.csv"

        def get_samples():

            n_chans = self.cam_sensors.get_value('n_chans')
            test_chan = random.choice(range(n_chans)[:self.n_chans_selected])
            requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=101,
                                                                     chans_around=2)
            expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
            # Get baseline 0 data, i.e. auto-corr of m000h
            test_baseline = 0
            # [CBF-REQ-0053]
            min_bandwithd_req = 770e6
            # [CBF-REQ-0126] CBF channel isolation
            cutoff = 53  # dB
            # Channel magnitude responses for each frequency
            chan_responses = []
            last_source_freq = None
            print_counts = 3
            req_chan_spacing=250e3

            if '4k' in self.instrument:
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
            dsim_set_success = False
            with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
                dsim_set_success = set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                                freq=expected_fc, fft_shift=fft_shift, gain=gain)
            if not dsim_set_success:
                Aqf.failed('Failed to configure digitise simulator levels')
                return False
            try:
                Aqf.step('Randomly select a frequency channel to test. Capture an initial correlator '
                         'SPEAD accumulation, determine the number of frequency channels')
                initial_dump = get_clean_dump(self)
                self.assertIsInstance(initial_dump, dict)
            except Exception:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            else:

                bls_to_test = eval(self.cam_sensors.get_value('bls_ordering'))[test_baseline]
                Aqf.progress('Randomly selected frequency channel to test: {} and '
                             'selected baseline {} / {} to test.'.format(test_chan, test_baseline,
                                bls_to_test))
                Aqf.equals(np.shape(initial_dump['xeng_raw'])[0], self.n_chans_selected,
                           'Confirm that the number of channels in the SPEAD accumulation, is equal '
                           'to the number of frequency channels as calculated: {}'.format(
                               np.shape(initial_dump['xeng_raw'])[0]))

                Aqf.is_true(self.cam_sensors.get_value('bandwidth') >= min_bandwithd_req,
                            'Channelise total bandwidth {}Hz shall be >= {}Hz.'.format(
                                self.cam_sensors.get_value('bandwidth'), min_bandwithd_req))
                chan_spacing = self.cam_sensors.get_value('bandwidth') / n_chans
                chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100),
                                    chan_spacing + (chan_spacing * 1 / 100)]
                Aqf.step('Confirm that the number of calculated channel '
                         'frequency step is within requirement.')
                msg = ('Verify that the calculated channel '
                       'frequency ({} Hz)step size is between {} and {} Hz'.format(chan_spacing,
                        req_chan_spacing / 2, req_chan_spacing))
                Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

                Aqf.step('Confirm that the channelisation spacing and confirm that it is '
                         'within the maximum tolerance.')
                msg = ('Channelisation spacing is within maximum tolerance of 1% of the '
                       'channel spacing.')
                Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

            Aqf.step('Sweep the digitiser simulator over the centre frequencies of at '
                     'least all the channels that fall within the complete L-band')

            for i, freq in enumerate(requested_test_freqs):
                if i < print_counts:
                    Aqf.progress('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(
                        i + 1, len(requested_test_freqs), freq / 1e6))
                elif i == print_counts:
                    Aqf.progress('.' * print_counts)
                elif i >= (len(requested_test_freqs) - print_counts):
                    Aqf.progress('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(
                        i + 1, len(requested_test_freqs), freq / 1e6))
                else:
                    LOGGER.debug('Getting channel response for freq %s @ %s: %s MHz.' % (
                        i + 1, len(requested_test_freqs), freq / 1e6))

                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency

                if this_source_freq == last_source_freq:
                    LOGGER.debug('Skipping channel response for freq %s @ %s: %s MHz.\n'
                                'Digitiser frequency is same as previous.' % (
                                    i + 1, len(requested_test_freqs), freq / 1e6))
                    continue  # Already calculated this one
                else:
                    last_source_freq = this_source_freq

                try:
                    this_freq_dump = self.receiver.get_clean_dump()
                    #get_clean_dump(self)
                    self.assertIsInstance(this_freq_dump, dict)
                except AssertionError:
                    errmsg = ('Could not retrieve clean SPEAD accumulation')
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                    return False
                else:
                    # No of spead heap discards relevant to vacc
                    discards = 0
                    max_wait_dumps = 100
                    deng_timestamp = self.dhost.registers.sys_clkcounter.read().get('timestamp')
                    while True:
                        try:
                            queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                            self.assertIsInstance(queued_dump, dict)
                        except Exception:
                            errmsg = 'Could not retrieve clean accumulation.'
                            LOGGER.exception(errmsg)
                            Aqf.failed(errmsg)
                        else:
                            timestamp_diff = np.abs(queued_dump['dump_timestamp'] - deng_timestamp)
                            if (timestamp_diff < 0.5):
                                msg = ('Received correct accumulation timestamp: %s, relevant to '
                                       'DEngine timestamp: %s (Difference %.2f)' % (
                                        queued_dump['dump_timestamp'], deng_timestamp, timestamp_diff))
                                LOGGER.info(msg)
                                break

                            if discards > max_wait_dumps:
                                errmsg = ('Could not get accumulation with correct timestamp within %s '
                                           'accumulation periods.' % max_wait_dumps)
                                Aqf.failed(errmsg)
                                LOGGER.error(errmsg)
                                break
                            else:
                                msg = ('Discarding subsequent dumps (%s) with dump timestamp (%s) '
                                       'and DEngine timestamp (%s) with difference of %s.' %(discards,
                                        queued_dump['dump_timestamp'], deng_timestamp, timestamp_diff))
                                LOGGER.info(msg)
                        discards += 1


                    this_freq_response = normalised_magnitude(
                        queued_dump['xeng_raw'][:, test_baseline, :])
                    chan_responses.append(this_freq_response)

            chan_responses = np.array(chan_responses)
            requested_test_freqs = np.asarray(requested_test_freqs)
            np.savetxt("CBF_Efficiency_Data.csv", zip(chan_responses[:, test_chan],
                requested_test_freqs), delimiter=",")

        def efficiency_calc(f, P_dB, binwidth, debug=False):
            # Adapted from SSalie
            # Sidelobe & scalloping loss requires f to be normalized to bins
            # Normalize the filter response
            Aqf.step("Measure/record the filter-bank spectral response from a channel")
            P_dB -= P_dB.max()
            f = f - f[P_dB>-3].mean() # CHANGED: center on zero

            # It's critical to get precise estimates of critical points so to minimize measurement resolution impact, up-sample!
            _f_, _P_dB_ = f, P_dB
            _f10_ = np.linspace(f[0], f[-1], len(f)*10) # up-sample 10x
            # CHANGED: slightly better than np.interp(_f10_, f, P_dB) e.g. for poorly sampled data
            P_dB = scipy.interpolate.interp1d(f, P_dB, "quadratic", bounds_error=False)(_f10_)
            f = _f10_

            # Measure critical bandwidths
            f_HPBW = f[P_dB >= -3.0];
            f_HABW = f[P_dB >= -6.0] # CHANGED: with better interpolation don't need earlier "fudged" 3.05 & 6.05
            HPBW = (f_HPBW[-1] - f_HPBW[0])/binwidth
            HABW = (f_HABW[-1] - f_HABW[0])/binwidth
            h = 10**(P_dB / 10.)
            NEBW = np.sum(h[:-1] * np.diff(f)) / binwidth # Noise Equivalent BW
            Aqf.step("Determine the Half Power Bandwidth as well as the Noise Equivalent Bandwidth "
                      "for each swept channel")
            Aqf.progress("Half Power Bandwidth: %s, Noise Equivalent Bandwidth: %s" % (HPBW, NEBW))

            Aqf.step("Compute the efficiency as the ratio of Half Power Bandwidth to the Noise "
                     "Equivalent Bandwidth: efficiency = HPBW/NEBW")
            _efficiency = HPBW / NEBW
            Aqf.more(_efficiency, .98, "Efficiency factor = {:.3f}".format(_efficiency))
            # Measure critical points
            pk = f.searchsorted(f[P_dB > -6].mean()) # The peak
            ch = f.searchsorted(f[0] + binwidth) # Channel-to-channel separation intervals
            SL = P_dB[pk+ch//2-1] # Scalloping loss at mid-point between channel peaks
            SL80 = P_dB[pk:pk + int((0.8*ch)//2-1)].min() # Max scalloping loss within 80% of a channel
            DDP = np.diff( scipy.signal.medfilt(np.diff(P_dB), (ch//16)*2+1) ) # Smooth it over 1/8th of a bin width to get rid of main lobe ripples
            mn = pk+ch//2 + (DDP[pk+ch//2:]>0.01).argmax() # The first large inflection point after the peak is the null
            SLL = P_dB[mn:].max() # The nearest one is typically the peak sidelobe
            # Upper half of the channel & the excluding main lobe
            plt.figure()
            plt.subplot(211);
            plt.title("Efficiency factor = {:.3f}".format(_efficiency))
            plt.plot(_f_, _P_dB_, label='Channel Response');
            plt.plot(f[pk:], P_dB[pk:], 'g.', label='Peak')
            plt.plot(f[mn:], P_dB[mn:], 'r.', label='After Null')
            plt.legend()
            plt.grid(True);
            plt.subplot(212, sharex=plt.gca());
            plt.plot(f[1:-1], DDP, label='Data diff');
            plt.grid(True)
            plt.legend()
            if debug:
                plt.show()


            cap = ("SLL = %.f, SL = %.1f(%.f), NE/3dB/6dB BW = %.2f/%.2f/%.2f, HPBW/NEBW = %4f, "% (
                SLL, SL, SL80, NEBW, HPBW, HABW, HPBW/NEBW))
            filename ='{}/{}.png'.format(self.logs_path, self._testMethodName)
            Aqf.matplotlib_fig(filename, caption=cap, autoscale=True)

        try:
            pfb_data = np.loadtxt(csv_filename, delimiter=",", unpack=False)
            Aqf.step("Retrieve channelisation (Frequencies and Power_dB) data results from CSV file")
        except IOError:
            try:
                get_samples()
                csv_file = max(glob.iglob(csv_filename), key=os.path.getctime)
                assert 'CBF' in csv_file
                pfb_data = np.loadtxt(csv_file, delimiter=',', unpack=False)
            except Exception:
                msg = 'Failed to load CBF_Efficiency_Data.csv file'
                LOGGER.exception(msg)
                Aqf.failed(msg)
                return

        chan_responses, requested_test_freqs = pfb_data[:,0][1:], pfb_data[:,1][1:]
        # Summarize isn't clever enough to cope with the spurious spike in first sample
        requested_test_freqs = np.asarray(requested_test_freqs)
        chan_responses = 10*np.log10(np.abs(np.asarray(chan_responses)))
        try:
            binwidth = self.cam_sensors.get_value('bandwidth') / (self.n_chans_selected - 1)
            efficiency_calc(requested_test_freqs, chan_responses, binwidth)
        except Exception:
            msg = "Could not compute the data, rerun test"
            LOGGER.exception(msg)
            Aqf.failed(msg)
        # else:
        #     subprocess.check_call(["rm", csv_filename])
