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

import gc
import glob
import os
import Queue
import random
import socket
import struct
import subprocess
import sys
import time
import unittest
from ast import literal_eval as evaluate
from datetime import datetime

import corr2
import katcp
import matplotlib.pyplot as plt
import ntplib
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal
from dotenv import find_dotenv, load_dotenv
from katcp.testutils import start_thread_with_cleanup
from mkat_fpga_tests import CorrelatorFixture, add_cleanup
from mkat_fpga_tests.aqf_utils import *
from mkat_fpga_tests.utils import *
from nosekatreport import (Aqf, aqf_requirements, aqf_vr, decorators,
                           generic_test, instrument_1k, instrument_4k,
                           instrument_32k, manual_test, slow, system, untested)
from termcolor import colored

from Corr_RX import CorrRx
from descriptions import TestProcedure
from Logger import LoggingClass
from power_logger import PowerLogger

load_dotenv(find_dotenv())

# How long to wait for a correlator dump to arrive in tests
DUMP_TIMEOUT = 10
SET_DSIM_EPOCH = False
DSIM_TIMEOUT = 60


# ToDo MM (2017-07-21) Improve the logging for debugging
@cls_end_aqf
@system("meerkat")
class test_CBF(unittest.TestCase, LoggingClass, AqfReporter, UtilsClass):
    """ Unit-testing class for mkat_fpga_tests"""

    cur_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    _katreport_dir = os.path.join(cur_path, "katreport")
    _csv_filename = os.path.join(cur_path, "docs/Manual_Tests.csv")
    _images_dir = os.path.join(cur_path, "docs/manual_tests_images")
    if os.path.exists(_csv_filename):
        csv_manual_tests = CSV_Reader(_csv_filename, set_index="Verification Event Number")

    def setUp(self):
        global SET_DSIM_EPOCH
        super(test_CBF, self).setUp()
        AqfReporter.__init__(self)
        self.receiver = None
        self._dsim_set = False
        self.logs_path = None
        self.corr_fix = CorrelatorFixture(logLevel=self.logger.root.level)
        try:
            self.logs_path = self.create_logs_directory()
            self.conf_file = self.corr_fix.test_config
            self.corr_fix.katcp_client = self.conf_file["instrument_params"]["katcp_client"]
            self.katcp_req = self.corr_fix.katcp_rct.req
            self.assertIsInstance(self.katcp_req, katcp.resource_client.AttrMappingProxy)
            self.katcp_req_sensors = self.corr_fix.katcp_rct_sensor.req
            self.assertIsInstance(self.katcp_req_sensors, katcp.resource_client.AttrMappingProxy)
            self.Note("Connecting to katcp client on %s" % self.corr_fix.katcp_client)
            self.cam_sensors = GetSensors(self.corr_fix)
        except AttributeError:
            errmsg = "Is the instrument up??"
            Aqf.failed(errmsg)
            sys.exit(errmsg)
        except Exception:
            errmsg = "Failed to connect to katcp/read test config file - Is the instrument up??"
            Aqf.Failed(errmsg)
            sys.exit(errmsg)

        errmsg = "Failed to instantiate the dsim, investigate"
        try:
            self.dhost = self.corr_fix.dhost
            if not isinstance(self.dhost, corr2.dsimhost_fpga.FpgaDsimHost):
                raise AssertionError(errmsg)
        except Exception:
            self.Error(errmsg, exc_info=True)
            sys.exit(errmsg)
        else:
            # See: https://docs.python.org/2/library/functions.html#super
            if SET_DSIM_EPOCH is False:
                try:
                    self.logger.info("This should only run once...")
                    self.fhosts, self.xhosts = (self.get_hosts("fhost"), self.get_hosts("xhost"))
                    if not self.dhost.is_running():
                        errmsg = "Dsim is not running, ensure dsim is running before test commences"
                        Aqf.end(message=errmsg)
                        sys.exit(errmsg)
                    self.dhost.get_system_information(filename=self.dhost.config.get("bitstream"))
                    if not isinstance(self.corr_fix.instrument, str):
                        self.Error("Issues with the defined instrument, figure it out")
                    # cbf_title_report(self.corr_fix.instrument)
                    # Disable warning messages(logs) once
                    disable_warnings_messages()
                    self.assertIsInstance(
                        self.corr_fix.katcp_rct,
                        katcp.resource_client.ThreadSafeKATCPClientResourceWrapper,
                        msg="katcp connection could not be established, investigate!!!")
                    reply, informs = self.katcp_req.sensor_value("synchronisation-epoch")
                    self.assertTrue(reply.reply_ok(),
                        "Failed to set Digitiser sync epoch via CAM interface.")
                    sync_time = float(informs[0].arguments[-1])
                    self.assertIsInstance(sync_time, float, msg="Issues with reading Sync epoch")
                    reply, informs = self.katcp_req.sync_epoch(sync_time)
                    self.assertTrue(reply.reply_ok(), msg="Failed to set digitiser sync epoch")
                    self.logger.info("Digitiser sync epoch set successfully")
                    SET_DSIM_EPOCH = self._dsim_set = True
                except Exception:
                    self.Error(errmsg, exc_info=True)


    # This needs proper testing
    def tearDown(self):
        try:
            self.katcp_req = None
            assert not self.receiver
        except AssertionError:
            self.logger.info("Cleaning up the receiver!!!!")
            add_cleanup(self.receiver.stop)
            self.receiver = None
            del self.receiver


    def set_instrument(self, acc_time=None, **kwargs):
        self.receiver = None
        acc_timeout = 60
        self.errmsg = None
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)
        self.addCleanup(init_dsim_sources, self.dhost)

        try:
            self.Step("Confirm running instrument, else start a new instrument")
            self.instrument = self.cam_sensors.get_value("instrument_state").split("_")[0]
            self.Progress(
                "Currently running instrument %s-%s as per /etc/corr" % (
                    self.corr_fix.array_name,
                    self.instrument))
            self._systems_tests()
        except Exception:
            errmsg = "No running instrument on array: %s, Exiting...." % self.corr_fix.array_name
            self.Error(errmsg, exc_info=True)
            Aqf.end(message=errmsg)
            sys.exit(errmsg)

        if self._dsim_set:
            self.Step("Configure a digitiser simulator to be used as input source to F-Engines.")
            self.Progress("Digitiser Simulator running on host: %s" % self.dhost.host)

        try:
            n_ants = int(self.cam_sensors.get_value("n_ants"))
            n_chans = int(self.cam_sensors.get_value("n_chans"))
            # This logic can be improved
            if acc_time:
                pass
            elif n_ants == 4:
                acc_time = 0.5
            else:
                acc_time = n_ants / 32.0
            reply, informs = self.katcp_req.accumulation_length(acc_time, timeout=acc_timeout)
            self.assertTrue(reply.reply_ok())
            acc_time = float(reply.arguments[-1])
            self.Step("Set and confirm accumulation period via CAM interface.")
            self.Progress("Accumulation time set to {:.3f} seconds".format(acc_time))
        except Exception:
            self.Error("Failed to set accumulation time.", exc_info=True)

        try:
            self._output_product = self.conf_file["instrument_params"]["output_product"]
            data_output_ip, data_output_port = self.cam_sensors.get_value(
                self._output_product.replace("-", "_") + "_destination"
            ).split(":")
            self.Step(
                "Starting SPEAD receiver listening on %s:%s, CBF output product: %s"
                % (data_output_ip, data_output_port, self._output_product)
            )
            katcp_ip = self.corr_fix.katcp_client
            katcp_port = int(self.corr_fix.katcp_rct.port)
            self.Step("Connected to katcp on %s" % katcp_ip)
            # ToDo maybe select stop channels depending on the no of ants
            # This logic can be improved
            start_channels = int(self.conf_file["instrument_params"].get("start_channels", 0))
            if n_ants == 64 and n_chans == 4096:
                stop_channels = 2047
            elif n_chans == 1024:
                stop_channels = 1023
            else:
                stop_channels = int(self.conf_file["instrument_params"].get("stop_channels", 2047))
            self.Step(
                "Starting receiver on port %s, will only capture channels between %s-%s"
                % (data_output_port, start_channels, stop_channels)
            )
            self.Note(
                "Configuring SPEAD receiver to capture %s channels from %s to %s."
                % (stop_channels - start_channels + 1, start_channels, stop_channels)
            )
            self.receiver = CorrRx(
                product_name=self._output_product,
                katcp_ip=katcp_ip,
                katcp_port=katcp_port,
                port=data_output_port,
                channels=(start_channels, stop_channels),
            )
            self.receiver.setName("CorrRx Thread")
            self.errmsg = "Failed to create SPEAD data receiver"
            self.assertIsInstance(self.receiver, CorrRx), self.errmsg
            start_thread_with_cleanup(self, self.receiver, timeout=10, start_timeout=1)
            self.errmsg = "Spead Receiver not Running, possible "
            self.assertTrue(self.receiver.isAlive(), msg=self.errmsg)
            self.corr_fix.start_x_data
            self.logger.info(
                "Getting a test dump to confirm number of channels else, test fails "
                "if cannot retrieve dump"
            )
            _test_dump = self.receiver.get_clean_dump()
            self.errmsg = "Getting empty dumps!!!!"
            self.assertIsInstance(_test_dump, dict, self.errmsg)
            self.n_chans_selected = int(_test_dump.get("n_chans_selected",
                self.cam_sensors.get_value("n_chans"))
            )
            self.logger.info(
                "Confirmed number of channels %s, from initial dump" % self.n_chans_selected
            )
        except Exception:
            self.Error(self.errmsg, exc_info=True)
            return False
        else:
            # Run system tests before each test is ran
            self.addCleanup(self._systems_tests)
            self.addCleanup(self.corr_fix.stop_x_data)
            self.addCleanup(self.receiver.stop)
            self.addCleanup(executed_by)
            self.addCleanup(gc.collect)
            return True


    @array_release_x
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_course(self):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                n_chans = self.n_chans_selected
                test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
                heading("CBF Channelisation Wideband Coarse L-band")
                num_discards = 1
                if n_chans >= 2048:
                    self._test_channelisation(
                        test_chan, no_channels=n_chans,
                        req_chan_spacing=250e3, num_discards=num_discards
                    )
                else:
                    self._test_channelisation(
                        test_chan, no_channels=n_chans,
                        req_chan_spacing=1000e3, num_discards=num_discards
                    )
            else:
                self.Failed(self.errmsg)


    @instrument_32k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine(self):
        # Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success and self.cam_sensors.get_value("n_chans") >= 32768:
                n_chans = self.n_chans_selected
                test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
                heading("CBF Channelisation Wideband Fine L-band")
                self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @slow
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_course_sfdr_peaks(self):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                heading("CBF Channelisation Wideband Coarse SFDR L-band")
                n_ch_to_test = int(self.conf_file["instrument_params"].get("sfdr_ch_to_test",
                    self.n_chans_selected))
                self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=n_ch_to_test)  # Hz
            else:
                self.Failed(self.errmsg)


    @slow
    @instrument_32k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine_sfdr_peaks(self):
        # Aqf.procedure(TestProcedure.ChannelisationSFDR)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success and self.cam_sensors.get_value("n_chans") >= 32768:
                heading("CBF Channelisation Wideband Fine SFDR L-band")
                n_ch_to_test = int(self.conf_file["instrument_params"].get("sfdr_ch_to_test",
                    self.n_chans_selected))
                self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=n_ch_to_test)  # Hz
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.3.46")
    @aqf_requirements("CBF-REQ-0164", "CBF-REQ-0191")
    def test_power_consumption(self):
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            self.Step("Test is being qualified by CBF.V.3.30")


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.10")
    @aqf_requirements("CBF-REQ-0127")
    def test_lband_efficiency(self):
        Aqf.procedure(TestProcedure.LBandEfficiency)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_efficiency()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.A.IF")
    @aqf_requirements("TBD")
    def test_linearity(self):
        Aqf.procedure(TestProcedure.Linearity)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_linearity(
                    test_channel=100, cw_start_scale=1, noise_scale=0.001,
                    gain="10+j", fft_shift=8191, max_steps=20
                )
            else:
                self.Failed(self.errmsg)

    @array_release_x
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.34")
    @aqf_requirements("CBF-REQ-0094", "CBF-REQ-0117", "CBF-REQ-0118", "CBF-REQ-0123", "CBF-REQ-0183")
    def test_beamforming(self):
        Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_beamforming()
            else:
                self.Failed(self.errmsg)

    @array_release_x
    # Test still under development, Alec will put it under test_informal
    @instrument_1k
    @instrument_4k
    def test_beamforming_timeseries(self):
        # Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_beamforming_timeseries()
            else:
                self.Failed(self.errmsg)


    @wipd  # Test still under development, Alec will put it under test_informal
    @instrument_1k
    @instrument_4k
    def test_group_delay(self):
        # Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_group_delay()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.4")
    @aqf_requirements("CBF-REQ-0087", "CBF-REQ-0225", "CBF-REQ-0104")
    def test_baseline_correlation_product(self):
        Aqf.procedure(TestProcedure.BaselineCorrelation)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_product_baselines()
                self._test_back2back_consistency()
                self._test_freq_scan_consistency()
                self._test_spead_verify()
                self._test_product_baseline_leakage()
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.3.62")
    @aqf_requirements("CBF-REQ-0238")
    def test_imaging_data_product_set(self):
        Aqf.procedure(TestProcedure.ImagingDataProductSet)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_baseline=True)
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.3.67")
    @aqf_requirements("CBF-REQ-0120")
    def test_tied_array_aux_baseline_correlation_products(self):
        Aqf.procedure(TestProcedure.TiedArrayAuxBaselineCorrelationProducts)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_baseline=True, _tiedarray=True)
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.3.64")
    @aqf_requirements("CBF-REQ-0242")
    def test_tied_array_voltage_data_product_set(self):
        Aqf.procedure(TestProcedure.TiedArrayVoltageDataProductSet)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_data_product(_tiedarray=True)
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.7")
    @aqf_requirements("CBF-REQ-0096")
    def test_accumulation_length(self):
        Aqf.procedure(TestProcedure.VectorAcc)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                if "32k" in self.instrument:
                    self.Step(
                        "Testing maximum channels to %s due to quantiser snap-block and "
                        "system performance limitations." % self.n_chans_selected
                    )
                chan_index = self.n_chans_selected
                n_chans = self.cam_sensors.get_value("n_chans")
                test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
                self._test_vacc(
                    test_chan,
                    chan_index,
                    acc_time=(0.998
                        if self.cam_sensors.get_value("n_ants") == 4
                        else 2 * n_ants / 32.0))
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.9")
    @aqf_requirements("CBF-REQ-0119")
    def test_gain_correction(self):
        Aqf.procedure(TestProcedure.GainCorr)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_gain_correction()
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.4.23")
    @aqf_requirements("CBF-REQ-0013")
    def test_product_switch(self):
        Aqf.procedure(TestProcedure.ProductSwitching)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            self.Failed("This requirement is currently not being tested in this release.")
            # _running_inst = which_instrument(self, instrument)
            # instrument_success = self.set_instrument()
            # if instrument_success:
            #     with RunTestWithTimeout(300):
            #         self._test_product_switch(instrument)
            # else:
            #     self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.31")
    @aqf_requirements("CBF-REQ-0066", "CBF-REQ-0072", "CBF-REQ-0077", "CBF-REQ-0110", "CBF-REQ-0200")
    def test_delay_phase_compensation_control(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation_Control)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_delays_control()
                self.clear_all_delays()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.32")
    @aqf_requirements("CBF-REQ-0112", "CBF-REQ-0128", "CBF-REQ-0185", "CBF-REQ-0187", "CBF-REQ-0188")
    def test_delay_phase_compensation_functional(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument(acc_time=(0.5
                if self.cam_sensors.sensors.n_ants.get_value() == 4
                else int(self.conf_file["instrument_params"]["delay_test_acc_time"])))
            if instrument_success:
                self._test_delay_tracking()
                self._test_delay_rate()
                self._test_fringe_rate()
                self._test_fringe_offset()
                self._test_delay_inputs()
                self.clear_all_delays()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.27")
    @aqf_requirements("CBF-REQ-0178")
    def test_report_configuration(self):
        Aqf.procedure(TestProcedure.ReportConfiguration)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_report_config()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.29")
    @aqf_requirements("CBF-REQ-0067")
    def test_systematic_error_reporting(self):
        Aqf.procedure(TestProcedure.PFBFaultDetection)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_fft_overflow()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.28")
    @aqf_requirements("CBF-REQ-0157")
    def test_fault_detection(self):
        Aqf.procedure(TestProcedure.LinkFaultDetection)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                # self._test_network_link_error()
                # self._test_memory_error()
                heading("Processing Pipeline Failures")
                self.Note("Test is being qualified by CBF.V.3.29")
                heading("HMC Memory errors")
                self.Note("See waiver")
                heading("Network Link errors")
                self.Note("See waiver")
            else:
                self.Failed(self.errmsg)


    @generic_test
    @aqf_vr("CBF.V.3.26")
    @aqf_requirements("CBF-REQ-0056", "CBF-REQ-0068", "CBF-REQ-0069")
    def test_monitor_sensors(self):
        Aqf.procedure(TestProcedure.MonitorSensors)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._test_sensor_values()
                # self._test_host_sensors_status()
            else:
                self.Failed(self.errmsg)


    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.38")
    @aqf_requirements("CBF-REQ-0203")
    def test_time_synchronisation(self):
        Aqf.procedure(TestProcedure.TimeSync)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            self._test_time_sync()


    @generic_test
    @aqf_vr("CBF.V.4.26")
    @aqf_requirements("CBF-REQ-0083", "CBF-REQ-0084", "CBF-REQ-0085", "CBF-REQ-0086", "CBF-REQ-0221")
    def test_antenna_voltage_buffer(self):
        Aqf.procedure(TestProcedure.VoltageBuffer)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if instrument_success:
                self._small_voltage_buffer()
            else:
                self.Failed(self.errmsg)


    # ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------MANUAL TESTS-----------------------------------------
    # ---------------------------------------------------------------------------------------------------

    # Perhaps, enlist all manual tests here with VE & REQ

    @manual_test
    @aqf_vr("CBF.V.3.56")
    @aqf_requirements("CBF-REQ-0228")
    def test__subarray(self):
        self._test_global_manual("CBF.V.3.56")


    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.37")
    @aqf_requirements("CBF-REQ-0071", "CBF-REQ-0204")
    def test__control(self):
        self._test_global_manual("CBF.V.3.37")
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.37*"))
        caption_list = ["Screenshot of the command executed and reply: CAM interface"]
        Report_Images(image_files, caption_list)

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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.18*"))
        caption_list = [
            "Mellanox SX1710 switches and status LEDs visible from front of rack.",
            "Dell PowerEdge servers and status via front panel display visible.",
            "AP8981 PDUs have status LEDs visible from the back of the rack.",
        ]
        Report_Images(image_files, caption_list)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.19")
    @aqf_requirements("CBF-REQ-0182")
    def test__interchangeability_ve(self):
        self._test_global_manual("CBF.V.3.19")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.20")
    @aqf_requirements("CBF-REQ-0168", "CBF-REQ-0171")
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
    @aqf_requirements("CBF-REQ-0147", " CBF-REQ-0148")
    def test__item_handling_ve(self):
        self._test_global_manual("CBF.V.3.22")

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.23")
    @aqf_requirements("CBF-REQ-0152", "CBF-REQ-0153", "CBF-REQ-0154", "CBF-REQ-0155", "CBF-REQ-0184")
    def test__item_marking_and_labelling_ve(self):
        self._test_global_manual("CBF.V.3.23")
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.23*"))
        caption_list = [
            "Mellanox SX1710 - Supplier name and model number visible with switch installed in rack.",
            "Dell PowerEdge servers - Supplier name, model number and serial number visible with "
            "server installed in rack.",
            "SKARAB Processing nodes.",
            "All data switch port numbers are labelled.",
            "All internal CBF cables are labelled.",
            "All internal CBF cables are labelled.",
            "HMC Mezzanine SRUs are labelled as specified but supplier name is obscured by heatsink",
            "QSFP+ Mezzanine SRUs are labelled as specified",
            "HMC mezzanine supplier name is obscured by heatsink.",
        ]
        Report_Images(image_files, caption_list)

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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.25*"))
        caption_list = [
            "Screenshot of the command executed via CAM interface (log-level)"] * len(image_files)
        Report_Images(image_files, caption_list)

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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.39*"))
        caption_list = [
            "Rear doors of all CBF racks are perforated",
            "Front doors of all CBF racks are perforated"
            ]
        Report_Images(image_files, caption_list)

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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.44*"))
        caption_list = [
            "All equipment labels are still attached on {}".format(i.split("/")[-1].split(".jpg")[0])
            for i in image_files
        ]
        Report_Images(image_files, caption_list)

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
    @aqf_requirements("CBF-REQ-0179", "CBF-REQ-0180", "CBF-REQ-0190", " CBF-REQ-0194")
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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.54*"))
        caption_list = [
            "LRU replacement: {}".format(i.split("/")[-1].split(".jpg")[0])
            for i in image_files
        ]
        Report_Images(image_files, caption_list)

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
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.6.9*"))
        caption_list = [
            "Cables are bundled separately but the separation distance is not more than "
            "500mm due to space constraints in the racks."
        ] * len(image_files)
        Report_Images(image_files, caption_list)

    @manual_test
    @generic_test
    @aqf_vr("CBF.V.6.10")
    @aqf_requirements("CBF-REQ-0139")
    def test__design_standards_ve(self):
        self._test_global_manual("CBF.V.6.10")
        image_files = sorted(glob.glob(self._images_dir + "/CBF.V.6.10*"))
        caption_list = ["CBF processing nodes contains an integrated power filter."]
        Report_Images(image_files, caption_list)

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

    # ----------------------------------------------NOT TESTED-----------------------------------------
    # ---------------------------------------------------------------------------------------------------

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

    @array_release_x
    @generic_test
    @manual_test
    @aqf_vr("CBF.V.A.IF")
    def test__informal(self):
        Aqf.procedure(
            "This verification event pertains to tests that are executed, "
            "but do not verify any formal requirements."
            "The procedures and results shall be available in the Qualification Test Report."
        )
        # self._test_informal()

    # -----------------------------------------------------------------------------------------------------

    #################################################################
    #                       Test Methods                            #
    #################################################################

    def _test_channelisation(self, test_chan=1500, no_channels=None, req_chan_spacing=None, num_discards=3):
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

        if "1k" in self.instrument:
            cw_scale = 0.9
            awgn_scale = 0.085
            gain = "7+0j"
            fft_shift = 8191
            test_chan = 596
        elif "4k" in self.instrument:
            cw_scale = 0.9
            awgn_scale = 0.085
            gain = "7+0j"
            fft_shift = 8191
            # cw_scale = 0.9
            # awgn_scale = 0.1
            # gain = '7+0j'
            # fft_shift = 8191
            # cw_scale = 0.9
            # awgn_scale = 0.0
            # gain = '7+0j'
            # fft_shift = 8191
        elif "32k":
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = "11+0j"
            fft_shift = 32767
        else:
            msg = "Instrument not found: {}".format(self.instrument)
            self.logger.exception(msg)
            self.Failed(msg)

        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=101, chans_around=2)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        # Why is this necessary
        # http://library.nrao.edu/public/memos/ovlbi/OVLBI_038.pdf
        # https://www.prosoundtraining.com/2010/03/11/hand-in-hand-phase-and-group-delay/
        self.Note("Residual delay is excluded from this test.")
        self.Step(
            "Digitiser simulator configured to generate a continuous wave (cwg0), "
            "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                cw_scale, awgn_scale, gain, fft_shift
            )
        )
        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale, cw_scale=cw_scale, freq=expected_fc, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        try:
            self.Step(
                "Randomly select a frequency channel to test. Capture an initial correlator "
                "SPEAD accumulation, determine the number of frequency channels"
            )
            # initial_dump = self.receiver.get_clean_dump(discard=num_discards * 10)
            initial_dump = self.receiver.get_clean_dump(discard=5)

            self.assertIsInstance(initial_dump, dict)
            reply, informs = self.cam_sensors.req.sensor_value("{}-n-accs".format(self._output_product))
            assert reply.reply_ok()
            n_accs = int(informs[0].arguments[-1])
        except Exception:
            errmsg = "Could not retrieve initial clean SPEAD accumulation: Queue is Empty."
            self.Error(errmsg, exc_info=True)
            return

        else:
            bls_to_test = evaluate(self.cam_sensors.get_value("bls_ordering"))[test_baseline]
            self.Progress(
                "Randomly selected frequency channel to test: {} and "
                "selected baseline {} / {} to test.".format(test_chan, test_baseline, bls_to_test)
            )
            # Aqf.equals(4096, no_channels,
            #           'Confirm that the number of channels in the SPEAD accumulation, is equal '
            #           'to the number of frequency channels as calculated: {}'.format(
            #              no_channels))
            self.Step(
                "The CBF, when configured to produce the Imaging data product set and Wideband "
                "Fine resolution channelisation, shall channelise a total bandwidth of >= %s" % (
                    min_bandwithd_req)
            )
            Aqf.is_true(
                self.cam_sensors.get_value("bandwidth") >= min_bandwithd_req,
                "Channelise total bandwidth {}Hz shall be >= {}Hz.".format(
                    self.cam_sensors.get_value("bandwidth"), min_bandwithd_req
                ),
            )
            # TODO (MM) 2016-10-27, As per JM
            # Channel spacing is reported as 209.266kHz. This is probably spot-on, considering we're
            # using a dsim that's not actually sampling at 1712MHz. But this is problematic for the
            # test report. We would be getting 1712MHz/8192=208.984375kHz on site.
            # Maybe we should be reporting this as a fraction of total sampling rate rather than
            # an absolute value? ie 1/4096=2.44140625e-4 I will speak to TA about how to handle this.
            # chan_spacing = 856e6 / np.shape(initial_dump['xeng_raw'])[0]
            chan_spacing = self.cam_sensors.get_value("bandwidth") / self.cam_sensors.get_value("n_chans")
            chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100), chan_spacing + (chan_spacing * 1 / 100)]
            self.Step(
                "CBF-REQ-0043 and CBF-REQ-0053 Confirm that the number of calculated channel "
                "frequency step is within requirement."
            )
            msg = "Verify that the calculated channel frequency (%s Hz)step size is between %s and " "%s Hz" % (
                chan_spacing,
                req_chan_spacing / 2,
                req_chan_spacing,
            )
            Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

            self.Step(
                "CBF-REQ-0046 and CBF-REQ-0047 Confirm that the channelisation spacing and "
                "confirm that it is within the maximum tolerance."
            )
            msg = "Channelisation spacing is within maximum tolerance of 1% of the " "channel spacing."
            Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)
            initial_dump = self.receiver.get_clean_dump(discard=num_discards)
            initial_freq_response = normalised_magnitude(initial_dump["xeng_raw"][:, test_baseline, :])
            where_is_the_tone = np.argmax(initial_freq_response)
            max_tone_val = np.max(initial_freq_response)
            # 1) I think the channelisation tests might still be saturating.
            # Could you include a dBFS peak value in the output?
            # (take the peak auto correlation output value and divide it by the number of accumulations;
            # you should get a value in the range 0-16129).
            value = np.max(magnetise(initial_dump["xeng_raw"][:, test_baseline, :]))
            valuedBFS = 20*np.log10(abs(value)/n_accs)


            self.Note(
                "Single peak found at channel %s, with max power of %.5f(%.5fdB)"
                % (where_is_the_tone, max_tone_val, 10 * np.log10(max_tone_val))
            )


            plt_filename = "{}/{}_overall_channel_resolution_Initial_capture.png".format(
                self.logs_path, self._testMethodName
            )
            plt_title = "Initial Overall frequency response at %s" % test_chan
            caption = (
                "An overall frequency response at the centre frequency %s,"
                "and selected baseline %s to test. Digitiser simulator is configured to "
                "generate a continuous wave, with cw scale: %s, awgn scale: %s, Eq gain: %s "
                "and FFT shift: %s" % (test_chan, test_baseline, cw_scale, awgn_scale, gain, fft_shift)
            )
            aqf_plot_channels(initial_freq_response, plt_filename, plt_title, caption=caption, ylimits=(-100, 1))




        self.Step(
            "Sweep the digitiser simulator over the centre frequencies of at "
            "least all the channels that fall within the complete L-band"
        )
        failure_count = 0
        for i, freq in enumerate(requested_test_freqs):
            _msg = "Getting channel response for freq {} @ {}: {:.3f} MHz.".format(
                i + 1, len(requested_test_freqs), freq / 1e6
            )
            if i < print_counts:
                self.Progress(_msg)
            elif i == print_counts:
                self.Progress("." * print_counts)
            elif i >= (len(requested_test_freqs) - print_counts):
                self.Progress(_msg)
            else:
                self.logger.debug(_msg)

            self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
            # self.dhost.sine_sources.sin_1.set(frequency=freq, scale=cw_scale)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency

            if this_source_freq == last_source_freq:
                self.logger.debug(
                    "Skipping channel response for freq %s @ %s: %s MHz.\n"
                    "Digitiser frequency is same as previous." % (i + 1, len(requested_test_freqs), freq / 1e6)
                )
                continue  # Already calculated this one
            else:
                last_source_freq = this_source_freq

            try:
                this_freq_dump = self.receiver.get_clean_dump(discard=num_discards)
                self.assertIsInstance(this_freq_dump, dict)
            except AssertionError:
                failure_count += 1
                errmsg = "Could not retrieve clean accumulation for freq (%s @ %s: %sMHz)." % (
                    i + 1,
                    len(requested_test_freqs),
                    freq / 1e6,
                )
                self.Error(errmsg, exc_info=True)
                if failure_count >= 5:
                    _errmsg = "Cannot continue running the test, Not receiving clean accumulations."
                    self.Failed(_errmsg)
                    return False
            else:
                # No of spead heap discards relevant to vacc
                discards = 0
                max_wait_dumps = 50
                while True:
                    try:
                        queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                        self.assertIsInstance(queued_dump, dict)
                        deng_timestamp = float(self.dhost.registers.sys_clkcounter.read().get("timestamp"))
                        self.assertIsInstance(deng_timestamp, float)
                    except Exception:
                        errmsg = "Could not retrieve clean queued accumulation for freq(%s @ %s: " "%s MHz)." % (
                            i + 1,
                            len(requested_test_freqs),
                            freq / 1e6,
                        )
                        self.Error(errmsg, exc_info=True)
                        break
                    else:
                        timestamp_diff = np.abs(queued_dump["dump_timestamp"] - deng_timestamp)
                        # print colored(timestamp_diff, 'red')
                        if timestamp_diff < 2:
                            msg = (
                                "Received correct accumulation timestamp: %s, relevant to "
                                "DEngine timestamp: %s (Difference %.2f)"
                                % (queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                            )
                            self.logger.info(_msg)
                            self.logger.info(msg)
                            break

                        if discards > max_wait_dumps:
                            errmsg = (
                                "Could not get accumulation with correct timestamp within %s "
                                "accumulation periods." % max_wait_dumps
                            )
                            self.Failed(errmsg)
                            if discards > 10:
                                return
                            break
                        else:
                            msg = (
                                "Discarding subsequent dumps (%s) with dump timestamp (%s) "
                                "and DEngine timestamp (%s) with difference of %s."
                                % (discards, queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                            )
                            self.logger.info(msg)
                        deng_timestamp = None
                    discards += 1

                this_freq_data = queued_dump["xeng_raw"]
                this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                # print("{} {} ".format(np.max(this_freq_response), np.argmax(this_freq_response)))
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            # Plot an overall frequency response at the centre frequency just as
            # a sanity check

            if np.abs(freq - expected_fc) < 0.1:
                plt_filename = "{}/{}_overall_channel_resolution.png".format(self.logs_path, self._testMethodName)
                plt_title = "Overall frequency response at {} at {:.3f}MHz.".format(test_chan, this_source_freq / 1e6)
                max_peak = np.max(loggerise(this_freq_response))
                self.Note(
                    "Single peak found at channel %s, with max power of %s (%fdB) midway "
                    "channelisation, to confirm if there is no offset."
                    % (np.argmax(this_freq_response), np.max(this_freq_response), max_peak)
                )
                new_cutoff = max_peak - cutoff
                y_axis_limits = (-100, 1)
                caption = (
                    "An overall frequency response at the centre frequency, and ({:.3f}dB) "
                    "and selected baseline {} / {} to test. CBF channel isolation [max channel"
                    " peak ({:.3f}dB) - ({}dB) cut-off] when "
                    "digitiser simulator is configured to generate a continuous wave, with "
                    "cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}".format(
                        new_cutoff, test_baseline, bls_to_test, max_peak, cutoff, cw_scale, awgn_scale, gain, fft_shift
                    )
                )
                aqf_plot_channels(
                    this_freq_response,
                    plt_filename,
                    plt_title,
                    caption=caption,
                    ylimits=y_axis_limits,
                    cutoff=new_cutoff,
                )

        if not where_is_the_tone == test_chan:
            self.Note(
                "We expect the channel response at %s, but in essence it is in channel %s, ie "
                "There's a channel offset of %s" % (test_chan, where_is_the_tone, np.abs(test_chan - where_is_the_tone))
            )
            test_chan += np.abs(test_chan - where_is_the_tone)

        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        df = self.cam_sensors.delta_f
        try:
            rand_chan_response = len(chan_responses[random.randrange(len(chan_responses))])
            # assert rand_chan_response == self.n_chans_selected
        except AssertionError:
            errmsg = (
                "Number of channels (%s) found on the spead data is inconsistent with the "
                "number of channels (%s) expected." % (rand_chan_response, self.n_chans_selected)
            )
            self.Error(errmsg, exc_info=True)
        else:
            csv_filename = "/".join([self._katreport_dir, r"CBF_Efficiency_Data.csv"])
            np.savetxt(csv_filename, zip(chan_responses[:, test_chan], requested_test_freqs), delimiter=",")
            plt_filename = "{}/{}_Channel_Response.png".format(self.logs_path, self._testMethodName)
            plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90, normalise=True, no_clip=True)
            plt_caption = (
                "Frequency channel {} @ {}MHz response vs source frequency and "
                "selected baseline {} / {} to test.".format(test_chan, expected_fc / 1e6, test_baseline, bls_to_test)
            )
            plt_title = "Channel {} @ {:.3f}MHz response.".format(test_chan, expected_fc / 1e6)
            # Plot channel response with -53dB cutoff horizontal line
            aqf_plot_and_save(
                freqs=actual_test_freqs[1:-1],
                data=plot_data[1:-1],
                df=df,
                expected_fc=expected_fc,
                plot_filename=plt_filename,
                plt_title=plt_title,
                caption=plt_caption,
                cutoff=-cutoff,
            )
            try:
                # CBF-REQ-0126
                pass_bw_min_max = np.argwhere((np.abs(plot_data) >= 3.0) & (np.abs(plot_data) <= 3.3))
                pass_bw = float(np.abs(actual_test_freqs[pass_bw_min_max[0]] - actual_test_freqs[pass_bw_min_max[-1]]))

                att_bw_min_max = [
                    np.argwhere(plot_data == i)[0][0]
                    for i in plot_data
                    if (abs(i) >= (cutoff - 1)) and (abs(i) <= (cutoff + 1))
                ]
                att_bw = actual_test_freqs[att_bw_min_max[-1]] - actual_test_freqs[att_bw_min_max[0]]

            except Exception:
                msg = (
                    "Could not compute if, CBF performs channelisation such that the 53dB "
                    "attenuation bandwidth is less/equal to 2x the pass bandwidth"
                )
                self.Error(msg, exc_info=True)
            else:
                msg = (
                    "The CBF shall perform channelisation such that the 53dB attenuation bandwidth(%s)"
                    "is less/equal to 2x the pass bandwidth(%s)" % (att_bw, pass_bw)
                )
                Aqf.is_true(att_bw >= pass_bw, msg)

            # Get responses for central 80% of channel
            df = self.cam_sensors.delta_f
            central_indices = (actual_test_freqs <= expected_fc + 0.4 * df) & (
                actual_test_freqs >= expected_fc - 0.4 * df
            )
            central_chan_responses = chan_responses[central_indices]
            central_chan_test_freqs = actual_test_freqs[central_indices]

            # Plot channel response for central 80% of channel
            graph_name_central = "{}/{}_central.png".format(self.logs_path, self._testMethodName)
            plot_data_central = loggerise(
                central_chan_responses[:, test_chan], dynamic_range=90, normalise=True, no_clip=True
            )

            n_chans = self.n_chans_selected
            caption = (
                "Channel {} central response vs source frequency on max channels {} and "
                "selected baseline {} / {} to test.".format(test_chan, n_chans, test_baseline, bls_to_test)
            )
            plt_title = "Channel {} @ {:.3f} MHz response @ 80%".format(test_chan, expected_fc / 1e6)

            aqf_plot_and_save(
                central_chan_test_freqs,
                plot_data_central,
                df,
                expected_fc,
                graph_name_central,
                plt_title,
                caption=caption,
            )

            self.Step(
                "Test that the peak channeliser response to input frequencies in central 80% of "
                "the test channel frequency band are all in the test channel"
            )
            fault_freqs = []
            fault_channels = []
            for i, freq in enumerate(central_chan_test_freqs):
                max_chan = np.argmax(np.abs(central_chan_responses[i]))
                if max_chan != test_chan:
                    fault_freqs.append(freq)
                    fault_channels.append(max_chan)
            if fault_freqs:
                self.Failed(
                    "The following input frequencies (first and last): {!r} "
                    "respectively had peak channeliser responses in channels "
                    "{!r}\n, and not test channel {} as expected.".format(
                        fault_freqs[1::-1], set(sorted(fault_channels)), test_chan
                    )
                )

                self.logger.error(
                    "The following input frequencies: %s respectively had "
                    "peak channeliser responses in channels %s, not "
                    "channel %s as expected." % (fault_freqs, set(sorted(fault_channels)), test_chan)
                )

            Aqf.less(
                np.max(np.abs(central_chan_responses[:, test_chan])),
                0.99,
                "Confirm that the VACC output is at < 99% of maximum value, if fails "
                "then it is probably over-ranging.",
            )

            max_central_chan_response = np.max(10 * np.log10(central_chan_responses[:, test_chan]))
            min_central_chan_response = np.min(10 * np.log10(central_chan_responses[:, test_chan]))
            chan_ripple = max_central_chan_response - min_central_chan_response
            acceptable_ripple_lt = 1.5
            Aqf.hop(
                "80% channel cut-off ripple at {:.2f} dB, should be less than {} dB".format(
                    chan_ripple, acceptable_ripple_lt
                )
            )

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
            self.Step(
                "Confirm that the response at channel-edges are -3 dB "
                "relative to the channel centre at {:.3f} Hz, actual source freq "
                "{:.3f} Hz".format(expected_fc, fc_src_freq)
            )

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
            legends = [
                "Channel {} / Sample {} \n@ {:.3f} MHz".format(
                    ((test_chan + i) - 1), v, self.cam_sensors.ch_center_freqs[test_chan + i] / 1e6
                )
                for i, v in zip(range(no_of_responses), center_bin)
            ]
            # center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.n_chans_selected / 1e3))
            center_bin.append("Channel spacing: {:.3f}kHz".format(chan_spacing / 1e3))

            channel_response_list = [chan_responses[:, test_chan + i - 1] for i in range(no_of_responses)]
            np.savetxt("Boop.csv", channel_response_list, delimiter=",")
            plot_title = "PFB Channel Response"
            plot_filename = "{}/{}_adjacent_channels.png".format(self.logs_path, self._testMethodName)

            caption = (
                "Sample PFB central channel response between channel {} and selected baseline "
                "{}/{},with channelisation spacing of {:.3f}kHz within tolerance of 1%, with "
                "the digitiser simulator configured to generate a continuous wave, with cw scale:"
                " {}, awgn scale: {}, Eq gain: {} and FFT shift: {}".format(
                    test_chan, test_baseline, bls_to_test, chan_spacing / 1e3, cw_scale, awgn_scale, gain, fft_shift
                )
            )

            aqf_plot_channels(
                zip(channel_response_list, legends),
                plot_filename,
                plot_title,
                normalise=True,
                caption=caption,
                cutoff=-cutoff_edge,
                vlines=center_bin,
                xlabel="Sample Steps",
                ylimits=y_axis_limits,
            )

            self.Step(
                "Measure the power difference between the middle of the center and the middle of "
                "the next adjacent bins and confirm that is > -%sdB" % cutoff
            )
            for bin_num, chan_resp in enumerate(channel_response_list, 1):
                power_diff = np.max(loggerise(chan_resp)) - cutoff
                msg = "Confirm that the power difference (%.2fdB) in bin %s is more than %sdB" % (
                    power_diff,
                    bin_num,
                    -cutoff,
                )
                Aqf.less(power_diff, -cutoff, msg)

            # Plot Central PFB channel response with ylimit 0 to -6dB
            y_axis_limits = (-7, 1)
            plot_filename = "{}/{}_central_adjacent_channels.png".format(self.logs_path, self._testMethodName)
            plot_title = "PFB Central Channel Response"
            caption = (
                "Sample PFB central channel response between channel {} and selected baseline "
                "{}/{}, with the digitiser simulator configured to generate a continuous wave, "
                "with cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}".format(
                    test_chan, test_baseline, bls_to_test, cw_scale, awgn_scale, gain, fft_shift
                )
            )

            aqf_plot_channels(
                zip(channel_response_list, legends),
                plot_filename,
                plot_title,
                normalise=True,
                caption=caption,
                cutoff=-1.5,
                xlabel="Sample Steps",
                ylimits=y_axis_limits,
            )

            Aqf.is_true(
                low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
                "Confirm that the relative response at the low band-edge "
                "(-{co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq "
                "{co_low_src_freq}) is within the range of {desired_cutoff_resp} +- 1% "
                "relative to channel centre response.".format(**locals()),
            )

            Aqf.is_true(
                low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
                "Confirm that the relative response at the high band-edge "
                "(-{co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq "
                "{co_high_src_freq}) is within the range of {desired_cutoff_resp} +- 1% "
                "relative to channel centre response.".format(**locals()),
            )

    def _test_sfdr_peaks(self, required_chan_spacing, no_channels, cutoff=53, plots_debug=False, log_power=True):

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
                power_logger.setName("CBF Power Consumption")
                self.addCleanup(power_logger.stop)
            except Exception:
                errmsg = "Failed to start power usage logging."
                self.Error(errmsg, exc_info=True)

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
        msg = (
            "This tests confirms that the correct channels have the peak response to each"
            " frequency and that no other channels have significant relative power, while logging "
            "the power usage of the CBF in the background."
        )
        self.Step(msg)
        if log_power:
            self.Progress("Logging power usage in the background.")

        if "4k" in self.instrument:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = "11+0j"
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = "11+0j"
            fft_shift = 32767

        self.Step(
            "Digitiser simulator configured to generate a continuous wave, "
            "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                cw_scale, awgn_scale, gain, fft_shift
            )
        )

        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale,
            cw_scale=cw_scale,
            freq=self.cam_sensors.get_value("bandwidth") / 2.0,
            fft_shift=fft_shift,
            gain=gain,
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        self.Step(
            "Capture an initial correlator SPEAD accumulation, determine the "
            "number of frequency channels.")
        try:
            initial_dump = self.receiver.get_clean_dump()
            self.assertIsInstance(initial_dump, dict)
        except AssertionError:
            self.Error("Could not retrieve clean SPEAD accumulation: Queue is Empty.", exc_info=True)
        else:
            # Aqf.equals(np.shape(initial_dump['xeng_raw'])[0], no_channels,
            #           'Captured an initial correlator SPEAD accumulation, '
            #           'determine the number of channels and processing bandwidth: '
            #           '{}Hz.'.format(self.cam_sensors.get_value('bandwidth')))
            # chan_spacing = (self.cam_sensors.get_value('bandwidth') / np.shape(initial_dump['xeng_raw'])[0])
            chan_spacing = self.cam_sensors.get_value("bandwidth") / self.cam_sensors.get_value("n_chans")
            # [CBF-REQ-0043]
            calc_channel = (required_chan_spacing / 2) <= chan_spacing <= required_chan_spacing
            self.Step("Confirm that the number of calculated channel frequency step is within requirement.")
            msg = ("Confirm that the calculated channel frequency step size is between {} and "
                   "{} Hz".format(required_chan_spacing / 2, required_chan_spacing))
            Aqf.is_true(calc_channel, msg)

        self.Step(
            "Sweep a digitiser simulator tone over the all channels that fall within the "
            "complete L-band.")
        channel_response_lst = []
        print_counts = 4

        start_chan = 1  # skip DC channel since dsim puts out zeros for freq=0
        failure_count = 0
        # if self.n_chans_selected != self.cam_sensors.get_value('n_chans'):
        #    _msg = 'Due to system performance the test will sweep a limited number (ie %s) of channels' % (
        #        self.n_chans_selected)
        #    self.Note(_msg)
        #    channel_freqs = self.cam_sensors.ch_center_freqs[start_chan:self.n_chans_selected]
        # else:
        channel_freqs = self.cam_sensors.ch_center_freqs[start_chan:no_channels]

        for channel, channel_f0 in enumerate(channel_freqs, start_chan):
            if channel < print_counts:
                self.Progress(
                    "Getting channel response for freq %s @ %s: %.3f MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )
            elif channel == print_counts:
                self.Progress("...")
            elif channel > (len(channel_freqs) - print_counts):
                self.Progress(
                    "Getting channel response for freq %s @ %s: %.3f MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )
            else:
                self.logger.info(
                    "Getting channel response for freq %s @ %s: %s MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)
            self.dhost.sine_sources.sin_1.set(frequency=0, scale=0)
            # self.dhost.sine_sources.sin_corr.set(frequency=0, scale=0)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            try:
                this_freq_dump = self.receiver.get_clean_dump()
                self.assertIsInstance(this_freq_dump, dict)
            except AssertionError:
                self.Error("Could not retrieve clean SPEAD accumulation", exc_info=True)
                if failure_count >= 5:
                    _errmsg = "Giving up the test, failed to capture accumulations after 5 tries."
                    self.Failed(_errmsg)
                    failure_count = 0
                    return False
                failure_count += 1
            else:
                this_freq_data = this_freq_dump["xeng_raw"]
                this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                # List of channels to be plotted
                chans_to_plot = (n_chans // 10, n_chans // 2, 9 * n_chans // 10)
                if channel in chans_to_plot:
                    channel_response_lst.append(this_freq_response)

                max_chan = np.argmax(this_freq_response)
                max_channels.append(max_chan)
                # Find responses that are more than -cutoff relative to max
                new_cutoff = np.max(loggerise(this_freq_response)) + cutoff
                unwanted_cutoff = this_freq_response[max_chan] / 10 ** (new_cutoff / 100.0)
                extra_responses = [
                    i
                    for i, resp in enumerate(loggerise(this_freq_response))
                    if i != max_chan and resp >= unwanted_cutoff
                ]

                plt_title = "Frequency response at {}".format(channel)
                plt_filename = "{}/{}_channel_{}_resp.png".format(self.logs_path,
                    self._testMethodName, channel)
                if extra_responses:
                    msg = "Weirdly found an extra responses on channel %s" % (channel)
                    self.Note(msg)
                    plt_title = "Extra responses found around {}".format(channel)
                    plt_filename = "{}_extra_responses.png".format(self._testMethodName)
                    plots_debug = True

                extra_peaks.append(extra_responses)
                if plots_debug:
                    plots_debug = False
                    new_cutoff = np.max(loggerise(this_freq_response)) - cutoff
                    aqf_plot_channels(
                        this_freq_response, plt_filename, plt_title, log_dynamic_range=90,
                        hlines=new_cutoff
                    )

        for channel, channel_resp in zip(chans_to_plot, channel_response_lst):
            plt_filename = "{}/{}_channel_{}_resp.png".format(self.logs_path, self._testMethodName, channel)
            test_freq_mega = channel_freqs[channel] / 1e6
            plt_title = "Frequency response at {} @ {:.3f} MHz".format(channel, test_freq_mega)
            caption = (
                "An overall frequency response at channel {} @ {:.3f}MHz, "
                "when digitiser simulator is configured to generate a continuous wave, "
                "with cw scale: {}. awgn scale: {}, eq gain: {}, fft shift: {}".format(
                    channel, test_freq_mega, cw_scale, awgn_scale, gain, fft_shift
                )
            )

            new_cutoff = np.max(loggerise(channel_resp)) - cutoff
            aqf_plot_channels(
                channel_resp, plt_filename, plt_title, log_dynamic_range=90, caption=caption, hlines=new_cutoff
            )

        channel_range = range(start_chan, len(max_channels) + start_chan)
        self.Step("Check that the correct channels have the peak response to each frequency")
        msg = (
            "Confirm that the correct channel(s) (eg expected channel %s vs actual channel %s) "
            "have the peak response to each frequency" % (channel_range[1], max_channels[1])
        )

        if max_channels == channel_range:
            self.Passed(msg)
        else:
            Aqf.array_almost_equal(max_channels[1:], channel_range[1:], msg)

        msg = "Confirm that no other channels response more than -%s dB.\n" % cutoff
        if extra_peaks == [[]] * len(max_channels):
            self.Passed(msg)
        else:
            self.logger.debug("Expected: %s\n\nGot: %s" % (extra_peaks, [[]] * len(max_channels)))
            self.Failed(msg)
        if power_logger:
            power_logger.stop()
            start_timestamp = power_logger.start_timestamp
            power_log_file = power_logger.log_file_name
            power_logger.join()
            try:
                heading("CBF Power Consumption")
                self._process_power_log(start_timestamp, power_log_file)
            except Exception:
                self.Error("Failed to read/decode the PDU log.", exc_info=True)

    def _test_spead_verify(self):
        """This test verifies if a cw tone is only applied to a single input 0,
            figure out if VACC is rooted by 1
        """
        heading("SPEAD Accumulation Verification")
        cw_scale = 0.035
        freq = 300e6
        self.Step(
            "Digitiser simulator configured to generate cw tone with frequency: {}MHz, "
            "scale:{} on input 0".format(freq / 1e6, cw_scale)
        )
        init_dsim_sources(self.dhost)
        self.logger.info("Set cw tone on pole 0")
        self.dhost.sine_sources.sin_0.set(scale=cw_scale, frequency=freq)
        try:
            self.Step("Capture a correlator SPEAD accumulation and, ")
            dump = self.receiver.get_clean_dump(discard=50)
            self.assertIsInstance(dump, dict)
        except AssertionError:
            self.Error("Could not retrieve clean SPEAD accumulation, as Queue is Empty.",
                exc_info=True)
        else:
            vacc_offset = get_vacc_offset(dump["xeng_raw"])
            msg = (
                "Confirm that the auto-correlation in baseline 0 contains Non-Zeros, "
                "and baseline 1 is Zeros, when cw tone is only outputted on input 0."
            )
            Aqf.equals(vacc_offset, 0, msg)

            # TODO Plot baseline
            self.Step("Digitiser simulator reset to Zeros, before next test")
            self.Step(
                "Digitiser simulator configured to generate cw tone with frequency: {}Mhz, "
                "scale:{} on input 1".format(freq / 1e6, cw_scale)
            )
            init_dsim_sources(self.dhost)
            self.logger.info("Set cw tone on pole 1")
            self.dhost.sine_sources.sin_1.set(scale=cw_scale, frequency=freq)
            self.Step("Capture a correlator SPEAD accumulation and,")
            dump = self.receiver.get_clean_dump(discard=50)
            vacc_offset = get_vacc_offset(dump["xeng_raw"])
            msg = (
                "Confirm that the auto-correlation in baseline 1 contains non-Zeros, "
                "and baseline 0 is Zeros, when cw tone is only outputted on input 1."
            )
            Aqf.equals(vacc_offset, 1, msg)
            init_dsim_sources(self.dhost)

    def _test_product_baselines(self):
        heading("CBF Baseline Correlation Products")
        if "4k" in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = "113+0j"
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = "344+0j"
            fft_shift = 4095

        self.Step(
            "Digitiser simulator configured to generate Gaussian noise, "
            "with scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
        )

        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale,
            freq=self.cam_sensors.get_value("bandwidth") / 2.0,
            fft_shift=fft_shift,
            gain=gain,
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        try:
            self.Step("Change CBF input labels and confirm via CAM interface.")
            reply_, _ = self.katcp_req.input_labels()
            self.assertTrue(reply_.reply_ok())
            ori_source_name = reply_.arguments[1:]
            self.Progress("Original source names: {}".format(", ".join(ori_source_name)))
        except Exception:
            self.Error("Failed to retrieve input labels via CAM interface", exc_info=True)
        try:
            local_src_names = self.cam_sensors.custom_input_labels
            reply, _ = self.katcp_req.input_labels(*local_src_names)
            self.assertTrue(reply.reply_ok())
        except Exception:
            self.Error("Could not retrieve new source names via CAM interface:\n %s" % (str(reply)))
        else:
            source_names = reply.arguments[1:]
            msg = "Source names changed to: {}".format(", ".join(source_names))
            self.Passed(msg)

        try:
            if self.cam_sensors.sensors.n_ants.value > 16:
                _discards = 60
            else:
                _discards = 30

            self.Step(
                "Capture an initial correlator SPEAD accumulation while discarding {} "
                "accumulations, and retrieve list of all the correlator input labels via "
                "Cam interface.".format(_discards)
            )
            test_dump = self.receiver.get_clean_dump(discard=_discards)
            # test_dump = self.receiver.get_clean_dump()
            self.assertIsInstance(test_dump, dict)
        except AssertionError:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
        else:
            # Get bls ordering from get dump
            self.Step(
                "Get list of all possible baselines (including redundant baselines) present "
                "in the correlator output from SPEAD accumulation"
            )

            bls_ordering = evaluate(self.cam_sensors.get_value("bls_ordering"))
            input_labels = sorted(self.cam_sensors.input_labels)
            inputs_to_plot = random.shuffle(input_labels)
            inputs_to_plot = input_labels[:8]
            baselines_lookup = self.get_baselines_lookup()
            present_baselines = sorted(baselines_lookup.keys())
            possible_baselines = set()
            [possible_baselines.add((li, lj)) for li in input_labels for lj in input_labels]

            test_bl = sorted(list(possible_baselines))
            self.Step(
                "Confirm that each baseline (or its reverse-order counterpart) is present in "
                "the correlator output"
            )

            baseline_is_present = {}
            for test_bl in possible_baselines:
                baseline_is_present[test_bl] = test_bl in present_baselines or test_bl[::-1] in present_baselines
            # Select some baselines to plot
            plot_baselines = (
                (input_labels[0], input_labels[0]),
                (input_labels[0], input_labels[1]),
                (input_labels[0], input_labels[2]),
                (input_labels[-1], input_labels[-1]),
                (input_labels[-1], input_labels[-2]),
            )
            plot_baseline_inds = []
            for bl in plot_baselines:
                if bl in baselines_lookup:
                    plot_baseline_inds.append(baselines_lookup[bl])
                else:
                    plot_baseline_inds.append(baselines_lookup[bl[::-1]])

            plot_baseline_legends = tuple(
                "{bl[0]}, {bl[1]}: {ind}".format(bl=bl, ind=ind)
                for bl, ind in zip(plot_baselines, plot_baseline_inds)
            )

            msg = "Confirm that all baselines are present in correlator output."
            Aqf.is_true(all(baseline_is_present.values()), msg)
            test_data = test_dump["xeng_raw"]
            self.Step(
                "Expect all baselines and all channels to be " "non-zero with Digitiser Simulator set to output AWGN."
            )
            msg = "Confirm that no baselines have all-zero visibilities."
            Aqf.is_false(zero_baselines(test_data), msg)
            msg = "Confirm that all baseline visibilities are non-zero across all channels"
            if not nonzero_baselines(test_data) == all_nonzero_baselines(test_data):
                self.Failed(msg)
            else:
                self.Passed(msg)
            self.Step("Save initial f-engine equalisations, and ensure they are " "restored at the end of the test")
            initial_equalisations = self.get_gain_all()
            self.Progress("Stored original F-engine equalisations.")

            def set_zero_gains():
                try:
                    for _input in input_labels:
                        reply, _ = self.katcp_req.gain(_input, 0)
                        self.assertTrue(reply.reply_ok())
                except Exception:
                    self.Error("Failed to set equalisations on all F-engines", exc_info=True)
                else:
                    self.Passed("All the inputs equalisations have been set to Zero.")

            def read_zero_gains():
                try:
                    for _input in input_labels:
                        reply, _ = self.katcp_req.gain(_input)
                        self.assertTrue(reply.reply_ok())
                        eq_values = reply.arguments[-1]
                        self.assertEqual(eq_values, "0j")
                except Exception:
                    self.Failed("Failed to retrieve gains/equalisations.")
                else:
                    msg = "Confirm that all the inputs equalisations have been set to 'Zero'."
                    Aqf.equals(eq_values, "0j", msg)

            self.Step("Set all inputs gains to 'Zero', and confirm that output product is all-zero")
            set_zero_gains()
            read_zero_gains()

            test_data = self.receiver.get_clean_dump(discard=_discards)

            Aqf.is_false(
                nonzero_baselines(test_data["xeng_raw"]),
                "Confirm that all baseline visibilities are 'Zero' after " "{} discards.\n".format(
                    _discards),
            )
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

            bls_msg = (
                "Iterate through input combinations, verifying for each that "
                "the correct output appears in the correct baseline product.\n"
            )
            self.Step(bls_msg)
            # dataFrame = pd.DataFrame(index=sorted(input_labels),
            #                          columns=list(sorted(present_baselines)))

            for count, inp in enumerate(input_labels, start=1):
                if count > 10:
                    break
                old_eq = complex(initial_equalisations)
                self.Step(
                    "Iteratively set gain/equalisation correction on relevant " "input %s set to %s." % (inp, old_eq)
                )
                try:
                    reply, _ = self.katcp_req.gain(inp, old_eq)
                    self.assertTrue(reply.reply_ok())
                except AssertionError:
                    errmsg = "%s: Failed to set gain/eq of %s for input %s" % (str(reply), old_eq, inp)
                    self.Error(errmsg, exc_info=True)
                else:
                    msg = "Gain/Equalisation correction on input %s set to %s." % (inp, old_eq)
                    self.Passed(msg)
                    zero_inputs.remove(inp)
                    nonzero_inputs.add(inp)
                    expected_z_bls, expected_nz_bls = calc_zero_and_nonzero_baselines(nonzero_inputs)
                    try:
                        self.Step(
                            "Retrieving SPEAD accumulation and confirm if gain/equalisation "
                            "correction has been applied."
                        )
                        test_dump = self.receiver.get_clean_dump(discard=_discards)
                        # test_dump = self.receiver.get_clean_dump()
                        self.assertIsInstance(test_dump, dict)
                    except Exception:
                        errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        test_data = test_dump["xeng_raw"]
                        # plot baseline channel response
                        if inp in inputs_to_plot:
                            plot_data = [
                                normalised_magnitude(test_data[:, i, :])
                                # plot_data = [loggerise(test_data[:, i, :])
                                for i in plot_baseline_inds
                            ]
                            plot_filename = "{}/{}_channel_resp_{}.png".format(
                                self.logs_path, self._testMethodName.replace(" ", "_"), inp
                            )

                            plot_title = "Baseline Correlation Products on input: %s" % inp

                            _caption = (
                                "Baseline Correlation Products on input:{} {} with the "
                                "following non-zero inputs:\n {} \n "
                                "and\nzero inputs:\n {}".format(
                                    inp, bls_msg, ", ".join(sorted(nonzero_inputs)),
                                    ", ".join(sorted(zero_inputs))
                                )
                            )

                            aqf_plot_channels(
                                zip(plot_data, plot_baseline_legends),
                                plot_filename,
                                plot_title,
                                log_dynamic_range=None,
                                log_normalise_to=1,
                                caption=_caption,
                                ylimits=(-0.1, np.max(plot_data) + 0.1),
                            )
                        actual_nz_bls_indices = all_nonzero_baselines(test_data)
                        actual_nz_bls = set([tuple(bls_ordering[i]) for i in actual_nz_bls_indices])

                        actual_z_bls_indices = zero_baselines(test_data)
                        actual_z_bls = set([tuple(bls_ordering[i]) for i in actual_z_bls_indices])
                        msg = "Confirm that the expected baseline visibilities are non-zero with non-zero inputs"
                        self.Step(msg)
                        msg = msg + " (%s) and," % (sorted(nonzero_inputs))
                        if actual_nz_bls == expected_nz_bls:
                            self.Passed(msg)
                        else:
                            self.Failed(msg)

                        msg = "Confirm that the expected baselines visibilities are 'Zeros'.\n"
                        self.Step(msg)
                        if actual_z_bls == expected_z_bls:
                            self.Passed(msg)
                        else:
                            self.Failed(msg)

                        # Sum of all baselines powers expected to be non zeros
                        # sum_of_bl_powers = (
                        #     [normalised_magnitude(test_data[:, expected_bl, :])
                        #      for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                        #                          for expected_nz_bl_ind in sorted(expected_nz_bls)]])
                        test_data = None
                        # dataFrame.loc[inp][sorted(
                        #     [i for i in expected_nz_bls])[-1]] = np.sum(sum_of_bl_powers)

            # dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')

    def _test_back2back_consistency(self):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect.
        """
        heading("Spead Accumulation Back-to-Back Consistency")
        self.Step("Randomly select a channel to test.")
        n_chans = self.cam_sensors.get_value("n_chans")
        test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
        test_baseline = 0  # auto-corr
        self.Progress("Randomly selected test channel %s and bls %s" % (test_chan, test_baseline))
        self.Step("Calculate a list of frequencies to test")
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        source_period_in_samples = self.n_chans_selected * 2
        cw_scale = 0.675
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=cw_scale, repeat_n=source_period_in_samples)
        self.Step(
            "Digitiser simulator configured to generate periodic wave "
            "({:.3f}Hz with FFT-length {}) in order for each FFT to be "
            "identical.".format(expected_fc / 1e6, source_period_in_samples)
        )

        try:
            _discards = (20 if self.cam_sensors.sensors.n_ants.value > 16 else 10)
            this_freq_dump = self.receiver.get_clean_dump(discard=_discards)
            assert isinstance(this_freq_dump, dict)
        except AssertionError:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
            return False
        else:
            self.Step(
                "Sweep the digitiser simulator over the selected/requested frequencies " "within the complete L-band"
            )
            for i, freq in enumerate(requested_test_freqs):
                Aqf.hop(
                    "Getting channel response for freq {}/{} @ {:.3f} MHz.".format(
                        i + 1, len(requested_test_freqs), freq / 1e6
                    )
                )
                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale, repeat_n=source_period_in_samples)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency
                dumps_data = []
                chan_responses = []
                self.Step(
                    "Getting SPEAD accumulation and confirm that the difference between"
                    " subsequent accumulation is Zero."
                )
                for dump_no in range(3):
                    if dump_no == 0:
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(discard=_discards)
                            assert isinstance(this_freq_dump, dict)
                        except AssertionError:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                            return False
                        else:
                            initial_max_freq = np.max(this_freq_dump["xeng_raw"])
                    else:
                        try:
                            this_freq_dump = self.receiver.get_clean_dump(discard=_discards)
                            assert isinstance(this_freq_dump, dict)
                        except AssertionError:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)

                    try:
                        this_freq_data = this_freq_dump["xeng_raw"]
                        assert isinstance(this_freq_data, np.ndarray)
                    except AssertionError:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                        return False
                    dumps_data.append(this_freq_data)
                    this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)

                # Maximum difference between the initial max frequency and the last max freq
                dumps_comp = np.max(dumps_data[-1]) - initial_max_freq
                msg = (
                    "Confirm that the maximum difference between the subsequent SPEAD accumulations"
                    " with the same frequency input ({}Hz) is 'Zero' on baseline {}.".format(
                        this_source_freq, test_baseline
                    )
                )

                if not Aqf.equals(dumps_comp, 0, msg):
                    legends = ["dump #{}".format(x) for x in range(len(chan_responses))]
                    plot_filename = "{}/{}_chan_resp_{}.png".format(self.logs_path, self._testMethodName, i + 1)
                    plot_title = "Frequency Response {} @ {:.3f}MHz".format(test_chan, this_source_freq / 1e6)
                    caption = (
                        "Comparison of back-to-back SPEAD accumulations with digitiser simulator "
                        "configured to generate periodic wave ({:.3f}Hz with FFT-length {}) "
                        "in order for each FFT to be identical".format(this_source_freq, source_period_in_samples)
                    )
                    aqf_plot_channels(
                        zip(chan_responses, legends),
                        plot_filename,
                        plot_title,
                        log_dynamic_range=90,
                        log_normalise_to=1,
                        normalise=False,
                        caption=caption,
                    )

    def _test_freq_scan_consistency(self, threshold=1e-1):
        """This test confirms if the identical frequency scans produce equal results."""
        heading("Spead Accumulation Frequency Consistency")
        self.Step("Randomly select a channel to test.")
        n_chans = self.cam_sensors.get_value("n_chans")
        test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        self.Step(
            "Randomly selected Frequency channel {} @ {:.3f}MHz for testing, and calculate a "
            "list of frequencies to test".format(test_chan, expected_fc / 1e6)
        )
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=3, chans_around=1)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        source_period_in_samples = self.n_chans_selected * 2

        try:
            test_dump = self.receiver.get_clean_dump(discard=50)
            assert isinstance(test_dump, dict)
        except Exception:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
            return
        else:
            cw_scale = 0.675
            self.Step("Digitiser simulator configured to generate continuous wave")
            self.Step(
                "Sweeping the digitiser simulator over the centre frequencies of at "
                "least all channels that fall within the complete L-band: {} Hz".format(expected_fc)
            )
            for scan_i in range(3):
                scan_dumps = []
                frequencies = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        Aqf.hop(
                            "Getting channel response for freq {} @ {}: {} MHz.".format(
                                i + 1, len(requested_test_freqs), freq / 1e6
                            )
                        )
                        self.dhost.sine_sources.sin_0.set(
                            frequency=freq, scale=cw_scale, repeat_n=source_period_in_samples
                        )
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            # this_freq_dump = self.receiver.get_clean_dump()
                            this_freq_dump = self.receiver.get_clean_dump(discard=20)
                            assert isinstance(this_freq_dump, dict)
                        except Exception:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                        else:
                            initial_max_freq = np.max(this_freq_dump["xeng_raw"])
                            this_freq_data = this_freq_dump["xeng_raw"]
                            initial_max_freq_list.append(initial_max_freq)
                    else:
                        self.dhost.sine_sources.sin_0.set(
                            frequency=freq, scale=cw_scale, repeat_n=source_period_in_samples
                        )
                        freq_val = self.dhost.sine_sources.sin_0.frequency
                        try:
                            # this_freq_dump = self.receiver.get_clean_dump()
                            this_freq_dump = self.receiver.get_clean_dump(discard=20)
                            assert isinstance(this_freq_dump, dict)
                        except Exception:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                        else:
                            this_freq_data = this_freq_dump["xeng_raw"]

                    this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    chan_responses.append(this_freq_response)
                    scan_dumps.append(this_freq_data)
                    frequencies.append(freq_val)

            for scan_i in range(1, len(scans)):
                for freq_i, freq_x in zip(range(len(scans[0])), frequencies):
                    s0 = scans[0][freq_i]
                    s1 = scans[scan_i][freq_i]
                    norm_fac = initial_max_freq_list[freq_i]
                    # TODO Convert to a less-verbose comparison for Aqf.
                    # E.g. test all the frequencies and only save the error cases,
                    # then have a final Aqf-check so that there is only one step
                    # (not n_chan) in the report.
                    max_freq_scan = np.max(np.abs(s1 - s0)) / norm_fac

                    msg = (
                        "Confirm that identical frequency ({:.3f} MHz) scans between subsequent "
                        "SPEAD accumulations produce equal results.".format(freq_x / 1e6)
                    )

                    if not Aqf.less(np.abs(max_freq_scan), np.abs(np.log10(threshold)), msg):
                        legends = ["Freq scan #{}".format(x) for x in range(len(chan_responses))]
                        caption = (
                            "A comparison of frequency sweeping from {:.3f}Mhz to {:.3f}Mhz "
                            "scan channelisation and also, {}".format(
                                requested_test_freqs[0] / 1e6, requested_test_freqs[-1] / 1e6, expected_fc, msg
                            )
                        )

                        aqf_plot_channels(
                            zip(chan_responses, legends),
                            plot_filename="{}/{}_chan_resp.png".format(self.logs_path, self._testMethodName),
                            caption=caption,
                        )

    def _test_restart_consistency(self, instrument, no_channels):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect on CBF restart.
        """
        self.Step(self._testMethodDoc)
        threshold = 1.0e1  #
        test_baseline = 0
        n_chans = self.cam_sensors.get_value("n_chans")
        test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
        requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        self.Step(
            "Sweeping the digitiser simulator over {:.3f}MHz of the channels that "
            "fall within {} complete L-band".format(np.max(requested_test_freqs) / 1e6, test_chan)
        )

        if "4k" in self.instrument:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = "11+0j"
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = "11+0j"
            fft_shift = 32767

        self.Step(
            "Digitiser simulator configured to generate a continuous wave, "
            "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                cw_scale, awgn_scale, gain, fft_shift
            )
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
            freq=expected_fc, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        try:
            this_freq_dump = self.receiver.get_clean_dump()
        except Queue.Empty:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
        else:
            # Plot an overall frequency response at the centre frequency just as
            # a sanity check
            init_source_freq = normalised_magnitude(this_freq_dump["xeng_raw"][:, test_baseline, :])
            filename = "{}/{}_channel_response.png".format(self.logs_path, self._testMethodName)
            title = "Frequency response at {} @ {:.3f} MHz.\n".format(test_chan, expected_fc / 1e6)
            caption = "An overall frequency response at the centre frequency."
            aqf_plot_channels(init_source_freq, filename, title, caption=caption)
            restart_retries = 5

            def _restart_instrument(retries=restart_retries):
                if not self.corr_fix.stop_x_data():
                    self.Failed("Could not stop x data from capturing.")
                with ignored(Exception):
                    # deprogram_hosts(self)
                    self.Failed("Fix deprogram Hosts")

                corr_init = False
                _empty = True
                with ignored(Queue.Empty):
                    self.receiver.get_clean_dump()
                    _empty = False

                Aqf.is_true(_empty, "Confirm that the SPEAD accumulations have stopped being produced.")

                self.corr_fix.halt_array

                while retries and not corr_init:
                    self.Step("Re-initialising the {} instrument".format(instrument))
                    with ignored(Exception):
                        corr_init = self.set_instrument()

                    retries -= 1
                    if retries == 0:
                        errmsg = "Could not restart the correlator after %s tries." % (retries)
                        self.Error(errmsg, exc_info=True)

                if corr_init.keys()[0] is not True and retries == 0:
                    msg = "Could not restart {} after {} tries.".format(instrument, retries)
                    Aqf.end(passed=False, message=msg)
                else:
                    startx = self.corr_fix.start_x_data
                    if not startx:
                        self.Failed("Failed to enable/start output product capturing.")
                    host = (self.xhosts + self.fhosts)[random.randrange(len(self.xhosts + self.fhosts))]
                    msg = (
                        "Confirm that the instrument is initialised by checking if a "
                        "random host: {} is programmed and running.".format(host.host)
                    )
                    Aqf.is_true(host, msg)

                    try:
                        self.assertIsInstance(self.receiver, CorrRx)
                        freq_dump = self.receiver.get_clean_dump()
                        assert np.shape(freq_dump["xeng_raw"])[0] == self.n_chans_selected
                    except Queue.Empty:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                        return False
                    except AssertionError:
                        errmsg = (
                            "Correlator Receiver could not be instantiated or No of channels "
                            "(%s) in the spead data is inconsistent with the no of"
                            " channels (%s) expected" % (np.shape(freq_dump["xeng_raw"])[0], self.n_chans_selected)
                        )
                        self.Error(errmsg, exc_info=True)
                        return False
                    else:
                        msg = (
                            "Confirm that the data product has the same number of frequency "
                            "channels {no_channels} corresponding to the {instrument} "
                            "instrument product".format(**locals())
                        )
                        try:
                            spead_chans = self.receiver.get_clean_dump()
                        except Queue.Empty:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                        else:
                            Aqf.equals(4096, no_channels, msg)
                            return True

            initial_max_freq_list = []
            scans = []
            channel_responses = []
            for scan_i in range(3):
                if scan_i:
                    self.Step("#{scan_i}: Initialising {instrument} instrument".format(**locals()))
                    intrument_success = _restart_instrument()
                    if not intrument_success:
                        msg = "Failed to restart the correlator successfully."
                        self.Failed(msg)
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
                            Aqf.hop(
                                "Getting Frequency SPEAD accumulation #{} with Digitiser simulator "
                                "configured to generate cw at {:.3f}MHz".format(i, freq / 1e6)
                            )
                            try:
                                this_freq_dump = self.receiver.get_clean_dump()
                            except Queue.Empty:
                                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                                self.Error(errmsg, exc_info=True)
                        initial_max_freq = np.max(this_freq_dump["xeng_raw"])
                        this_freq_data = this_freq_dump["xeng_raw"]
                        initial_max_freq_list.append(initial_max_freq)
                        freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    else:
                        msg = (
                            "Getting Frequency SPEAD accumulation #{} with digitiser simulator "
                            "configured to generate cw at {:.3f}MHz".format(i, freq / 1e6)
                        )
                        Aqf.hop(msg)
                        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=0.125)
                        try:
                            this_freq_dump = self.receiver.get_clean_dump()
                        except Queue.Empty:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                        else:
                            this_freq_data = this_freq_dump["xeng_raw"]
                            freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                    scan_dumps.append(this_freq_data)
                    channel_responses.append(freq_response)

            normalised_init_freq = np.array(initial_max_freq_list)
            for comp in range(1, len(normalised_init_freq)):
                v0 = np.array(normalised_init_freq[comp - 1])
                v1 = np.array(normalised_init_freq[comp])

            correct_init_freq = np.abs(np.max(v0 - v1))

            diff_scans_dumps = []
            for comparison in range(1, len(scans)):
                s0 = np.array(scans[comparison - 1])
                s1 = np.array(scans[comparison])
                diff_scans_dumps.append(np.max(s0 - s1))

            diff_scans_comp = np.max(np.array(diff_scans_dumps) / correct_init_freq)

            msg = (
                "Confirm that CBF restart SPEAD accumulations comparison results "
                "with the same frequency input differ by no more than {:.3f}dB "
                "threshold.".format(threshold)
            )

            if not Aqf.less(diff_scans_comp, threshold, msg):
                legends = ["Channel Response #{}".format(x) for x in range(len(channel_responses))]
                plot_filename = "{}/{}_chan_resp.png".format(self.logs_path, self._testMethodName)
                caption = "Confirm that results are consistent on CBF restart"
                plot_title = "CBF restart consistency channel response {}".format(test_chan)
                aqf_plot_channels(zip(channel_responses, legends), plot_filename, plot_title, caption=caption)

    def _test_delay_tracking(self):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Delay tracking"
        heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            num_int = setup_data["num_int"]
            int_time = self.cam_sensors.get_value("int_time")
            # katcp_port = self.cam_sensors.get_value('katcp_port')
            no_chans = range(self.n_chans_selected)
            sampling_period = self.cam_sensors.sample_period
            test_delays = [0, sampling_period, 1.5 * sampling_period, 1.9 * sampling_period]
            test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
            # num_inputs = len(self.cam_sensors.input_labels)
            delays = [0] * setup_data["num_inputs"]
            self.Step("Delays to be set (iteratively) %s for testing purposes\n" % (test_delays))

            def get_expected_phases():
                expected_phases = []
                for delay in test_delays:
                    # phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                    phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                    phases -= np.max(phases) / 2.0
                    expected_phases.append(phases)
                return zip(test_delays_ns, expected_phases)

            def get_actual_phases():
                actual_phases_list = []
                # chan_responses = []
                for count, delay in enumerate(test_delays, 1):
                    delays[setup_data["test_source_ind"]] = delay
                    delay_coefficients = ["{},0:0,0".format(dv) for dv in delays]
                    try:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        this_freq_dump = self.receiver.get_clean_dump(discard=0)
                        self.assertIsInstance(this_freq_dump, dict), errmsg
                        t_apply = this_freq_dump["dump_timestamp"] + (num_int * int_time)
                        t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
                        curr_time = time.time()
                        curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
                        self.Step("Delay #%s will be applied with the following parameters:" % count)
                        msg = (
                            "On baseline %s and input %s, Current cmc time: %s (%s)"
                            ", Current Dump timestamp: %s (%s), "
                            "Delay(s) will be applied @ %s (%s), Delay to be applied: %s"
                            % (
                                setup_data["baseline_index"],
                                setup_data["test_source"],
                                curr_time,
                                curr_time_readable,
                                this_freq_dump["dump_timestamp"],
                                this_freq_dump["dump_timestamp_readable"],
                                t_apply,
                                t_apply_readable,
                                delay,
                            )
                        )
                        self.Progress(msg)
                        self.Step(
                            "Execute delays via CAM interface and calculate the amount of time "
                            "it takes to load the delays"
                        )
                        self.logger.info("Setting a delay of %s via cam interface" % delay)
                        load_strt_time = time.time()
                        reply, _informs = self.katcp_req.delays("antenna-channelised-voltage", t_apply, *delay_coefficients)
                        load_done_time = time.time()
                        formated_reply = str(reply).replace("\_", " ")
                        errmsg = (
                            "CAM Reply: %s: Failed to set delays via CAM interface with "
                            "load-time: %s vs Current cmc time: %s" % (formated_reply, t_apply, time.time())
                        )
                        self.assertTrue(reply.reply_ok(), errmsg)
                        cmd_load_time = round(load_done_time - load_strt_time, 3)
                        self.Step("Delay load command took {} seconds".format(cmd_load_time))
                        # _give_up = int(num_int * int_time * 3)
                        # while True:
                        #    _give_up -= 1
                        #    try:
                        #        self.logger.info('Waiting for the delays to be updated on sensors: '
                        #                    '%s retry' % _give_up)
                        #        try:
                        #            reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                        #        except:
                        #            reply, informs = self.katcp_req.sensor_value()
                        # self.assertTrue(reply.reply_ok())
                        #    except Exception:
                        #        self.Error('Weirdly I couldnt get the sensor values', exc_info=True)
                        #    else:
                        #        delays_updated = list(set([int(i.arguments[-1]) for i in informs
                        #                                if '.cd.delay' in i.arguments[2]]))[0]
                        #        if delays_updated:
                        #            self.logger.info('%s delay(s) have been successfully set' % delay)
                        #            msg = ('Delays set successfully via CAM interface: Reply: %s' %
                        #                    formated_reply)
                        #            self.Passed(msg)
                        #            break
                        #    if _give_up == 0:
                        #        msg = ("Could not confirm the delays in the time stipulated, exiting")
                        #        self.logger.error(msg)
                        #        self.Failed(msg)
                        #        break
                        #    time.sleep(1)

                        # Tested elsewhere:
                        # cam_max_load_time = setup_data['cam_max_load_time']
                        # msg = ('Time it took to load delays {}s is less than {}s with an '
                        #      'integration time of {:.3f}s'
                        #      .format(cmd_load_time, cam_max_load_time, int_time))
                        # Aqf.less(cmd_load_time, cam_max_load_time, msg)
                    except Exception:
                        self.Error(errmsg, exc_info=True)

                    try:
                        _num_discards = num_int + 2
                        self.Step(
                            "Getting SPEAD accumulation(while discarding %s dumps) containing "
                            "the change in delay(s) on input: %s baseline: %s."
                            % (_num_discards, setup_data["test_source"], setup_data["baseline_index"])
                        )
                        self.logger.info("Getting dump...")
                        dump = self.receiver.get_clean_dump(discard=_num_discards)
                        self.logger.info("Done...")
                        assert isinstance(dump, dict)
                        self.Progress(
                            "Readable time stamp received on SPEAD accumulation: %s "
                            "after %s number of discards \n" % (dump["dump_timestamp_readable"], _num_discards)
                        )
                    except Exception:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        # # this_freq_data = this_freq_dump['xeng_raw']
                        # # this_freq_response = normalised_magnitude(
                        # #    this_freq_data[:, setup_data['test_source_ind'], :])
                        # # chan_responses.append(this_freq_response)
                        data = complexise(dump["xeng_raw"][:, setup_data["baseline_index"], :])
                        phases = np.angle(data)
                        # # actual_channel_responses = zip(test_delays, chan_responses)
                        # # return zip(actual_phases_list, actual_channel_responses)
                        actual_phases_list.append(phases)
                return actual_phases_list

            expected_phases = get_expected_phases()
            actual_phases = get_actual_phases()

            try:
                if set([float(0)]) in [set(i) for i in actual_phases[1:]] or not actual_phases:
                    self.Failed(
                        "Delays could not be applied at time_apply: {} "
                        "possibly in the past.\n".format(setup_data["t_apply"])
                    )
                else:
                    # actual_phases = [phases for phases, response in actual_data]
                    # actual_response = [response for phases, response in actual_data]
                    plot_title = "CBF Delay Compensation"
                    caption = (
                        "Actual and expected Unwrapped Correlation Phase [Delay tracking].\n"
                        "Note: Dashed line indicates expected value and solid line "
                        "indicates actual values received from SPEAD accumulation."
                    )
                    plot_filename = "{}/{}_test_delay_tracking.png".format(self.logs_path, self._testMethodName)
                    plot_units = "secs"

                    aqf_plot_phase_results(
                        no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption
                    )

                    nc_sel = self.n_chans_selected
                    expected_phases_ = [phase[:nc_sel] for _rads, phase in expected_phases]

                    degree = 1.0
                    decimal = len(str(degree).split(".")[-1])
                    try:
                        for i, delay in enumerate(test_delays):
                            delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
                            delta_expected = np.max(expected_phases_[i]) - np.min(expected_phases_[i])
                            abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                            # abs_diff = np.abs(delta_expected - delta_actual)
                            msg = (
                                "Confirm that if difference expected({:.5f}) "
                                "and actual({:.5f}) phases are equal at delay {:.5f}ns within "
                                "{} degree.".format(delta_expected, delta_actual, delay * 1e9, degree)
                            )
                            Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                            Aqf.less(
                                abs_diff,
                                degree,
                                "Confirm that the maximum difference ({:.3f} degree/"
                                " {:.3f} rad) between expected phase and actual phase between "
                                "integrations is less than {} degree.\n".format(abs_diff, np.deg2rad(abs_diff), degree),
                            )
                            try:
                                delta_actual_s = delta_actual - (delta_actual % degree)
                                delta_expected_s = delta_expected - (delta_expected % degree)
                                np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
                            except AssertionError:
                                msg = (
                                    "Difference expected({:.5f}) phases"
                                    " and actual({:.5f}) phases are 'Not almost equal' "
                                    "within {} degree when delay of {}ns is applied.".format(
                                        delta_expected, delta_actual, degree, delay * 1e9
                                    )
                                )
                                self.Step(msg)

                                caption = (
                                    "The figure above shows, The difference between expected({:.5f}) "
                                    "phases and actual({:.5f}) phases are 'Not almost equal' within {} "
                                    "degree when a delay of {:.5f}s is applied. Therefore CBF-REQ-0128 and"
                                    ", CBF-REQ-0187 are not verified.".format(
                                        delta_expected, delta_actual, degree, delay
                                    )
                                )

                                actual_phases_i = (delta_actual, actual_phases[i])
                                if len(expected_phases[i]) == 2:
                                    expected_phases_i = (delta_expected, expected_phases[i][-1])
                                else:
                                    expected_phases_i = (delta_expected, expected_phases[i])
                                aqf_plot_phase_results(
                                    no_chans,
                                    actual_phases_i,
                                    expected_phases_i,
                                    plot_filename="{}/{}_{}_delay_tracking.png".format(
                                        self.logs_path, self._testMethodName, i
                                    ),
                                    plot_title=("Delay offset:\n" "Actual vs Expected Phase Response"),
                                    plot_units=plot_units,
                                    caption=caption,
                                )

                        for delay, count in zip(test_delays, range(1, len(expected_phases))):
                            msg = (
                                "Confirm that when a delay of {} clock "
                                "cycle({:.5f} ns) is introduced there is a phase change "
                                "of {:.3f} degrees as expected to within {} degree.".format(
                                    (count + 1) * 0.5, delay * 1e9, np.rad2deg(np.pi) * (count + 1) * 0.5, degree
                                )
                            )
                            try:
                                Aqf.array_abs_error(
                                    actual_phases[count][5:-5], expected_phases_[count][5:-5], msg, degree
                                )
                            except Exception:
                                Aqf.array_abs_error(
                                    actual_phases[count][5:-5],
                                    expected_phases_[count][5 : -5 + len(actual_phases[count])],
                                    msg,
                                    degree,
                                )
                    except Exception:
                        self.Error("Error occurred, this shouldnt happen", exc_info=True)
                        return
            except Exception:
                self.Error("Error occurred, this shouldnt happen", exc_info=True)
                return

    def _test_sensor_values(self):
        """
        Report sensor values
        """
        heading("Monitor Sensors: Report Sensor Values")

        def report_sensor_list(self):
            self.Step(
                "Confirm that the number of sensors available on the primary " "and sub array interface is consistent."
            )
            try:
                reply, informs = self.katcp_req.sensor_list(timeout=60)
            except BaseException:
                errmsg = "CAM interface connection encountered errors."
                self.Error(errmsg, exc_info=True)
            else:
                msg = (
                    "Confirm that the number of sensors are equal "
                    "to the number of sensors listed on the running instrument.\n"
                )
                Aqf.equals(int(reply.arguments[-1]), len(informs), msg)

        def report_time_sync(self):
            self.Step("Confirm that the time synchronous is implemented on primary interface")
            try:
                reply, informs = self.corr_fix.rct.req.sensor_value("time.synchronised")
            except BaseException:
                self.Failed("CBF report time sync could not be retrieved from primary interface.")
            else:
                Aqf.is_true(reply.reply_ok(), "CBF report time sync implemented in this release.")

            msg = "Confirm that the CBF can report time sync status via CAM interface. "
            try:
                reply, informs = self.corr_fix.rct.req.sensor_value("time.synchronised")
            except BaseException:
                self.Error(msg, exc_info=True)
            else:
                msg = msg + "CAM Reply: {}\n".format(str(informs[0]))
                if not reply.reply_ok():
                    self.Failed(msg)
                self.Passed(msg)

        def report_small_buffer(self):
            self.Step("Confirm that the Transient Buffer ready is implemented.")
            try:
                assert self.katcp_req.transient_buffer_trigger.is_active()
            except Exception:
                errmsg = "CBF Transient buffer ready for triggering" "'Not' implemented in this release.\n"
                self.Error(errmsg, exc_info=True)
            else:
                self.Passed("CBF Transient buffer ready for triggering" " implemented in this release.\n")

        def report_primary_sensors(self):
            self.Step("Confirm that all primary sensors are nominal.")
            for sensor in self.corr_fix.rct.sensor.values():
                msg = "Primary sensor: {}, current status: {}".format(sensor.name, sensor.get_status())
                Aqf.equals(sensor.get_status(), "nominal", msg)

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
        heading("Systematic Errors Reporting: FFT Overflow")
        # TODO MM, Simplify the test
        ch_list = self.cam_sensors.ch_center_freqs
        cw_freq = ch_list[int(self.n_chans_selected / 2)]

        if "4k" in self.instrument:
            cw_scale = 0.7
            awgn_scale = 0.085
            gain = "7+0j"
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = "11+0j"
            fft_shift = 32767

        self.Step(
            "Digitiser simulator configured to generate a continuous wave (cwg0), "
            "with cw scale: {}, cw frequency: {}, awgn scale: {}, eq gain: {}, "
            "fft shift: {}".format(cw_scale, cw_freq, awgn_scale, gain, fft_shift)
        )

        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale, cw_scale=cw_scale, freq=cw_freq, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        try:
            self.Step("Get the current FFT Shift before manipulation.")
            fft_shift = int(self.get_fftshift_all())
            assert fft_shift
            self.Progress("Current system FFT Shift: %s" % fft_shift)
        except Exception:
            self.Error("Could not get the F-Engine FFT Shift value", exc_info=True)
            return

        try:
            self.Step("Confirm all F-engines do not contain PFB errors/warnings")
            for i in range(3):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=60)
                except BaseException:
                    reply, informs = self.katcp_req.sensor_value(timeout=60)
                self.assertTrue(reply.reply_ok())
        except Exception:
            msg = "Failed to retrieve sensor values via CAM interface"
            self.Error(msg, exc_info=True)
            return
        else:
            pfb_status = list(set([i.arguments[-2] for i in informs if "pfb.or0-err-cnt" in i.arguments[2]]))[0]
            Aqf.equals(pfb_status, "nominal", "Confirm that all F-Engines report nominal PFB status")

        try:
            self.Step("Set an FFT shift of 0 on all f-engines, and confirm if system integrity is affected")
            reply, informs = self.katcp_req.fft_shift(0)
            self.assertTrue(reply.reply_ok())
        except AssertionError:
            msg = "Could not set FFT shift for all F-Engine hosts"
            self.Error(msg, exc_info=True)
            return

        try:
            msg = "Waiting for sensors to trigger."
            Aqf.wait(60, msg)

            self.Step("Check if all F-engines contain(s) PFB errors/warnings")
            for i in range(3):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                except BaseException:
                    reply, informs = self.katcp_req.sensor_value()
                self.assertTrue(reply.reply_ok())
        except Exception:
            msg = "Failed to retrieve sensor values via CAM interface"
            self.Error(msg, exc_info=True)
            return
        else:
            pfb_status = list(set(
                [i.arguments[-2] for i in informs if "pfb.or0-err-cnt" in i.arguments[2]]
                ))[0]
            Aqf.equals(pfb_status, "warn", "Confirm that all F-Engines report warnings/errors PFB status")

        try:
            self.Step("Restore original FFT Shift values")
            reply, informs = self.katcp_req.fft_shift(fft_shift)
            self.assertTrue(reply.reply_ok())
            self.Passed("FFT Shift: %s restored." % fft_shift)
        except Exception:
            self.Error("Could not set the F-Engine FFT Shift value", exc_info=True)
            return

    def _test_memory_error(self):
        pass

    def _test_network_link_error(self):
        heading("Fault Detection: Network Link Errors")

        def int2ip(n):
            return socket.inet_ntoa(struct.pack("!I", n))

        def ip2int(ipstr):
            return struct.unpack("!I", socket.inet_aton(ipstr))[0]

        def get_spead_data():
            try:
                self.receiver.get_clean_dump()
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
            else:
                msg = (
                    "Confirm that the SPEAD accumulation is being produced by "
                    "instrument but not verified.\n")
                self.Passed(msg)

        # Record the current multicast destination of one of the F-engine data
        # ethernet ports,
        def get_host_ip(host):
            try:
                int_ip = host.registers.iptx_base.read()["data"].get("reg")
                assert isinstance(int_ip, int)
                return int2ip(int_ip)
            except BaseException:
                self.Failed("Failed to retrieve multicast destination from %s".format(
                    host.host.upper()))
                return

        def get_lru_status(self, host):

            if host in self.fhosts:
                engine_type = "feng"
            else:
                engine_type = "xeng"

            try:
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(
                        "%s-%s-lru-ok".format(host.host, engine_type)
                    )
                except BaseException:
                    reply, informs = self.katcp_req.sensor_value("%s-%s-lru-ok".format(host.host,
                        engine_type))
            except BaseException:
                self.Failed("Could not get sensor attributes on %s".format(host.host))
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
                host.registers.iptx_base.write(**{"reg": int(ip_new)})
                changed_ip = host.registers.iptx_base.read()["data"].get("reg")
                assert isinstance(changed_ip, int)
                changed_ip = int2ip(changed_ip)
            except BaseException:
                self.Failed("Failed to write new multicast destination on %s" % host.host)
            else:
                self.Passed(
                    "Confirm that the multicast destination address for %s has been changed "
                    "from %s to %s." % (host.host, ip_old, changed_ip)
                )

        def report_lru_status(self, host, get_lru_status):
            Aqf.wait(30, "Wait until the sensors have been updated with new changes")
            if get_lru_status(self, host) == 1:
                self.Passed(
                    "Confirm that the X-engine %s LRU sensor is OKAY and "
                    "that the X-eng is receiving feasible data." % (host.host)
                )
            elif get_lru_status(self, host) == 0:
                self.Passed(
                    "Confirm that the X-engine %s LRU sensor is reporting a "
                    "failure and that the X-eng is not receiving feasible "
                    "data." % (host.host)
                )
            else:
                self.Failed("Failed to read %s sensor" % (host.host))

        fhost = self.fhosts[random.randrange(len(self.fhosts))]
        # xhost = self.xhosts[random.randrange(len(self.xhosts))]
        ip_new = "239.101.2.250"

        self.Step(
            "Randomly selected %s host that is being used to produce the test "
            "data product on which to trigger the link error." % (fhost.host)
        )
        current_ip = get_host_ip(fhost)
        if not current_ip:
            self.Failed("Failed to retrieve multicast destination address of %s" % fhost.host)
        elif current_ip != ip_new:
            self.Passed("Current multicast destination address for %s: %s." % (fhost.host,
                current_ip))
        else:
            self.Failed("Multicast destination address of %s" % (fhost.host))

        self.Note("Debug code")
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)

        # write_new_ip(fhost, ip_new, current_ip)
        # time.sleep(30 / 2)
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)

        # self.Step('Restoring the multicast destination from %s to the original %s' % (
        #     human_readable_ip(ip_new), human_readable_ip(current_ip)))

        # write_new_ip(fhost, current_ip, ip_new, get_host_ip, human_readable_ip)
        # report_lru_status(self, xhost, get_lru_status)
        # get_spead_data(self)
        #

    def _test_host_sensors_status(self):
        heading("Monitor Sensors: Processing Node's Sensor Status")

        self.Step(
            "This test confirms that each processing node's sensor (Temp, Voltage, Current, "
            "Fan) has not FAILED, Reports only errors."
        )
        try:
            for i in range(2):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                except BaseException:
                    reply, informs = self.katcp_req.sensor_value()
                self.assertTrue(reply.reply_ok())
        except AssertionError:
            errmsg = "Failed to retrieve sensors via CAM interface"
            self.Error(errmsg, exc_info=True)
            return
        else:
            for i in informs:
                if i.arguments[2].startswith("xhost") or i.arguments[2].startswith("fhost"):
                    if i.arguments[-2].lower() != "nominal":
                        self.Note(" contains a ".join(i.arguments[2:-1]))

    def _test_vacc(self, test_chan, chan_index=None, acc_time=0.998):
        """Test vector accumulator"""
        # Choose a test frequency around the centre of the band.
        test_freq = self.cam_sensors.get_value("bandwidth") / 2.0

        test_input = self.cam_sensors.input_labels[0]
        eq_scaling = 30
        acc_times = [acc_time / 2, acc_time]
        # acc_times = [acc_time/2, acc_time, acc_time*2]
        n_chans = self.cam_sensors.get_value("n_chans")
        try:
            internal_accumulations = int(self.cam_sensors.get_value("xeng_acc_len"))
        except Exception:
            self.Error("Failed to retrieve X-engine accumulation length", exc_info=True)
        try:
            initial_dump = self.receiver.get_clean_dump()
            assert isinstance(initial_dump, dict)
        except Exception:
            self.Error("Could not retrieve clean SPEAD accumulation: Queue is Empty.",
                exc_info=True)
            return

        delta_acc_t = self.cam_sensors.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = abs(
            np.argmin(np.abs(self.cam_sensors.ch_center_freqs[:chan_index] - test_freq)) - test_chan
        )
        self.Step("Selected test input {} and test frequency channel {}".format(test_input, test_freq_channel))
        eqs = np.zeros(n_chans, dtype=np.complex)
        eqs[test_freq_channel] = eq_scaling
        self.restore_initial_equalisations()
        try:
            reply, _informs = self.katcp_req.gain(test_input, *list(eqs))
            self.assertTrue(reply.reply_ok())
            Aqf.hop("Gain successfully set on input %s via CAM interface." % test_input)
        except Exception:
            errmsg = "Gains/Eq could not be set on input %s via CAM interface" % test_input
            self.Error(errmsg, exc_info=True)

        self.Step(
            "Configured Digitiser simulator output(cw0 @ {:.3f}MHz) to be periodic in "
            "FFT-length: {} in order for each FFT to be identical".format(test_freq / 1e6, n_chans * 2)
        )

        cw_scale = 0.125
        # The re-quantiser outputs signed int (8bit), but the snapshot code
        # normalises it to floats between -1:1. Since we want to calculate the
        # output of the vacc which sums integers, denormalise the snapshot
        # output back to ints.
        # q_denorm = 128
        # quantiser_spectrum = get_quant_snapshot(self, test_input) * q_denorm
        try:
            # Make dsim output periodic in FFT-length so that each FFT is identical
            self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
            self.dhost.sine_sources.sin_1.set(frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
            assert self.dhost.sine_sources.sin_0.repeat == n_chans * 2
        except AssertionError:
            errmsg = "Failed to make the DEng output periodic in FFT-length so that each FFT is identical"
            self.Error(errmsg, exc_info=True)
        try:
            reply, informs = self.katcp_req.quantiser_snapshot(test_input)
            self.assertTrue(reply.reply_ok())
            informs = informs[0]
        except Exception:
            errmsg = (
                "Failed to retrieve quantiser snapshot of input %s via "
                "CAM Interface: \nReply %s" % (test_input, str(reply).replace("_", " "),))
            self.Error(errmsg, exc_info=True)
            return
        else:
            quantiser_spectrum = np.array(evaluate(informs.arguments[-1]))
            if chan_index:
                quantiser_spectrum = quantiser_spectrum[:chan_index]
            # Check that the spectrum is not zero in the test channel
            # Aqf.is_true(quantiser_spectrum[test_freq_channel] != 0,
            # 'Check that the spectrum is not zero in the test channel')
            # Check that the spectrum is zero except in the test channel
            Aqf.is_true(
                np.all(quantiser_spectrum[0:test_freq_channel] == 0),
                ("Confirm that the spectrum is zero except in the test channel:"
                 " [0:test_freq_channel]"),
            )
            Aqf.is_true(
                np.all(quantiser_spectrum[test_freq_channel + 1 :] == 0),
                ("Confirm that the spectrum is zero except in the test channel:"
                 " [test_freq_channel+1:]"),
            )
            self.Step(
                "FFT Window [{} samples] = {:.3f} micro seconds, Internal Accumulations = {}, "
                "One VACC accumulation = {}s".format(
                    n_chans * 2, self.cam_sensors.fft_period * 1e6, internal_accumulations, delta_acc_t
                )
            )

            chan_response = []
            for vacc_accumulations, acc_time in zip(test_acc_lens, acc_times):
                try:
                    reply = self.katcp_req.accumulation_length(acc_time, timeout=60)
                    assert reply.succeeded
                except Exception:
                    self.Failed(
                        "Failed to set accumulation length of {} after maximum vacc "
                        "sync attempts.".format(vacc_accumulations)
                    )
                else:
                    internal_acc = 2 * internal_accumulations * n_chans
                    accum_len = int(np.ceil((acc_time * self.cam_sensors.get_value("sample")) / internal_acc))
                    Aqf.almost_equals(
                        vacc_accumulations,
                        accum_len,
                        1,
                        "Confirm that vacc length was set successfully with"
                        " {}, which equates to an accumulation time of {:.6f}s".format(
                            vacc_accumulations, vacc_accumulations * delta_acc_t
                        ),
                    )
                    no_accs = internal_accumulations * vacc_accumulations
                    expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
                    try:
                        dump = self.receiver.get_clean_dump()
                        assert isinstance(dump, dict)
                    except Exception:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        actual_response = complexise(dump["xeng_raw"][:, 0, :])
                        actual_response_ = loggerise(dump["xeng_raw"][:, 0, :])
                        actual_response_mag = normalised_magnitude(dump["xeng_raw"][:, 0, :])
                        chan_response.append(actual_response_mag)
                        # Check that the accumulator response is equal to the expected response
                        caption = (
                            "Accumulators actual response is equal to the expected response for {} "
                            "accumulation length with a periodic cw tone every {} samples"
                            " at frequency of {:.3f} MHz with scale {}.".format(
                                test_acc_lens, n_chans * 2, test_freq / 1e6, cw_scale
                            )
                        )

                        plot_filename = "{}/{}_chan_resp_{}_vacc.png".format(
                            self.logs_path, self._testMethodName, int(vacc_accumulations)
                        )
                        plot_title = "Vector Accumulation Length: channel %s" % test_freq_channel
                        msg = (
                            "Confirm that the accumulator actual response is "
                            "equal to the expected response for {} accumulation length".format(vacc_accumulations)
                        )

                        if not Aqf.array_abs_error(
                            expected_response[:chan_index], actual_response_mag[:chan_index], msg
                        ):
                            aqf_plot_channels(
                                actual_response_mag,
                                plot_filename,
                                plot_title,
                                log_normalise_to=0,
                                normalise=0,
                                caption=caption,
                            )

    def _test_product_switch(self, instrument):
        self.Step(
            "Confirm that the SPEAD accumulations are being produced when Digitiser simulator is "
            "configured to output correlated noise"
        )
        self.dhost.noise_sources.noise_corr.set(scale=0.25)
        with ignored(Queue.Empty):
            self.receiver.get_clean_dump()

        self.Step("Capture stopped, deprogramming hosts by halting the katcp connection.")
        self.corr_fix.stop_x_data()
        self.corr_fix.halt_array

        no_channels = self.n_chans_selected
        self.Step("Re-initialising {instrument} instrument".format(**locals()))
        corr_init = False
        retries = 5
        start_time = time.time()
        self.Step("Correlator initialisation timer-started: %s" % start_time)
        while retries and not corr_init:
            try:
                self.set_instrument()
                self.corr_fix.start_x_data
                corr_init = True
                retries -= 1
                if corr_init:
                    end_time = time.time()
                    msg = "Correlator initialisation (%s) timer end: %s" % (instrument, end_time)
                    self.Step(msg)
                    self.logger.info(msg + " within %s retries" % (retries))
            except BaseException:
                retries -= 1
                if retries == 0:
                    errmsg = "Could not restart the correlator after %s tries." % (retries)
                    self.Error(errmsg, exc_info=True)

        if corr_init:
            host = self.xhosts[random.randrange(len(self.xhosts))]
            Aqf.is_true(
                host.is_running(),
                "Confirm that the instrument is initialised by checking if " "%s is programmed." % host.host,
            )
            self.set_instrument()
            try:
                Aqf.hop(
                    "Capturing SPEAD Accumulation after re-initialisation to confirm "
                    "that the instrument activated is valid."
                )
                self.assertIsInstance(self.receiver, CorrRx)
                re_dump = self.receiver.get_clean_dump()
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
            except AttributeError:
                errmsg = "Could not retrieve clean SPEAD accumulation: Receiver could not " "be instantiated"
                self.Error(errmsg, exc_info=True)
            else:
                msg = "Confirm that the SPEAD accumulations are being produced after instrument " "re-initialisation."
                Aqf.is_true(re_dump, msg)

                msg = (
                    "Confirm that the data product has the number of frequency channels %s "
                    "corresponding to the %s instrument product" % (no_channels, instrument)
                )
                Aqf.equals(4096, self.cam_sensors.get_value("no_chans"), msg)

                final_time = end_time - start_time - float(self.corr_fix.halt_wait_time)
                minute = 60.0
                msg = "Confirm that instrument switching to %s " "time is less than one minute" % instrument
                Aqf.less(final_time, minute, msg)

    def _test_delay_rate(self):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Delay Rate"
        heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            # delay_rate = ((setup_data['sample_period'] / self.cam_sensors.get_value('int_time']) *
            # np.random.rand() * (dump_counts - 3))
            # delay_rate = 3.98195128768e-09
            # _rate = get_delay_bounds(self.corr_fix.correlator).get('min_delay_rate')
            delay_rate = 0.7 * (self.cam_sensors.sample_period / self.cam_sensors.get_value("int_time"))
            delay_value = 0
            fringe_offset = 0
            fringe_rate = 0
            load_time = setup_data["t_apply"]
            delay_rates = [0] * setup_data["num_inputs"]
            delay_rates[setup_data["test_source_ind"]] = delay_rate
            delay_coefficients = ["0,{}:0,0".format(fr) for fr in delay_rates]
            self.Step("Calculate the parameters to be used for setting Fringe(s)/Delay(s).")
            self.Progress(
                "Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s "
                % (delay_rate, delay_value, fringe_offset, fringe_rate)
            )

            try:
                actual_data = self._get_actual_data(setup_data, dump_counts, delay_coefficients)
                actual_phases = [phases for phases, response in actual_data]
            except TypeError:
                errmsg = "Could not retrieve actual delay rate data. Aborting test"
                self.Error(errmsg, exc_info=True)
                return
            else:
                expected_phases = self._get_expected_data(
                    setup_data, dump_counts, delay_coefficients, actual_phases)

                no_chans = range(self.n_chans_selected)
                plot_units = "ns/s"
                plot_title = "Calculated delay rate {}{} [.7(sample period/integration time)]".format(
                    delay_rate * 1e9, plot_units)
                plot_filename = "{}/{}_delay_rate.png".format(self.logs_path, self._testMethodName)
                caption = (
                    "Actual vs Expected Unwrapped Correlation Phase [Delay Rate].\n"
                    "Note: Dashed line indicates expected value and solid line indicates "
                    "actual values received from SPEAD accumulation."
                )

                msg = "Observe the change in the phase slope, and confirm the phase change is as expected."
                self.Step(msg)
                actual_phases_ = np.unwrap(actual_phases)
                degree = 1.0
                radians = (degree / 360) * np.pi * 2
                decimal = len(str(degree).split(".")[-1])
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                expected_phases_ = expected_phases_[:, 0 : self.n_chans_selected]
                for i in range(0, len(expected_phases_) - 1):
                    delta_expected = np.abs(np.max(expected_phases_[i + 1] - expected_phases_[i]))
                    delta_actual = np.abs(np.max(actual_phases_[i + 1] - actual_phases_[i]))
                    # abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    abs_diff = np.abs(delta_expected - delta_actual)
                    msg = (
                        "Confirm that if difference (radians) between expected({:.3f}) "
                        "phases and actual({:.3f}) phases are 'Almost Equal' "
                        "within {} degree when delay rate of {} is applied.".format(
                            delta_expected, delta_actual, degree, delay_rate
                        )
                    )
                    Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                    msg = (
                        "Confirm that the maximum difference ({:.3f} "
                        "degree/{:.3f} rad) between expected phase and actual phase "
                        "between integrations is less than {} degree.".format(
                            np.rad2deg(abs_diff), abs_diff, degree)
                    )
                    Aqf.less(abs_diff, radians, msg)

                    try:
                        abs_error = np.max(actual_phases_[i] - expected_phases_[i])
                    except ValueError:
                        abs_error = np.max(actual_phases_[i] - expected_phases_[i][: len(actual_phases_[i])])
                    msg = (
                        "Confirm that the absolute maximum difference ({:.3f} "
                        "degree/{:.3f} rad) between expected phase and actual phase "
                        "is less than {} degree.".format(np.rad2deg(abs_error), abs_error, degree)
                    )
                    Aqf.less(abs_error, radians, msg)

                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

                    except AssertionError:
                        self.Step(
                            "Difference  between expected({:.3f}) "
                            "phases and actual({:.3f}) phases are "
                            "'Not almost equal' within {} degree when delay rate "
                            "of {} is applied.".format(delta_expected, delta_actual, degree, delay_rate)
                        )
                        caption = (
                            "Difference expected({:.3f}) and actual({:.3f})"
                            " phases are not equal within {} degree when delay rate of {} "
                            "is applied.".format(delta_expected, delta_actual, degree, delay_rate)
                        )

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans,
                            actual_phases_i,
                            expected_phases_i,
                            plot_filename="{}/{}_{}_delay_rate.png".format(self.logs_path, self._testMethodName, i),
                            plot_title="Delay Rate:\nActual vs Expected Phase Response",
                            plot_units=plot_units,
                            caption=caption,
                        )

                aqf_plot_phase_results(
                    no_chans,
                    actual_phases,
                    expected_phases,
                    plot_filename,
                    plot_title,
                    plot_units,
                    caption,
                    dump_counts,
                )

    def _test_fringe_rate(self):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Fringe rate"
        heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            _rand_gen = self.cam_sensors.get_value("int_time") * np.random.rand() * dump_counts
            fringe_rate = (np.pi / 8.0) / _rand_gen
            delay_value = 0
            delay_rate = 0
            fringe_offset = 0
            load_time = setup_data["t_apply"]
            fringe_rates = [0] * setup_data["num_inputs"]
            fringe_rates[setup_data["test_source_ind"]] = fringe_rate
            delay_coefficients = ["0,0:0,{}".format(fr) for fr in fringe_rates]

            self.Step("Calculate the parameters to be used for setting Fringe(s)/Delay(s).")
            self.Progress(
                "Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s "
                % (delay_rate, delay_value, fringe_offset, fringe_rate)
            )
            try:
                actual_data = self._get_actual_data(setup_data, dump_counts, delay_coefficients)
                actual_phases = [phases for phases, response in actual_data]

            except TypeError:
                errmsg = "Could not retrieve actual delay rate data. Aborting test: Exception: {}".format(e)
                self.Error(errmsg, exc_info=True)
                return
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts, delay_coefficients, actual_phases)

                no_chans = range(self.n_chans_selected)
                plot_units = "rads/sec"
                plot_title = "Randomly generated fringe rate {} {}".format(fringe_rate, plot_units)
                plot_filename = "{}/{}_fringe_rate.png".format(self.logs_path, self._testMethodName)
                caption = (
                    "Actual vs Expected Unwrapped Correlation Phase [Fringe Rate].\n"
                    "Note: Dashed line indicates expected value and solid line "
                    "indicates actual values received from SPEAD accumulation."
                )

                degree = 1.0
                radians = (degree / 360) * np.pi * 2
                decimal = len(str(degree).split(".")[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                msg = "Observe the change in the phase slope, and confirm the phase change is as " "expected."
                self.Step(msg)
                for i in range(0, len(expected_phases_) - 1):
                    try:
                        delta_expected = np.max(expected_phases_[i + 1] - expected_phases_[i])
                        delta_actual = np.max(actual_phases_[i + 1] - actual_phases_[i])
                    except IndexError:
                        errmsg = "Failed: Index is out of bounds"
                        self.Error(errmsg, exc_info=True)
                    else:
                        abs_diff = np.abs(delta_expected - delta_actual)
                        # abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                        msg = (
                            "Confirm that the difference between expected({:.3f}) "
                            "phases and actual({:.3f}) phases are 'Almost Equal' within "
                            "{} degree when fringe rate of {} is applied.".format(
                                delta_expected, delta_actual, degree, fringe_rate
                            )
                        )
                        Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                        msg = (
                            "Confirm that the maximum difference ({:.3f} "
                            "deg / {:.3f} rad) between expected phase and actual phase "
                            "between integrations is less than {} degree\n".format(
                                np.rad2deg(abs_diff), abs_diff, degree
                            )
                        )
                        Aqf.less(abs_diff, radians, msg)

                        try:
                            delta_actual_s = delta_actual - (delta_actual % degree)
                            delta_expected_s = delta_expected - (delta_expected % degree)
                            np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
                        except AssertionError:
                            self.Step(
                                "Difference between expected({:.3f}) phases and actual({:.3f}) "
                                "phases are 'Not almost equal' within {} degree when fringe rate "
                                "of {} is applied.".format(delta_expected, delta_actual, degree, fringe_rate)
                            )

                            caption = (
                                "Difference expected({:.3f}) and "
                                "actual({:.3f}) phases are not equal within {} degree when "
                                "fringe rate of {} is applied.".format(
                                    delta_expected, delta_actual, degree, fringe_rate
                                )
                            )

                            actual_phases_i = (delta_actual, actual_phases[i])
                            if len(expected_phases[i]) == 2:
                                expected_phases_i = (delta_expected, expected_phases[i][-1])
                            else:
                                expected_phases_i = (delta_expected, expected_phases[i])

                            aqf_plot_phase_results(
                                no_chans,
                                actual_phases_i,
                                expected_phases_i,
                                plot_filename="{}/{}_fringe_rate_{}.png".format(
                                    self.logs_path, self._testMethodName, i
                                ),
                                plot_title="Fringe Rate: Actual vs Expected Phase Response",
                                plot_units=plot_units,
                                caption=caption,
                            )

                aqf_plot_phase_results(
                    no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption
                )

    def _test_fringe_offset(self):
        msg = "CBF Delay and Phase Compensation Functional VR: Fringe offset"
        heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            fringe_offset = (np.pi / 2.0) * np.random.rand() * dump_counts
            # fringe_offset = 1.22796022444
            delay_value = 0
            delay_rate = 0
            fringe_rate = 0
            load_time = setup_data["t_apply"]
            fringe_offsets = [0] * setup_data["num_inputs"]
            fringe_offsets[setup_data["test_source_ind"]] = fringe_offset
            delay_coefficients = ["0,0:{},0".format(fo) for fo in fringe_offsets]

            self.Step("Calculate the parameters to be used for setting Fringe(s)/Delay(s).")
            self.Progress(
                "Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s "
                % (delay_rate, delay_value, fringe_offset, fringe_rate))

            try:
                actual_data = self._get_actual_data(setup_data, dump_counts, delay_coefficients)
                actual_phases = [phases for phases, response in actual_data]
            except TypeError:
                self.Error("Could not retrieve actual delay rate data. Aborting test", exc_info=True)
                return
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts, delay_coefficients, actual_phases)
                no_chans = range(self.n_chans_selected)
                plot_units = "rads"
                plot_title = "Randomly generated fringe offset {:.3f} {}".format(fringe_offset, plot_units)
                plot_filename = "{}/{}_fringe_offset.png".format(self.logs_path, self._testMethodName)
                caption = (
                    "Actual vs Expected Unwrapped Correlation Phase [Fringe Offset].\n"
                    "Note: Dashed line indicates expected value and solid line "
                    "indicates actual values received from SPEAD accumulation. "
                    "Values are rounded off to 3 decimals places"
                )

                # Ignoring first dump because the delays might not be set for full
                # integration.
                degree = 1.0
                decimal = len(str(degree).split(".")[-1])
                actual_phases_ = np.unwrap(actual_phases)
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                msg = "Observe the change in the phase slope, and confirm the phase change is as " "expected."
                self.Step(msg)
                for i in range(1, len(expected_phases) - 1):
                    delta_expected = np.abs(np.max(expected_phases_[i]))
                    delta_actual = np.abs(np.max(actual_phases_[i]))
                    # abs_diff = np.abs(delta_expected - delta_actual)
                    abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                    msg = (
                        "Confirm that the difference between expected({:.3f})"
                        " phases and actual({:.3f}) phases are 'Almost Equal' "
                        "within {:.3f} degree when fringe offset of {:.3f} is "
                        "applied.".format(delta_expected, delta_actual, degree, fringe_offset)
                    )

                    Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                    Aqf.less(
                        abs_diff,
                        degree,
                        "Confirm that the maximum difference({:.3f} "
                        "degrees/{:.3f}rads) between expected phase and actual phase "
                        "between integrations is less than {:.3f} degree\n".format(
                            abs_diff, np.deg2rad(abs_diff), degree
                        ),
                    )
                    try:
                        delta_actual_s = delta_actual - (delta_actual % degree)
                        delta_expected_s = delta_expected - (delta_expected % degree)
                        np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

                    except AssertionError:
                        self.Step(
                            "Difference between expected({:.5f}) phases "
                            "and actual({:.5f}) phases are 'Not almost equal' "
                            "within {} degree when fringe offset of {} is applied.".format(
                                delta_expected, delta_actual, degree, fringe_offset
                            )
                        )

                        caption = (
                            "Difference expected({:.3f}) and actual({:.3f}) "
                            "phases are not equal within {:.3f} degree when fringe offset "
                            "of {:.3f} {} is applied.".format(
                                delta_expected, delta_actual, degree, fringe_offset, plot_units
                            )
                        )

                        actual_phases_i = (delta_actual, actual_phases[i])
                        if len(expected_phases[i]) == 2:
                            expected_phases_i = (delta_expected, expected_phases[i][-1])
                        else:
                            expected_phases_i = (delta_expected, expected_phases[i])
                        aqf_plot_phase_results(
                            no_chans,
                            actual_phases_i,
                            expected_phases_i,
                            plot_filename="{}/{}_{}_fringe_offset.png".format(self.logs_path,
                                self._testMethodName, i),
                            plot_title=("Fringe Offset:\nActual vs Expected Phase Response"),
                            plot_units=plot_units,
                            caption=caption,
                        )

                aqf_plot_phase_results(
                    no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption
                )

    def _test_delay_inputs(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial:
        Delay applied to the correct input
        """
        msg = "CBF Delay and Phase Compensation Functional VR: Delays applied to the correct input"
        heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            self.Step(
                "The test will sweep through four(4) randomly selected baselines, select and "
                "set a delay value, Confirm if the delay set is as expected."
            )
            input_labels = self.cam_sensors.input_labels
            random.shuffle(input_labels)
            input_labels = input_labels[4:]
            for delayed_input in input_labels:
                test_delay_val = random.randrange(self.cam_sensors.sample_period, step=0.83e-10, int=float)
                # test_delay_val = self.cam_sensors.sample_period  # Pi
                expected_phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * test_delay_val
                expected_phases -= np.max(expected_phases) / 2.0
                self.Step("Clear all coarse and fine delays for all inputs before testing input %s." % delayed_input)
                delays_cleared = True  # clear_all_delays(self)
                if not delays_cleared:
                    self.Failed("Delays were not completely cleared, data might be corrupted.\n")
                else:
                    self.Passed("Cleared all previously applied delays prior to test.\n")
                    delays = [0] * setup_data["num_inputs"]
                    # Get index for input to delay
                    test_source_idx = input_labels.index(delayed_input)
                    self.Step("Selected input to test: {}".format(delayed_input))
                    delays[test_source_idx] = test_delay_val
                    self.Step("Randomly selected delay value ({}) relevant to sampling period".format(test_delay_val))
                    delay_coefficients = ["{},0:0,0".format(dv) for dv in delays]
                    int_time = setup_data["int_time"]
                    num_int = setup_data["num_int"]
                    try:
                        this_freq_dump = self.receiver.get_clean_dump()
                        t_apply = this_freq_dump["dump_timestamp"] + (num_int * int_time)
                        t_apply_readable = this_freq_dump["dump_timestamp_readable"]
                        self.Step("Delays will be applied with the following parameters:")
                        self.Progress("Current cmc time: %s (%s)" % (time.time(), time.strftime("%H:%M:%S")))
                        self.Progress(
                            "Current Dump timestamp: %s (%s)"
                            % (this_freq_dump["dump_timestamp"], this_freq_dump["dump_timestamp_readable"])
                        )
                        self.Progress("Time delays will be applied: %s (%s)" % (t_apply, t_apply_readable))
                        self.Progress("Delay coefficients: %s" % delay_coefficients)
                        reply, _informs = self.katcp_req.delays("antenna-channelised-voltage", t_apply, *delay_coefficients)
                        self.assertTrue(reply.reply_ok())
                    except Exception:
                        self.Failed("Failed to execute katcp requests!")
                        return
                    else:
                        Aqf.is_true(reply.reply_ok(), str(reply).replace("\_", " "))
                        self.Passed("Delays where successfully applied on input: {}".format(delayed_input))
                    try:
                        self.Step(
                            "Getting SPEAD accumulation (while discarding subsequent dumps) containing "
                            "the change in delay(s) on input: %s." % (test_source_idx)
                        )
                        dump = self.receiver.get_clean_dump(discard=35)
                    except Exception:
                        self.Error("Could not retrieve clean SPEAD accumulation: Queue is Empty.",
                            exc_info=True)
                    else:
                        sorted_bls = self.get_baselines_lookup(this_freq_dump, sorted_lookup=True)
                        degree = 1.0
                        self.Step("Maximum expected delay: %s" % np.max(expected_phases))
                        for b_line in sorted_bls:
                            b_line_val = b_line[1]
                            b_line_dump = dump["xeng_raw"][:, b_line_val, :]
                            b_line_phase = np.angle(complexise(b_line_dump))
                            # np.deg2rad(1) = 0.017 ie error should be withing 2 decimals
                            b_line_phase_max = round(np.max(b_line_phase), 2)
                            if (delayed_input in b_line[0]) and b_line[0] != (delayed_input, delayed_input):
                                msg = "Confirm that the baseline(s) {} expected delay is within 1 " "degree.".format(
                                    b_line[0]
                                )
                                Aqf.array_abs_error(
                                    np.abs(b_line_phase[5:-5]), np.abs(expected_phases[5:-5]), msg, degree
                                )
                            else:
                                # TODO Readdress this failure and calculate
                                if b_line_phase_max != 0.0:
                                    desc = (
                                        "Checking baseline {}, index: {}, phase offset found, "
                                        "maximum error value = {} rads".format(b_line[0], b_line_val, b_line_phase_max)
                                    )
                                    self.Failed(desc)



    def _test_min_max_delays(self):
        delays_cleared = self.clear_all_delays()
        setup_data = self._delays_setup()

        num_int = setup_data["num_int"]
        int_time = self.cam_sensors.get_value("int_time")
        if setup_data:
            self.Step("Clear all coarse and fine delays for all inputs before test commences.")
            if not delays_cleared:
                self.Failed("Delays were not completely cleared, data might be corrupted.\n")
            else:
                dump_counts = 5
                delay_bounds = get_delay_bounds(self.correlator)
                for _name, _values in sorted(delay_bounds.iteritems()):
                    _new_name = _name.title().replace("_", " ")
                    self.Step("Calculate the parameters to be used for setting %s." % _new_name)
                    delay_coefficients = 0
                    dump = self.receiver.get_clean_dump()
                    t_apply = dump["dump_timestamp"] + num_int * int_time
                    setup_data["t_apply"] = t_apply
                    no_inputs = [0] * setup_data["num_inputs"]
                    input_source = setup_data["test_source"]
                    no_inputs[setup_data["test_source_ind"]] = _values * dump_counts
                    if "delay_value" in _name:
                        delay_coefficients = ["{},0:0,0".format(dv) for dv in no_inputs]
                    if "delay_rate" in _name:
                        delay_coefficients = ["0,{}:0,0".format(dr) for dr in no_inputs]
                    if "phase_offset" in _name:
                        delay_coefficients = ["0,0:{},0".format(fo) for fo in no_inputs]
                    else:
                        delay_coefficients = ["0,0:0,{}".format(fr) for fr in no_inputs]

                    self.Progress(
                        "%s of %s will be set on input %s. Note: All other parameters "
                        "will be set to zero" % (_name.title(), _values, input_source)
                    )
                    try:
                        actual_data = self._get_actual_data(setup_data, dump_counts, delay_coefficients)
                    except TypeError:
                        self.Error("Failed to set the delays/fringes", exc_info=True)
                    else:
                        self.Step("Confirm that the %s where successfully set" % _new_name)
                        reply, informs = self.katcp_req.delays("antenna-channelised-voltage", )
                        msg = (
                            "%s where successfully set via CAM interface."
                            "\n\t\t\t    Reply: %s\n\n" % (_new_name, reply,))
                        Aqf.is_true(reply.reply_ok(), msg)


    def _test_delays_control(self):
        delays_cleared = self.clear_all_delays()
        setup_data = self._delays_setup()

        num_int = setup_data["num_int"]
        int_time = self.cam_sensors.get_value("int_time")
        self.Step("Disable Delays and/or Phases for all inputs.")
        if not delays_cleared:
            self.Failed("Delays were not completely cleared, data might be corrupted.\n")
        else:
            self.Passed("Confirm that the user can disable Delays and/or Phase changes via CAM interface.")
        dump = self.receiver.get_clean_dump()
        t_apply = dump["dump_timestamp"] + num_int * int_time
        no_inputs = [0] * setup_data["num_inputs"]
        input_source = setup_data["test_source"]
        no_inputs[setup_data["test_source_ind"]] = self.cam_sensors.sample_period * 2
        delay_coefficients = ["{},0:0,0".format(dv) for dv in no_inputs]
        try:
            self.Step(
                "Request and enable Delays and/or Phases Corrections on input (%s) "
                "via CAM interface." % input_source
            )
            load_strt_time = time.time()
            reply_, _informs = self.katcp_req.delays(t_apply, *delay_coefficients, timeout=30)
            load_done_time = time.time()
            msg = "Delay/Fringe(s) set via CAM interface reply : %s" % str(reply_)
            self.assertTrue(reply_.reply_ok())
            cmd_load_time = round(load_done_time - load_strt_time, 3)
            Aqf.is_true(reply_.reply_ok(), msg)
            self.Step("Fringe/Delay load command took {} seconds".format(cmd_load_time))
            # _give_up = int(num_int * int_time * 3)
            # while True:
            #    _give_up -= 1
            #    try:
            #        self.logger.info('Waiting for the delays to be updated')
            #        try:
            #            reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
            #        except:
            #            reply, informs = self.katcp_req.sensor_value()
            # self.assertTrue(reply.reply_ok())
            #    except Exception:
            #        self.Error('Weirdly I couldnt get the sensor values', exc_info=True)
            #    else:
            #        delays_updated = list(set([int(i.arguments[-1]) for i in informs
            #                                if '.cd.delay' in i.arguments[2]]))[0]
            #        if delays_updated:
            #            self.logger.info('Delays have been successfully set')
            #            break
            #    if _give_up == 0:
            #        self.logger.error("Could not confirm the delays in the time stipulated, exiting")
            #        break
            #    time.sleep(1)

        except Exception:
            errmsg = (
                "Failed to set delays via CAM interface with load-time: %s, "
                "Delay coefficients: %s" % (setup_data["t_apply"], delay_coefficients,))
            self.Error(errmsg, exc_info=True)
            return
        else:
            cam_max_load_time = setup_data["cam_max_load_time"]
            msg = "Time it took to load delay/fringe(s) %s is less than %ss" % (cmd_load_time,
                cam_max_load_time)
            Aqf.less(cmd_load_time, cam_max_load_time, msg)



    def _test_report_config(self):
        """CBF Report configuration"""
        import spead2
        import casperfpga
        test_config = self.corr_fix._test_config_file

        def git_revision_short_hash(mod_name=None, dir_name=None):
            return (
                subprocess.check_output(
                    ["git", "--git-dir=%s/.git" % dir_name, "--work-tree=%s" % mod_name,
                    "rev-parse", "--short", "HEAD"]
                ).strip()
                if mod_name and dir_name
                else None
            )

        def get_skarab_config(_timeout=30):
            from casperfpga import utils as fpgautils

            self.Step("List of all processing nodes")
            self.Progress("D-Engine :{}".format(self.dhost.host))
            try:
                fhosts, xhosts = self.fhosts, self.xhosts
            except AttributeError:
                fhosts = [self.corr_fix.corr_config["fengine"]["hosts"]]
                xhosts = [self.corr_fix.corr_config["xengine"]["hosts"]]

            self.Progress("List of F-Engines :{}".format(", ".join(fhosts)))
            self.Progress("List of X-Engines :{}\n".format(", ".join(xhosts)))
            self._hosts = list(
                np.concatenate(
                    [i.get("hosts", None).split(",")
                    for i in self.corr_fix.corr_config.values() if i.get("hosts")]
                )
            )
            skarabs = FPGA_Connect(self._hosts)
            if skarabs:
                version_info = fpgautils.threaded_fpga_operation(
                    skarabs,
                    timeout=_timeout,
                    target_function=(lambda fpga: fpga.transport.get_skarab_version_info(), [], {}),
                )
                # ToDo (MM) Get a list of all skarabs available including ip's and
                # leaf the host is connected to.
                # subprocess.check_output(['bash', 'scripts/find-skarabs-arp.sh'])
                for _host, _versions in version_info.iteritems():
                    self.Step("%s [R3000-0000] Software/Hardware Version Information" % _host)
                    self.Progress("IP Address: %s" % (socket.gethostbyname(_host)))
                    for _name, _version in _versions.iteritems():
                        try:
                            assert isinstance(_version, str)
                            _name = _name.title().replace("_", " ")
                            if _name.startswith("Microblaze Hardware"):
                                self.Progress("%s [M1200-0070]: %s\n" % (_name, _version))
                            elif _name.startswith("Microblaze Software"):
                                self.Progress("%s [M1200-0071]: %s" % (_name, _version))
                            elif _name.startswith("Spartan"):
                                self.Progress("%s [M1200-0069]: %s" % (_name, _version))
                        except BaseException:
                            pass

        def get_package_versions():
            corr2_name = corr2.__name__
            corr2_version = corr2.__version__
            corr2_pn = "M1200-0046"
            try:
                assert "devel" in corr2_version
                corr2_version = "".join([i for i in corr2_version.split(".") if len(i) == 7])
                corr2_link = "https://github.com/ska-sa/%s/commit/%s" % (corr2_name, corr2_version)
            except Exception:
                corr2_link = "Not Version Controlled at this time."

            casper_name = casperfpga.__name__
            casper_version = casperfpga.__version__
            casper_pn = "M1200-0055"
            try:
                assert "dev" in casper_version
                casper_version = "".join([i for i in casper_version.split(".") if len(i) == 7])
                casper_link = "https://github.com/ska-sa/%s/commit/%s" % (casper_name, casper_version)
            except Exception:
                casper_link = "Not Version Controlled at this time."

            katcp_name = katcp.__name__
            katcp_version = katcp.__version__
            katcp_pn = "M1200-0053"
            try:
                assert "dev" in katcp_version
                katcp_version = "".join([i for i in katcp_version.split(".") if len(i) == 7])
                assert len(katcp_version) == 7
                katcp_link = "https://github.com/ska-sa/%s-python/commit/%s" % (katcp_name, katcp_version)
            except Exception:
                katcp_link = "https://github.com/ska-sa/%s/releases/tag/v%s" % (katcp_name, katcp_version)

            spead2_name = spead2.__name__
            spead2_version = spead2.__version__
            spead2_pn = "M1200-0056"
            try:
                assert "dev" in spead2_version
                assert len(spead2_version) == 7
                spead2_version = "".join([i for i in spead2_version.split(".") if len(i) == 7])
                spead2_link = "https://github.com/ska-sa/%s/commit/%s" % (spead2_name, spead2_version)
            except Exception:
                spead2_link = "https://github.com/ska-sa/%s/releases/tag/v%s" % (spead2_name, spead2_version)

            try:
                bitstream_dir = self.corr_fix.configd.get("xengine").get("bitstream")
                mkat_dir, _ = os.path.split(os.path.split(os.path.dirname(os.path.realpath(bitstream_dir)))[0])
                _, mkat_name = os.path.split(mkat_dir)
                assert mkat_name
                mkat_version = git_revision_short_hash(dir_name=mkat_dir, mod_name=mkat_name)
                assert len(mkat_version) == 7
                mkat_link = "https://github.com/ska-sa/%s/commit/%s" % (mkat_name, mkat_version)
            except Exception:
                mkat_name = "mkat_fpga"
                mkat_link = "Not Version Controlled at this time."
                mkat_version = "Not Version Controlled at this time."

            try:
                test_dir, test_name = os.path.split(os.path.dirname(os.path.realpath(__file__)))
                testing_version = git_revision_short_hash(dir_name=test_dir, mod_name=test_name)
                assert len(testing_version) == 7
                testing_link = "https://github.com/ska-sa/%s/commit/%s" % (test_name, testing_version)
            except AssertionError:
                testing_link = "Not Version Controlled at this time."

            try:
                with open("/etc/cmc.conf") as f:
                    cmc_conf = f.readlines()
                templates_loc = [i.strip().split("=") for i in cmc_conf if i.startswith("CORR_TEMPLATE")][0][-1]
                # template_name = template_name.replace('_', ' ').title()
                config_dir = os.path.split(templates_loc)[0]
                config_dir_name = os.path.split(config_dir)[-1]
                config_version = git_revision_short_hash(dir_name=config_dir, mod_name=config_dir_name)
                config_pn = "M1200-0063"
                assert len(config_version) == 7
                config_link = "https://github.com/ska-sa/%s/commit/%s" % (config_dir_name, config_version)
            except Exception:
                config_dir_name = "mkat_config_templates"
                config_version = "Not Version Controlled"
                config_link = "Not Version Controlled"

            return {
                corr2_name: [corr2_version, corr2_link, corr2_pn],
                casper_name: [casper_version, casper_link, casper_pn],
                katcp_name: [katcp_version, katcp_link, katcp_pn],
                spead2_name: [spead2_version, spead2_link, spead2_pn],
                mkat_name: [mkat_version, mkat_link, "None"],
                test_name: [testing_version, testing_link, "None"],
                config_dir_name: [config_version, config_link, "None"],
            }

        def get_gateware_info():
            try:
                reply, informs = self.katcp_req.version_list()
                self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Failed("Could not retrieve CBF Gate-ware Version Information")
            else:
                for inform in informs:
                    if [s for s in inform.arguments if "xengine-firmware" in s]:
                        _hash = inform.arguments[-1].split(" ")
                        _hash = "".join([i.replace("[", "").replace("]", "") for i in _hash if 40 < len(i) < 42])
                        self.Progress("%s: %s" % (inform.arguments[0], _hash))
                        self.Progress("X/B-ENGINE (CBF) : M1200-0067")
                    elif [s for s in inform.arguments if "fengine-firmware" in s]:
                        _hash = inform.arguments[-1].split(" ")
                        _hash = "".join([i.replace("[", "").replace("]", "") for i in _hash if 40 < len(i) < 42])
                        self.Progress("%s: %s" % (inform.arguments[0], _hash))
                        self.Progress("F-ENGINE (CBF) : M1200-0064")
                    else:
                        self.Progress(": ".join(inform.arguments))
                self.Progress("CMC KATCP_C : M1200-0047")
                self.Progress("CMC CBF SCRIPTS : M1200-0048")
                self.Progress("CORRELATOR MASTER CONTROLLER (CMC) : M1200-0012")

        heading("CBF CMC Operating System.")
        self.Progress("CBF OS: %s | CMC OS P/N: M1200-0045" % " ".join(os.uname()))

        heading("CBF Software Packages Version Information.")
        self.Progress("CORRELATOR BEAMFORMER GATEWARE (CBF) : M1200-0041")
        get_gateware_info()

        heading("CBF Git Version Information.")
        self.Progress("CORRELATOR BEAMFORMER SOFTWARE : M1200-0036")
        packages_info = get_package_versions()
        for name, repo_dir in packages_info.iteritems():
            try:
                if name and (len(repo_dir[0]) == 7):
                    self.Progress(
                        "Repo: %s | Part Number: %s | Git Commit: %s | GitHub: %s"
                        % (name, repo_dir[2], repo_dir[0], repo_dir[1])
                    )
                else:
                    self.Progress("Repo: %s | Git Tag: %s | GitHub: %s" % (name, repo_dir[0], repo_dir[1]))
            except Exception:
                pass

        heading("CBF Processing Node Version Information")

        get_skarab_config()

    def _test_data_product(self, _baseline=False, _tiedarray=False):
        """CBF Imaging Data Product Set"""
        # Put some correlated noise on both outputs
        if "4k" in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = "113+0j"
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = "344+0j"
            fft_shift = 4095

        self.Step("Configure a digitiser simulator to generate correlated noise.")
        self.Progress(
            "Digitiser simulator configured to generate Gaussian noise with scale: {}, "
            "gain: {} and fft shift: {}.".format(awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        if _baseline:
            self.Step("Configure the CBF to generate Baseline-Correlation-Products(If available).")
            try:
                self.Progress(
                    "Retrieving initial SPEAD accumulation, in-order to confirm the number of "
                    "channels in the SPEAD data."
                )
                test_dump = self.receiver.get_clean_dump()
                assert isinstance(test_dump, dict)
            except Exception:
                errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
                self.Error(errmsg, exc_info=True)
                return
            else:
                exp_channels = test_dump["xeng_raw"].shape[0]
                no_channels = self.cam_sensors.get_value("n_chans")
                # Get baseline 0 data, i.e. auto-corr of m000h
                test_baseline = 0
                test_bls = evaluate(self.cam_sensors.get_value("bls_ordering"))[test_baseline]
                Aqf.equals(
                    exp_channels,
                    no_channels,
                    "Confirm that the baseline-correlation-products has the same number of "
                    "frequency channels ({}) corresponding to the {} "
                    "instrument currently running,".format(no_channels, self.instrument),
                )
                self.Passed(
                    "and confirm that imaging data product set has been "
                    "implemented for instrument: {}.".format(self.instrument)
                )

                response = normalised_magnitude(test_dump["xeng_raw"][:, test_baseline, :])
                plot_filename = "{}/{}_channel_response_.png".format(self.logs_path, self._testMethodName)

                caption = (
                    "An overall frequency response at {} baseline, "
                    "when digitiser simulator is configured to generate Gaussian noise, "
                    "with scale: {}, eq gain: {} and fft shift: {}".format(test_bls, awgn_scale, gain, fft_shift)
                )
                aqf_plot_channels(response, plot_filename, log_dynamic_range=90, caption=caption)

        if _tiedarray:
            try:
                self.logger.info("Checking if Docker is running!!!")
                output = subprocess.check_output(["docker", "run", "hello-world"])
                self.logger.info(output)
            except subprocess.CalledProcessError:
                errmsg = "Cannot connect to the Docker daemon. Is the docker daemon running on this host?"
                self.Failed(errmsg)
                return False

            try:
                # Set custom source names
                local_src_names = self.cam_sensors.custom_input_labels
                reply, informs = self.katcp_req.input_labels(*local_src_names)
                self.assertTrue(reply.reply_ok())
                labels = reply.arguments[1:]
                beams = ["tied-array-channelised-voltage.0x", "tied-array-channelised-voltage.0y"]
                running_instrument = self.instrument
                assert running_instrument is not False
                msg = "Running instrument currently does not have beamforming capabilities."
                assert not running_instrument.endswith("32k"), msg
                self.Step("Discontinue any capturing of %s and %s, if active." % (beams[0], beams[1]))
                reply, informs = self.katcp_req.capture_stop(beams[0])
                self.assertTrue(reply.reply_ok())
                reply, informs = self.katcp_req.capture_stop(beams[1])
                self.assertTrue(reply.reply_ok())

                # Get instrument parameters
                bw = self.cam_sensors.get_value("bandwidth")
                nr_ch = self.cam_sensors.get_value("n_chans")
                ants = self.cam_sensors.get_value("n_ants")
                ch_list = self.cam_sensors.ch_center_freqs
                ch_bw = ch_list[1]
                dsim_factor = float(self.conf_file["instrument_params"]["sample_freq"]) / self.cam_sensors.get_value(
                    "scale_factor_timestamp"
                )
                substreams = self.cam_sensors.get_value("n_xengs")
            except AssertionError:
                errmsg = "%s" % str(reply).replace("\_", " ")
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                self.Error("Error occurred", exc_info=True)
                return False

            self.Progress("Bandwidth = {}Hz".format(bw * dsim_factor))
            self.Progress("Number of channels = {}".format(nr_ch))
            self.Progress("Channel spacing = {}Hz".format(ch_bw * dsim_factor))

            beam = beams[0]
            try:
                beam_name = beam.replace("-", "_").replace(".", "_")
                beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
                beam_ip = beam_ip.split("+")[0]
                start_beam_ip = beam_ip
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_capture"])
                start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
                if start_substream + n_substrms_to_cap_m > substreams:
                    errmsg = (
                        "Substream start + substreams to process "
                        "is more than substreams available: {}. "
                        "Fix in test configuration file".format(substeams)
                    )
                    self.Failed(errmsg)
                    return False
                ticks_between_spectra = self.cam_sensors.get_value(
                    "antenna_channelised_voltage_n_samples_between_spectra"
                )
                assert isinstance(ticks_between_spectra, int)
                spectra_per_heap = self.cam_sensors.get_value(beam_name + "_spectra_per_heap")
                assert isinstance(spectra_per_heap, int)
                ch_per_substream = self.cam_sensors.get_value(beam_name + "_n_chans_per_substream")
                assert isinstance(ch_per_substream, int)
            except AssertionError:
                errmsg = "%s" % str(reply).replace("\_", " ")
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                self.Error("Error occurred", exc_info=True)
                return False

            # Compute the start IP address according to substream start index
            beam_ip = int2ip(ip2int(beam_ip) + start_substream)
            # Compute spectrum parameters
            strt_ch_idx = start_substream * ch_per_substream
            strt_freq = ch_list[strt_ch_idx] * dsim_factor
            self.Step("Start a KAT SDP docker ingest node for beam captures")
            docker_status = self.start_katsdpingest_docker(
                beam_ip,
                beam_port,
                n_substrms_to_cap_m,
                nr_ch,
                ticks_between_spectra,
                ch_per_substream,
                spectra_per_heap,
            )
            if docker_status:
                self.Progress(
                    "KAT SDP Ingest Node started. Capturing {} substream/s "
                    "starting at {}".format(n_substrms_to_cap_m, beam_ip)
                )
            else:
                self.Failed("KAT SDP Ingest Node failed to start")

            self.Step("Set beamformer quantiser gain for selected beam to 1")
            self.set_beam_quant_gain(beam, 1)

            beam_dict = {}
            beam_pol = beam[-1]
            for label in labels:
                if label.find(beam_pol) != -1:
                    beam_dict[label] = 0.0

            self.Progress("Only one antenna gain is set to 1, the reset are set to zero")
            weight = 1.0
            beam_dict = self.populate_beam_dict( 1, weight, beam_dict)
            try:
                bf_raw, cap_ts, bf_ts, in_wgts = self.capture_beam_data(beam, beam_dict)
            except TypeError:
                errmsg = (
                    "Failed to capture beam data: Confirm that Docker container is "
                    "running and also confirm the igmp version = 2 "
                )
                self.Error(errmsg, exc_info=True)
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
                    plot_filename="{}/{}_beam_response_{}.png".format(self.logs_path, self._testMethodName, beam),
                    plot_title=(
                        "Beam = {}, Spectrum Start Frequency = {} MHz\n"
                        "Number of Channels Captured = {}\n"
                        "Integrated over {} captures".format(
                            beam, strt_freq / 1e6, n_substrms_to_cap_m * ch_per_substream, nc
                        )
                    ),
                    log_dynamic_range=90,
                    log_normalise_to=1,
                    caption=("Tied Array Beamformer data captured during Baseline Correlation Product test."),
                    plot_type="bf",
                )
            except Exception:
                self.Failed("Failed to plot the diagram")

        if _baseline and _tiedarray:
            captured_bw = bw * self.n_chans_selected / float(nr_ch)
            baseline_ch_bw = captured_bw / test_dump["xeng_raw"].shape[0]
            beam_ch_bw = bw / len(cap_mag[0])
            msg = (
                "Confirm that the baseline-correlation-product channel width"
                " {}Hz is the same as the tied-array-channelised-voltage channel width "
                "{}Hz".format(baseline_ch_bw, beam_ch_bw)
            )
            Aqf.almost_equals(baseline_ch_bw, beam_ch_bw, 1e-3, msg)

    def _test_time_sync(self):
        self.Step("Request NTP pool address used.")
        try:
            host_ip = "192.168.194.2"
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        except ntplib.NTPException:
            host_ip = "192.168.1.21"
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        req_sync_time = 5e-3
        msg = (
            "Confirm that the CBF synchronised time is within {}s of "
            "UTC time as provided via PTP (NTP server: {}) on the CBF-TRF "
            "interface.".format(req_sync_time, host_ip)
        )
        Aqf.less(ntp_offset, req_sync_time, msg)

    def _test_gain_correction(self):
        """CBF Gain Correction"""
        if "4k" in self.instrument:
            # 4K
            awgn_scale = 0.0645
            # gain = 113
            gain = 30
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = 344
            fft_shift = 4095

        self.Step("Configure a digitiser simulator to generate correlated noise.")
        self.Progress(
            "Digitiser simulator configured to generate Gaussian noise, "
            "with scale: %s, eq gain: %s, fft shift: %s" % (awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=0.0,
            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            self.Failed("Failed to configure digitiser simulator levels")
            return False

        # Set per channel gain vectors for chosen input.
        source = random.randrange(len(self.cam_sensors.input_labels))
        test_input = random.choice(self.cam_sensors.input_labels)
        self.Step("Randomly selected input to test: %s" % (test_input))
        n_chans = self.cam_sensors.get_value("n_chans")
        rand_ch = random.choice(range(n_chans)[: self.n_chans_selected])
        gain_vector = [gain] * n_chans
        base_gain = gain
        try:
            reply, informs = self.katcp_req.gain(test_input, base_gain)
            self.assertTrue(reply.reply_ok())
        except Exception:
            self.Failed("Gain correction on %s could not be set to %s.: " "KATCP Reply: %s" % (test_input, gain, reply))
            return False

        _discards = 5
        try:
            initial_dump = self.receiver.get_clean_dump(discard=_discards)
            self.assertIsInstance(initial_dump, dict)
            assert np.any(initial_dump["xeng_raw"])
        except Exception:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
            return
        else:
            # Get auto correlation index of the selected input
            bls_order = evaluate(self.cam_sensors.get_value("bls_ordering"))
            for idx, val in enumerate(bls_order):
                if val[0] == test_input and val[1] == test_input:
                    auto_corr_idx = idx
            initial_resp = np.abs(complexise(initial_dump["xeng_raw"][:, auto_corr_idx, :]))
            initial_resp = 10 * np.log10(initial_resp)
            prev_resp = initial_resp
            chan_resp = []
            legends = []
            found = False
            fnd_less_one = False
            count = 0
            self.Step(
                "Note: Gains are relative to reference channels, and are increased "
                "iteratively until output power is increased by more than 6dB."
            )
            # Reset gain vectors for all channels
            try:
                reply, informs = self.katcp_req.gain(test_input, *gain_vector, timeout=60)
                self.assertTrue(reply.reply_ok())
            except Exception:
                self.Failed(
                    "Gain correction on %s could not be set to %s.: " "KATCP Reply: %s" % (test_input, gain, reply)
                )
                return False
            while not found:
                if not fnd_less_one:
                    target = 1
                    gain_inc = 5
                else:
                    target = 6
                    gain_inc = 400
                gain = gain + gain_inc
                gain_vector[rand_ch] = gain
                try:
                    reply, _ = self.katcp_req.gain(test_input, *gain_vector)
                    self.assertTrue(reply.reply_ok())
                    reply, _ = self.katcp_req.gain(test_input)
                    self.assertTrue(reply.reply_ok())
                except AssertionError:
                    self.Failed(
                        "Gain correction on %s could not be set to %s.: " "KATCP Reply: %s" % (test_input, gain, reply)
                    )
                else:
                    msg = "Gain correction on input %s, channel %s set to %s." % (
                        test_input,
                        rand_ch,
                        reply.arguments[rand_ch + 1],
                    )
                    self.Passed(msg)
                    try:
                        dump = self.receiver.get_clean_dump(discard=_discards)
                        self.assertIsInstance(dump, dict)
                    except AssertionError:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        response = np.abs(complexise(dump["xeng_raw"][:, auto_corr_idx, :]))
                        response = 10 * np.log10(response)
                        self.Progress("Maximum value found in channel {}".format(np.argmax(response)))
                        # resp_diff = response[rand_ch] - initial_resp[rand_ch]
                        resp_diff = response[rand_ch] - prev_resp[rand_ch]
                        prev_resp = response
                        if resp_diff < target:
                            msg = (
                                "Output power increased by less than 1 dB "
                                "(actual = {:.2f} dB) with a gain "
                                "increment of {}.".format(resp_diff, complex(gain_inc))
                            )
                            self.Passed(msg)
                            fnd_less_one = True
                            chan_resp.append(response)
                            legends.append("Gain set to %s" % (complex(gain)))
                        elif fnd_less_one and (resp_diff > target):
                            msg = (
                                "Output power increased by more than 6 dB "
                                "(actual = {:.2f} dB) with a gain "
                                "increment of {}.".format(resp_diff, complex(gain_inc))
                            )
                            self.Passed(msg)
                            found = True
                            chan_resp.append(response)
                            legends.append("Gain set to %s" % (complex(gain)))
                        else:
                            pass
                count += 1
                if count == 7:
                    self.Failed("Gains to change output power by less than 1 and more than 6 dB " "could not be found.")
                    found = True

            if chan_resp != []:
                zipped_data = zip(chan_resp, legends)
                zipped_data.reverse()
                aqf_plot_channels(
                    zipped_data,
                    plot_filename="{}/{}_chan_resp.png".format(self.logs_path, self._testMethodName),
                    plot_title="Channel Response Gain Correction for channel %s" % (rand_ch),
                    log_dynamic_range=90,
                    log_normalise_to=1,
                    caption="Gain Correction channel response, gain varied for channel %s, "
                    "all remaining channels are set to %s" % (rand_ch, complex(base_gain)),
                )
            else:
                self.Failed("Could not retrieve channel response with gain/eq corrections.")

    def _test_beamforming(self):
        """
        Apply weights and capture beamformer data, Verify that weights are correctly applied.
        """
        # Main test code
        # TODO AR
        # Neccessarry to compare output products with capture-list output products?

        try:
            output = subprocess.check_output(["docker", "run", "hello-world"])
            self.logger.info(output)
        except subprocess.CalledProcessError:
            errmsg = "Cannot connect to the Docker daemon. Is the docker daemon running on this host?"
            self.Failed(errmsg)
            return False

        try:
            # Set custom source names
            local_src_names = self.cam_sensors.custom_input_labels
            reply, informs = self.katcp_req.input_labels(*local_src_names)
            self.assertTrue(reply.reply_ok())
            labels = reply.arguments[1:]
            beams = ["tied-array-channelised-voltage.0x", "tied-array-channelised-voltage.0y"]
            # running_instrument = self.corr_fix.get_running_instrument()
            # assert running_instrument is not False
            # msg = 'Running instrument currently does not have beamforming capabilities.'
            # assert running_instrument.endswith('4k'), msg
            self.Step("Discontinue any capturing of %s and %s, if active." % (beams[0], beams[1]))
            reply, informs = self.katcp_req.capture_stop(beams[0])
            self.assertTrue(reply.reply_ok())
            reply, informs = self.katcp_req.capture_stop(beams[1])
            self.assertTrue(reply.reply_ok())

            # Get instrument parameters
            bw = self.cam_sensors.get_value("bandwidth")
            nr_ch = self.cam_sensors.get_value("n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]
            dsim_factor = float(
                self.conf_file["instrument_params"]["sample_freq"]) / self.cam_sensors.get_value(
                "scale_factor_timestamp"
            )
            substreams = self.cam_sensors.get_value("n_xengs")
            # For substream alignment test only print out 5 results
            align_print_modulo = int(substreams / 4)
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            self.Error("Error Occurred", exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * dsim_factor))

        beam = beams[0]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_capture"])
            start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            if start_substream + n_substrms_to_cap_m > substreams:
                errmsg = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substeams)
                )
                self.Failed(errmsg)
                return False
            ticks_between_spectra = self.cam_sensors.get_value("antenna_channelised_voltage_n_samples_between_spectra")
            assert isinstance(ticks_between_spectra, int)
            spectra_per_heap = self.cam_sensors.get_value(beam_name + "_spectra_per_heap")
            assert isinstance(spectra_per_heap, int)
            ch_per_substream = self.cam_sensors.get_value(beam_name + "_n_chans_per_substream")
            assert isinstance(ch_per_substream, int)
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            self.Error("Error Occurred", exc_info=True)
            return False

        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_freq = ch_list[strt_ch_idx] * dsim_factor
        self.Step("Start a KAT SDP docker ingest node for beam captures")
        docker_status = self.start_katsdpingest_docker(
            beam_ip,
            beam_port,
            n_substrms_to_cap_m,
            nr_ch,
            ticks_between_spectra,
            ch_per_substream,
            spectra_per_heap,
        )
        if docker_status:
            self.Progress(
                "KAT SDP Ingest Node started. Capturing {} substream/s "
                "starting at {}".format(n_substrms_to_cap_m, beam_ip)
            )
        else:
            self.Failed("KAT SDP Ingest Node failed to start")

        # Create a katcp client to connect to katcpingest
        if os.uname()[1] == "cmc2":
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc2"]
        elif os.uname()[1] == "cmc3":
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc3"]
        else:
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node"]
        ingst_nd_p = self.corr_fix._test_config_file["beamformer"]["ingest_node_port"]
        _timeout = 10
        try:
            import katcp

            ingest_kcp_client = katcp.BlockingClient(ingst_nd, ingst_nd_p)
            ingest_kcp_client.setDaemon(True)
            ingest_kcp_client.start()
            self.addCleanup(ingest_kcp_client.stop)
            is_connected = ingest_kcp_client.wait_connected(_timeout)
            if not is_connected:
                errmsg = "Could not connect to %s:%s, timed out." % (ingst_nd, ingst_nd_p)
                ingest_kcp_client.stop()
                raise RuntimeError(errmsg)
        except Exception:
            self.Error("error occurred", exc_info=True)

        def substreams_to_capture(lbeam, lbeam_ip, lsubstrms_to_cap, lbeam_port):
            """ Set ingest node capture substreams """
            try:
                self.logger.info(
                    "Setting ingest node to capture beam, substreams: {}, {}+{}:{}".format(
                        lbeam, lbeam_ip, lsubstrms_to_cap - 1, lbeam_port
                    )
                )
                reply, informs = ingest_kcp_client.blocking_request(
                    katcp.Message.request(
                        "substreams-to-capture", "{}+{}:{}".format(
                            lbeam_ip, lsubstrms_to_cap - 1, lbeam_port)
                    ),
                    timeout=_timeout,
                )
                self.assertTrue(reply.reply_ok())
            except Exception:
                errmsg = "Failed to issues ingest node capture-init: {}".format(str(reply))
                self.Error(errmsg, exc_info=True)

        for beam in beams:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap = n_substrms_to_cap_m
            # Compute the start IP address according to substream start index
            beam_ip = int2ip(ip2int(beam_ip) + start_substream)
            substreams_to_capture(beam, beam_ip, n_substrms_to_cap, beam_port)
            Aqf.hop("")
            Aqf.hop("")
            self.Step("Testing beam: {}".format(beam))

            def get_beam_data(
                beam,
                beam_dict=None,
                inp_ref_lvl=0,
                beam_quant_gain=1,
                act_wgts=None,
                exp_cw_ch=-1,
                s_ch_idx=0,
                s_substream=start_substream,
                subs_to_cap=n_substrms_to_cap,
                max_cap_retries=5,
                conf_data_type=False,
                avg_only=False,
                data_only=False,
            ):
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
                bf_raw_str = s_substream * ch_per_substream
                bf_raw_end = bf_raw_str + ch_per_substream * subs_to_cap

                # Capture beam data, retry if more than 20% of heaps dropped or empty data
                retries = 0
                while retries < max_cap_retries:
                    if retries == max_cap_retries - 1:
                        self.Failed("Error capturing beam data.")
                        return False
                    retries += 1
                    try:
                        bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(beam,
                            beam_dict, ingest_kcp_client)
                        # Set beamdict to None in case the capture needs to be retried.
                        # The beam weights have already been set.
                        beam_dict = None
                        if (len(in_wgts) == 0) and (isinstance(act_wgts, dict)):
                            in_wgts = act_wgts.copy()
                    except Exception:
                        self.Failed(
                            "Confirm that the Docker container is running and also confirm the "
                            "igmp version = 2"
                        )
                        return False

                    data_type = bf_raw.dtype.name
                    # Cut selected partitions out of bf_flags
                    flags = bf_flags[s_substream : s_substream + subs_to_cap]
                    # self.Step('Finding missed heaps for all partitions.')
                    if flags.size == 0:
                        self.logger.warning("Beam data empty. Capture failed. Retrying...")
                    else:
                        missed_err = False
                        for part in flags:
                            missed_heaps = np.where(part > 0)[0]
                            missed_perc = missed_heaps.size / part.size
                            perc = 0.60
                            if missed_perc > perc:
                                self.Progress("Missed heap percentage = {}%%".format(missed_perc * 100))
                                self.Progress("Missed heaps = {}".format(missed_heaps))
                                self.Progress(
                                    "Beam captured missed more than %s%% heaps. Retrying..." % (
                                        perc * 100))
                                missed_err = True
                                # break
                        # Good capture, break out of loop
                        if not missed_err:
                            break

                # Print missed heaps
                idx = s_substream
                for part in flags:
                    missed_heaps = np.where(part > 0)[0]
                    if missed_heaps.size > 0:
                        self.logger.info("Missed heaps for substream {} at heap indexes {}".format(
                            idx, missed_heaps))
                    idx += 1
                # Combine all missed heap flags. These heaps will be discarded
                flags = np.sum(flags, axis=0)
                # cap = [0] * num_caps
                # cap = [0] * len(bf_raw.shape[1])
                cap = []
                cap_idx = 0
                raw_idx = 0
                try:
                    for heap_flag in flags:
                        if heap_flag == 0:
                            for raw_idx in range(raw_idx, raw_idx + spectra_per_heap):
                                cap.append(np.array(complexise(bf_raw[bf_raw_str:bf_raw_end, raw_idx, :])))
                                cap_idx += 1
                            raw_idx += 1
                        else:
                            if raw_idx == 0:
                                raw_idx = spectra_per_heap
                            else:
                                raw_idx = raw_idx + spectra_per_heap
                except Exception:
                    errmsg = "Failed to capture beam data"
                    self.Error(errmsg, exc_info=True)

                if conf_data_type:
                    self.Step("Confirm that the data type of the beamforming data for one channel.")
                    msg = "Beamformer data type is {}, example value for one channel: {}".format(
                        data_type, cap[0][0])
                    Aqf.equals(data_type, "int8", msg)

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
                labels = ""
                label_values = in_wgts.values()
                if label_values[1:] == label_values[:-1]:
                    labels += "All inputs = {}\n".format(label_values[0])
                else:
                    tmp = {}
                    for key, val in in_wgts.items():
                        if val not in tmp.values():
                            tmp[key] = val
                        else:
                            for k, v in tmp.items():
                                if val == v:
                                    tmp.pop(k)
                            tmp["Multiple Inputs"] = val
                    for key in tmp:
                        labels += (key + " = {}\n").format(tmp[key])
                labels += "Mean = {:0.2f}dB\n".format(cap_db_mean)

                failed = False
                if inp_ref_lvl == 0:
                    # Get the voltage level for one antenna. Gain for one input
                    # should be set to 1, the rest should be 0
                    inp_ref_lvl = np.mean(cap_avg)
                    self.Step("Input ref level: {}".format(inp_ref_lvl))
                    self.Step(
                        "Reference level measured by setting the "
                        "gain for one antenna to 1 and the rest to 0. "
                        "Reference level = {:.3f}dB".format(20 * np.log10(inp_ref_lvl))
                    )
                    self.Step(
                        "Reference level averaged over {} channels. "
                        "Channel averages determined over {} "
                        "samples.".format(n_substrms_to_cap * ch_per_substream, cap_idx)
                    )
                    expected = 0
                else:
                    delta = 0.2
                    expected = np.sum([inp_ref_lvl * in_wgts[key] for key in in_wgts]) * beam_quant_gain
                    expected = 20 * np.log10(expected)

                    if exp_cw_ch != -1:
                        local_substream = s_ch_idx / ch_per_substream
                        # Find cw in expected channel, all other channels must be at expected level
                        max_val_ch = np.argmax(cap_db)
                        max_val = np.max(cap_db)
                        if max_val_ch == (exp_cw_ch - s_ch_idx):
                            msg = (
                                "CW at {:.3f}MHz found in channel {}, magnitude = {:.1f}dB, "
                                "spectrum mean = {:.1f}dB".format(
                                    ch_list[exp_cw_ch] / 1e6, exp_cw_ch, max_val, cap_db_mean
                                )
                            )
                            self.logger.info(msg)
                            if local_substream % align_print_modulo == 0:
                                self.Passed(msg)
                        else:
                            failed = True
                            self.Failed(
                                "CW at {:.3f}MHz not found in channel {}. "
                                "Maximum value of {}dB found in channel {}. "
                                "Mean spectrum value = {}dB".format(
                                    ch_list[exp_cw_ch] / 1e6, exp_cw_ch, max_val,
                                    max_val_ch + s_ch_idx, cap_db_mean
                                )
                            )

                        spikes = np.where(cap_db > expected + delta)[0]
                        if len(spikes == 1):
                            msg = "No spikes found in sub spectrum."
                            self.logger.info(msg)
                            if local_substream % align_print_modulo == 0:
                                self.Passed(msg)
                        else:
                            failed = True
                            self.Failed("Spikes found at: {}".format(spikes))
                    else:
                        self.Step(
                            "Expected value is calculated by taking the reference input level "
                            "and multiplying by the channel weights and quantiser gain."
                        )
                        labels += "Expected = {:.2f}dB\n".format(expected)
                        msg = (
                            "Confirm that the expected voltage level ({:.3f}dB) is within "
                            "{}dB of the measured mean value ({:.3f}dB)".format(
                                expected, delta, cap_db_mean)
                        )
                        Aqf.almost_equals(cap_db_mean, expected, delta, msg)
                return cap_avg, labels, inp_ref_lvl, expected, cap_idx, in_wgts, failed

            # Setting DSIM to generate noise
            if "4k" in self.instrument:
                # 4K
                awgn_scale = 0.0645
                cw_scale = 0.0
                gain = "113+0j"
                fft_shift = 511
            else:
                # 32K
                awgn_scale = 0.063
                cw_scale = 0.0
                gain = "344+0j"
                fft_shift = 4095

            self.Progress(
                "Digitiser simulator configured to generate Gaussian noise: "
                "Noise scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
            )
            dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
                freq=0, fft_shift=fft_shift, gain=gain)
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False

            # Only one antenna gain is set to 1, this will be used as the reference
            # input level
            # Set beamformer quantiser gain for selected beam to 1 quant gain reversed TODO: Fix
            bq_gain = self.set_beam_quant_gain(beam, 1)
            # Generating a dictionary to contain beam weights
            beam_dict = {}
            act_wgts = {}
            beam_pol = beam[-1]
            for label in labels:
                if label.find(beam_pol) != -1:
                    beam_dict[label] = 0.0
            if len(beam_dict) == 0:
                self.Failed("Beam dictionary not created, beam labels or beam name incorrect")
                return False
            ants = self.cam_sensors.get_value("n_ants")
            ref_input = np.random.randint(ants)
            # Find reference input label
            for key in beam_dict:
                if int(filter(str.isdigit, key)) == ref_input:
                    ref_input_label = key
                    break
            self.Step("{} used as a randomised reference input for this test".format(ref_input_label))
            weight = 1.0
            beam_dict = self.populate_beam_dict_idx(ref_input, weight, beam_dict)
            beam_data = []
            beam_lbls = []
            self.Step("Testing individual beam weights.")
            try:
                # Calculate reference level by not specifying ref level
                # Use weights from previous test
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(beam,
                    beam_dict=beam_dict, conf_data_type=True)
            except Exception:
                errmsg = "Failed to retrieve beamformer data"
                self.Failed(errmsg)
                return False
            beam_data.append(d)
            beam_lbls.append(l)

            # Characterise beam weight application:
            self.Step("Characterising beam weight application.")
            self.Step(
                "Step weight for one input and plot the mean value for all channels "
                "against expected value.")
            self.Step("Expected value calculated by multiplying reference value with weight.")
            weight = 0.1
            mean_vals = []
            exp_mean_vals = []
            weight_lbls = []
            # Create a list of weights to send
            weight_list = [0] * ants

            retry_cnt = 0
            while weight <= 4:
                weight_list[ref_input] = round(weight, 1)

                # Set weight for reference input, the rest are all zero
                # TODO: check that this actually checks that the correct weight has been set
                self.logger.info(
                    "Confirm that antenna input ({}) weight has been set to the desired weight.".format(
                        ref_input_label)
                )
                try:
                    reply, informs = self.katcp_req.beam_weights(beam, *weight_list, timeout=60)
                    self.assertTrue(reply.reply_ok())
                    actual_weight = float(reply.arguments[1 + ref_input])
                    retry_cnt = 0
                except AssertionError:
                    retry_cnt += 1
                    self.Failed("Beam weight not successfully set: {}".format(reply))
                    if retry_cnt == 5:
                        self.Failed("Beam weight could not be set after 5 retries... Exiting test.")
                        return False
                    continue
                except Exception:
                    retry_cnt += 1
                    errmsg = "Test failed"
                    self.Error(errmsg, exc_info=True)
                    if retry_cnt == 5:
                        self.Failed("Beam weight could not be set after 5 retries... Exiting test.")
                        return False
                    continue
                else:
                    self.Passed("Antenna input {} weight set to {}".format(key, actual_weight))

                # Get mean beam data
                try:
                    cap_data, act_wgts = get_beam_data(beam, avg_only=True)
                    cap_mean = np.mean(cap_data)
                    exp_mean = rl * actual_weight
                    mean_vals.append(cap_mean)
                    exp_mean_vals.append(exp_mean)
                    weight_lbls.append(weight)
                    self.Progress(
                        "Captured mean value = {:.2f}, Calculated mean value "
                        "(using reference value) = {:.2f}".format(cap_mean, exp_mean)
                    )
                except Exception:
                    errmsg = "Failed to retrieve beamformer data"
                    self.Failed(errmsg)
                    return
                if round(weight, 1) < 1:
                    weight += 0.1
                else:
                    weight += 0.5
            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(
                (
                    (
                        mean_vals,
                        "Captured mean beam output.\nStepping one input weight,\nwith remaining weights set to 0.",
                    ),
                    (
                        exp_mean_vals,
                        "Value calculated from reference,\nwhere reference measured at\nan input weight of 1.",
                    ),
                ),
                plot_filename="{}/{}_weight_application_{}.png".format(self.logs_path, self._testMethodName, beam),
                plot_title=("Beam = {}\n" "Expected vs Actual Mean Beam Output for Input Weight.".format(beam)),
                log_dynamic_range=None,  # 90, log_normalise_to=1,
                ylabel="Mean Beam Output",
                xlabel="{} Weight".format(ref_input_label),
                xvals=weight_lbls,
            )

            # Test weight application across all antennas
            self.Step("Testing weight application across all antennas.")
            weight = 0.4 / ants
            beam_dict = self.populate_beam_dict( -1, weight, beam_dict)
            try:
                d, l, rl, exp1, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except Exception:
                errmsg = "Failed to retrieve beamformer data:"
                self.Failed(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)
            weight = 1.0 / ants
            beam_dict = self.populate_beam_dict( -1, weight, beam_dict)
            try:
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except Exception:
                errmsg = "Failed to retrieve beamformer data"
                self.Failed(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)
            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(
                zip(np.square(beam_data), beam_lbls),
                plot_filename="{}/{}_chan_resp_{}.png".format(self.logs_path, self._testMethodName,
                    beam),
                plot_title=(
                    "Beam = {}\nSpectrum Start Frequency = {} MHz\n"
                    "Number of Channels Captured = {}"
                    "\nIntegrated over {} captures".format(
                        beam, strt_freq / 1e6, n_substrms_to_cap * ch_per_substream, nc
                    )
                ),
                log_dynamic_range=90,
                log_normalise_to=1,
                caption="Captured beamformer data",
                hlines=[exp0, exp1],
                plot_type="bf",
                hline_strt_idx=1,
            )

            self.Step("Testing quantiser gain adjustment.")
            # Level adjust after beamforming gain has already been set to 1
            beam_data = []
            beam_lbls = []
            try:
                # Recalculate reference level by not specifying ref level
                # Use weights from previous test
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(beam, beam_quant_gain=bq_gain,
                    act_wgts=act_wgts)
            except Exception:
                errmsg = "Failed to retrieve beamformer data:"
                self.Failed(errmsg)
                return
            beam_data.append(d)
            l += "Level adjust gain={}".format(bq_gain)
            beam_lbls.append(l)

            # Set level adjust after beamforming gain to 0.5
            bq_gain = self.set_beam_quant_gain(beam, 0.5)
            try:
                d, l, rl, exp1, nc, act_wgts, dummy = get_beam_data(
                    beam, inp_ref_lvl=rl, beam_quant_gain=bq_gain, act_wgts=act_wgts
                )
            except Exception:
                errmsg = "Failed to retrieve beamformer data:"
                self.Failed(errmsg)
                return
            beam_data.append(d)
            l += "Level adjust gain={}".format(bq_gain)
            beam_lbls.append(l)

            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(
                zip(np.square(beam_data), beam_lbls),
                plot_filename="{}/{}_level_adjust_after_bf_{}.png".format(self.logs_path, self._testMethodName, beam),
                plot_title=(
                    "Beam = {}\nSpectrum Start Frequency = {} MHz\n"
                    "Number of Channels Captured = {}"
                    "\nIntegrated over {} captures".format(
                        beam, strt_freq / 1e6, n_substrms_to_cap * ch_per_substream, nc
                    )
                ),
                log_dynamic_range=90,
                log_normalise_to=1,
                caption="Captured beamformer data with level adjust after beam-forming gain set.",
                hlines=exp1,
                plot_type="bf",
                hline_strt_idx=1,
            )

            self.Step("Checking beamformer substream alignment by injecting a CW in each substream.")
            self.Step(
                "Stepping through {} substreams and checking that the CW is in the correct "
                "position.".format(substreams)
            )
            # Reset quantiser gain
            bq_gain = self.set_beam_quant_gain(beam, 1)
            if "4k" in self.instrument:
                # 4K
                awgn_scale = 0.0645
                cw_scale = 0.01
                gain = "113+0j"
                fft_shift = 511
            else:
                # 32K
                awgn_scale = 0.063
                cw_scale = 0.01
                gain = "344+0j"
                fft_shift = 4095

            self.Progress(
                "Digitiser simulator configured to generate a stepping "
                "Constant Wave and Gaussian noise, "
                "CW scale: {}, Noise scale: {}, eq gain: {}, fft shift: {}".format(
                    cw_scale, awgn_scale, gain, fft_shift
                )
            )
            self.Step("This test will take a long time... check log for progress.")
            self.Step(
                "Only 5 results will be printed, all {} substreams will be tested. "
                "All errors will be displayed".format(substreams)
            )
            aligned_failed = False
            for substream in range(substreams):
                # Get substream start channel index
                strt_ch_idx = substream * ch_per_substream
                n_substrms_to_cap = 1
                # Compute the start IP address according to substream
                beam_ip = int2ip(ip2int(start_beam_ip) + substream)
                substreams_to_capture(beam, beam_ip, n_substrms_to_cap, beam_port)
                msg = "Capturing 1 substream at {}".format(beam_ip)
                self.logger.info(msg)
                if substream % align_print_modulo == 0:
                    self.Passed(msg)

                # Step dsim CW
                dsim_set_success = False
                cw_ch = strt_ch_idx + int(ch_per_substream / 4)
                freq = ch_list[cw_ch]
                dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
                    freq=freq, fft_shift=fft_shift, gain=gain
                )
                if not dsim_set_success:
                    self.Failed("Failed to configure digitise simulator levels")
                    return False

                try:
                    d, l, rl, exp0, nc, act_wgts, failed = get_beam_data(
                        beam,
                        inp_ref_lvl=rl,
                        act_wgts=act_wgts,
                        exp_cw_ch=cw_ch,
                        s_ch_idx=strt_ch_idx,
                        s_substream=substream,
                        subs_to_cap=n_substrms_to_cap,
                    )
                    if failed:
                        aligned_failed = True
                except Exception:
                    errmsg = "Failed to retrieve beamformer data"
                    self.Failed(errmsg)
                    return False
            if aligned_failed:
                self.Failed("Beamformer substream alignment test failed.")
            else:
                self.Passed("All beamformer substreams correctly aligned.")

        # Close any KAT SDP ingest nodes
        try:
            if ingest_kcp_client:
                ingest_kcp_client.stop()
        except BaseException:
            pass
        self.stop_katsdpingest_docker()

    def _test_beamforming_timeseries(self, beam_idx=0):
        """
        Perform a time series analysis of the beamforming data
        """
        # Main test code

        try:
            output = subprocess.check_output(["docker", "run", "hello-world"])
            self.logger.info(output)
        except subprocess.CalledProcessError:
            errmsg = "Cannot connect to the Docker daemon. Is the docker daemon running on this host?"
            self.Failed(errmsg)
            return False

        try:
            # TODO: custom source names not working?
            # Set custom source names
            # local_src_names = self.cam_sensors.custom_input_labels
            # reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            self.assertTrue(reply.reply_ok())
            # labels = reply.arguments[1:]
            labels = self.cam_sensors.input_labels
            beams = ["tied-array-channelised-voltage.0x", "tied-array-channelised-voltage.0y"]
            running_instrument = self.instrument
            assert running_instrument is not False
            msg = "Running instrument currently does not have beamforming capabilities."
            assert not running_instrument.endswith("32k"), msg
            self.Step("Discontinue any capturing of %s and %s, if active." % (beams[0], beams[1]))
            reply, informs = self.katcp_req.capture_stop(beams[0], timeout=60)
            self.assertTrue(reply.reply_ok())
            reply, informs = self.katcp_req.capture_stop(beams[1], timeout=60)
            self.assertTrue(reply.reply_ok())

            # Get instrument parameters
            bw = self.cam_sensors.get_value("bandwidth")
            nr_ch = self.cam_sensors.get_value("n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]
            dsim_factor = float(self.conf_file["instrument_params"]["sample_freq"]) / self.cam_sensors.get_value(
                "scale_factor_timestamp"
            )
            substreams = self.cam_sensors.get_value("n_xengs")
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            self.Error("Error Occurred", exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * dsim_factor))

        beam = beams[beam_idx]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_capture"])
            start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            if start_substream + n_substrms_to_cap_m > substreams:
                errmsg = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substeams)
                )
                self.Failed(errmsg)
                return False
            ticks_between_spectra = self.cam_sensors.get_value("antenna_channelised_voltage_n_samples_between_spectra")
            assert isinstance(ticks_between_spectra, int)
            spectra_per_heap = self.cam_sensors.get_value(beam_name + "_spectra_per_heap")
            assert isinstance(spectra_per_heap, int)
            ch_per_substream = self.cam_sensors.get_value(beam_name + "_n_chans_per_substream")
            assert isinstance(ch_per_substream, int)
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            self.Error("Error Occurred", exc_info=True)
            return False

        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_freq = ch_list[strt_ch_idx] * dsim_factor
        self.Step("Start a KAT SDP docker ingest node for beam captures")
        docker_status = self.start_katsdpingest_docker(
            beam_ip,
            beam_port,
            n_substrms_to_cap_m,
            nr_ch,
            ticks_between_spectra,
            ch_per_substream,
            spectra_per_heap,
        )
        if docker_status:
            self.Progress(
                "KAT SDP Ingest Node started. Capturing {} substream/s "
                "starting at {}".format(n_substrms_to_cap_m, beam_ip)
            )
        else:
            self.Failed("KAT SDP Ingest Node failed to start")

        # Determine CW frequency
        center_bin_offset = float(self.conf_file["beamformer"]["center_bin_offset"])
        center_bin_offset_freq = ch_bw * center_bin_offset
        cw_ch = strt_ch_idx + int(ch_per_substream / 4)

        # Setting DSIM to generate off center bin CW time sequence
        if "4k" in self.instrument:
            # 4K
            _capture_time = 0.1
            awgn_scale = 0.085
            cw_scale = 0.9
            gain = 7
            fft_shift = 8191
        elif "1k" in self.instrument:
            #
            _capture_time = 2
            awgn_scale = 0.085
            cw_scale = 0.9
            gain = "30+0j"
            fft_shift = 1023
            cw_ch = 65
        else:
            # 32K
            _capture_time = 0.1
            awgn_scale = 0.063
            cw_scale = 0.01
            gain = "344+0j"
            fft_shift = 4095

        freq = ch_list[cw_ch] + center_bin_offset_freq

        self.Step(
            "Generating time analysis plots of beam for channel {} containing a "
            "CW offset from center of a bin.".format(cw_ch)
        )
        self.Progress(
            "Digitiser simulator configured to generate a "
            "Constant Wave at {} Hz offset from the center "
            "of a bin by {} Hz.".format(freq, center_bin_offset_freq)
        )
        self.Progress(
            "CW scale: {}, Noise scale: {}, eq gain: {}, fft shift: {}".format(cw_scale, awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
            freq=freq, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        beam_quant_gain = 1.0 / ants
        self.Step("Set beamformer quantiser gain for selected beam to {}".format(beam_quant_gain))
        self.set_beam_quant_gain(beam, beam_quant_gain)

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # TODO: Currently setting weights is broken
        # self.Progress("Only one antenna gain is set to 1, the reset are set to zero")
        ref_input = np.random.randint(ants)
        ref_input = 1
        # Find reference input label
        for key in beam_dict:
            if int(filter(str.isdigit, key)) == ref_input:
                ref_input_label = key
                break
        self.Step("{} used as a randomised reference input for this test".format(ref_input_label))
        weight = 1.0
        # beam_dict = self.populate_beam_dict_idx(ref_input, weight, beam_dict)
        beam_dict = self.populate_beam_dict(-1, weight, beam_dict)
        try:
            # Currently setting weights is broken
            bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(beam, beam_dict, capture_time=_capture_time)
            # bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data( beam, capture_time=0.1)
            # Close any KAT SDP ingest nodes
            self.stop_katsdpingest_docker()
        except TypeError:
            errmsg = (
                "Failed to capture beam data\n\n Confirm that Docker container is "
                "running and also confirm the igmp version = 2 "
            )
            self.Error(errmsg, exc_info=True)
            return False

        flags = bf_flags[start_substream : start_substream + n_substrms_to_cap_m]
        # self.Step('Finding missed heaps for all partitions.')
        if flags.size == 0:
            self.logger.warning("Beam data empty. Capture failed. Retrying...")
            self.Failed("Beam data empty. Capture failed. Retrying...")
        else:
            missed_err = False
            for part in flags:
                missed_heaps = np.where(part > 0)[0]
                missed_perc = missed_heaps.size / part.size
                perc = 0.50
                if missed_perc > perc:
                    self.Progress("Missed heap percentage = {}%%".format(missed_perc * 100))
                    self.Progress("Missed heaps = {}".format(missed_heaps))
                    self.logger.warning("Beam captured missed more than %s%% heaps. Retrying..." % (perc * 100))
                    self.Failed("Beam captured missed more than %s%% heaps. Retrying..." % (perc * 100))
            # Print missed heaps
            idx = start_substream
            for part in flags:
                missed_heaps = np.where(part > 0)[0]
                if missed_heaps.size > 0:
                    self.logger.info("Missed heaps for substream {} at heap indexes {}".format(idx, missed_heaps))
                idx += 1
            # Combine all missed heap flags. These heaps will be discarded
            flags = np.sum(flags, axis=0)
            # Find longest run of uninterrupted data
            # Create an array that is 1 where flags is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(flags, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            # Find max run
            max_run = ranges[np.argmax(np.diff(ranges))]
            bf_raw_strt = max_run[0] * spectra_per_heap
            bf_raw_stop = max_run[1] * spectra_per_heap
            bf_raw = bf_raw[:, bf_raw_strt:bf_raw_stop, :]
            bf_ts = bf_ts[bf_raw_strt:bf_raw_stop]

            np.save("skarab_bf_data_plus.np", bf_raw)
            # return True
            from bf_time_analysis import analyse_beam_data

            analyse_beam_data(
                bf_raw,
                dsim_settings=[freq, cw_scale, awgn_scale],
                cbf_settings=[fft_shift, gain],
                do_save=True,
                spectra_use="all",
                chans_to_use=n_substrms_to_cap_m * ch_per_substream,
                xlim=[20, 21],
                dsim_factor=1.0,
                ref_input_label=ref_input_label,
                bandwidth=bw,
            )

            # aqf_plot_channels(beam_data[0:50, cw_ch-strt_ch_idx],
            #                  plot_filename='{}/{}_beam_cw_offset_from_centerbin_{}.png'.format(self.logs_path,
            #                    self._testMethodName, beam),
            #                  plot_title=('Beam = {}\n'
            #                    'Input = CW offset by {} Hz from the center of bin {}'
            #                    .format(beam, center_bin_offset_freq, cw_ch)),
            #                  log_dynamic_range=None, #90, log_normalise_to=1,
            #                  ylabel='Beam Output',
            #                  xlabel='Samples')

    def _test_group_delay(self, beam_idx=0):
        """

        Parameters
        ----------
        manual : Manually set the offset from the future_dump point.
        manual_offset : Offset in adc sample clocks.
        future_dump : Dump in which impulse is expected


        Returns
        -------
        """

        try:
            output = subprocess.check_output(["docker", "run", "hello-world"])
            self.logger.info(output)
        except subprocess.CalledProcessError:
            errmsg = "Cannot connect to the Docker daemon. Is the docker daemon running on this host?"
            self.Failed(errmsg)
            return False

        try:
            # Set custom source names
            # local_src_names = self.cam_sensors.custom_input_labels
            # reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            # self.assertTrue(reply.reply_ok())
            # labels = reply.arguments[1:]
            labels = self.cam_sensors.input_labels
            beams = ["tied-array-channelised-voltage.0x", "tied-array-channelised-voltage.0y"]
            running_instrument = self.instrument
            assert running_instrument is not False
            # msg = 'Running instrument currently does not have beamforming capabilities.'
            # assert running_instrument.endswith('1k'), msg
            self.Step("Discontinue any capturing of %s and %s, if active." % (beams[0], beams[1]))
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[0])
            self.assertTrue(reply.reply_ok())
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beams[1])
            self.assertTrue(reply.reply_ok())
            sync_time = self.cam_sensors.get_value("sync_time")

            # Get instrument parameters
            bw = self.cam_sensors.get_value("bandwidth")
            nr_ch = self.cam_sensors.get_value("n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]
            scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
            dsim_factor = float(
                self.conf_file["instrument_params"]["sample_freq"]) / scale_factor_timestamp
            substreams = self.cam_sensors.get_value("n_xengs")
        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * dsim_factor))

        beam = beams[beam_idx]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_capture"])
            start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            if start_substream + n_substrms_to_cap_m > substreams:
                errmsg = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substeams)
                )
                self.Failed(errmsg)
                return False
            ticks_between_spectra = self.cam_sensors.get_value("antenna_channelised_voltage_n_samples_between_spectra")
            assert isinstance(ticks_between_spectra, int)
            spectra_per_heap = self.cam_sensors.get_value(beam_name + "_spectra_per_heap")
            assert isinstance(spectra_per_heap, int)
            ch_per_substream = self.cam_sensors.get_value(beam_name + "_n_chans_per_substream")
            assert isinstance(ch_per_substream, int)
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)
            return False

        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_freq = ch_list[strt_ch_idx] * dsim_factor
        self.Step("Start a KAT SDP docker ingest node for beam captures")
        docker_status = self.start_katsdpingest_docker(
            beam_ip,
            beam_port,
            n_substrms_to_cap_m,
            nr_ch,
            ticks_between_spectra,
            ch_per_substream,
            spectra_per_heap,
        )
        if docker_status:
            self.Progress(
                "KAT SDP Ingest Node started. Capturing {} substream/s "
                "starting at {}".format(n_substrms_to_cap_m, beam_ip)
            )
        else:
            self.Failed("KAT SDP Ingest Node failed to start")
        # Create a katcp client to connect to katcpingest
        if os.uname()[1] == "cmc2":
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc2"]
        elif os.uname()[1] == "cmc3":
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node_cmc3"]
        else:
            ingst_nd = self.corr_fix._test_config_file["beamformer"]["ingest_node"]
        ingst_nd_p = self.corr_fix._test_config_file["beamformer"]["ingest_node_port"]
        _timeout = 10
        try:
            import katcp

            ingest_kcp_client = katcp.BlockingClient(ingst_nd, ingst_nd_p)
            ingest_kcp_client.setDaemon(True)
            ingest_kcp_client.start()
            self.addCleanup(ingest_kcp_client.stop)
            is_connected = ingest_kcp_client.wait_connected(_timeout)
            if not is_connected:
                errmsg = "Could not connect to %s:%s, timed out." % (ingst_nd, ingst_nd_p)
                ingest_kcp_client.stop()
                raise RuntimeError(errmsg)
        except Exception:
            self.Error("Could not connect to katcp client", exc_info=True)

        beam_quant_gain = 1.0 / ants
        # self.Step("Set beamformer quantiser gain for selected beam to {}".format(beam_quant_gain))
        # self.set_beam_quant_gain(beam, beam_quant_gain)

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # Currently setting weights is broken
        # self.Progress("Only one antenna gain is set to 1, the reset are set to zero")
        ref_input = np.random.randint(ants)
        ref_input = 1
        # Find reference input label
        for key in beam_dict:
            if int(filter(str.isdigit, key)) == ref_input:
                ref_input_label = key
                break
        self.Step("{} used as a randomised reference input for this test".format(ref_input_label))
        weight = 1.0
        beam_dict = self.populate_beam_dict_idx(ref_input, weight, beam_dict)
        # To Do: set beam weights

        def get_beam_data():
            try:
                bf_raw, bf_flags, bf_ts, in_wgts = capture_beam_data(
                    self, beam, ingest_kcp_client=ingest_kcp_client, stop_only=True
                )
            except Exception:
                errmsg = (
                    "Failed to capture beam data: Confirm that Docker container is "
                    "running and also confirm the igmp version = 2 "
                )
                self.Error(errmsg, exc_info=True)
                return False

            flags = bf_flags[start_substream : start_substream + n_substrms_to_cap_m]
            # self.Step('Finding missed heaps for all partitions.')
            if flags.size == 0:
                self.logger.warning("Beam data empty. Capture failed.")
                return None, None
            else:
                for part in flags:
                    missed_heaps = np.where(part > 0)[0]
                    missed_perc = missed_heaps.size / part.size
                    perc = 0.50
                    if missed_perc > perc:
                        self.Progress("Missed heap percentage = {}%%".format(missed_perc * 100))
                        self.Progress("Missed heaps = {}".format(missed_heaps))
                        self.Failed("Beam captured missed more than %s%% heaps. Retrying..." % (perc * 100))
                        return None, None
            # Print missed heaps
            idx = start_substream
            for part in flags:
                missed_heaps = np.where(part > 0)[0]
                if missed_heaps.size > 0:
                    self.logger.info("Missed heaps for substream {} at heap indexes {}".format(idx, missed_heaps))
                idx += 1
            # Combine all missed heap flags. These heaps will be discarded
            flags = np.sum(flags, axis=0)
            # Find longest run of uninterrupted data
            # Create an array that is 1 where flags is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(flags, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            # Find max run
            max_run = ranges[np.argmax(np.diff(ranges))]
            bf_raw_strt = max_run[0] * spectra_per_heap
            bf_raw_stop = max_run[1] * spectra_per_heap
            bf_raw = bf_raw[:, bf_raw_strt:bf_raw_stop, :]
            bf_ts = bf_ts[bf_raw_strt:bf_raw_stop]
            return bf_raw, bf_ts

        def load_dsim_impulse(load_timestamp, offset=0):
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=2)
            self.dhost.registers.src_sel_cntrl.write(src_sel_1=0)
            self.dhost.registers.impulse_delay_correction.write(reg=16)
            load_timestamp = load_timestamp + offset
            # lt_abs_t = datetime.fromtimestamp(
            #    sync_time + load_timestamp / scale_factor_timestamp)
            # curr_t = datetime.fromtimestamp(time.time())
            # self.Progress('Current time      = {}:{}.{}'.format(curr_t.minute,
            #                                            curr_t.second,
            #                                            curr_t.microsecond))
            # self.Progress('Impulse load time = {}:{}.{}'.format(lt_abs_t.minute,
            #                                            lt_abs_t.second,
            #                                            lt_abs_t.microsecond))
            # if ((abs(curr_t.minute - lt_abs_t.minute) > 1) and
            #    (abs(curr_t.second - lt_abs_t.second) > 1)):
            #    self.Failed('Timestamp drift too big. Resynchronise digitiser simulator.')
            # Digitiser simulator local clock factor of 8 slower
            # (FPGA clock = sample clock / 8).
            load_timestamp = load_timestamp / 8.0
            if not load_timestamp.is_integer():
                self.Failed("Timestamp received in accumulation not divisible" " by 8: {:.15f}".format(
                    load_timestamp))
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

        def get_dsim_mcount(spectra_ref_mcount):
            # Get the current mcount and shift it to the start of a spectra
            dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()["data"]["reg"]
            dsim_loc_msw = self.dhost.registers.local_time_msw.read()["data"]["reg"]
            reg_size = 32
            dsim_loc_time = dsim_loc_msw * pow(2, reg_size) + dsim_loc_lsw
            if not (spectra_ref_mcount / 8.0).is_integer():
                self.Failed("Spectra reference mcount is not divisible" " by 8: {:.15f}".format(
                    spectra_ref_mcount))
            dsim_loc_time = dsim_loc_time * 8
            # Shift current dsim time to the edge of a spectra
            dsim_spectra_time = dsim_loc_time - (
                dsim_loc_time - spectra_ref_mcount) % ticks_between_spectra
            return dsim_spectra_time

        dsim_set_success = self.set_input_levels(awgn_scale=0.0, cw_scale=0.0, freq=0,
            fft_shift=0, gain="32767+0j")
        self.dhost.outputs.out_1.scale_output(0)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        out_func = []
        num_pulse_caps = 500
        # num_pulse_int = 2
        # pulse_step must be divisible by 8
        pulse_step = 16
        points_around_trg = 16
        chan_str = 0
        chan_stp = 511
        load_lead_time = 0.01
        load_lead_mcount = 8 * int(load_lead_time * scale_factor_timestamp / 8)
        for pulse_cap in range(num_pulse_caps):
            beam_retries = 5
            while beam_retries > 0:
                # mcount_list = []
                beam_retries -= 1
                # Get an mcount at the start of a spectrum
                _ = self.capture_beam_data(beam, ingest_kcp_client=ingest_kcp_client, start_only=True)
                time.sleep(0.005)
                bf_raw, bf_ts = get_beam_data()
                if np.all(bf_raw) is None or np.all(bf_ts) is None:
                    break
                spectra_ref_mcount = bf_ts[-1]
                # Start beam capture
                _ = self.capture_beam_data(beam, ingest_kcp_client=ingest_kcp_client, start_only=True)
                # Get current mcount
                # for pulse_int in range(num_pulse_int):
                curr_mcount = get_dsim_mcount(spectra_ref_mcount)
                future_mcount = load_lead_mcount + curr_mcount + pulse_step * pulse_cap
                load_dsim_impulse(future_mcount)
                # mcount_list.append(future_mcount)
                # while get_dsim_mcount(spectra_ref_mcount) < future_mcount:
                time.sleep(load_lead_time)
                bf_raw, bf_ts = get_beam_data()
                if np.all(bf_raw) is not None and np.all(bf_ts) is not None:
                    break
            # beam_retries = 5
            # while beam_retries > 0:
            #    beam_retries -= 1
            #    _ = self.capture_beam_data( beam, ingest_kcp_client=ingest_kcp_client, start_only=True)
            #    time.sleep(0.01)
            #    bf_raw, bf_ts = get_beam_data()
            #    if np.all(bf_raw) != None and np.all(bf_ts) != None:
            #        curr_mcount = bf_ts[-1]
            #        future_mcount = 0.5 * scale_factor_timestamp + curr_mcount + pulse_step*pulse_cap
            #        future_mcount = 8*int(future_mcount/8)
            #        load_dsim_impulse(future_mcount)
            #        _ = self.capture_beam_data( beam, ingest_kcp_client=ingest_kcp_client, start_only=True)
            #        time.sleep(0.2)
            #        bf_raw, bf_ts = get_beam_data()
            #    if np.all(bf_raw) != None and np.all(bf_ts) != None:
            #        break
            else:
                self.Failed("Beam data capture failed.")
                break
            # num_found = 0
            # captured_list = []
            # for trgt_mcount in mcount_list[:-1]:
            try:
                assert future_mcount
            except Exception:
                return False
            trgt_spectra_idx = np.where(bf_ts > future_mcount)[0]
            if trgt_spectra_idx.size == 0:
                self.logger.warning(
                    "Target spectra timestamp too late by {} seconds".format(
                        (future_mcount - bf_ts[-1]) / scale_factor_timestamp
                    )
                )
            elif trgt_spectra_idx.size == bf_ts.size:
                self.logger.warning(
                    "Target spectra timestamp too early by {} seconds".format(
                        (bf_ts[0] - future_mcount) / scale_factor_timestamp
                    )
                )
            else:
                trgt_spectra_idx = trgt_spectra_idx[0] - 1
                # num_found += 1
                self.Progress(
                    "Target specra found at index {} of beam capture "
                    "containing {} spectra".format(trgt_spectra_idx, bf_ts.shape[0])
                )
                # trgt_cap_list = []
                for i in range(trgt_spectra_idx - points_around_trg, trgt_spectra_idx + 1):
                    spectra_mean_val = np.sum(np.abs(complexise(bf_raw[chan_str:chan_stp, i, :]))) / (
                        chan_stp - chan_str
                    )
                    spectra_ts = bf_ts[i]
                    ts_delta = int(spectra_ts) - future_mcount
                    # trgt_cap_list.append([ts_delta,spectra_mean_val])
                    out_func.append([ts_delta, spectra_mean_val])
                # captured_list.append(trgt_cap_list)
                # print ('{}:{}'.format(ts_delta,spectra_mean_val))
        else:
            # Remove any values which don't make sense, these happend when a capture missed the target mcount
            rem_index = np.where((np.sum(out_func, axis=1)) > 30000)
            out_func = np.delete(out_func, rem_index, axis=0)
            x = [x[0] for x in out_func]
            y = [y[1] for y in out_func]
            plt.scatter(x, y)
            plt.show()
            # import IPython;IPython.embed()

        # Close any KAT SDP ingest nodes
        try:
            if ingest_kcp_client:
                ingest_kcp_client.stop()
        except BaseException:
            pass
        self.stop_katsdpingest_docker()

        # Check ADC snapshot for pulse
        # self.correlator.est_sync_epoch()
        # sync_time = self.cam_sensors.get_value('sync_time')
        # bf_raw, bf_ts = get_beam_data()
        # curr_mcount = bf_ts[-1]
        # future_mcount = 1 * scale_factor_timestamp + curr_mcount
        # load_dsim_impulse(future_mcount)
        # unix_time = sync_time + (future_mcount/scale_factor_timestamp)
        # error_mcount = self.correlator.mcnt_from_time(unix_time) - future_mcount
        # unix_time = sync_time + (future_mcount-error_mcount-4000)/scale_factor_timestamp
        # a = self.correlator.fops.get_adc_snapshot(unix_time=unix_time)[labels[1]].data
        # print time.time()
        # print unix_time
        # print self.correlator.mcnt_from_time(unix_time) - future_mcount
        # print np.argmax(a)

    def _bf_efficiency(self):

        local_src_names = self.cam_sensors.custom_input_labels
        try:
            reply, informs = self.katcp_req.capture_stop("beam_0x", timeout=60)
            reply, informs = self.katcp_req.capture_stop("beam_0y", timeout=60)
            reply, informs = self.katcp_req.capture_stop("c856M4k", timeout=60)
            reply, informs = self.katcp_req.input_labels(*local_src_names)
            if reply.reply_ok():
                labels = reply.arguments[1:]
            else:
                raise Exception
        except Exception:
            self.Failed(e)
            return
        bw = self.cam_sensors.get_value("bandwidth")
        ch_list = self.cam_sensors.ch_center_freqs
        nr_ch = self.n_chans_selected

        # Start of test. Setting required partitions and center frequency
        partitions = 1
        part_size = bw / 16
        target_cfreq = bw + bw * 0.5
        target_pb = partitions * part_size
        ch_bw = bw / nr_ch
        beams = ("beam_0x", "beam_0y")
        beam = beams[1]

        # Set beamformer quantiser gain for selected beam to 1
        self.set_beam_quant_gain(beam, 1)

        if "4k" in self.instrument:
            # 4K
            awgn_scale = 0.032
            gain = "226+0j"
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = "344+0j"
            fft_shift = 4095

        self.Step(
            "Digitiser simulator configured to generate Gaussian noise, "
            "with scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=0.0,
            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                inp = label
                break
        try:
            reply, informs = self.katcp_req.quantiser_snapshot(inp, timeout=60)
        except Exception:
            self.Failed("Failed to grab quantiser snapshot.")
        quant_snap = [evaluate(v) for v in (reply.arguments[1:][1:])]
        try:
            reply, informs = self.katcp_req.adc_snapshot(inp, timeout=60)
        except Exception:
            self.Failed("Failed to grab adc snapshot.")
        fpga = self.correlator.fhosts[0]
        adc_data = fpga.get_adc_snapshots()["p0"].data
        p_std = np.std(adc_data)
        p_levels = p_std * 512
        aqf_plot_histogram(
            adc_data,
            plot_filename="{}/{}_adc_hist_{}.png".format(self.logs_path, self._testMethodName, inp),
            plot_title=(
                "ADC Histogram for input {}\nNoise Profile: "
                "Std Dev: {:.3f} equates to {:.1f} levels "
                "toggling.".format(inp, p_std, p_levels)
            ),
            caption="ADC input histogram for beamformer efficiency test, "
            "with the digitiser simulator noise scale at {}, "
            "quantiser gain at {} and fft shift at {}.".format(awgn_scale, gain, fft_shift),
            bins=256,
            ranges=(-1, 1),
        )
        p_std = np.std(quant_snap)
        aqf_plot_histogram(
            np.abs(quant_snap),
            plot_filename="{}/{}_quant_hist_{}.png".format(self.logs_path, self._testMethodName, inp),
            plot_title=("Quantiser Histogram for input {}\n " "Standard Deviation: {:.3f}".format(inp, p_std)),
            caption="Quantiser histogram for beamformer efficiency test, "
            "with the digitiser simulator noise scale at {}, "
            "quantiser gain at {} and fft shift at {}.".format(awgn_scale, gain, fft_shift),
            bins=64,
            ranges=(0, 1.5),
        )

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # Only one antenna gain is set to 1, this will be used as the reference
        # input level
        weight = 1.0
        beam_dict = self.populate_beam_dict( 1, weight, beam_dict)
        try:
            bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = capture_beam_data(
                self, beam, beam_dict, target_pb, target_cfreq, capture_time=0.3
            )
        except TypeError:
            errmsg = "Failed to capture beam data:"
            self.Failed(errmsg)
            self.logger.info(errmsg)
            return
        Aqf.hop("Packaging beamformer data.")
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
        self.Step("Calculating time series mean.")
        ch_mean = cap.mean(axis=0)
        self.Step("Calculating time series standard deviation")
        ch_std = cap.std(axis=0, ddof=1)
        ch_bw = self.cam_sensors.delta_f
        acc_time = self.cam_sensors.fft_period
        sqrt_bw_at = np.sqrt(ch_bw * acc_time)
        self.Step("Calculating channel efficiency.")
        eff = 1 / ((ch_std / ch_mean) * sqrt_bw_at)
        self.Step("Beamformer mean efficiency for {} channels = {:.2f}%".format(nr_ch, 100 * eff.mean()))
        plt_filename = "{}/{}_beamformer_efficiency.png".format(self.logs_path, self._testMethodName)
        plt_title = "Beamformer Efficiency per Channel\n " "Mean Efficiency = {:.2f}%".format(100 * eff.mean())
        caption = (
            "Beamformer efficiency per channel calculated over {} samples "
            "with a channel bandwidth of {:.2f}Hz and a FFT window length "
            "of {:.3f} micro seconds per sample.".format(cap_idx, ch_bw, acc_time * 1000000.0)
        )
        aqf_plot_channels(
            eff * 100,
            plt_filename,
            plt_title,
            caption=caption,
            log_dynamic_range=None,
            hlines=95,
            ylimits=(90, 105),
            plot_type="eff",
        )

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
            lt_abs_t = datetime.datetime.fromtimestamp(sync_time + load_timestamp / scale_factor_timestamp)
            print "Impulse load time = {}:{}.{}".format(lt_abs_t.minute, lt_abs_t.second, lt_abs_t.microsecond)
            print "Number of dumps in future = {:.10f}".format((load_timestamp - dump_ts) / dump_ticks)
            # Digitiser simulator local clock factor of 8 slower
            # (FPGA clock = sample clock / 8).
            load_timestamp = load_timestamp / 8
            if not load_timestamp.is_integer():
                self.Failed("Timestamp received in accumulation not divisible" " by 8: {:.15f}".format(load_timestamp))
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
        #     reply, informs = self.katcp_req.accumulation_length(1, timeout=60)
        #     if not reply.reply_ok():
        #         raise Exception
        # except:
        #     errmsg = 'Failed to set accumulation time withing {}s'.format(
        #         reply)
        #     self.Error(errmsg, exc_info=True)
        #     self.Failed(errmsg)
        #     return False

        dsim_set_success = self.set_input_levels(
            self.corr_fix, self.dhost, awgn_scale=0.0, cw_scale=0.0, freq=100000000, fft_shift=0, gain="32767+0j"
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        self.dhost.outputs.out_1.scale_output(0)
        dump = self.receiver.get_clean_dump()
        baseline_lookup = self.get_baselines_lookup(dump)
        sync_time = self.cam_sensors.get_values("sync_epoch")
        scale_factor_timestamp = self.cam_sensors.get_values("scale_factor_timestamp")
        inp = self.cam_sensors.get_values("input_labels")[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        # fft_sliding_window = dump['n_chans'].value * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = self.cam_sensors.get_values("int_time") * self.cam_sensors.get_values("adc_sample_rate")
        # print dump_ticks
        dump_ticks = self.cam_sensors.get_values("n_accs") * self.cam_sensors.get_values("n_chans") * 2
        # print dump_ticks
        # print ['adc_sample_rate'].value
        # print dump['timestamp']
        if not (dump_ticks / 8.0).is_integer():
            self.Failed("Number of ticks per dump is not divisible" " by 8: {:.3f}".format(dump_ticks))
        # Create a linear array spaced by 8 for finding dump timestamp edge
        tick_array = np.linspace(-dump_ticks / 2, dump_ticks / 2, num=(dump_ticks / 8) + 1)
        # num=fft_sliding_window+1)
        # Offset into tick array to step impulse.
        tckar_idx = len(tick_array) / 2
        tckar_upper_idx = len(tick_array) - 1
        tckar_lower_idx = 0
        future_ticks = dump_ticks * future_dump
        found = False
        # prev_imp_loc = 0
        first_run = True
        split_found = False
        single_step = False
        while not found:
            if manual:
                found = True
                offset = manual_offset
            else:
                offset = tick_array[int(tckar_idx)]
            dump = self.receiver.get_clean_dump()
            print dump["timestamp"]
            dump_ts = dump["timestamp"]
            dump_abs_t = datetime.datetime.fromtimestamp(sync_time + dump_ts / scale_factor_timestamp)
            print "Start dump time = {}:{}.{}".format(dump_abs_t.minute, dump_abs_t.second, dump_abs_t.microsecond)
            load_timestamp = dump_ts + future_ticks
            load_dsim_impulse(load_timestamp, offset)
            dump_list = []
            cnt = 0
            for i in range(future_dump):
                cnt += 1
                dump = self.receiver.data_queue.get()
                print dump["timestamp"]
                dval = dump["xeng_raw"]
                auto_corr = dval[:, inp_autocorr_idx, :]
                curr_ts = dump["timestamp"]
                delta_ts = curr_ts - dump_ts
                dump_ts = curr_ts
                if delta_ts != dump_ticks:
                    self.Failed(
                        "Accumulation dropped, Expected timestamp = {}, "
                        "received timestamp = {}".format(dump_ts + dump_ticks, curr_ts)
                    )
                print "Maximum value found in dump {} = {}, average = {}".format(
                    cnt, np.max(auto_corr), np.average(auto_corr)
                )
                dump_list.append(dval)
            # Find dump containing impulse, check that other dumps are empty.
            # val_found = 0
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
            if dumps_nzero == 2:
                self.Step("Two dumps found containing impulse.")
                # Only start stepping by one once the split is close
                # split_found = True
            elif dumps_nzero > 2:
                self.Failed("Invalid data found in dumps.")
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
                    self.Failed("Impulse not where expected.")
                    found = True
            else:
                idx_diff = abs(tckar_idx_prev - tckar_idx)
                tckar_idx_prev = tckar_idx
                if single_step and (dumps_nzero == 1):
                    found = True
                    print "Edge of dump found at offset {} (ticks)".format(offset)
                elif ((idx_diff < 10) and (dumps_nzero == 2)) or single_step:
                    single_step = True
                    tckar_idx += 1
                elif imp_loc == future_dump - 1:
                    tckar_lower_idx = tckar_idx
                    tckar_idx = tckar_idx + (tckar_upper_idx - tckar_idx) / 2
                elif imp_loc == future_dump:
                    tckar_upper_idx = tckar_idx
                    tckar_idx = tckar_idx - (tckar_idx - tckar_lower_idx) / 2
                else:
                    self.Failed("Impulse not where expected.")
                    found = True
            print "Tick array index = {}, Diff = {}".format(tckar_idx, tckar_idx - tckar_idx_prev)

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
            self.Step("Checking timestamp accuracy: {}\n".format(self.corr_fix.get_running_instrument()))
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
            lt_abs_t = datetime.datetime.fromtimestamp(sync_time + load_timestamp / scale_factor_timestamp)
            print "Impulse load time = {}:{}.{}".format(lt_abs_t.minute, lt_abs_t.second, lt_abs_t.microsecond)
            print "Number of dumps in future = {:.10f}".format((load_timestamp - dump_ts) / dump_ticks)
            # Digitiser simulator local clock factor of 8 slower
            # (FPGA clock = sample clock / 8).
            load_timestamp = load_timestamp / 8
            if not load_timestamp.is_integer():
                self.Failed("Timestamp received in accumulation not divisible" " by 8: {:.15f}".format(load_timestamp))
            load_timestamp = int(load_timestamp)
            reg_size = 32
            load_ts_lsw = load_timestamp & (pow(2, reg_size) - 1)
            load_ts_msw = load_timestamp >> reg_size
            self.dhost.registers.impulse_load_time_lsw.write(reg=load_ts_lsw)
            self.dhost.registers.impulse_load_time_msw.write(reg=load_ts_msw)

        dsim_set_success = self.set_input_levels(
            self.corr_fix, self.dhost, awgn_scale=0.0, cw_scale=0.0, freq=100000000, fft_shift=0, gain="32767+0j"
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        self.dhost.outputs.out_1.scale_output(0)
        dump = self.receiver.get_clean_dump()
        baseline_lookup = self.get_baselines_lookup(dump)
        sync_time = self.cam_sensors.get_value("sync_epoch")
        scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
        inp = self.cam_sensors.input_labels[0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        # fft_sliding_window = self.n_chans_selected * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = self.cam_sensors.get_value("int_time") * self.cam_sensors.get_value("adc_sample_rate")
        dump_ticks = self.cam_sensors.get_value("n_accs") * self.n_chans_selected * 2
        input_spec_ticks = self.n_chans_selected * 2
        if not (dump_ticks / 8.0).is_integer():
            self.Failed("Number of ticks per dump is not divisible" " by 8: {:.3f}".format(dump_ticks))
        future_ticks = dump_ticks * future_dump
        shift_set = [[[], []] for x in range(5)]
        for shift in range(len(shift_set)):
            set_offset = offset + 1024 * shift
            list0 = []
            list1 = []
            for step in range(shift_nr):
                set_offset = set_offset + input_spec_ticks
                dump = self.receiver.get_clean_dump()
                dump_ts = dump["timestamp"]
                sync_time = self.cam_sensors.get_value("sync_epoch")
                scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
                dump_abs_t = datetime.datetime.fromtimestamp(sync_time + dump_ts / scale_factor_timestamp)
                print "Start dump time = {}:{}.{}".format(dump_abs_t.minute, dump_abs_t.second, dump_abs_t.microsecond)
                load_timestamp = dump_ts + future_ticks
                load_dsim_impulse(load_timestamp, set_offset)
                dump_list = []
                cnt = 0
                for i in range(future_dump):
                    cnt += 1
                    dump = self.receiver.data_queue.get()
                    print dump["timestamp"]
                    dval = dump["xeng_raw"]
                    auto_corr = dval[:, inp_autocorr_idx, :]
                    curr_ts = dump["timestamp"]
                    delta_ts = curr_ts - dump_ts
                    dump_ts = curr_ts
                    if delta_ts != dump_ticks:
                        self.Failed(
                            "Accumulation dropped, Expected timestamp = {}, "
                            "received timestamp = {}"
                            "".format(dump_ts + dump_ticks, curr_ts)
                        )
                    print "Maximum value found in dump {} = {}, average = {}" "".format(
                        cnt, np.max(auto_corr), np.average(auto_corr)
                    )
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
    #         assert evaluate(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument()
    #         _running_inst = self.corr_fix.get_running_instrument()
    #         if instrument_success:
    #             fft_shift = pow(2, 15) - 1
    #             self._set_input_levels_and_gain(profile='cw', cw_freq=200000000, cw_margin=0.6,
    #                                             trgt_bits=5, trgt_q_std=0.30, fft_shift=fft_shift)
    #         else:
    #             self.Failed(self.errmsg)

    def _small_voltage_buffer(self):
        channel_list = self.cam_sensors.ch_center_freqs
        # Choose a frequency 3 quarters through the band
        cw_chan_set = int(self.n_chans_selected * 3 / 4)
        cw_freq = channel_list[cw_chan_set]
        dsim_clk_factor = 1.712e9 / self.cam_sensors.sample_period
        bandwidth = self.cam_sensors.get_value("bandwidth")
        # eff_freq = (cw_freq + bandwidth) * dsim_clk_factor
        channel_bandwidth = self.cam_sensors.delta_f
        input_labels = self.cam_sensors.input_labels

        if "4k" in self.instrument:
            # 4K
            cw_scale = 0.675
            awgn_scale = 0.05
            gain = "11+0j"
            fft_shift = 8191
        else:
            # 32K
            cw_scale = 0.375
            awgn_scale = 0.085
            gain = "11+0j"
            fft_shift = 32767

        self.Step(
            "Digitiser simulator configured to generate a continuous wave at %s Hz (channel=%s),"
            " with cw scale: %s, awgn scale: %s, eq gain: %s, fft shift: %s"
            % (cw_freq, cw_chan_set, cw_scale, awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
            freq=cw_freq, fft_shift=fft_shift, gain=gain, cw_src=0
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False

        try:
            self.Step("Confirm that the `Transient Buffer ready` is implemented.")
            reply, informs = self.katcp_req.transient_buffer_trigger(timeout=120)
            self.assertTrue(reply.reply_ok())
            self.Passed("Transient buffer trigger present.")
        except Exception:
            self.Failed("Transient buffer trigger failed. \nReply: %s" % str(reply).replace("_", " "))

        try:
            self.Step("Randomly select an input to capture ADC snapshot")
            input_label = random.choice(input_labels)
            self.Progress("Selected input %s to capture ADC snapshot from" % input_label)
            self.Step("Capture an ADC snapshot and confirm the fft length")
            reply, informs = self.katcp_req.adc_snapshot(input_label, timeout=60)
            self.assertTrue(reply.reply_ok())
            informs = informs[0]
        except Exception:
            self.Error("Failed to capture ADC snapshot. \nReply: %s" % str(reply).replace("_", " "),
                exc_info=True)
            return
        else:
            adc_data = evaluate(informs.arguments[-1])
            fft_len = len(adc_data)
            self.Progress("ADC capture length: {}".format(fft_len))
            fft_real = np.abs(np.fft.fft(adc_data))
            fft_pos = fft_real[0 : int(fft_len / 2)]
            cw_chan = np.argmax(fft_pos)
            cw_freq_found = cw_chan / (fft_len / 2) * bandwidth
            msg = (
                "Confirm that the expected frequency: {}Hz and measured frequency: "
                "{}Hz matches to within a channel bandwidth: {:.3f}Hz".format(cw_freq_found,
                    cw_freq, channel_bandwidth)
            )
            Aqf.almost_equals(cw_freq_found, cw_freq, channel_bandwidth, msg)
            aqf_plot_channels(
                np.log10(fft_pos),
                plot_filename="{}/{}_fft_{}.png".format(self.logs_path, self._testMethodName, input_label),
                plot_title=(
                    "Input Frequency = %s Hz\nMeasured Frequency at FFT bin %s "
                    "= %sHz" % (cw_freq, cw_chan, cw_freq_found)
                ),
                log_dynamic_range=None,
                caption=(
                    "FFT of captured small voltage buffer. %s voltage points captured "
                    "on input %s. Input bandwidth = %sHz" % (fft_len, input_label, bandwidth)
                ),
                xlabel="FFT bins",
            )

    def _test_efficiency(self):

        csv_filename = "/".join([self._katreport_dir, r"CBF_Efficiency_Data.csv"])

        def get_samples():

            n_chans = self.cam_sensors.get_value("n_chans")
            test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
            requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=101, chans_around=2)
            expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
            # Get baseline 0 data, i.e. auto-corr of m000h
            test_baseline = 0
            # [CBF-REQ-0053]
            min_bandwithd_req = 770e6
            # Channel magnitude responses for each frequency
            chan_responses = []
            last_source_freq = None
            print_counts = 3
            req_chan_spacing = 250e3

            if "4k" in self.instrument:
                # 4K
                cw_scale = 0.675
                awgn_scale = 0.05
                gain = "11+0j"
                fft_shift = 8191
            else:
                # 32K
                cw_scale = 0.375
                awgn_scale = 0.085
                gain = "11+0j"
                fft_shift = 32767

            self.Step(
                "Digitiser simulator configured to generate a continuous wave, "
                "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                    cw_scale, awgn_scale, gain, fft_shift
                )
            )
            dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
                freq=expected_fc, fft_shift=fft_shift, gain=gain
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False
            try:
                self.Step(
                    "Randomly select a frequency channel to test. Capture an initial correlator "
                    "SPEAD accumulation, determine the number of frequency channels"
                )
                initial_dump = self.receiver.get_clean_dump()
                self.assertIsInstance(initial_dump, dict)
            except Exception:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
                self.Failed(errmsg)
            else:

                bls_to_test = evaluate(self.cam_sensors.get_value("bls_ordering"))[test_baseline]
                self.Progress(
                    "Randomly selected frequency channel to test: {} and "
                    "selected baseline {} / {} to test.".format(test_chan, test_baseline, bls_to_test)
                )
                Aqf.equals(
                    np.shape(initial_dump["xeng_raw"])[0],
                    self.n_chans_selected,
                    "Confirm that the number of channels in the SPEAD accumulation, is equal "
                    "to the number of frequency channels as calculated: {}".format(
                        np.shape(initial_dump["xeng_raw"])[0]
                    ),
                )

                Aqf.is_true(
                    self.cam_sensors.get_value("bandwidth") >= min_bandwithd_req,
                    "Channelise total bandwidth {}Hz shall be >= {}Hz.".format(
                        self.cam_sensors.get_value("bandwidth"), min_bandwithd_req
                    ),
                )
                chan_spacing = self.cam_sensors.get_value("bandwidth") / n_chans
                chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100), chan_spacing + (chan_spacing * 1 / 100)]
                self.Step("Confirm that the number of calculated channel " "frequency step is within requirement.")
                msg = "Verify that the calculated channel " "frequency ({} Hz)step size is between {} and {} Hz".format(
                    chan_spacing, req_chan_spacing / 2, req_chan_spacing
                )
                Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

                self.Step(
                    "Confirm that the channelisation spacing and confirm that it is " "within the maximum tolerance."
                )
                msg = "Channelisation spacing is within maximum tolerance of 1% of the " "channel spacing."
                Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

            self.Step(
                "Sweep the digitiser simulator over the centre frequencies of at "
                "least all the channels that fall within the complete L-band"
            )

            for i, freq in enumerate(requested_test_freqs):
                if i < print_counts:
                    self.Progress(
                        "Getting channel response for freq {} @ {}: {:.3f} MHz.".format(
                            i + 1, len(requested_test_freqs), freq / 1e6
                        )
                    )
                elif i == print_counts:
                    self.Progress("." * print_counts)
                elif i >= (len(requested_test_freqs) - print_counts):
                    self.Progress(
                        "Getting channel response for freq {} @ {}: {:.3f} MHz.".format(
                            i + 1, len(requested_test_freqs), freq / 1e6
                        )
                    )
                else:
                    self.logger.debug(
                        "Getting channel response for freq %s @ %s: %s MHz."
                        % (i + 1, len(requested_test_freqs), freq / 1e6)
                    )

                self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
                this_source_freq = self.dhost.sine_sources.sin_0.frequency

                if this_source_freq == last_source_freq:
                    self.logger.debug(
                        "Skipping channel response for freq %s @ %s: %s MHz.\n"
                        "Digitiser frequency is same as previous." % (i + 1, len(requested_test_freqs), freq / 1e6)
                    )
                    continue  # Already calculated this one
                else:
                    last_source_freq = this_source_freq

                try:
                    this_freq_dump = self.receiver.get_clean_dump()
                    # self.receiver.get_clean_dump()
                    self.assertIsInstance(this_freq_dump, dict)
                except AssertionError:
                    self.Error("Could not retrieve clean SPEAD accumulation", exc_info=True)
                    return False
                else:
                    # No of spead heap discards relevant to vacc
                    discards = 0
                    max_wait_dumps = 100
                    deng_timestamp = self.dhost.registers.sys_clkcounter.read().get("timestamp")
                    while True:
                        try:
                            queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                            self.assertIsInstance(queued_dump, dict)
                        except Exception:
                            self.Error("Could not retrieve clean accumulation.", exc_info=True)
                        else:
                            timestamp_diff = np.abs(queued_dump["dump_timestamp"] - deng_timestamp)
                            if timestamp_diff < 0.5:
                                msg = (
                                    "Received correct accumulation timestamp: %s, relevant to "
                                    "DEngine timestamp: %s (Difference %.2f)"
                                    % (queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                                )
                                self.logger.info(msg)
                                break

                            if discards > max_wait_dumps:
                                errmsg = (
                                    "Could not get accumulation with correct timestamp within %s "
                                    "accumulation periods." % max_wait_dumps
                                )
                                self.Failed(errmsg)
                                break
                            else:
                                msg = (
                                    "Discarding subsequent dumps (%s) with dump timestamp (%s) "
                                    "and DEngine timestamp (%s) with difference of %s."
                                    % (discards, queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                                )
                                self.logger.info(msg)
                        discards += 1

                    this_freq_response = normalised_magnitude(queued_dump["xeng_raw"][:, test_baseline, :])
                    chan_responses.append(this_freq_response)

            chan_responses = np.array(chan_responses)
            requested_test_freqs = np.asarray(requested_test_freqs)
            csv_filename = "/".join([self._katreport_dir, r"CBF_Efficiency_Data.csv"])
            np.savetxt(csv_filename, zip(chan_responses[:, test_chan], requested_test_freqs), delimiter=",")

        def efficiency_calc(f, P_dB, binwidth, debug=False):
            # Adapted from SSalie
            # Sidelobe & scalloping loss requires f to be normalized to bins
            # Normalize the filter response
            self.Step("Measure/record the filter-bank spectral response from a channel")
            P_dB -= P_dB.max()
            f = f - f[P_dB > -3].mean()  # CHANGED: center on zero

            # It's critical to get precise estimates of critical points so to minimize measurement
            # resolution impact, up-sample!
            _f_, _P_dB_ = f, P_dB
            _f10_ = np.linspace(f[0], f[-1], len(f) * 10)  # up-sample 10x
            # CHANGED: slightly better than np.interp(_f10_, f, P_dB) e.g. for poorly sampled data
            P_dB = scipy.interpolate.interp1d(f, P_dB, "quadratic", bounds_error=False)(_f10_)
            f = _f10_

            # Measure critical bandwidths
            f_HPBW = f[P_dB >= -3.0]
            # CHANGED: with better interpolation don't need earlier "fudged" 3.05 & 6.05
            f_HABW = f[P_dB >= -6.0]
            HPBW = (f_HPBW[-1] - f_HPBW[0]) / binwidth
            HABW = (f_HABW[-1] - f_HABW[0]) / binwidth
            h = 10 ** (P_dB / 10.0)
            NEBW = np.sum(h[:-1] * np.diff(f)) / binwidth  # Noise Equivalent BW
            self.Step(
                "Determine the Half Power Bandwidth as well as the Noise Equivalent Bandwidth " "for each swept channel"
            )
            self.Progress("Half Power Bandwidth: %s, Noise Equivalent Bandwidth: %s" % (HPBW, NEBW))

            self.Step(
                "Compute the efficiency as the ratio of Half Power Bandwidth to the Noise "
                "Equivalent Bandwidth: efficiency = HPBW/NEBW"
            )
            _efficiency = HPBW / NEBW
            Aqf.more(_efficiency, 0.98, "Efficiency factor = {:.3f}".format(_efficiency))
            # Measure critical points
            pk = f.searchsorted(f[P_dB > -6].mean())  # The peak
            # Channel-to-channel separation intervals
            ch = f.searchsorted(f[0] + binwidth)
            # Scalloping loss at mid-point between channel peaks
            SL = P_dB[pk + ch // 2 - 1]
            # Max scalloping loss within 80% of a channel
            SL80 = P_dB[pk : pk + int((0.8 * ch) // 2 - 1)].min()
            # Smooth it over 1/8th of a bin width to get rid of main lobe ripples
            DDP = np.diff(scipy.signal.medfilt(np.diff(P_dB), (ch // 16) * 2 + 1))
            # The first large inflection point after the peak is the null
            mn = pk + ch // 2 + (DDP[pk + ch // 2 :] > 0.01).argmax()
            # The nearest one is typically the peak sidelobe
            SLL = P_dB[mn:].max()
            # Upper half of the channel & the excluding main lobe
            plt.figure()
            plt.subplot(211)
            plt.title("Efficiency factor = {:.3f}".format(_efficiency))
            plt.plot(_f_, _P_dB_, label="Channel Response")
            plt.plot(f[pk:], P_dB[pk:], "g.", label="Peak")
            plt.plot(f[mn:], P_dB[mn:], "r.", label="After Null")
            plt.legend()
            plt.grid(True)
            plt.subplot(212, sharex=plt.gca())
            plt.plot(f[1:-1], DDP, label="Data diff")
            plt.grid(True)
            plt.legend()
            if debug:
                plt.show()

            cap = "SLL = %.f, SL = %.1f(%.f), NE/3dB/6dB BW = %.2f/%.2f/%.2f, HPBW/NEBW = %4f, " % (
                SLL,
                SL,
                SL80,
                NEBW,
                HPBW,
                HABW,
                HPBW / NEBW,
            )
            filename = "{}/{}.png".format(self.logs_path, self._testMethodName)
            Aqf.matplotlib_fig(filename, caption=cap, autoscale=True)

        try:
            pfb_data = np.loadtxt(csv_filename, delimiter=",", unpack=False)
            self.Step("Retrieved channelisation (Frequencies and Power_dB) data results from CSV file")
        except IOError:
            try:
                get_samples()
                csv_file = max(glob.iglob(csv_filename), key=os.path.getctime)
                assert "CBF" in csv_file
                pfb_data = np.loadtxt(csv_file, delimiter=",", unpack=False)
            except Exception:
                msg = "Failed to load CBF_Efficiency_Data.csv file"
                self.Error(msg, exc_info=True)
                return

        chan_responses, requested_test_freqs = pfb_data[:, 0][1:], pfb_data[:, 1][1:]
        # Summarize isn't clever enough to cope with the spurious spike in first sample
        requested_test_freqs = np.asarray(requested_test_freqs)
        chan_responses = 10 * np.log10(np.abs(np.asarray(chan_responses)))
        try:
            binwidth = self.cam_sensors.get_value("bandwidth") / (self.n_chans_selected - 1)
            efficiency_calc(requested_test_freqs, chan_responses, binwidth)
        except Exception:
            msg = "Could not compute the data, rerun test"
            self.Error(msg, exc_info=True)
            self.Failed(msg)
        # else:
        #     subprocess.check_call(["rm", csv_filename])

    def _test_product_baseline_leakage(self):
        heading("CBF Baseline Correlation Product Leakage")
        if "4k" in self.instrument:
            # 4K
            awgn_scale = 0.0645
            gain = "113+0j"
            fft_shift = 511
        else:
            # 32K
            awgn_scale = 0.063
            gain = "344+0j"
            fft_shift = 4095

        self.Step(
            "Digitiser simulator configured to generate Gaussian noise, "
            "with scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, corr_noise=False,
            fft_shift=fft_shift, gain=gain
        )

        self.Step(
            "Capture an initial correlator SPEAD accumulation, and retrieve list "
            "of all the correlator input labels via Cam interface."
        )
        try:
            test_dump = self.receiver.get_clean_dump(discard=50)
            self.assertIsInstance(test_dump, dict)
        except AssertionError:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
        else:
            # Get bls ordering from get dump
            self.Step(
                "Get list of all possible baselines (including redundant baselines) present "
                "in the correlator output from SPEAD accumulation"
            )

            bls_ordering = evaluate(self.cam_sensors.get_value("bls_ordering"))
            input_labels = sorted(self.cam_sensors.input_labels)
            inputs_to_plot = random.shuffle(input_labels)
            inputs_to_plot = input_labels[:8]
            bls_to_plot = [0, 2, 4, 8, 11, 14, 23, 33]
            baselines_lookup = self.get_baselines_lookup()
            present_baselines = sorted(baselines_lookup.keys())
            possible_baselines = set()
            _ = [possible_baselines.add((li, lj)) for li in input_labels for lj in input_labels]

            test_bl = sorted(list(possible_baselines))
            self.Step(
                "Confirm that each baseline (or its reverse-order counterpart) is present in " "the correlator output"
            )

            baseline_is_present = {}
            for test_bl in possible_baselines:
                baseline_is_present[test_bl] = test_bl in present_baselines or test_bl[::-1] in present_baselines
            # Select some baselines to plot
            plot_baselines = (
                (input_labels[0], input_labels[0]),
                (input_labels[0], input_labels[1]),
                (input_labels[0], input_labels[2]),
                (input_labels[-1], input_labels[-1]),
                (input_labels[-1], input_labels[-2]),
            )
            plot_baseline_inds = []
            for bl in plot_baselines:
                if bl in baselines_lookup:
                    plot_baseline_inds.append(baselines_lookup[bl])
                else:
                    plot_baseline_inds.append(baselines_lookup[bl[::-1]])

            plot_baseline_legends = tuple(
                "{bl[0]}, {bl[1]}: {ind}".format(bl=bl, ind=ind) for bl, ind in zip(plot_baselines, plot_baseline_inds)
            )

            msg = "Confirm that all baselines are present in correlator output."
            Aqf.is_true(all(baseline_is_present.values()), msg)
            test_data = test_dump["xeng_raw"]
            self.Step(
                "Expect all baselines and all channels to be " "non-zero with Digitiser Simulator set to output AWGN."
            )
            msg = "Confirm that no baselines have all-zero visibilities."
            Aqf.is_false(zero_baselines(test_data), msg)

            msg = "Confirm that all baseline visibilities are non-zero across all channels"
            Aqf.equals(nonzero_baselines(test_data), all_nonzero_baselines(test_data), msg)

            def prt_arr(array, print_len=4):
                try:
                    if len(array) < print_len:
                        print_len = len(array)
                        out_arr = array[:print_len]
                        out_arr = ", ".join([str(e) for e in out_arr])
                    else:
                        out_arr = array[:print_len]
                        out_arr = ", ".join([str(e) for e in out_arr]) + ", ..."
                except BaseException:
                    out_arr = str(array)

                return out_arr

            ref_auto = True
            ref_x = True
            ref_y = True
            idnt = " " * 28
            auto_phase = []
            auto_mag = []
            cross_phase = []
            cross_mag = []
            for inputs, index in baselines_lookup.iteritems():
                # Auto correlations
                if inputs[0][-1] == inputs[1][-1]:
                    test_data_complex = complexise(test_data[:, index, :])
                    phase = np.angle(test_data_complex)
                    mag = np.abs(test_data_complex)
                    auto_phase.append(phase)
                    auto_mag.append(mag)
                    if ref_auto:
                        ref_auto_phase = phase
                        ref_auto_mag = mag
                        ref_auto = False
                        self.Step(
                            "Using {}, baseline {}, as an auto-correlation "
                            "reference with:\n{}\n{}".format(
                                inputs, index,
                                idnt + "Magnitude: " + prt_arr(mag),
                                idnt + "Phase:  " + prt_arr(phase)
                            )
                        )
                    else:
                        phase_match = ref_auto_phase == phase
                        mag_match = ref_auto_mag == mag
                        if not (np.all(mag_match)):
                            err_idx = np.where(np.invert(mag_match))
                            err_arr = np.take(mag, err_idx)[0]
                            ref_arr = np.take(ref_auto_mag, err_idx)[0]
                            err_idx = err_idx[0]
                            self.Failed(
                                "{}, baseline {}, auto-correlation magnitudes do "
                                "not match:\n{}\n{}\n{}".format(
                                    inputs,
                                    index,
                                    idnt + "Error indices:    " + prt_arr(err_idx),
                                    idnt + "Reference values: " + prt_arr(ref_arr),
                                    idnt + "Magnitude values: " + prt_arr(err_arr),
                                )
                            )
                        elif not (np.all(phase_match)):
                            err_idx = np.where(np.invert(phase_match))
                            err_arr = np.take(phase, err_idx)[0]
                            ref_arr = np.take(ref_auto_phase, err_idx)[0]
                            err_idx = err_idx[0]
                            self.Failed(
                                "{}, baseline {}, auto-correlation phases do not match:\n{}\n{}\n{}"
                                "".format(
                                    inputs,
                                    index,
                                    idnt + "Error indices:    " + prt_arr(err_idx),
                                    idnt + "Reference values: " + prt_arr(ref_arr),
                                    idnt + "Phase values:     " + prt_arr(err_arr),
                                )
                            )
                        else:
                            self.Passed(
                                "{}, baseline {}, is an auto-correlation, magnitude and phase matches:\n{}\n{}"
                                "".format(
                                    inputs,
                                    index,
                                    idnt + "Magnitude values: " + prt_arr(mag),
                                    idnt + "Phase values:     " + prt_arr(phase),
                                )
                            )

            for inputs, index in baselines_lookup.iteritems():
                # Cross correlations
                if inputs[0][-1] != inputs[1][-1]:
                    test_data_complex = complexise(test_data[:, index, :])
                    phase = np.angle(test_data_complex)
                    mag = np.abs(test_data_complex)
                    cross_phase.append(phase)
                    cross_mag.append(mag)
                    if inputs[0][-1] == "x":
                        if ref_x:
                            ref_phase_x = phase
                            ref_mag_x = mag
                            ref_x = False
                            self.Step(
                                "Using {}, baseline {}, as a x-pol cross-correlation reference with:\n{}\n{}"
                                "".format(
                                    inputs,
                                    index,
                                    idnt + "Magnitude: " + prt_arr(mag),
                                    idnt + "Phase:     " + prt_arr(phase),
                                )
                            )
                        else:
                            phase_match = ref_phase_x == phase
                            mag_match = ref_mag_x == mag
                            if not (np.all(mag_match)):
                                err_idx = np.where(np.invert(mag_match))
                                err_arr = np.take(mag, err_idx)[0]
                                ref_arr = np.take(ref_auto_mag, err_idx)[0]
                                err_idx = err_idx[0]
                                self.Failed(
                                    "{}, baseline {}, x-pol cross-correlation magnitudes do not match:\n{}\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Error indices:    " + prt_arr(err_idx),
                                        idnt + "Reference values: " + prt_arr(ref_arr),
                                        idnt + "Magnitude values: " + prt_arr(err_arr),
                                    )
                                )
                            elif not (np.all(phase_match)):
                                err_idx = np.where(np.invert(phase_match))
                                err_arr = np.take(phase, err_idx)[0]
                                ref_arr = np.take(ref_auto_phase, err_idx)[0]
                                err_idx = err_idx[0]
                                self.Failed(
                                    "{}, baseline {}, x-pol cross-correlation phases do not match:\n{}\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Error indices:    " + prt_arr(err_idx),
                                        idnt + "Reference values: " + prt_arr(ref_arr),
                                        idnt + "Phase values:     " + prt_arr(err_arr),
                                    )
                                )
                            else:
                                self.Passed(
                                    "{}, baseline {}, is a x-poll cross-correlation, magnitude and phase matches:\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Magnitude values: " + prt_arr(mag),
                                        idnt + "Phase values:     " + prt_arr(phase),
                                    )
                                )

                    else:
                        if ref_y:
                            ref_phase_y = phase
                            ref_mag_y = mag
                            ref_y = False
                            self.Step(
                                "Using {}, baseline {}, as a y-pol cross-correlation reference with:\n{}\n{}"
                                "".format(
                                    inputs,
                                    index,
                                    idnt + "Magnitude: " + prt_arr(mag),
                                    idnt + "Phase:     " + prt_arr(phase),
                                )
                            )
                        else:
                            phase_match = ref_phase_y == phase
                            mag_match = ref_mag_y == mag
                            if False and not (np.all(mag_match)):
                                err_idx = np.where(np.invert(mag_match))
                                err_arr = np.take(mag, err_idx)[0]
                                ref_arr = np.take(ref_auto_mag, err_idx)[0]
                                err_idx = err_idx[0]
                                self.Failed(
                                    "{}, baseline {}, y-pol cross-correlation magnitudes do not match:\n{}\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Error indices:    " + prt_arr(err_idx),
                                        idnt + "Reference values: " + prt_arr(ref_arr),
                                        idnt + "Magnitude values: " + prt_arr(err_arr),
                                    )
                                )
                            elif not (np.all(phase_match)):
                                err_idx = np.where(np.invert(phase_match))
                                err_arr = np.take(phase, err_idx)[0]
                                ref_arr = np.take(ref_auto_phase, err_idx)[0]
                                err_idx = err_idx[0]
                                self.Failed(
                                    "{}, baseline {}, y-pol cross-correlation phases do not match:\n{}\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Error indices:    " + prt_arr(err_idx),
                                        idnt + "Reference values: " + prt_arr(ref_arr),
                                        idnt + "Phase values:     " + prt_arr(err_arr),
                                    )
                                )
                            else:
                                self.Passed(
                                    "{}, baseline {}, is a y-poll cross-correlation, magnitude and phase matches:\n{}\n{}"
                                    "".format(
                                        inputs,
                                        index,
                                        idnt + "Magnitude values: " + prt_arr(mag),
                                        idnt + "Phase values:     " + prt_arr(phase),
                                    )
                                )

            plt_filename = "{}/{}_autocorrelation_channel_response.png".format(self.logs_path, self._testMethodName)
            plt_caption = "Channel responses for all auto correlation baselines."
            plt_title = "Channel responses for all auto correlation baselines."
            aqf_plot_channels(auto_mag, plot_filename=plt_filename, plot_title=plt_title)

    def _test_linearity(self, test_channel, cw_start_scale, noise_scale, gain, fft_shift, max_steps):
        # # Get instrument parameters
        # bw = self.cam_sensors.get_value('bandwidth')
        # nr_ch = self.cam_sensors.get_value('n_chans')
        # ants = self.cam_sensors.get_value('n_ants')
        # ch_bw = ch_list[1]
        # scale_factor_timestamp = self.cam_sensors.get_value('scale_factor_timestamp')
        # dsim_factor = (float(self.conf_file['instrument_params']['sample_freq'])/
        #                scale_factor_timestamp)
        # substreams = self.cam_sensors.get_value('n_xengs')

        ch_list = self.cam_sensors.ch_center_freqs

        def get_cw_val(cw_scale, noise_scale, gain, fft_shift, test_channel, inp, f_offset=50000):
            self.Step(
                "Digitiser simulator configured to generate a continuous wave, "
                "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                    cw_scale, noise_scale, gain, fft_shift
                )
            )
            dsim_set_success = self.set_input_levels(
                awgn_scale=noise_scale,
                cw_scale=cw_scale,
                freq=ch_list[test_channel] + f_offset,
                fft_shift=fft_shift,
                gain=gain,
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False

            try:
                dump = self.receiver.get_clean_dump(DUMP_TIMEOUT)
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
            try:
                baseline_lookup = self.get_baselines_lookup(dump)
                # Choose baseline for phase comparison
                baseline_index = baseline_lookup[(inp, inp)]
            except KeyError:
                self.Failed("Initial SPEAD accumulation does not contain correct baseline " "ordering format.")
                return False
            data = dump["xeng_raw"]
            freq_response = complexise(data[:, baseline_index, :])
            if freq_response[test_channel] == 0:
                return 0
            else:
                return 10 * np.log10(np.abs(freq_response[test_channel]))

        Aqf.hop("Requesting input labels.")
        try:
            # Build dictionary with inputs and
            # which fhosts they are associated with.
            reply, informs = self.katcp_req.input_labels()
            if reply.reply_ok():
                inp = reply.arguments[1:][0]
        except Exception:
            self.Failed("Failed to get input labels. KATCP Reply: {}".format(reply))
            return False
        Aqf.hop("Sampling input {}".format(inp))
        cw_scale = cw_start_scale
        cw_delta = 0.1
        threshold = 10 * np.log10(pow(2, 30))
        curr_val = threshold
        Aqf.hop("Finding starting cw input scale...")
        max_cnt = max_steps
        while (curr_val >= threshold) and max_cnt:
            prev_val = curr_val
            curr_val = get_cw_val(cw_scale, noise_scale, gain, fft_shift, test_channel, inp)
            cw_scale -= cw_delta
            if cw_scale < 0:
                max_cnt = 0
                cw_scale = 0
            else:
                max_cnt -= 1
        cw_start_scale = cw_scale + cw_delta
        Aqf.hop("Starting cw input scale set to {}".format(cw_start_scale))
        cw_scale = cw_start_scale
        output_power = []
        x_val_array = []
        # Find closes point to this power to place linear expected line.
        exp_step = 6
        exp_y_lvl = 70
        exp_y_dlt = exp_step / 2
        exp_y_lvl_lwr = exp_y_lvl - exp_y_dlt
        exp_y_lvl_upr = exp_y_lvl + exp_y_dlt
        exp_y_val = 0
        exp_x_val = 0
        min_cnt_val = 3
        min_cnt = min_cnt_val
        max_cnt = max_steps
        while min_cnt and max_cnt:
            curr_val = get_cw_val(cw_scale, noise_scale, gain, fft_shift, test_channel, inp)
            if exp_y_lvl_lwr < curr_val < exp_y_lvl_upr:
                exp_y_val = curr_val
                exp_x_val = 20 * np.log10(cw_scale)
            step = curr_val - prev_val
            if curr_val == 0:
                break
            if np.abs(step) < 0.2:
                min_cnt -= 1
            else:
                min_cnt = min_cnt_val
            x_val_array.append(20 * np.log10(cw_scale))
            self.Step("CW power = {}dB, Step = {}dB, channel = {}".format(curr_val, step, test_channel))
            prev_val = curr_val
            output_power.append(curr_val)
            cw_scale = cw_scale / 2
            max_cnt -= 1
        output_power = np.array(output_power)
        output_power_max = output_power.max()
        output_power = output_power - output_power_max
        exp_y_val = exp_y_val - output_power_max

        plt_filename = "{}_cbf_response_{}_{}_{}.png".format(self._testMethodName, gain, noise_scale, cw_start_scale)
        plt_title = "CBF Response (Linearity Test)"
        caption = (
            "Digitiser Simulator start scale: {}, end scale: {}. Scale "
            "halved for every step. FFT Shift: {}, Quantiser Gain: {}, "
            "Noise scale: {}".format(cw_start_scale, cw_scale * 2, fft_shift, gain, noise_scale)
        )
        m = 1
        c = exp_y_val - m * exp_x_val
        y_exp = []
        for x in x_val_array:
            y_exp.append(m * x + c)
        # import IPython;IPython.embed()
        aqf_plot_xy(
            zip(([x_val_array, output_power], [x_val_array, y_exp]), ["Response", "Expected"]),
            plt_filename,
            plt_title,
            caption,
            xlabel="Input Power [dBm]",
            ylabel="Integrated Output Power [dBfs]",
        )
        Aqf.end(passed=True, message="Linearity plot generated.")

    # def get_clean_dump(self):
    #     retries = 20
    #     while retries:
    #         retries -= 1
    #         try:
    #             dump = self.receiver.get_clean_dump(discard=0)
    #             assert hasattr(self.dhost.registers, "sys_clkcounter"), "Dhost is broken, missing sys_clkcounter"
    #             dhost_timestamp = self.dhost.registers.sys_clkcounter.read().get("timestamp")
    #             errmsg = "Queue is empty will retry (%s) ie EMPTY DUMPS!!!!!!!!!!!!!!!!!!!!!" % retries
    #             assert isinstance(dump, dict), errmsg
    #             discard = 0
    #             while True:
    #                 dump = self.receiver.data_queue.get(timeout=10)
    #                 assert isinstance(dump, dict), errmsg
    #                 dump_timestamp = dump["dump_timestamp"]
    #                 time_diff = np.abs(dump_timestamp - dhost_timestamp)
    #                 if time_diff < 1:
    #                     msg = (
    #                         "Yeyyyyyyyyy: Dump timestamp (%s) in-sync with digitiser sync epoch (%s)"
    #                         " [diff: %s] within %s retries and discarded %s dumps"
    #                         % (dump_timestamp, dhost_timestamp, time_diff, retries, discard)
    #                     )
    #                     self.logger.info(msg)
    #                     break
    #                 else:
    #                     msg = "Dump timestamp (%s) is not in-sync with digitiser sync epoch (%s) [diff: %s]" % (
    #                         dump_timestamp,
    #                         dhost_timestamp,
    #                         time_diff,
    #                     )
    #                     self.logger.info(msg)
    #                 if discard > 10:
    #                     errmsg = "Could not retrieve clean queued SPEAD accumulation."
    #                     raise AssertionError(errmsg)
    #                 discard += 1
    #         except AssertionError:
    #             errmsg = "Could not retrieve clean queued SPEAD accumulation."
    #             self.logger.warning(errmsg)
    #         except Queue.Empty:
    #             errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
    #             self.Error(errmsg, exc_info=True)
    #             if retries < 15:
    #                 self.logger.exception("Exiting brutally with no Accumulation")
    #                 return False
    #         else:
    #             return dump
