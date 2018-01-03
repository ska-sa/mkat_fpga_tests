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

import corr2
import csv
import gc
import katcp
import logging
import os
import Queue
import random
import socket
import subprocess
import sys
import telnetlib
import textwrap
import threading
import time
import unittest

import matplotlib.pyplot as plt
import ntplib
import numpy as np
import pandas as pd

from concurrent.futures import TimeoutError
from corr2.corr_rx import CorrRx
from corr2.fxcorrelator_xengops import VaccSynchAttemptsMaxedOut
from katcp.testutils import start_thread_with_cleanup

# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
# from memory_profiler import profile as DetectMemLeaks
from datetime import datetime

# Todo MM 07-09-2017
# perhaps import mkat_fpga_tests.utils as Utils
# and mkat_fpga_tests.aqf_utils as AQF_Utils instead
from mkat_fpga_tests import correlator_fixture

from mkat_fpga_tests.aqf_utils import *
from mkat_fpga_tests.utils import *
from nosekatreport import *
from descriptions import TestProcedure
from power_logger import PowerLogger

LOGGER = logging.getLogger('mkat_fpga_tests')
# LOGGER = logging.getLogger(__name__)

# How long to wait for a correlator dump to arrive in tests
DUMP_TIMEOUT = 10

# ToDo MM (2017-07-21) Improve the logging for debugging
have_subscribed = False
set_dsim_epoch = False
dsim_timeout = 60


@cls_end_aqf
@system('all')
class test_CBF(unittest.TestCase):
    """ Unit-testing class for mkat_fpga_tests"""

    receiver = None

    def setUp(self):
        global have_subscribed, set_dsim_epoch
        self._dsim_set = False
        self.corr_fix = correlator_fixture
        try:
            self.conf_file = self.corr_fix.test_config
            self.corr_fix.katcp_clt = self.conf_file['inst_param']['katcp_client']
        except Exception:
            errmsg = 'Failed to read test config file.'
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        try:
            self.dhost = self.corr_fix.dhost
            assert isinstance(self.dhost, corr2.dsimhost_fpga.FpgaDsimHost)
            assert self.dhost.is_running()
            self.dhost.get_system_information()
            self._dsim_set = True
        except AssertionError:
            errmsg = 'For some apparent reason, DSim is not running'
            LOGGER.exception(errmsg)
            Aqf.end(message=errmsg)
            sys.exit(errmsg)
        except Exception:
            errmsg = ('Failed to connect to retrieve digitiser simulator information, ensure that '
                      'the correct digitiser simulator is running.')
            LOGGER.exception(errmsg)
            Aqf.end(message=errmsg)
            sys.exit(errmsg)
        else:
            self.logs_path = None
            self.addCleanup(executed_by)
            self.logs_path = create_logs_directory(self)

            # See: https://docs.python.org/2/library/functions.html#super
            super(test_CBF, self).setUp()
            self._hosts = list(np.concatenate(
                [i.get('hosts', None).split(',') for i in self.corr_fix.corr_config.values()
                if i.get('hosts')]))
            if have_subscribed is False:
                subscribed = self.corr_fix.subscribe_multicast
                if subscribed:
                    LOGGER.info('Multicast subscription successful.')
                    have_subscribed = True
            if set_dsim_epoch is False:
                try:
                    assert isinstance(self.corr_fix.instrument, str)
                    # cbf_title_report(self.corr_fix.instrument)
                    # Disable warning messages(logs) once
                    disable_warnings_messages()
                    self.assertIsInstance(self.corr_fix.katcp_rct,
                        katcp.resource_client.ThreadSafeKATCPClientResourceWrapper)
                    reply, informs = self.corr_fix.katcp_rct.req.sensor_value('synchronisation-epoch')
                    assert reply.reply_ok()
                    sync_time = float(informs[0].arguments[-1])
                    assert isinstance(sync_time, float)
                    reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch(sync_time)
                    assert reply.reply_ok()
                    LOGGER.info('Digitiser sync epoch set to %s' %reply.arguments[-1])
                except AssertionError as e:
                    errmsg = 'Failed to set Digitiser sync epoch via CAM interface. %s' % str(e)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                except AttributeError as e:
                    errmsg = 'Attribute Error: %s' % (str(e))
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                else:
                    set_dsim_epoch = True

    def set_instrument(self, instrument, acc_time=0.5, queue_size=3, **kwargs):
        acc_timeout = 60
        self.errmsg = None
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)
        self.addCleanup(init_dsim_sources, self.dhost)
        try:
            self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
        except Exception:
            pass
        else:
            LOGGER.info('Spead2 capturing thread clean-up: %s' % threading._active)
            self.receiver.stop()
            self.receiver = None
            del self.receiver

        for retry in retryloop(3, timeout=30):
            try:
                instrument_state = self.corr_fix.ensure_instrument(instrument, **kwargs)
                self.errmsg = ('Could not initialise instrument or ensure running instrument: %s'
                    %instrument)
                assert instrument_state, self.errmsg
            except AssertionError:
                LOGGER.error(self.errmsg)
                try:
                    retry()
                except RetryError:
                    return False

        if self._dsim_set:
                Aqf.step('Configure a digitiser simulator to be used as input source to F-Engines.')
                Aqf.progress('Digitiser Simulator running on host: %s' % self.dhost.host)
        try:
            reply, informs = self.corr_fix.katcp_rct.req.accumulation_length(acc_time,
                timeout=acc_timeout)
            assert reply.reply_ok()
        except (TimeoutError, VaccSynchAttemptsMaxedOut):
            self.corr_fix.halt_array
            self.corr_fix.ensure_instrument(instrument)
            self.errmsg = ('Timed-Out/VACC did not trigger: Failed to set accumulation time within '
                           '%s, SubArray will be halted and restarted with next test' % (acc_timeout))
            LOGGER.error(self.errmsg)
            return False
        except AssertionError:
            self.errmsg = ('%s, Will try to re-initialise instrument: %s' % (str(reply), instrument))
            LOGGER.error(self.errmsg)
            self.corr_fix.halt_array
            self.corr_fix.ensure_instrument(instrument)
            with ignored(Exception):
                reply, informs = self.corr_fix.katcp_rct.req.accumulation_length(acc_time,
                timeout=acc_timeout)
        except Exception as e:
            self.errmsg = ('Failed to set accumulation time due to :%s' % str(e))
            self.corr_fix.halt_array
            LOGGER.exception(self.errmsg)
            return False
        else:
            acc_time = float(reply.arguments[-1])
            Aqf.step('Set and confirm accumulation period via CAM interface.')
            Aqf.progress('Accumulation time set to {:.3f} seconds'.format(acc_time))
            try:
                self.correlator = self.corr_fix.correlator
                self.assertIsInstance(self.correlator, corr2.fxcorrelator.FxCorrelator)
            except AssertionError, e:
                self.errmsg = 'Failed to instantiate a correlator object: %s' % str(e)
                LOGGER.error(self.errmsg)
                return False
            try:
                corrRx_port = int(self.conf_file['inst_param']['corr_rx_port'])
            except Exception:
                corrRx_port = 8888
                errmsg = ('Failed to retrieve corr rx port from config file.'
                          'Setting it to default port: %s' % (corrRx_port))
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
            try:

                output_product = parameters(self)['output_product']
                Aqf.step('Initiate SPEAD receiver on port %s, and CBF output product %s' % (
                    corrRx_port, output_product))
                if corrRx_port == 8888:
                    self.receiver = CorrRx(product_name=output_product,
                        port=corrRx_port, queue_size=queue_size)
                    LOGGER.info('Running lab testing and listening to corr2_servlet on localhost')
                else:
                    servlet_ip = str(self.conf_file['inst_param']['corr2_servlet_ip'])
                    servlet_port = int(self.corr_fix.katcp_rct.port)
                    LOGGER.info('Running site testing and listening to corr2_servlet on %s' %servlet_ip)
                    self.receiver = CorrRx(product_name=output_product, servlet_ip=servlet_ip,
                        servlet_port=servlet_port, port=corrRx_port, queue_size=queue_size)

                self.receiver.setName('CorrRx Thread')
                self.errmsg = 'Failed to create SPEAD data receiver'
                self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                start_thread_with_cleanup(self, self.receiver, timeout=10, start_timeout=1)
                self.errmsg = 'Failed to subscribe to multicast IPs.'
                _IP, _PORT = self.corr_fix.corr_config['xengine']['output_destinations_base'].split(':')
                assert self.receiver.confirm_multicast_subs(mul_ip=_IP) is 'Successful', self.errmsg
                self.errmsg = 'Spead Receiver not Running, possible '
                assert self.receiver.isAlive(), self.errmsg
            except AssertionError:
                LOGGER.error(self.errmsg)
                return False
            except Exception as e:
                Aqf.failed('%s' % str(e))
                LOGGER.exception('%s' % str(e))
                return False
            else:
                self.corr_freqs = CorrelatorFrequencyInfo(self.correlator.configd)
                self.corr_fix.start_x_data
                self.addCleanup(self.corr_fix.stop_x_data)
                self.addCleanup(gc.collect)
                self.addCleanup(self.receiver.stop)
                clear_host_status(self)
                # Run system tests before each test is ran
                # self.addCleanup(self._systems_tests)
                # self._systems_tests()
                return True

    @instrument_bc8n856M4k
    @aqf_vr('TP.C.1.19')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043")
    def test_bc8n856M4k_channelisation(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.2)
            _running_inst = self.corr_fix.get_running_instrument()
            self._testMethodDoc = TestProcedure.Channelisation
            if instrument_success and _running_inst:
                n_chans = self.corr_freqs.n_chans
                test_chan = random.randrange(start=n_chans % 100, stop=n_chans - 1)
                self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)
            else:
                Aqf.failed(self.errmsg)

    @instrument_bc16n856M4k
    @site_only
    @aqf_vr('TP.C.1.19')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_bc16n856M4k_channelisation(self, instrument='bc16n856M4k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                test_chan = random.randrange(self.corr_freqs.n_chans)
                self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)
            else:
                Aqf.failed(self.errmsg)

    @site_only
    @instrument_bc32n856M4k
    @aqf_vr('TP.C.1.19')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043")
    def test_bc32n856M4k_channelisation(self, instrument='bc32n856M4k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                n_chans = self.corr_freqs.n_chans
                test_chan = random.randrange(start=n_chans % 100, stop=n_chans - 1)
                self._test_channelisation(test_chan, no_channels=4096, req_chan_spacing=250e3)
            else:
                Aqf.failed(self.errmsg)

    @instrument_bc8n856M32k
    @aqf_vr('TP.C.1.20')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049")
    def test_bc8n856M32k_channelisation(self, instrument='bc8n856M32k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.5, force_reinit=True)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                test_chan = random.randrange(self.corr_freqs.n_chans)
                self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
            else:
                Aqf.failed(self.errmsg)

    @site_only
    @instrument_bc16n856M32k
    @aqf_vr('TP.C.1.20')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049")
    def test_bc16n856M32k_channelisation(self, instrument='bc16n856M32k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.5)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                test_chan = random.randrange(self.corr_freqs.n_chans)
                self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
            else:
                Aqf.failed(self.errmsg)

    @site_only
    @instrument_bc32n856M32k
    @aqf_vr('TP.C.1.20')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049")
    def test_bc32n856M32k_channelisation(self, instrument='bc32n856M32k'):
        Aqf.procedure(TestProcedure.Channelisation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.5)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                test_chan = random.randrange(self.corr_freqs.n_chans)
                self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
            else:
                Aqf.failed(self.errmsg)

    @slow
    @instrument_bc8n856M4k
    @aqf_vr('TP.C.1.19', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc8n856M4k_channelisation_sfdr_peaks(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.2)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @site_only
    @instrument_bc16n856M4k
    @aqf_vr('TP.C.1.19', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc16n856M4k_channelisation_sfdr_peaks(self, instrument='bc16n856M4k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.3)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @site_only
    @instrument_bc32n856M4k
    @aqf_vr('TP.C.1.19', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc32n856M4k_channelisation_sfdr_peaks(self, instrument='bc32n856M4k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=250e3, no_channels=4096)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @instrument_bc8n856M32k
    @aqf_vr('TP.C.1.20', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc8n856M32k_channelisation_sfdr_peaks(self, instrument='bc8n856M32k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.2)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @site_only
    @instrument_bc16n856M32k
    @aqf_vr('TP.C.1.20', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc16n856M32k_channelisation_sfdr_peaks(self, instrument='bc16n856M32k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.5)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @slow
    @site_only
    @instrument_bc32n856M32k
    @aqf_vr('TP.C.1.20', 'TP.C.4.1')
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0053", "CBF-REQ-0050")
    @aqf_requirements("CBF-REQ-0049", "CBF-REQ-0164", "CBF-REQ-0191")
    def test_bc32n856M32k_channelisation_sfdr_peaks(self, instrument='bc32n856M32k'):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        Aqf.procedure(TestProcedure.PowerConsumption)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, acc_time=0.5)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=32768)  # Hz
            else:
                Aqf.failed(self.errmsg)

    @instrument_bc8n856M4k
    @aqf_vr('TP.C.1.37', 'TP.C.1.51', 'TP.C.1.35')
    @aqf_requirements("CBF-REQ-0117", "CBF-REQ-0094", "CBF-REQ-0118", "CBF-REQ-0123", "CBF-REQ-0183")
    def test_bc8n856M4k_beamforming(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_beamforming()
            else:
                Aqf.failed(self.errmsg)

    @site_only
    @instrument_bc16n856M4k
    @aqf_vr('TP.C.1.37', 'TP.C.1.51', 'TP.C.1.35')
    @aqf_requirements("CBF-REQ-0117", "CBF-REQ-0094", "CBF-REQ-0118", "CBF-REQ-0123", "CBF-REQ-0092")
    @aqf_requirements("CBF-REQ-0183")
    def test_bc16n856M4k_beamforming(self, instrument='bc16n856M4k'):
        Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, force_reinit=True)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_beamforming()
            else:
                Aqf.failed(self.errmsg)

    @site_only
    @instrument_bc32n856M4k
    @aqf_vr('TP.C.1.37', 'TP.C.1.51', 'TP.C.1.35')
    @aqf_requirements("CBF-REQ-0117", "CBF-REQ-0094", "CBF-REQ-0118", "CBF-REQ-0123", "CBF-REQ-0092")
    @aqf_requirements("CBF-REQ-0183")
    def test_bc32n856M4k_beamforming(self, instrument='bc32n856M4k'):
        Aqf.procedure(TestProcedure.Beamformer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            instrument_success = self.set_instrument(instrument, force_reinit=True)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_beamforming()
            else:
                Aqf.failed(self.errmsg)

    # def _test_bc8n856M32k_input_levels(self, instrument='bc8n856M32k'):
    #     """
    #     Testing Digitiser simulator input levels
    #     Set input levels to requested values and check that the ADC and the
    #     quantiser block do not see saturated samples.
    #     """
    #     Aqf.procedure(TestProcedure.Channelisation)
    #     try:
    #         assert eval(os.getenv('DRY_RUN', 'False'))
    #     except AssertionError:
    #         instrument_success = self.set_instrument(instrument)
    #         _running_inst = self.corr_fix.get_running_instrument()
    #         if instrument_success and _running_inst:
    #             fft_shift = pow(2, 15) - 1
    #             self._set_input_levels_and_gain(profile='cw', cw_freq=200000000, cw_margin=0.6,
    #                                             trgt_bits=5, trgt_q_std=0.30, fft_shift=fft_shift)
    #         else:
    #             Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.30')
    @aqf_requirements("CBF-REQ-0087", "CBF-REQ-0225", "CBF-REQ-0104")
    def test__baseline_correlation_product(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.BaselineCorrelation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst, acc_time=0.5)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_product_baselines()
                self._test_back2back_consistency()
                self._test_freq_scan_consistency()
                self._test_spead_verify()
                # self._test_restart_consistency(instrument, no_channels=4096)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.33')
    @aqf_requirements("CBF-REQ-0030", "CBF-REQ-0002", "CBF-REQ-0206", "CBF-REQ-0216")
    @aqf_requirements("CBF-REQ-0120", "CBF-REQ-0213", "CBF-REQ-0223", "CBF-REQ-0006", "CBF-REQ-0092")
    def test__data_product(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.DataProduct)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_data_product(instrument)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.31')
    @aqf_requirements("CBF-REQ-0096")
    def test__accumulation_length(self, instrument='bc8n856M4k'):
        # The CBF shall set the Baseline Correlation Products accumulation interval to a fixed time
        # in the range $$500 +0 -20ms$$.
        Aqf.procedure(TestProcedure.VectorAcc)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst, acc_time=0.99)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                Aqf.step('Testing maximum channels to 4096 due to quantiser snap-block limitations.')
                chan_index = 4096
                test_timeout = 3000000
                test_chan = random.randrange(chan_index)
                with RunTestWithTimeout(test_timeout):
                    self._test_vacc(test_chan, chan_index)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.29')
    @aqf_requirements("CBF-REQ-0119")
    def test__gain_correction(self, instrument='bc8n856M4k'):
        # The CBF shall apply gain correction per antenna, per polarisation, per frequency channel
        # with a range of at least $$\pm 6 \; dB$$ and a resolution of $$\le 1 \; db$$.
        Aqf.procedure(TestProcedure.GainCorr)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_gain_correction()
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.40')
    @aqf_requirements("CBF-REQ-0013")
    def test__product_switch(self, instrument='bc8n856M4k'):
        # The CBF shall, on request via the CAM interface, switch between Sub-Array data product
        #  combinations, using the same combination of Receptors, in less than 60 seconds.
        Aqf.procedure(TestProcedure.ProductSwitching)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                test_timeout = 300
                with RunTestWithTimeout(test_timeout):
                    self._test_product_switch(instrument)
            else:
                Aqf.failed(self.errmsg)


    @generic_test
    @aqf_vr('TP.C.1.43')
    @aqf_requirements("CBF-REQ-0071", "CBF-REQ-0204")
    def test__control(self, instrument='bc8n856M4k'):
        # The CBF shall, on request via the CAM interface, set the following parameters:
        #     a) Downconversion frequency
        #     b) Channelisation configuration
        #     c) Accumulation interval
        #     d) Re-quantiser settings (Gain)
        #     e) Complex gain correction
        #     f) Polarisation correction.
        # The CBF shall, on request via the CAM interface, report the requested setting of each
        # control parameter.
        Aqf.procedure(TestProcedure.Control)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            self._test_control_vr()

    @aqf_vr('TP.C.1.24')
    @generic_test
    # @aqf_vr('TP.C.1.24', 'TP.C.1.49', 'TP.C.1.54')
    @aqf_requirements("CBF-REQ-0066", "CBF-REQ-0072", "CBF-REQ-0077", "CBF-REQ-0110", "CBF-REQ-0200")
    @aqf_requirements("CBF-REQ-0112", "CBF-REQ-0128", "CBF-REQ-0185", "CBF-REQ-0187", "CBF-REQ-0188")
    def test__delay_phase_compensation(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_delay_tracking()
                self._test_delay_rate()
                self._test_fringe_rate()
                self._test_fringe_offset()
                self._test_delay_inputs()
                self._test_min_max_delays()
                clear_all_delays(self)
                restore_src_names(self)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.17')
    @aqf_requirements("CBF-REQ-0178")
    def test__report_configuration(self, instrument='bc8n856M4k'):
        # The CBF shall, on request via the CAM interface, report sensors that identify the installed
        # configuration of the CBF unambiguously, including hardware, software and firmware part
        # numbers and versions.

        Aqf.procedure(TestProcedure.ReportHWVersion)
        Aqf.procedure(TestProcedure.ReportSWVersion)
        Aqf.procedure(TestProcedure.ReportGitVersion)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_report_config(verbose=False)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.18')
    @aqf_requirements("CBF-REQ-0067", "CBF-REQ-0157")
    def test__systematic_error_reporting(self, instrument='bc8n856M4k'):
        # The CBF shall detect and flag data where the signal integrity has been compromised due to:
        #     a. Digitiser data acquisition and/or signal processing (e.g. ADC saturation),
        #     b. Signal processing and/or data manipulation performed in the CBF (e.g. FFT overflow).
        Aqf.procedure(TestProcedure.PFBFaultDetection)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_fft_overflow()
                clear_host_status(self)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.18')
    @aqf_requirements("CBF-REQ-0157")
    def test__fault_detection(self, instrument='bc8n856M4k'):
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
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_network_link_error()
                clear_host_status(self)
            else:
                Aqf.failed(self.errmsg)

    @generic_test
    @aqf_vr('TP.C.1.16')
    @aqf_requirements("CBF-REQ-0056", "CBF-REQ-0068", "CBF-REQ-0069")
    def test__monitor_sensors(self, instrument='bc8n856M4k'):
        # The CBF shall report the following transient search monitoring data:
        #     a) Transient buffer ready for triggering
        # The CBF shall, on request via the CAM interface, report sensor values.
        # The CBF shall, on request via the CAM interface, report time synchronisation status.

        Aqf.procedure(TestProcedure.ReportSensorStatus)
        Aqf.procedure(TestProcedure.ReportHostSensor)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._test_sensor_values()
                self._test_host_sensors_status()
            else:
                Aqf.failed(self.errmsg)
        clear_host_status(self)

    @generic_test
    @aqf_vr('TP.C.1.42')
    @aqf_requirements("CBF-REQ-0203")
    def test__time_synchronisation(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.TimeSync)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            self._test_time_sync()

    @generic_test
    @aqf_vr('TP.C.4.6')
    @aqf_requirements("CBF-REQ-0083", "CBF-REQ-0084", "CBF-REQ-0085", "CBF-REQ-0086", "CBF-REQ-0221")
    def test__antenna_voltage_buffer(self, instrument='bc8n856M4k'):
        Aqf.procedure(TestProcedure.VoltageBuffer)
        try:
            assert eval(os.getenv('DRY_RUN', 'False'))
        except AssertionError:
            _running_inst = which_instrument(self, instrument)
            instrument_success = self.set_instrument(_running_inst)
            _running_inst = self.corr_fix.get_running_instrument()
            if instrument_success and _running_inst:
                self._small_voltage_buffer()
            else:
                Aqf.failed(self.errmsg)


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
    def _systems_tests(self):
        """Checking system stability before and after use"""
        try:
            FNULL = open(os.devnull, 'w')
            subprocess.check_call(['pgrep', '-fol', 'corr2_sensor_servlet.py'], stdout=FNULL,
                stderr=FNULL)
        except subprocess.CalledProcessError:
            # Aqf.failed('Corr2_Sensor_Servlet PID could not be discovered, might not be running.')
            LOGGER.error('Corr2_Sensor_Servlet PID could not be discovered, might not be running.')
        except IOError:
            LOGGER.exception('Corr2_Sensor_Servlet PID could not be discovered, might not be running.')

        if not confirm_out_dest_ip(self):
            Aqf.failed('Output destination IP is not the same as the one stored in the register, '
                       'i.e. data is being spewed elsewhere.')
        # clear_host_status(self)
        set_default_eq(self)
        # ---------------------------------------------------------------
        Aqf.step('Checking system stability(sensors OK status) before and after testing')
        xeng_sensors = ['phy', 'qdr', 'lru', 'reorder', 'network-tx', 'network-rx']
        test_timeout = 30
        errmsg = 'Failed to retrieve X-Eng status: Timed-out after %s seconds.' % (test_timeout)
        try:
            with RunTestWithTimeout(test_timeout, errmsg):
                get_hosts_status(self, check_host_okay, xeng_sensors, engine_type='xeng')
        except Exception:
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        xeng_sensors.append('pfb')
        feng_sensors = xeng_sensors
        errmsg = ('Failed to retrieve F-Eng status: Timed-out after %s seconds.' % (test_timeout))
        try:
            with RunTestWithTimeout(test_timeout, errmsg):
                get_hosts_status(self, check_host_okay, feng_sensors, engine_type='feng')
        except Exception:
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)

    # ---------------------------------------------------------------
        if 'roach' in str(self._hosts):
            try:
                self.last_pfb_counts = get_pfb_counts(
                    get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
            except Exception:
                LOGGER.error('Failed to read correlator attribute, correlator might not be running.')


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

        _parameters = parameters(self)
        local_src_names = _parameters['custom_src_names']
        network_latency = _parameters['network_latency']
        self.corr_fix.issue_metadata
        source_names = _parameters['input_labels']
        # Get name for test_source_idx
        test_source = source_names[test_source_idx]
        ref_source = source_names[0]
        num_inputs = len(source_names)
        # Number of integrations
        num_int = 30
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
            Aqf.progress('Successfully retrieve initial spead accumulation')
            int_time = _parameters['int_time']
            synch_epoch = _parameters['synch_epoch']
            # n_accs = _parameters['n_accs']
            no_chans = range(_parameters['n_chans'])
            time_stamp = initial_dump['timestamp']
            # ticks_between_spectra = initial_dump['ticks_between_spectra'].value
            # int_time_ticks = n_accs * ticks_between_spectra
            t_apply = (initial_dump['dump_timestamp'] + num_int * int_time)
            t_apply_readable = time.strftime("%H:%M:%S", time.localtime(t_apply))
            try:
                baseline_lookup = get_baselines_lookup(self)
                # Choose baseline for phase comparison
                baseline_index = baseline_lookup[(ref_source, test_source)]
                Aqf.step('Get list of all the baselines present in the correlator output')
                Aqf.progress('Selected input and baseline for testing respectively: %s, %s.'%(
                    test_source, baseline_index))
                Aqf.progress('Time to apply delays (%s/%s) is set to %s integrations/accumulations in '
                     'the future.' % (t_apply,t_apply_readable, num_int))
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
                        'sample_period': self.corr_freqs.sample_period,
                        't_apply': t_apply,
                        'test_source': test_source,
                        'test_source_ind': test_source_idx,
                        'time_stamp': time_stamp,
                        'synch_epoch': synch_epoch,
                        'num_int': num_int,
                       }


    def _get_actual_data(self, setup_data, dump_counts, delay_coefficients, max_wait_dumps=30):
        cam_max_load_time = 1
        try:
            # Max time it takes to resync katcp (client connection)
            katcp_rsync_time = 0.9
            # Max network latency
            network_roundtrip = setup_data['network_latency'] + katcp_rsync_time

            # katcp_host = self.corr_fix.katcp_rct.host
            # katcp_port = self.corr_fix.katcp_rct.port
            # cmd_start_time = time.time()
            # os.system("/usr/local/bin/kcpcmd -s {}:{} delays {} {}".format(katcp_host, katcp_port,
            # setup_data['t_apply'] + 5, ' '.join(delay_coefficients)))
            # final_cmd_time = time.time() - cmd_start_time

            ### TODO MM 2016-07-05
            ## Disabled katcp resource client setting delays, instead setting them
            ## via telnet kcs interface.
            Aqf.step('Request Fringe/Delay(s) Corrections via CAM interface.')
            katcp_conn_time = time.time()
            reply, _informs = self.corr_fix.katcp_rct.req.delays(setup_data['t_apply'],
                                                                 *delay_coefficients, timeout=30)
            cmd_end_time = time.time()
            assert reply.reply_ok()
            actual_delay_coef = reply.arguments[1:]
            try:
                assert setup_data['num_inputs'] == len(actual_delay_coef)
            except:
                actual_delay_coef = None

            cmd_tot_time = katcp_conn_time + network_roundtrip
            final_cmd_time = abs(cmd_end_time - cmd_tot_time - katcp_rsync_time)
        except AssertionError:
            errmsg = str(reply).replace('\_', ' ')
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        except Exception:
            errmsg = ('%s: Failed to set delays via CAM interface with load-time: %s, '
                      'Delay coefficients: %s' % (str(reply), setup_data['t_apply'],
                        delay_coefficients))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return
        else:
            msg = ('Delay/Fringe(s) set via CAM interface reply : {}'.format(reply.arguments[1]))
            Aqf.is_true(reply.reply_ok(), msg)
            msg = ('Time it takes to load delay/fringe(s) is less than %s with integration time '
                   'of %s seconds\n' % (cam_max_load_time, setup_data['int_time']))
            Aqf.less(final_cmd_time, cam_max_load_time, msg)

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
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                # print np.max(
                #     np.angle(complexise(dump['xeng_raw'][:, setup_data['baseline_index'], :])))
                #Aqf.hop('Dump timestamp in ticks: {:20d}'.format(dump['timestamp']))

                if (np.abs(dump['dump_timestamp'] - last_discard) < 0.1 * setup_data['int_time']):
                    Aqf.passed('Received final accumulation before fringe '
                             'application with dump timestamp: %s, relevant to time apply: %s '
                             '(Difference %.2f)' % (dump['dump_timestamp'], setup_data['t_apply'],
                                (setup_data['t_apply'] - dump['dump_timestamp'])))
                    fringe_dumps.append(dump)
                    break

                if num_discards > max_wait_dumps:
                    Aqf.failed('Could not get accumulation with correct timestamp within %s '
                               'accumulation periods.' % max_wait_dumps)
                    break
                else:
                    difference = setup_data['t_apply'] - dump['dump_timestamp']
                    msg = ("Discarding (#%d) Spead accumulation with dump timestamp: %s"
                           "(and timestamp in ticks: %s), relevant to time to apply: %s"
                           "(Difference %.2f), Current epoch time: %s." % (num_discards,
                            dump['dump_timestamp'], dump['timestamp'], setup_data['t_apply'],
                            difference, time.time()))
                    if num_discards <= 3:
                        Aqf.progress(msg)
                    elif num_discards >= max_wait_dumps - 3:
                        Aqf.progress(msg)
                    elif num_discards == 4:
                        Aqf.progress('...')

        for i in xrange(dump_counts - 1):
            Aqf.progress('Getting subsequent SPEAD accumulation {}.'.format(i + 1))
            try:
                dump = self.receiver.data_queue.get()
                #dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue might be Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                # print np.max(np.angle(dump['xeng_raw'][:,setup_data['baseline_index'],:]))
                fringe_dumps.append(dump)

        chan_resp = []
        phases = []
        for acc in fringe_dumps:
            dval = acc['xeng_raw']
            freq_response = normalised_magnitude(
                dval[:, setup_data['baseline_index'], :])
            chan_resp.append(freq_response)

            data = complexise(dval[:, setup_data['baseline_index'], :])
            #print np.max(np.angle(data))
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
                    msg = ('Measured power for rack {} ({:.2f}kW) is more than {}kW'.format(
                            rack, watts, max_power_per_rack))
                    Aqf.less(watts, max_power_per_rack, msg)
                    phase = np.zeros(3)
                    for i, x in enumerate(phase):
                        phase[i] = curr[:, i].mean()
                    Aqf.progress('Average current per phase for rack {}: P1={:.2f}A, P2={:.2f}A, '
                            'P3={:.2f}A'.format(rack, phase[0], phase[1], phase[2]))
                    ph_m = np.max(phase)
                    max_diff = np.max([100 * (x / ph_m) for x in ph_m - phase])
                    max_diff = float('{:.1f}'.format(max_diff))
                    msg = ('Maximum difference in current per phase for rack {} ({:.1f}%) is '
                           'less than {}%'.format(rack, max_diff, max_power_diff_per_rack))
                    # Aqf.less(max_diff,max_power_diff_per_rack,msg)
                    # Aqf.waived(msg)
                watts = tot_power.mean()
                msg = 'Measured power for CBF ({:.2f}kW) is more than {}kW'.format(watts,
                    max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)
                watts = tot_power.max()
                msg = 'Measured peak power for CBF ({:.2f}kW) is more than {}kW'.format(watts,
                    max_power_cbf)
                Aqf.less(watts, max_power_cbf, msg)

    #################################################################
    #                       Test Methods                            #
    #################################################################

    def _test_channelisation(self, test_chan=1500, no_channels=None, req_chan_spacing=None):
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

        print_counts = 3
        spead_failure_counter = 0

        try:
            self.last_pfb_counts = get_pfb_counts(
                get_fftoverflow_qdrstatus(self.correlator)['fhosts'].items())
        except Exception:
            LOGGER.error('Failed to read correlator attribute, correlator might not be running.')

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
            _parameters = parameters(self)
            bls_to_test = _parameters['bls_ordering'][test_baseline]
            Aqf.progress('Randomly selected frequency channel to test: {} and '
                         'selected baseline {} / {} to test.'.format(test_chan, test_baseline,
                            bls_to_test))
            Aqf.equals(np.shape(initial_dump['xeng_raw'])[0], no_channels,
                       'Confirm that the number of channels in the SPEAD accumulation, is equal '
                       'to the number of frequency channels as calculated: {}'.format(
                           np.shape(initial_dump['xeng_raw'])[0]))

            Aqf.is_true(_parameters['bandwidth'] >= min_bandwithd_req,
                        'Channelise total bandwidth {}Hz shall be >= {}Hz.'.format(
                            _parameters['bandwidth'], min_bandwithd_req))
            # TODO (MM) 2016-10-27, As per JM
            # Channel spacing is reported as 209.266kHz. This is probably spot-on, considering we're
            # using a dsim that's not actually sampling at 1712MHz. But this is problematic for the
            # test report. We would be getting 1712MHz/8192=208.984375kHz on site.
            # Maybe we should be reporting this as a fraction of total sampling rate rather than
            # an absolute value? ie 1/4096=2.44140625e-4 I will speak to TA about how to handle this.

            # chan_spacing = initial_dump['bandwidth'].value / initial_dump['xeng_raw'].shape[0]
            chan_spacing = 856e6 / np.shape(initial_dump['xeng_raw'])[0]
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

        sync_time = _parameters['synch_epoch']
        int_time = _parameters['int_time']
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
                this_freq_dump = get_clean_dump(self)
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
        if 'roach' in str(self._hosts):
            # Test fft overflow and qdr status after
            Aqf.step('Check FFT overflow and QDR errors after channelisation.')
            check_fftoverflow_qdrstatus(self.correlator, self.last_pfb_counts)
        clear_host_status(self)
        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        df = self.corr_freqs.delta_f
        try:
            rand_chan_response = len(chan_responses[random.randrange(len(chan_responses))])
            assert rand_chan_response == _parameters['n_chans']
        except AssertionError:
            errmsg = ('Number of channels (%s) found on the spead data is inconsistent with the '
                      'number of channels (%s) expected.' %(rand_chan_response, _parameters['n_chans']))
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)
        else:
            plt_filename = '{}/{}_Channel_Response.png'.format(self.logs_path,
                self._testMethodName)
            plot_data = loggerise(chan_responses[:, test_chan], dynamic_range=90, normalise=True)
            plt_caption = ('Frequency channel {} @ {}MHz response vs source frequency and '
                           'selected baseline {} / {} to test.'.format(test_chan, expected_fc / 1e6,
                            test_baseline, bls_to_test))
            plt_title = 'Channel {} @ {:.3f}MHz response.'.format(test_chan, expected_fc / 1e6)
            # Plot channel response with -53dB cutoff horizontal line
            aqf_plot_and_save(freqs=actual_test_freqs, data=plot_data, df=df, expected_fc=expected_fc,
                              plot_filename=plt_filename, plt_title=plt_title, caption=plt_caption,
                              cutoff=-cutoff)

            # Get responses for central 80% of channel
            df = self.corr_freqs.delta_f
            central_indices = ((actual_test_freqs <= expected_fc + 0.4 * df) &
                               (actual_test_freqs >= expected_fc - 0.4 * df))
            central_chan_responses = chan_responses[central_indices]
            central_chan_test_freqs = actual_test_freqs[central_indices]

            # Plot channel response for central 80% of channel
            graph_name_central = '{}/{}_central.png'.format(self.logs_path, self._testMethodName)
            plot_data_central = loggerise(central_chan_responses[:, test_chan], dynamic_range=90,
                                          normalise=True)

            n_chans = self.corr_freqs.n_chans
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
                     'Confirm that theVACC output is at < 99% of maximum value, if fails '
                     'then it is probably over-ranging.')

            max_central_chan_response = np.max(10 * np.log10(
                central_chan_responses[:, test_chan]))
            min_central_chan_response = np.min(10 * np.log10(
                central_chan_responses[:, test_chan]))
            chan_ripple = max_central_chan_response - min_central_chan_response
            acceptable_ripple_lt = 1.5
            Aqf.less(chan_ripple, acceptable_ripple_lt,
                     'Confirm that theripple within 80% of cut-off '
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
            Aqf.step('Confirm that theresponse at channel-edges are -3 dB '
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

            # Plot PFB channel response with -6dB cuttoff horizontal line
            # TODO MM 2016-10-04 hard coded center bins, should probably fix
            no_of_responses = 3
            center_bin = [150, 250, 350]
            y_axis_limits = (-90, 1)
            legends = ['Channel {} / Sample {} \n@ {:.3f} MHz'.format(((test_chan + i) - 1), v,
                            self.corr_freqs.chan_freqs[test_chan + i] / 1e6)
                       for i, v in zip(range(no_of_responses), center_bin)]
            # TODO (MM) 2016-11-23
            # Hardcorded frequency bandwidth instead of reading it from spead
            center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.corr_freqs.n_chans / 1e3))
            # center_bin.append('Channel spacing: {:.3f}kHz'.format(chan_spacing/1e3))

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

            aqf_plot_channels(zip(channel_response_list, legends), plot_filename, plot_title,
                              normalise=True, caption=caption, cutoff=-cutoff_edge, vlines=center_bin,
                              xlabel='Sample Steps', ylimits=y_axis_limits)

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
                              normalise=True, caption=caption, xlabel='Sample Steps',
                              ylimits=y_axis_limits)

            Aqf.is_true(low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
                        'Confirm that therelative response at the low band-edge '
                        '({co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq '
                        '{co_low_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                        'relative to channel centre response.'.format(**locals()))

            Aqf.is_true(low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
                        'Confirm that therelative response at the high band-edge '
                        '({co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq '
                        '{co_high_src_freq}) is within the range of {desired_cutoff_resp} +- 1% '
                        'relative to channel centre response.'.format(**locals()))


    def _test_sfdr_peaks(self, required_chan_spacing, no_channels, cutoff=53, log_power=True):
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
        start_chan = 1  # skip DC channel since dsim puts out zeros for freq=0
        n_chans = self.corr_freqs.n_chans
        msg = ('This tests confirms that the correct channels have the peak response to each'
               ' frequency and that no other channels have significant relative power, while logging '
               'the power usage of the CBF in the background.')
        Aqf.step(msg)
        if log_power:
            Aqf.progress('Logging power usage in the background.')

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
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=self.corr_freqs.bandwidth / 2.0,
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
            _parameters = parameters(self)
            Aqf.equals(np.shape(initial_dump['xeng_raw'])[0], no_channels,
                       'Captured an initial correlator SPEAD accumulation, '
                       'determine the number of channels and processing bandwidth: '
                       '{}Hz.'.format(_parameters['bandwidth']))
            chan_spacing = (_parameters['bandwidth'] / np.shape(initial_dump['xeng_raw'])[0])
            # [CBF-REQ-0043]
            calc_channel = ((required_chan_spacing / 2) <= chan_spacing <= required_chan_spacing)
            Aqf.step('Confirm that the number of calculated channel '
                     'frequency step is within requirement.')
            mag = ('Confirm that the calculated channel frequency step size is between {} and '
                   '{} Hz'.format(required_chan_spacing / 2, required_chan_spacing))
            Aqf.is_true(calc_channel, msg)

        Aqf.step('Sweep the digitiser simulator over the all channels that fall '
                 'within the complete L-band.')
        spead_failure_counter = 0
        channel_response_lst = []
        print_counts = 4
        for channel, channel_f0 in enumerate(self.corr_freqs.chan_freqs[start_chan:], start_chan):
            if channel < print_counts:
                Aqf.progress('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(
                    channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            elif channel == print_counts:
                Aqf.progress('.' * 3)
            elif channel > (n_chans - print_counts):
                Aqf.progress('Getting channel response for freq {} @ {}: {:.3f} MHz.'.format(
                    channel, len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
            else:
                # LOGGER.info('Getting channel response for freq %s @ %s: %s MHz.' % (channel,
                #         len(self.corr_freqs.chan_freqs), channel_f0 / 1e6))
                pass

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            try:
                this_freq_dump = get_clean_dump(self)
                self.assertIsInstance(this_freq_dump, dict)
            except AssertionError:
                errmsg = ('Could not retrieve clean SPEAD accumulation')
                Aqf.failed(errmsg)
                return False
            else:
                this_freq_data = this_freq_dump['xeng_raw']
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
            plt_filename = '{}/{}_channel_{}_resp.png'.format(self.logs_path,
                self._testMethodName, channel)

            test_freq_mega = self.corr_freqs.chan_freqs[channel] / 1e6
            plt_title = 'Frequency response at {} @ {:.3f} MHz'.format(channel, test_freq_mega)
            caption = ('An overall frequency response at channel {} @ {:.3f}MHz, '
                       'when digitiser simulator is configured to generate a continuous wave, '
                       'with cw scale: {}. awgn scale: {}, eq gain: {}, fft shift: {}'.format(
                            channel, test_freq_mega, cw_scale, awgn_scale, gain, fft_shift))

            new_cutoff = np.max(loggerise(channel_resp)) - cutoff
            aqf_plot_channels(channel_resp, plt_filename, plt_title, log_dynamic_range=90,
                              caption=caption, hlines=new_cutoff)

        channel_range = range(start_chan, len(max_channels) + start_chan)
        Aqf.step('Confirm that the correct channels have the peak response to each frequency')
        if max_channels == channel_range:
            Aqf.passed('Confirm that the correct channels have the peak response to each frequency')
        else:
            msg = ('Confirm that the correct channels have the peak response to each frequency')
            Aqf.array_abs_error(max_channels[1:], channel_range[1:], msg, 1)

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
            self._process_power_log(start_timestamp, power_log_file)


    def _test_spead_verify(self):
        """This test verifies if a cw tone is only applied to a single input 0,
            figure out if VACC is rooted by 1
        """
        test_heading("SPEAD Accumulation Verification")
        cw_scale = 0.035
        freq = 300e6
        Aqf.step('Digitiser simulator configured to generate cw tone with frequency: {}MHz, '
                 'scale:{} on input 0'.format(freq / 1e6, cw_scale))
        self.dhost.sine_sources.sin_0.set(scale=cw_scale, frequency=freq)
        Aqf.step('Capture a correlator SPEAD accumulation.')
        try:
            dump = get_clean_dump(self)
            self.assertIsInstance(dump, dict)
        except AssertionError:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            vacc_offset = get_vacc_offset(dump['xeng_raw'])
            msg = ('Confirm that theauto-correlation in baseline 0 contains Non-Zeros, '
                   'and baseline 1 is Zeros, when cw tone is only outputted on input 0.')
            Aqf.equals(vacc_offset, 0, msg)
            # TODO Plot baseline
            Aqf.step('Digitiser simulator reset to Zeros, before next test')
            init_dsim_sources(self.dhost)
            Aqf.step('Digitiser simulator configured to generate cw tone with frequency: {}Mhz, '
                     'scale:{} on input 1'.format(freq / 1e6, cw_scale))
            self.dhost.sine_sources.sin_1.set(scale=cw_scale, frequency=freq)
            Aqf.step('Capture a correlator SPEAD accumulation.')
            dump = get_clean_dump(self)
            vacc_offset = get_vacc_offset(dump['xeng_raw'])
            msg = ('Confirm that theauto-correlation in baseline 1 contains non-Zeros, '
                   'and baseline 0 is Zeros, when cw tone is only outputted on input 1.')
            Aqf.equals(vacc_offset, 1, msg)
            init_dsim_sources(self.dhost)


    def _test_product_baselines(self):
        test_heading("CBF Baseline Correlation Products")
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

        Aqf.step('Digitiser simulator configured to generate Gaussian noise, '
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(awgn_scale, gain,
                                                                          fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            freq=self.corr_freqs.chan_freqs[1500],
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
            self.corr_fix.issue_metadata
            time.sleep(5)
            _parameters = parameters(self)
            self.corr_fix.issue_metadata
            local_src_names = _parameters['custom_src_names']
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
            _parameters = parameters(self)
            bls_ordering = _parameters['bls_ordering']
            input_labels = sorted(_parameters['input_labels'])
            baselines_lookup = get_baselines_lookup(self)
            present_baselines = sorted(baselines_lookup.keys())

            possible_baselines = set()
            _ = [possible_baselines.add((li, lj)) for li in input_labels for lj in input_labels]

            test_bl = sorted(list(possible_baselines))
            Aqf.step('Confirm that each baseline (or its reverse-order '
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
            Aqf.passed('Stored initial F-engine equalisations: %s'%initial_equalisations)

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

            test_data = get_clean_dump(self)

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
            dataFrame = pd.DataFrame(index=sorted(input_labels),
                                     columns=list(sorted(present_baselines)))

            for count, inp in enumerate(input_labels, start=1):
                old_eq = complex(initial_equalisations[inp])
                Aqf.step('[CBF-REQ-0071] Iteratively set gain/equalisation correction on relevant '
                         'input {} set to {}.'.format(inp, old_eq))
                try:
                    reply, informs = self.corr_fix.katcp_rct.req.gain(inp, old_eq)
                    assert reply.reply_ok()
                except Exception:
                    errmsg = '%s: Failed to set gain/eq of %s for input %s' %(str(reply), old_eq,
                        inp)
                    Aqf.failed(errmsg)
                    LOGGER.exception(errmsg)
                else:
                    msg = 'Gain/Equalisation correction on input {} set to {}.'.format(inp, old_eq)
                    Aqf.passed(msg)
                    zero_inputs.remove(inp)
                    nonzero_inputs.add(inp)
                    expected_z_bls, expected_nz_bls = (calc_zero_and_nonzero_baselines(
                        nonzero_inputs))
                    try:
                        Aqf.step('Retrieving SPEAD accumulation and confirm if gain/equlisation '
                                 'correction has been applied.')
                        test_dump = get_clean_dump(self)
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        test_data = test_dump['xeng_raw']
                        # plot baseline channel response
                        plot_data = [normalised_magnitude(test_data[:, i, :])
                                     # plot_data = [loggerise(test_data[:, i, :])
                                     for i in plot_baseline_inds]
                        plot_filename = '{}/{}_channel_resp_{}.png'.format(self.logs_path,
                            self._testMethodName.replace(' ', '_'), inp)

                        plot_title = ('Baseline Correlation Products on input: {}\n'
                                      'Bls channel response \'Non-Zero\' inputs:\n {}\n'
                                      '\'Zero\' inputs:\n {}'.format(inp,
                                      ' \n'.join(textwrap.wrap(', \n'.join(sorted(nonzero_inputs)))),
                                      ' \n'.join(textwrap.wrap(', \n'.join(sorted(zero_inputs))))))

                        caption = ('Baseline channel response on input:{} {} with the following non-zero'
                                   ' inputs:\n {} \n and\nzero inputs:\n {}'.format(inp, bls_msg,
                                        sorted(nonzero_inputs), sorted(zero_inputs)))

                        aqf_plot_channels(zip(plot_data, plot_baseline_legends), plot_filename,
                                          plot_title, log_dynamic_range=None, log_normalise_to=1,
                                          caption=caption, ylimits=(-0.1, np.max(plot_data) + 0.1))
                        actual_nz_bls_indices = all_nonzero_baselines(test_data)
                        actual_nz_bls = set(tuple(bls_ordering[i]) for i in actual_nz_bls_indices)

                        actual_z_bls_indices = zero_baselines(test_data)
                        actual_z_bls = set(tuple(bls_ordering[i]) for i in actual_z_bls_indices)
                        msg = ('Confirm that the expected baseline visibilities are non-zero with '
                               'non-zero inputs {} and,'.format(sorted(nonzero_inputs)))
                        Aqf.equals(actual_nz_bls, expected_nz_bls, msg)

                        msg = ('Confirm that theexpected baselines visibilities are \'Zeros\'.\n')
                        Aqf.equals(actual_z_bls, expected_z_bls, msg)

                        # Sum of all baselines powers expected to be non zeros
                        sum_of_bl_powers = (
                            [normalised_magnitude(test_data[:, expected_bl, :])
                             for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                                                 for expected_nz_bl_ind in sorted(expected_nz_bls)]])
                        test_data = None
                        dataFrame.loc[inp][sorted(
                            [i for i in expected_nz_bls])[-1]] = np.sum(sum_of_bl_powers)

            dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')


    def _test_back2back_consistency(self):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect.
        """
        test_heading("Spead Accumulation Back-to-Back Consistency")
        Aqf.step('Randomly select a channel to test.')
        test_chan = random.randrange(self.corr_freqs.n_chans)
        test_baseline = 0  # auto-corr
        Aqf.progress('Randomly selected test channel %s and bls %s'%(test_chan, test_baseline))
        Aqf.step('Calculate a list of frequencies to test')
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=9, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        source_period_in_samples = self.corr_freqs.n_chans * 2
        cw_scale = 0.675
        self.dhost.sine_sources.sin_0.set(frequency=expected_fc, scale=cw_scale,
                                          repeat_n=source_period_in_samples)
        Aqf.step('Digitiser simulator configured to generate periodic wave '
                 '({:.3f}Hz with FFT-length {}) in order for each FFT to be '
                 'identical.'.format(expected_fc / 1e6, source_period_in_samples))

        try:
            this_freq_dump = get_clean_dump(self)
            assert this_freq_dump is not None
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
                            assert this_freq_dump is not None
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
                        except AssertionError:
                            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                            Aqf.failed(errmsg)
                            LOGGER.exception(errmsg)

                    this_freq_data = this_freq_dump['xeng_raw']
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
        test_chan = random.randrange(self.corr_freqs.n_chans)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Randomly selected Frequency channel {} @ {:.3f}MHz for testing.'.format(
                test_chan, expected_fc / 1e6))
        Aqf.step('Calculate a list of frequencies to test')
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        source_period_in_samples = self.corr_freqs.n_chans * 2

        try:
            test_dump = get_clean_dump(self)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
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
                        except Queue.Empty:
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
                        except Queue.Empty:
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

        test_chan = random.randrange(self.corr_freqs.n_chans)
        requested_test_freqs = self.corr_freqs.calc_freq_samples(
            test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.corr_freqs.chan_freqs[test_chan]
        Aqf.step('Sweeping the digitiser simulator over {:.3f}MHz of the channels that '
                 'fall within {} complete L-band'.format(np.max(requested_test_freqs) / 1e6,
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
                    deprogram_hosts(self)

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
                    startx = self.corr_fix.start_x_data
                    if not startx:
                        Aqf.failed('Failed to enable/start output product capturing.')
                    host = (xhosts + fhosts)[random.randrange(len(xhosts + fhosts))]
                    msg = ('Confirm that the instrument is initialised by checking if a '
                           'random host: {} is programmed and running.'.format(host.host))
                    Aqf.is_true(host, msg)
                    _parameters = parameters(self)
                    try:
                        self.assertIsInstance(self.receiver, corr2.corr_rx.CorrRx)
                        freq_dump = get_clean_dump(self)
                        assert np.shape(freq_dump['xeng_raw'])[0] == _parameters['n_chans']
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    except AssertionError:
                        errmsg = ('Correlator Receiver could not be instantiated or No of channels '
                                  '(%s) in the spead data is inconsistent with the no of'
                                  ' channels (%s) expected' %(np.shape(freq_dump['xeng_raw'])[0],
                                     _parameters['n_chans']))
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
                            Aqf.equals(spead_chans['xeng_raw'].shape[0], no_channels, msg)
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
                        self.set_instrument(instrument)
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
            sampling_period = self.corr_freqs.sample_period
            no_chans = range(self.corr_freqs.n_chans)
            roundtrip = 300e-3
            cam_max_load_time = 1
            test_delays = [0, sampling_period, 1.5 * sampling_period, 2 * sampling_period]
            test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
            delays = [0] * setup_data['num_inputs']
            num_int = setup_data['num_int']
            Aqf.step('Delays to be set (iteratively) %s for testing purposes\n'%(
                test_delays))

            def get_expected_phases():
                expected_phases = []
                for delay in test_delays:
                    phases = self.corr_freqs.chan_freqs * 2 * np.pi * delay
                    phases -= np.max(phases) / 2.
                    expected_phases.append(phases)
                return zip(test_delays_ns, expected_phases)

            def get_actual_phases(_parameters):
                actual_phases_list = []
                # chan_responses = []
                int_time = _parameters['int_time']
                katcp_port = _parameters['katcp_port']
                for count, delay in enumerate(test_delays, 1):
                    delays[setup_data['test_source_ind']] = delay
                    delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                    try:
                        this_freq_dump = get_clean_dump(self)
                        t_apply = this_freq_dump['dump_timestamp'] + (num_int * int_time)
                        t_apply_readable = this_freq_dump['dump_timestamp_readable']
                        Aqf.step('Delay #%s will be applied with the following parameters:' %count)
                        msg = ('On baseline %s and input %s, Current epoch time: %s (%s)'
                              ', Current Dump timestamp: %s (%s), '
                              'Time delays will be applied: %s (%s), Delay to be applied: %s' %(
                            setup_data['baseline_index'], setup_data['test_source'],
                            time.time(), time.strftime("%H:%M:%S"), this_freq_dump['dump_timestamp'],
                            this_freq_dump['dump_timestamp_readable'], t_apply, t_apply_readable,
                            delay))
                        Aqf.progress(msg)
                        Aqf.step('Execute delays via CAM interface and calculate the amount of time '
                                 'it takes to load the delays')
                        reply, _informs = self.corr_fix.katcp_rct.req.delays(t_apply,
                                                                             *delay_coefficients)
                        cmd_start_time = time.time()
                        final_cmd_time = (time.time() - cmd_start_time - roundtrip)
                        formated_reply = str(reply).replace('\_', ' ')
                        assert reply.reply_ok()
                        # Aqf.step('Setting delay of %ss to be applied at epoch %s (or %s), relevant to '
                        #          'current time: %s \n with delay coefficients: %s' % (delay, t_apply,
                        #             t_apply_readable, time.time(), delay_coefficients))
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    except Exception:
                        errmsg = ('CAM Reply: %s: Failed to set delays via CAM interface with '
                                  'load-time: %s vs Current epoch time: %s' % (
                                    formated_reply, t_apply, time.time()))
                        LOGGER.exception(errmsg)
                        Aqf.failed(errmsg)
                    else:
                        Aqf.less(final_cmd_time, cam_max_load_time,
                            ': Time it takes to load delays is less '
                            'than {}s with integration time of {:.3f}s'.format(cam_max_load_time,
                                int_time))
                        Aqf.is_true(reply.reply_ok(),
                            '''Delays set successfully via CAM interface:
                            Reply: %s'''%(formated_reply))
                    try:
                        _num_discards = num_int + 5
                        Aqf.step('Getting SPEAD accumulation(while discarding %s dumps) containing '
                                 'the change in delay(s) on input: %s baseline: %s.'%(_num_discards,
                                    setup_data['test_source'], setup_data['baseline_index']))
                        dump = self.receiver.get_clean_dump(discard=_num_discards)
                        Aqf.progress('Readable time stamp received on SPEAD accumulation: %s '
                                     'after %s number of discards \n'%(
                                        dump['dump_timestamp_readable'], _num_discards))
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        # # this_freq_data = this_freq_dump['xeng_raw']
                        # # this_freq_response = normalised_magnitude(
                        # #    this_freq_data[:, setup_data['test_source_ind'], :])
                        # # chan_responses.append(this_freq_response)
                        data = complexise(dump['xeng_raw']
                                          [:, setup_data['baseline_index'], :])

                        phases = np.angle(data)
                        # # actual_channel_responses = zip(test_delays, chan_responses)
                        # # return zip(actual_phases_list, actual_channel_responses)
                        actual_phases_list.append(phases)
                return actual_phases_list

            expected_phases = get_expected_phases()
            _parameters = parameters(self)
            actual_phases = get_actual_phases(_parameters)
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
                    plot_filename = '{}/{}_phase_response.png'.format(self.logs_path,
                        self._testMethodName)
                    plot_units = 'secs'

                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                           plot_filename, plot_title, plot_units, caption)

                    expected_phases_ = [phase for _rads, phase in get_expected_phases()]

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
                                                       plot_filename='{}/{}_{}_phase_resp.png'.format(
                                                            self.logs_path, self._testMethodName, i),
                                                       plot_title=('Delay offset:\n'
                                                                   'Actual vs Expected Phase Response'),
                                                       plot_units=plot_units, caption=caption)

                        for delay, count in zip(test_delays, xrange(1, len(expected_phases))):
                            msg = ('Confirm that when a delay of {} clock '
                                   'cycle({:.5f} ns) is introduced there is a phase change '
                                   'of {:.3f} degrees as expected to within {} degree.'.format(
                                        (count + 1) * .5, delay * 1e9,
                                        np.rad2deg(np.pi) * (count + 1) * .5, degree))
                            Aqf.array_abs_error(actual_phases[count][1:-1],
                                                expected_phases_[count][1:-1], msg, degree)
                    except IndexError:
                        return
            except TypeError:
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
                msg = ('[CBF-REQ-0068] Confirm that the number of sensors are equal '
                       'to the number of sensors listed on the running instrument.\n')
                Aqf.equals(int(reply.arguments[-1]), len(informs), msg)

        def report_time_sync(self):
            Aqf.step('Confirm that thetime synchronous is implemented on primary interface')
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

        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
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
        Aqf.progress('Sensor poll time: {} seconds '.format(sensor_poll_time))

        def get_pfb_status(self):
            """Retrieve F-Engines PFB status
            :param: Object
            :rtype: Boolean or list
            """
            try:
                reply, informs = self.corr_fix.katcp_rct.req.sensor_value(timeout=60)
                assert reply.reply_ok()
            except Exception:
                LOGGER.exception('Failed to retrieve sensor values via CAM interface')
                return False
            else:
                hosts = [_i.host.lower() for _i in self.correlator.fhosts]
                try:
                    roach_dict = [getattr(self.corr_fix.katcp_rct.sensor, 'fhost{}_pfb_ok'.format(host))
                                  for host in range(len(hosts))]

                except AttributeError:
                    Aqf.failed('Failed to retrieve PFB status on F-hosts')
                    return False
                else:
                    pfb_status = [[' '.join(i.arguments[2:]) for i in informs
                                   if i.arguments[2] == 'fhost{}-pfb-ok'.format(host)]
                                  for host in range(len(hosts))]
                    return list(set([int(i[0].split()[-1]) for i in pfb_status]))[0]

        def confirm_pfb_status(self, get_pfb_status, fft_shift=0):
            Aqf.step('Set an FFT shift on all f-engines.')
            fft_shift_val = self.corr_fix.katcp_rct.req.fft_shift(shift_value=fft_shift)
            if fft_shift_val is None:
                Aqf.failed('Could not set FFT shift for all F-Engine hosts')
            else:
                msg = ('{} was set on all F-Engines.'.format(str(fft_shift_val)))
                Aqf.wait(self.correlator.sensor_poll_time * 2, msg)
                pfb_status = get_pfb_status(self)
                Aqf.step('Confirm that the sensors indicated that the F-Eng PFB has been set')
                if pfb_status == 1:
                    msg = ('Sensors indicate that F-Eng PFB status is OKAY')
                    Aqf.passed(msg)
                elif pfb_status == 0 and fft_shift == 0:
                    msg = ('Sensors indicate that there is an ERROR on the F-Eng PFB status.\n')
                    Aqf.passed(msg)
                elif pfb_status == 1 and fft_shift == 0:
                    msg = ('Sensors indicate that there is an ERROR on the F-Eng PFB status.\n')
                    Aqf.failed(msg)
                else:
                    Aqf.failed('Could not retrieve PFB sensor status')

        confirm_pfb_status(self, get_pfb_status, fft_shift=fft_shift)
        confirm_pfb_status(self, get_pfb_status)
        Aqf.step('Restoring previous FFT Shift values')
        confirm_pfb_status(self, get_pfb_status, fft_shift=fft_shift)
        clear_host_status(self)


    def _test_network_link_error(self):
        test_heading("Fault Detection: Network Link Errors")
        def get_spead_data(self):
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
                reply, _informs = host.katcprequest('wordread', request_args=(['iptx_base']))
            except:
                Aqf.failed('Failed to retrieve multicast destination from {}'.format(
                    host.host.upper()))
            else:
                if reply.reply_ok() and len(reply.arguments) > 1:
                    hex_ip_ = reply.arguments[-1]
                    if hex_ip_.startswith('0x'):
                        return hex_ip_
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
                Aqf.failed('Could not get sensor attributes on {}'.format(host.host.upper()))
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
                                                   request_args=(['iptx_base', '0', ip_new]))
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
                Aqf.passed('Confirm that the X-engine {} LRU sensor is OKAY and '
                           'that the X-eng is receiving feasible data.'.format(host.host.upper()))
            elif get_lru_status(self, host) == 0:
                Aqf.passed('Confirm that the X-engine {} LRU sensor is reporting a '
                           'failure and that the X-eng is not receiving feasible '
                           'data.'.format(host.host.upper()))
            else:
                Aqf.failed('Failed to read {} sensor'.format(host.host.upper()))

        fhost = self.correlator.fhosts[random.randrange(len(self.correlator.fhosts))]
        xhost = self.correlator.xhosts[random.randrange(len(self.correlator.xhosts))]
        ip_new = '0xefefefef'

        Aqf.step('Randomly selected {} host that is being used to produce the test '
                 'data product on which to trigger the link error.'.format(fhost.host.upper()))
        current_ip = get_host_ip(fhost)
        if not current_ip:
            Aqf.failed('Failed to retrieve multicast destination address of {}'.format(
                fhost.host.upper()))
        elif current_ip != ip_new:
            Aqf.passed('Current multicast destination address for {}: {}.'.format(fhost.host.upper(),
                human_readable_ip(current_ip)))
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


    def _test_host_sensors_status(self):
        test_heading('Monitor Sensors: Processing Node\'s Sensor Status')
        clear_host_status(self)
        Aqf.step('This test confirms that each processing node\'s sensor (Temp, Voltage, Current, '
                 'Fan) has not FAILED, it will fail if the are errors')
        Aqf.progress('Confirm if any F/X/B-Engines contain(s) any errors/faults via CAM interface.')
        try:
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
                        Aqf.failed(' contains an '.join(i.arguments[2:-1]))


    def _test_vacc(self, test_chan, chan_index=None, acc_time=0.998):
        """Test vector accumulator"""
        MAX_VACC_SYNCH_ATTEMPTS = corr2.fxcorrelator_xengops.MAX_VACC_SYNCH_ATTEMPTS
        # Choose a test frequency around the centre of the band.
        test_freq = self.corr_freqs.bandwidth / 2.
        _parameters = parameters(self)
        test_input = _parameters['input_labels'][0]
        eq_scaling = 30
        acc_times = [acc_time / 2, acc_time]
        #acc_times = [acc_time/2, acc_time, acc_time*2]
        n_chans = self.corr_freqs.n_chans
        try:
            internal_accumulations = int(_parameters['xeng_acc_len'])
            acc_len = int(self.corr_freqs.xeng_accumulation_len)
        except Exception as e:
            errmsg = 'Failed to retrieve X-engine accumulation length: %s.' %str(e)
            LOGGER.exception(errmsg)
            Aqf.failed(errmsg)

        delta_acc_t = self.corr_freqs.fft_period * internal_accumulations
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq_channel = abs(np.argmin(
                    np.abs(self.corr_freqs.chan_freqs[:chan_index] - test_freq)) - test_chan)
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
            self.dhost.sine_sources.sin_0.set(frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
            self.dhost.sine_sources.sin_1.set(frequency=test_freq, scale=cw_scale, repeat_n=n_chans * 2)
            assert self.dhost.sine_sources.sin_0.repeat == n_chans * 2
        except AssertionError:
            errmsg = 'Failed to make the DEng output periodic in FFT-length so that each FFT is identical'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        try:
            reply, informs = self.corr_fix.katcp_rct.req.quantiser_snapshot(test_input)
            informs = informs[0]
            assert reply.reply_ok()
        except Exception:
            errmsg = ('REPLY: %s: Failed to retrieve quantiser snapshot of input %s via '
                      'CAM Interface' %(str(reply), test_input))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            quantiser_spectrum = np.array(eval(informs.arguments[-1]))
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
                                                          self.corr_freqs.fft_period * 1e6,
                                                          internal_accumulations,
                                                          delta_acc_t))

            chan_response = []
            # TODO MM 2016-10-07 Fix tests to use cam interface instead of corr object
            for vacc_accumulations, acc_time in zip(test_acc_lens, acc_times):
                try:
                    # self.correlator.xops.set_acc_len(vacc_accumulations)
                    reply = self.corr_fix.katcp_rct.req.accumulation_length(acc_time, timeout=60)
                    self.assertIsInstance(reply, katcp.resource.KATCPReply)
                except (TimeoutError, VaccSynchAttemptsMaxedOut):
                    Aqf.failed('Failed to set accumulation length of {} after {} maximum vacc '
                               'sync attempts.'.format(vacc_accumulations, MAX_VACC_SYNCH_ATTEMPTS))
                else:
                    accum_len = int(np.ceil(
                            (acc_time * self.corr_freqs.sample_freq) / (2 * internal_accumulations *
                                                                        self.corr_freqs.n_chans)))
                    Aqf.almost_equals(vacc_accumulations, self.correlator.xops.get_acc_len(), 1,
                                      'Confirm that vacc length was set successfully with'
                                      ' {}, which equates to an accumulation time of {:.6f}s'.format(
                                            vacc_accumulations, vacc_accumulations * delta_acc_t))
                    no_accs = internal_accumulations * vacc_accumulations
                    expected_response = np.abs(quantiser_spectrum) ** 2 * no_accs
                    try:
                        dump = get_clean_dump(self)
                    except Queue.Empty:
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

                        plot_filename = ('{}/{}_chan_resp_{}_acc.png'.format(self.logs_path,
                                                                             self._testMethodName,
                                                                             int(vacc_accumulations)))
                        plot_title = ('Vector Accumulation Length: channel {}'.format(test_freq_channel))
                        msg = ('Confirm that the accumulator actual response is '
                               'equal to the expected response for {} accumulation length'.format(
                                    vacc_accumulations))
                        if not Aqf.array_abs_error(expected_response[:chan_index],
                                                   actual_response_mag[:chan_index], msg,
                                                   abs_error=0.1):
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

        no_channels = self.corr_freqs.n_chans
        Aqf.step('Re-initialising {instrument} instrument'.format(**locals()))
        corr_init = False
        retries = 5
        start_time = time.time()
        Aqf.step('Correlator initialisation timer-started: %s' %start_time)
        while retries and not corr_init:
            try:
                self.set_instrument(instrument)
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
            self.set_instrument(instrument)
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
                Aqf.equals(re_dump['xeng_raw'].shape[0], no_channels, msg)

                final_time = end_time - start_time - float(self.corr_fix.halt_wait_time)
                minute = 60.0
                msg = ('[CBF-REQ-0013] Confirm that instrument switching to %s '
                       'time is less than one minute' % instrument)
                Aqf.less(final_time, minute, msg)


    def _test_delay_rate(self, plot_diagram=True):
        msg = ("CBF Delay and Phase Compensation Functional VR: -- Delay Rate")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            dump_counts = 5
            # delay_rate = ((setup_data['sample_period'] / parameters(self)['int_time']) *
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
            Aqf.step('Calculate the parameters to be used for setting Fringe(s)/Delay(s).')
            Aqf.progress('Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s ' % (
                delay_rate, delay_value, fringe_offset, fringe_rate))

            try:
                actual_data, _delay_coefficients = self._get_actual_data(
                    setup_data, dump_counts, delay_coefficients)
            except TypeError:
                return
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
            plot_title = 'Randomly generated delay rate {} {}'.format(delay_rate * 1e9, plot_units)
            plot_filename = '{}/{}_phase_response.png'.format(self.logs_path,
                self._testMethodName)
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

                    abs_error = np.max(actual_phases_[i] - expected_phases_[i])
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
                            plot_filename='{}/{}_{}_phase_resp.png'.format(self.logs_path,
                                self._testMethodName, i),
                            plot_title='Delay Rate:\nActual vs Expected Phase Response',
                            plot_units=plot_units, caption=caption, )
                if plot_diagram:
                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases, plot_filename,
                                   plot_title, plot_units, caption, dump_counts)


    def _test_fringe_rate(self, plot_diagram=True):
        msg = ("CBF Delay and Phase Compensation Functional VR: -- Fringe rate")
        test_heading(msg)
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

            Aqf.step('Calculate the parameters to be used for setting Fringe(s)/Delay(s).')
            Aqf.progress('Delay Rate: %s, Delay Value: %s, Fringe Offset: %s, Fringe Rate: %s ' % (
                delay_rate, delay_value, fringe_offset, fringe_rate))
            try:
                actual_data, _delay_coefficients = self._get_actual_data(setup_data, dump_counts,
                    delay_coefficients)
            except TypeError:
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
                no_chans = range(self.corr_freqs.n_chans)
                plot_units = 'rads/sec'
                plot_title = 'Randomly generated fringe rate {} {}'.format(fringe_rate,
                                                                           plot_units)
                plot_filename = '{}/{}_phase_response.png'.format(self.logs_path,
                    self._testMethodName)
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
                                plot_filename='{}/{}_phase_resp_{}.png'.format(self.logs_path,
                                    self._testMethodName, i),
                                plot_title='Fringe Rate: Actual vs Expected Phase Response',
                                plot_units=plot_units, caption=caption, )

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
            except TypeError:
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
                no_chans = range(self.corr_freqs.n_chans)
                plot_units = 'rads'
                plot_title = 'Randomly generated fringe offset {:.3f} {}'.format(
                    fringe_offset, plot_units)
                plot_filename = '{}/{}_phase_response.png'.format(self.logs_path,
                    self._testMethodName)
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
                            plot_filename='{}/{}_{}_phase_resp.png'.format(self.logs_path,
                                self._testMethodName, i),
                            plot_title=('Fringe Offset:\nActual vs Expected Phase Response'),
                            plot_units=plot_units, caption=caption, )
                if plot_diagram:
                    aqf_plot_phase_results(no_chans, actual_phases, expected_phases,
                                       plot_filename, plot_title, plot_units, caption)

    def _test_delay_inputs(self):
        """
        CBF Delay Compensation/LO Fringe stopping polynomial:
        Delay applied to the correct input
        """
        msg = ("CBF Delay and Phase Compensation Functional VR: "
                "Delays applied to the correct input")
        test_heading(msg)
        setup_data = self._delays_setup()
        if setup_data:
            # (MM) 2016-07-12
            # Disabled source name randomisation due to the fact that some roach boards
            # are known to have QDR issues which results to test failures, hence
            # input1 has been statically assigned to be the testing input
            Aqf.step("The test will sweep through all baselines, randomly select and set a delay value,"
                     " Confirm if the delay set is as expected.")
            for delayed_input in source_names:
                test_delay_val = random.randrange(self.corr_freqs.sample_period, step=.83e-10, int=float)
                # test_delay_val = self.corr_freqs.sample_period  # Pi
                expected_phases = self.corr_freqs.chan_freqs * 2 * np.pi * test_delay_val
                expected_phases -= np.max(expected_phases) / 2.
                source_names = parameters(self)['input_labels']
                Aqf.step('Clear all coarse and fine delays for all inputs before test commences.')
                delays_cleared = clear_all_delays(self)
                if not delays_cleared:
                    Aqf.failed('Delays were not completely cleared, data might be corrupted.\n')
                else:
                    Aqf.passed('Cleared all previously applied delays prior to test.\n')
                    delays = [0] * setup_data['num_inputs']
                    # Get index for input to delay
                    test_source_idx = source_names.index(delayed_input)
                    Aqf.step('Selected input to test: {}'.format(delayed_input))
                    delays[test_source_idx] = test_delay_val
                    Aqf.step('Randomly selected delay value ({}) relevant to sampling period'.format(
                        test_delay_val))
                    delay_coefficients = ['{},0:0,0'.format(dv) for dv in delays]
                    int_time = setup_data['int_time']
                    sync_time = setup_data['synch_epoch']
                    num_int = setup_data['num_int']
                    try:
                        this_freq_dump = get_clean_dump(self)
                        t_apply = this_freq_dump['dump_timestamp'] + (num_int * int_time)
                        t_apply_readable = this_freq_dump['dump_timestamp_readable']
                        Aqf.step('Delays will be applied with the following parameters:')
                        Aqf.progress('Current epoch time: %s (%s)' %(time.time(), time.strftime("%H:%M:%S")))
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
                        Aqf.passed('Delays were applied on input: {} successfully'.format(delayed_input))
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
                                Aqf.array_abs_error(np.abs(b_line_phase[1:-1]),
                                                    np.abs(expected_phases[1:-1]), msg, degree)
                            else:
                                # TODO Readdress this failure and calculate
                                if b_line_phase_max != 0.0:
                                    desc = ('Checking baseline {}, index: {}, phase offset found, '
                                            'maximum error value = {} rads'.format(b_line[0], b_line_val,
                                                b_line_phase_max))
                                    Aqf.failed(desc)

    def _test_min_max_delays(self):
    pass

    def _test_report_config(self, verbose):
        """CBF Report configuration"""
        test_config = self.corr_fix._test_config_file

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
                Aqf.step('%s [R3000-0000] Software/Hardware Version Information' % _host.upper())
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

        def get_src_dir():

            import corr2
            import casperfpga
            import katcp
            import spead2

            corr2_dir, _ = os.path.split(os.path.split(corr2.__file__)[0])
            corr2_name = corr2.__name__
            corr2_version = corr2.__version__
            corr2_link = 'https://github.com/ska-sa/corr2/releases/tag/v%s' % corr2_version

            casper_dir, _ = os.path.split(os.path.split(casperfpga.__file__)[0])
            casper_name = casperfpga.__name__
            casper_version = casperfpga.__version__

            katcp_dir, _ = os.path.split(os.path.split(katcp.__file__)[0])
            katcp_name = katcp.__name__
            katcp_version = katcp.__version__
            katcp_link = 'https://github.com/ska-sa/katcp/releases/tag/v%s' % katcp_version

            spead2_dir, _ = os.path.split(os.path.split(spead2.__file__)[0])
            spead2_name = spead2.__name__
            spead2_version = spead2.__version__
            spead2_link = 'https://github.com/ska-sa/spead2/releases/tag/v%s' % spead2_version

            try:
                bitstream_dir = self.correlator.configd['xengine']['bitstream']
            except AttributeError:
                mkat_name = None
                mkat_dir = None
                Aqf.failed('Failed to retrieve mkat_fpga info')
            else:
                mkat_dir, _ = os.path.split(os.path.split(os.path.dirname(
                    os.path.realpath(bitstream_dir)))[0])
                _, mkat_name = os.path.split(mkat_dir)

            test_dir, test_name = os.path.split(os.path.dirname(
                os.path.realpath(__file__)))
            with open('/etc/cmc.conf') as f:
                cmc_conf =  f.readlines()
            template_name , template_loc  = [i.strip().split('=') for i in cmc_conf
                                            if i.startswith('CORR_TEMPLATE')][0]
            template_name = template_name.replace('_', ' ').title()
            template_dir, templates = os.path.split(template_loc)

            return {
                    corr2_name: [corr2_dir, corr2_version],
                    casper_name: [casper_dir, casper_version],
                    katcp_name: [katcp_dir, katcp_version],
                    spead2_name: [spead2_dir, [spead2_version, spead2_link]],
                    mkat_name: [mkat_dir, None],
                    test_name: [test_dir, None],
                    template_name: [template_dir, templates],
                    }

        def get_package_versions():
            FNULL = open(os.devnull, 'w')
            for name, repo_dir in get_src_dir().iteritems():
                try:
                    git_hash = subprocess.check_output(['git', '--git-dir=%s/.git' % repo_dir[0],
                                                        '--work-tree=%s' % repo_dir[0], 'rev-parse',
                                                        '--short', 'HEAD'], stderr=FNULL).strip()

                    git_branch = subprocess.check_output(['git', '--git-dir=%s/.git' % repo_dir[0],
                                                              '--work-tree=%s'% repo_dir[0], 'rev-parse',
                                                              '--abbrev-ref', 'HEAD'], stderr=FNULL).strip()

                    Aqf.progress('Repo: %s, Git Dir: %s, Branch: %s, Last Hash: %s'%(name,
                                 repo_dir[0], git_branch, git_hash))

                    git_diff = subprocess.check_output(
                        ['git', '--git-dir=%s/.git'%(repo_dir[0]), '--work-tree=%s'%(repo_dir[0]),
                         'diff', 'HEAD'])
                    if bool(git_diff):
                        Aqf.progress('Repo: %s: Contains changes not committed.\n' % name)
                    else:
                        Aqf.progress('Repo: %s: Up-to-date.\n' % name)
                except subprocess.CalledProcessError:
                    if name == 'spead2':
                        Aqf.progress('Repo: %s, Version: %s, GitTag: %s\n'%(name, repo_dir[1][0],
                            repo_dir[1][1]))
                    else:
                        Aqf.progress('Repo: %s, Version: %s, Branch: Dirty, Last Hash: Dirty\n'%(name,
                                                                                      repo_dir[1]))
                except AssertionError:
                    Aqf.failed('AssertionError occurred while retrieving git repo: %s\n' % name)
                except OSError:
                    Aqf.failed('OS Error occurred while retrieving gut repo: %s\n' % name)

        def get_gateware_info():
            try:
                reply, informs = self.corr_fix.katcp_rct.req.version_list()
                assert reply.reply_ok()
            except AssertionError:
                Aqf.failed('Could not retrieve CBF Gate-ware Version Information')
            else:
                for inform in informs:
                    Aqf.progress(': '.join(inform.arguments))
                print

            # ToDO add this info
            # CORRELATOR MASTER CONTROLLER (CMC)              M1200-0012
            #     CORRELATOR BEAMFORMER SOFTWARE              M1200-0036
            #     CMC OPERATING SYSTEM                        M1200-0045
            #     CMC CORR2 PACKAGE                           M1200-0046
            #     CMC KATCP_PYTHON PACKAGE                    M1200-0053
            #     CMC CASPERFPGA PACKAGE                      M1200-0055
            #     CMC SPEAD2 PACKAGE                          M1200-0056
            #     CMC CONFIGURATION FILES                     M1200-0063
            #     CMC KATCP_C PACKAGE                         M1200-0047
            #     CMC CBF SCRIPTS                             M1200-0048

            # CORRELATOR BEAMFORMER GATEWARE (CBF)            M1200-0041
            #     F-ENGINE (CBF)                              M1200-0064
            #     X-ENGINE (CBF)                              M1200-0065
            #     B-ENGINE (CBF)                              M1200-0066
            #     X/B-ENGINE (CBF)                            M1200-0067

        test_heading('CBF Software Packages Version Information.')
        get_gateware_info()
        test_heading('CBF Git Version Information.')
        get_package_versions()
        test_heading('CBF Processing Node Version Information')
        try:
            assert 'roach' in str(self._hosts)
        except Exception:
            get_skarab_config()

    def _test_data_product(self, instrument):
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

        Aqf.step('Configure a digitiser simulator to generate correlated noise.')
        Aqf.progress('Digitiser simulator configured to generate Gaussian noise with scale: {}, '
                 'gain: {} and fft shift: {}.'.format(awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        Aqf.step('Configure the CBF to simultaneously generate baseline-correlation-products '
                 ' as-well as tied-array-channelised-voltage Data Products (If available).')
        try:
            Aqf.progress('Retrieving initial SPEAD accumulation, in-order to confirm the number of '
                         'channels in the SPEAD data.')
            test_dump = get_clean_dump(self)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            _parameters = parameters(self)
            no_channels = _parameters['n_chans']
            # Get baseline 0 data, i.e. auto-corr of m000h
            test_baseline = 0
            test_bls = _parameters['bls_ordering'][test_baseline]
            Aqf.equals(test_dump['xeng_raw'].shape[0], no_channels,
                       'Confirm that the baseline-correlation-products has the same number of '
                       'frequency channels ({no_channels}) corresponding to the {instrument} '
                       'instrument currently running,'.format(**locals()))
            Aqf.passed('and confirm that imaging data product set has been '
                       'implemented for instrument: {}.'.format(instrument))

            response = normalised_magnitude(test_dump['xeng_raw'][:, test_baseline, :])
            plot_filename = '{}/{}.png'.format(self.logs_path, self._testMethodName)

            caption = ('An overall frequency response at {} baseline, '
                       'when digitiser simulator is configured to generate Gaussian noise, '
                       'with scale: {}, eq gain: {} and fft shift: {}'.format(test_bls, awgn_scale,
                        gain, fft_shift))
            aqf_plot_channels(response, plot_filename, log_dynamic_range=90, caption=caption)


            Aqf.step('Check if Tied-Array Data capture is available.')
            labels = _parameters['input_labels']
            beam0_output_product = _parameters['beam0_output_product']
            beam1_output_product = _parameters['beam1_output_product']
            try:
                running_instrument = self.corr_fix.get_running_instrument()
                assert running_instrument is not False
                msg = 'Instrument does not have beamforming capabilities.'
                assert running_instrument.endswith('4k'), msg
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam0_output_product)
                assert reply.reply_ok(), str(reply)
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam1_output_product)
                assert reply.reply_ok(), str(reply)
            except Exception:
                return
            except AssertionError:
                errmsg = '%s'%str(reply).replace('\_', ' ')
                LOGGER.exception(errmsg)
                Aqf.failed(errmsg)
                return False
            else:
                bw = self.corr_freqs.bandwidth
                nr_ch = self.corr_freqs.n_chans
                dsim_clk_factor = 1.712e9 / self.corr_freqs.sample_freq

                # Start of test. Setting required partitions and center frequency
                partitions = 2
                part_size = bw / 16
                target_cfreq = bw + bw * 0.5
                target_pb = partitions * part_size
                ch_bw = bw / nr_ch
                beams = (beam0_output_product, beam1_output_product)
                beam = beams[1]

                # Set beamformer quantiser gain for selected beam to 1
                set_beam_quant_gain(self, beam, 1)

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
                    beam_y = self.corr_fix.corr_config['beam1']['output_products']
                    beam_y_ip, beam_y_port = self.corr_fix.corr_config['beam1']['output_destinations_base'].split(':')
                    beam_ip = beam_y_ip
                    beam_port = beam_y_port
                    beam_name = beam_y.replace('-','_').replace('.','_')
                    parts_to_process = 2
                    part_strt_idx = 0

                    try:
                        output = subprocess.check_output(['sudo', 'docker', 'run', 'hello-world'])
                        LOGGER.info(output)
                    except subprocess.CalledProcessError:
                        errmsg = 'Cannot connect to the Docker daemon. Is the docker daemon running on this host?'
                        LOGGER.error(errmsg)
                        Aqf.failed(errmsg)
                        return
                    try:
                        katcp_rct = self.corr_fix.katcp_rct.sensors
                        nr_ch = katcp_rct.n_chans.get_value()
                        assert isinstance(nr_ch,int)
                        ticks_between_spectra = katcp_rct.antenna_channelised_voltage_n_samples_between_spectra.get_value()
                        assert isinstance(ticks_between_spectra,int)
                        spectra_per_heap = getattr(katcp_rct, '{}_spectra_per_heap'.format(beam_name)).get_value()
                        assert isinstance(spectra_per_heap,int)
                        ch_per_heap = getattr(katcp_rct, '{}_n_chans_per_substream'.format(beam_name)).get_value()
                        assert isinstance(ch_per_heap,int)
                        return False
                    except Exception as e:
                        errmsg = 'Exception: {}'.format(str(e))
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                        return False
                    try:
                        docker_status = start_katsdpingest_docker(self, beam_ip, beam_port,
                                                              parts_to_process, nr_ch,
                                                              ticks_between_spectra,
                                                              ch_per_heap, spectra_per_heap)
                        assert docker_status
                        Aqf.progress('KAT SDP ingest node started successfully.')
                    except Exception:
                        Aqf.failed('KAT SDP Ingest Node failed to start')


                    bf_raw, cap_ts, bf_ts, in_wgts, pb, cf = capture_beam_data(
                        self, beam, beam_dict, target_pb, target_cfreq)
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
                    baseline_ch_bw = 856e6 / test_dump['xeng_raw'].shape[0]
                    beam_ch_bw = pb / len(cap_mag[0])
                    msg = ('Confirm that the baseline-correlation-product channel width'
                           ' {}Hz is the same as the tied-array-channelised-voltage channel width '
                           '{}Hz'.format(baseline_ch_bw, beam_ch_bw))
                    Aqf.almost_equals(baseline_ch_bw, beam_ch_bw, 1e-3, msg)

                    # Square the voltage data. This is a hack as aqf_plot expects squared
                    # power data
                    aqf_plot_channels(
                        np.square(cap_avg),
                        plot_filename='{}/{}_beam_resp_{}.png'.format(self.logs_path,
                            self._testMethodName, beam),
                        plot_title=('Beam = {}, Passband = {} MHz\nCentre Frequency = {} MHz'
                                    '\nIntegrated over {} captures'.format(beam, pb / 1e6, cf / 1e6, nc)),
                        log_dynamic_range=90, log_normalise_to=1,
                        caption=('Tied Array Beamformer data captured during Baseline Correlation '
                                 'Product test.'), plot_type='bf')
                except Exception as e:
                    Aqf.failed(str(e))


    def _test_control_vr(self):
        """
        CBF Control
        Test confirms all available CAM commands, executes a handful commands to confirm is the command
        is usable.
        """
        replies = self.corr_fix.katcp_rct.req.help()
        replies_str = str(replies).replace('\_',' ').splitlines()
        Aqf.step('List of available CAM commands, current status(executable or abandoned),'
                 'replies(if any)\n')
        requests_test = ['accumulation_length','capture_list','client_list','delays',
                         'group_list', 'instrument_list', 'log_default', 'log_level', 'log_limit',
                         'log_local', 'log_override', 'system_info', 'transient_buffer_trigger',
                         'watchdog']

        for reply in replies.informs:
            req_name = str(reply.arguments[0].replace('-', '_'))
            req_name_status = getattr(self.corr_fix.katcp_rct.req, req_name).is_active()
            if req_name in requests_test:
                try:
                    reply, informs = getattr(self.corr_fix.katcp_rct.req, req_name)()
                    assert reply.reply_ok()
                    msg = ('%s: Available and executable via CAM interface, '
                           'reply from CAM interface: %s.' % (req_name.upper(), str(reply)))
                except:
                    msg = 'Failed to retrieve reply from: %s via CAM Interface.' % req_name.upper()
            else:
                msg = '%s: Available and executable via CAM interface.' % req_name.upper()
            Aqf.progress(msg)

        Aqf.progress('Down-conversion frequency has not been implemented '
                     'in this release.')
        Aqf.progress('CBF Polarisation Correction has not been implemented '
                    'in this release.')

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
        if self.corr_freqs.n_chans == 4096:
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
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitiser simulator levels')
            return False

        self.addCleanup(set_default_eq, self)
        source = random.randrange(len(self.correlator.fops.fengines))
        _discards = 40
        try:
            initial_dump = get_clean_dump(self)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation, as Queue is Empty.'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
        else:
            _parameters = parameters(self)
            test_input = random.choice(_parameters['input_labels'])
            Aqf.step('Randomly selected input to test: {}'.format(test_input))
            # Get auto correlation index of the selected input
            bls_order = _parameters['bls_ordering']
            for idx, val in enumerate(bls_order):
                if val[0] == test_input and val[1] == test_input:
                    auto_corr_idx = idx

            n_chans = _parameters['n_chans']
            rand_ch = random.randrange(n_chans)
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
                        Aqf.failed('Gain correction on {} could not be set to {}.: '
                                   'KATCP Reply: {}'.format(test_input, gain, reply))
                        return
                else:
                    msg = ('Gain correction on input {}, channel {} set to {}.'.format(
                        test_input, rand_ch, complex(gain)))
                    Aqf.passed(msg)
                    try:
                        dump = get_clean_dump(self)
                        assert dump
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
                            legends.append('Gain set to {}'.format(complex(gain)))
                        elif fnd_less_one and (resp_diff > target):
                            msg = ('Output power increased by more than 6 dB '
                                   '(actual = {:.2f} dB) with a gain '
                                   'increment of {}.'.format(resp_diff, complex(gain_inc)))
                            Aqf.passed(msg)
                            found = True
                            chan_resp.append(response)
                            legends.append('Gain set to {}'.format(complex(gain)))
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
                  plot_title='Channel Response Gain Correction for channel {}'.format(rand_ch),
                  log_dynamic_range=90, log_normalise_to=1,
                  caption='Gain Correction channel response, gain varied for channel {}, '
                          'all remaining channels are set to {}'.format(rand_ch, complex(base_gain)))
            else:
                Aqf.failed('Could not retrieve channel response with gain/eq corrections.')

    def _test_beamforming(self):
        """
        Apply weights and capture beamformer data, Verify that weights are correctly applied.
        """
        # Main test code
        # TODO AR
        # Neccessarry to compare output products with capture-list output products?
        # Test both beams
        beam_x = self.corr_fix.corr_config['beam0']['output_products']
        beam_x_ip, beam_x_port = self.corr_fix.corr_config['beam0']['output_destinations_base'].split(':')
        beam_y = self.corr_fix.corr_config['beam1']['output_products']
        beam_y_ip, beam_y_port = self.corr_fix.corr_config['beam1']['output_destinations_base'].split(':')
        beam = beam_y
        beam_ip = beam_y_ip
        beam_port = beam_y_port
        beam_name = beam_y.replace('-','_').replace('.','_')
        parts_to_process = 2
        part_strt_idx = 0

        try:
            output = subprocess.check_output(['sudo', 'docker', 'run', 'hello-world'])
            LOGGER.info(output)
        except subprocess.CalledProcessError:
            errmsg = 'Cannot connect to the Docker daemon. Is the docker daemon running on this host?'
            LOGGER.error(errmsg)
            Aqf.failed(errmsg)
            return

        Aqf.step('Getting current instrument parameters.')
        try:
            katcp_rct = self.corr_fix.katcp_rct.sensors
            ants = katcp_rct.n_ants.get_value()
            assert isinstance(ants,int)
            bw = katcp_rct.bandwidth.get_value()
            assert isinstance(bw,float)
            nr_ch = katcp_rct.n_chans.get_value()
            assert isinstance(nr_ch,int)
            ticks_between_spectra = katcp_rct.antenna_channelised_voltage_n_samples_between_spectra.get_value()
            assert isinstance(ticks_between_spectra,int)
            spectra_per_heap = getattr(katcp_rct, '{}_spectra_per_heap'.format(beam_name)).get_value()
            assert isinstance(spectra_per_heap,int)
            ch_per_heap = getattr(katcp_rct, '{}_n_chans_per_substream'.format(beam_name)).get_value()
            assert isinstance(ch_per_heap,int)
            ch_freq = bw / nr_ch
            f_start = 0.  # Center freq of the first bin
            ch_list = f_start + np.arange(nr_ch) * ch_freq
            partitions = int(nr_ch / ch_per_heap)
            part_size = bw / partitions
            # Setting required partitions and center frequency
            #target_cf = bw + bw * 0.5
            # Partitions are no longer selected. The full bandwidth is always selected
            # target_pb = partitions * part_size
            target_pb = bw
            target_cf = bw + target_pb / 2
        except AssertionError:
            errmsg = 'Failed to get instrument parameters'
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        Aqf.progress('Bandwidth = {}Hz'.format(bw))
        Aqf.progress('Number of channels = {}'.format(nr_ch))
        Aqf.progress('Channel spacing = {}Hz'.format(ch_freq))
        Aqf.step('Start a KAT SDP docker inject node for beam captures')
        docker_status = start_katsdpingest_docker(self, beam_ip, beam_port,
                                                  parts_to_process, nr_ch,
                                                  ticks_between_spectra,
                                                  ch_per_heap, spectra_per_heap)
        if docker_status:
            Aqf.progress('KAT SDP Ingest Node started')
        else:
            Aqf.failed('KAT SDP Ingest Node failed to start')

        # Set source names and stop all streams
        _parameters = parameters(self)
        local_src_names = _parameters['custom_src_names']
        try:
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam_x)
            assert reply.reply_ok()
            reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam_y)
            assert reply.reply_ok()
            #reply, informs = self.corr_fix.katcp_rct.req.capture_stop('c856M4k')
            #assert reply.reply_ok()
            reply, informs = self.corr_fix.katcp_rct.req.input_labels(*local_src_names)
            assert reply.reply_ok()
            labels = reply.arguments[1:]
        except AssertionError as e:
            errmsg = 'KatCP request failed: {}'.format(reply)
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False
        except Exception as e:
            errmsg = 'Exception: {}'.format(str(e))
            Aqf.failed(errmsg)
            LOGGER.exception(errmsg)
            return False



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

        Aqf.step('Configure a digitiser simulator to generate correlated noise.')
        Aqf.progress('Digitiser simulator configured to generate Gaussian noise, '
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(
            awgn_scale, gain, fft_shift))

        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale,
                                            cw_scale=0.0, freq=cw_freq, fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False


        def get_beam_data(beam, beam_dict, inp_ref_lvl=0, beam_quant_gain=1, num_caps=10000,
            max_cap_retries=5):

            # Determine slice of valid data in bf_raw
            bf_raw_str = part_strt_idx * ch_per_heap
            bf_raw_end = bf_raw_str + parts_to_process * ch_per_heap

            # Capture beam data, retry if more than 20% of heaps dropped or empty data
            retries = 0
            while retries < max_cap_retries:
                if retries == max_cap_retries - 1:
                    Aqf.failed('Error capturing beam data.')
                    return False
                retries += 1
                try:
                    bf_raw, bf_flags, bf_ts, in_wgts, pb, cf = capture_beam_data(self, beam,
                        beam_dict, target_pb, target_cf)

                except Exception as e:
                    Aqf.step('Confirm that the Docker container is running and also confirm the '
                        'igmp version = 2')
                    errmsg = 'Failed to capture beam data due to error: %s' % str(e)
                    Aqf.failed(errmsg)
                    LOGGER.error(errmsg)
                    return False

                data_type = bf_raw.dtype.name
                #for heaps in bf_flags:
                # Cut selected partitions out of bf_flags
                flags = bf_flags[part_strt_idx:part_strt_idx+parts_to_process]
                idx = part_strt_idx
                #Aqf.step('Finding missed heaps for all partitions.')
                if flags.size == 0:
                    LOGGER.warning('Beam data empty. Capture failed. Retrying...')
                else:
                    missed_err = False
                    for part in flags:
                        missed_heaps = np.where(part>0)[0]
                        missed_perc = missed_heaps.size/part.size
                        perc = 0.5
                        if missed_perc > perc:
                            LOGGER.warning('Beam captured missed more than %s% heaps. Retrying...'%(perc*100))
                            missed_err = True
                            break
                    # Good capture, break out of loop
                    if not missed_err:
                        break

            # Print missed heaps
            idx = part_strt_idx

            for part in flags:
                missed_heaps = np.where(part>0)[0]
                if missed_heaps.size > 0:
                    Aqf.progress('Missed heaps for partition {} at heap indexes {}'.format(idx,
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
            Aqf.step('Confirm that the data type of the beamforming data for one channel.')
            try:
                msg = ('Beamformer data type is {}, example value for one channel: {}'.format(
                    data_type, cap[0][0]))
            except Exception as e:
                errmsg = "Failed with exception: %s" %str(e)
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

            Aqf.equals(data_type, 'int8', msg)
            cap_mag = np.abs(cap)
            cap_avg = cap_mag.sum(axis=0) / cap_idx
            cap_db = 20 * np.log10(cap_avg)
            cap_db_mean = np.mean(cap_db)
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
            for key in in_wgts:
                labels += (key + "= {}\n").format(in_wgts[key])
            labels += 'Mean = {:0.2f}dB\n'.format(cap_db_mean)

            if inp_ref_lvl == 0:
                # Get the voltage level for one antenna. Gain for one input
                # should be set to 1, the rest should be 0
                inp_ref_lvl = np.mean(cap_avg)
                Aqf.step('Reference level measured by setting the '
                         'gain for one antenna to 1 and the rest to 0. '
                         'Reference level = {:.3f}dB'.format(20*np.log10(inp_ref_lvl)))
                Aqf.step('Reference level averaged over {} channels. '
                         'Channel averages determined over {} '
                         'samples.'.format(parts_to_process*ch_per_heap, cap_idx))
                expected = 0
            else:
                delta = 0.2
                expected = np.sum([inp_ref_lvl * in_wgts[key] for key in in_wgts]) * beam_quant_gain
                expected = 20 * np.log10(expected)

                Aqf.step('Expected value is calculated by taking the reference input level '
                         'and multiplying by the channel weights and quantiser gain.')
                labels += 'Expected = {:.2f}dB'.format(expected)
                msg = ('Confirm that the expected voltage level ({:.3f}dB) is within '
                    '{}dB of the measured mean value ({:.3f}dB)'.format(expected,delta, cap_db_mean))
                Aqf.almost_equals(cap_db_mean, expected, delta, msg)


            return cap_avg, labels, inp_ref_lvl, pb, cf, expected, cap_idx

        beam_data = []
        beam_lbls = []

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0
        if len(beam_dict) == 0:
            Aqf.failed('Beam dictionary not created, beam labels or beam name incorrect')
            return False

        # Only one antenna gain is set to 1, this will be used as the reference
        # input level
        # Set beamformer quantiser gain for selected beam to 1
        Aqf.step('Testing individual beam weights.')
        set_beam_quant_gain(self, beam, 1)
        weight = 1.0
        beam_dict = populate_beam_dict(self, 1, weight, beam_dict)
        rl = 0
        try:
            d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, rl)
        except TypeError, e:
            errmsg = 'Failed to retrieve beamformer data'
            Aqf.failed(errmsg)
            LOGGER.error(errmsg)
            return
        else:
            beam_data.append(d)
            beam_lbls.append(l)

            weight = 1.0 / ants
            beam_dict = populate_beam_dict(self, -1, weight, beam_dict)
            try:
                d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, rl)
            except IndexError, e:
                errmsg = 'Failed to retrieve beamformer data'
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)

            weight = 2.0 / ants
            beam_dict = populate_beam_dict(self, -1, weight, beam_dict)
            try:
                d, l, rl, pb, cf, exp1, nc = get_beam_data(beam, beam_dict, rl)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
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
                              plot_title=('Beam = {}, Passband = {} MHz\nCenter Frequency = {} MHz'
                                          '\nIntegrated over {} captures'.format(beam, pb / 1e6,
                                                                                 cf / 1e6, nc)),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='Captured beamformer data', hlines=[exp0, exp1],
                              plot_type='bf', hline_strt_idx=1)

            Aqf.step('Testing quantiser gain adjustment.')
            beam_data = []
            beam_lbls = []
            # Set beamformer quantiser gain for selected beam to 1/number inputs
            gain = 1.0
            gain = set_beam_quant_gain(self, beam, gain)
            weight = 1.0 / ants
            beam_dict = populate_beam_dict(self, -1, weight, beam_dict)
            rl = 0

            try:
                d, l, rl, pb, cf, exp0, nc = get_beam_data(beam, beam_dict, rl, gain)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            l += '\nLevel adjust gain={}'.format(gain)
            beam_lbls.append(l)

            gain = 0.5
            gain = set_beam_quant_gain(self, beam, gain)

            try:
                d, l, rl, pb, cf, exp1, nc = get_beam_data(beam, beam_dict, rl, gain)
            except Exception as e:
                errmsg = 'Failed to retrieve beamformer data: %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.error(errmsg)
                return
            beam_data.append(d)
            l += '\nLevel adjust gain={}'.format(gain)
            beam_lbls.append(l)

            # Square the voltage data. This is a hack as aqf_plot expects squared
            # power data
            aqf_plot_channels(zip(np.square(beam_data), beam_lbls),
                              plot_filename='{}/{}_level_adjust_after_bf_{}.png'.format(self.logs_path,
                                self._testMethodName, beam),
                              plot_title=('Beam = {}, Passband = {} MHz\nCentre Frequency = {} MHz'
                                          '\nIntegrated over {} captures'.format(beam, pb / 1e6,
                                                                                 cf / 1e6, nc)),
                              log_dynamic_range=90, log_normalise_to=1,
                              caption='Captured beamformer data with level adjust after beam-forming gain set.',
                              hlines=exp1, plot_type='bf', hline_strt_idx=1)

        # Close any KAT SDP ingest nodes
        stop_katsdpingest_docker(self)

    def _test_cap_beam(self, instrument='bc8n856M4k'):
        """Testing timestamp accuracy
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

    # def _test_bc8n856M4k_beamforming_ch(self, instrument='bc8n856M4k'):
    #     """CBF Beamformer channel accuracy

    #     Apply weights and capture beamformer data.
    #     Verify that weights are correctly applied.
    #     """
    #     instrument_success = self.set_instrument(instrument)
    #     _running_inst = self.corr_fix.get_running_instrument()
    #     if instrument_success and _running_inst:
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

        local_src_names = parameters(self)['custom_src_names']
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
        set_beam_quant_gain(self, beam, 1)

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
        ch_bw = self.corr_freqs.delta_f
        acc_time = self.corr_freqs.fft_period
        sqrt_bw_at = np.sqrt(ch_bw * acc_time)
        Aqf.step('Calculating channel efficiency.')
        eff = 1 / ((ch_std / ch_mean) * sqrt_bw_at)
        Aqf.step('Beamformer mean efficiency for {} channels = {:.2f}%'
                 ''.format(nr_ch, 100 * eff.mean()))
        plt_filename = '{}/{}_beamformer_efficiency.png'.format(self.logs_path,
            self._testMethodName)
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
        sync_time = parameters(self)['synch_epoch']
        scale_factor_timestamp = parameters(self)['scale_factor_timestamp']
        inp = parameters(self)['input_labels'][0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = dump['n_chans'].value * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = parameters(self)['int_time'] * parameters(self)['adc_sample_rate']
        # print dump_ticks
        dump_ticks = parameters(self)['n_accs'] * parameters(self)['n_chans'] * 2
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

    def _test_timestamp_shift(self, instrument='bc8n856M4k'):
        """Testing timestamp accuracy
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

        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self.corr_fix, self.dhost,
                                            awgn_scale=0.0,
                                            cw_scale=0.0, freq=100000000,
                                            fft_shift=0, gain='32767+0j')
        self.dhost.outputs.out_1.scale_output(0)
        dump = get_clean_dump(self)
        baseline_lookup = get_baselines_lookup(self, dump)
        sync_time = parameters(self)['synch_epoch'].value
        scale_factor_timestamp = parameters(self)['scale_factor_timestamp']
        inp = parameters(self)['input_labels'][0][0]
        inp_autocorr_idx = baseline_lookup[(inp, inp)]
        # FFT input sliding window size = 8 spectra
        fft_sliding_window = parameters(self)['n_chans'] * 2 * 8
        # Get number of ticks per dump and ensure it is divisible by 8 (FPGA
        # clock runs 8 times slower)
        dump_ticks = parameters(self)['int_time'] * parameters(self)['adc_sample_rate']
        dump_ticks = parameters(self)['n_accs'] * parameters(self)['n_chans'] * 2
        input_spec_ticks = parameters(self)['n_chans'] * 2
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
                sync_time = parameters(self)['synch_epoch']
                scale_factor_timestamp = parameters(self)['scale_factor_timestamp']
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

    def _test_input_levels(self, instrument='bc8n856M4k'):
        """Testing Digitiser simulator input levels
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        if self.set_instrument(instrument):
            Aqf.step('Setting and checking Digitiser simulator input levels: {}\n'.format(
                self.corr_fix.get_running_instrument()))
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

            bw = self.corr_freqs.bandwidth
            nr_ch = self.corr_freqs.n_chans
            ch_bw = self.corr_freqs.chan_freqs[1]
            ch_list = self.corr_freqs.chan_freqs
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
                n_accs = parameters(self)['n_accs']
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
                                    parameters(self)['n_chans']))
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
                            parameters(self)['n_chans']))
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
                 'with scale: {}, eq gain: {}, fft shift: {}'.format(awgn_scale, gain,
                                                                          fft_shift))
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, cw_scale=0.0,
                                            fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            inp = parameters(self)
            assert inp
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
                           plot_filename='{}/{}_adc_hist_{}.png'.format(self.logs_path,
                                self._testMethodName, inp),
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
                           plot_filename='{}/{}_quant_hist_{}.png'.format(self.logs_path,
                                self._testMethodName, inp),
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
                dump = get_clean_dump(self)
            except Queue.Empty:
                errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                baseline_lookup = get_baselines_lookup(self, dump)
                inp_autocorr_idx = baseline_lookup[(inp, inp)]
                acc_time = parameters(self)['int_time']
                ch_bw = self.corr_freqs.delta_f
                dval = dump['xeng_raw']
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
                        dump = self.receiver.data_queue.get()
                    except Queue.Empty:
                        errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
                        Aqf.failed(errmsg)
                        LOGGER.exception(errmsg)
                    else:
                        dval = dump['xeng_raw']
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

        plt_filename = '{}/{}_correlator_efficiency.png'.format(self.logs_path,
            self._testMethodName)
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
        dsim_set_success = False
        with RunTestWithTimeout(dsim_timeout, errmsg='D-Engine configuration timed out, failing test'):
            dsim_set_success =set_input_levels(self, awgn_scale=awgn_scale, cw_scale=cw_scale,
                                            freq=cw_freq, fft_shift=fft_shift, gain=gain, cw_src=0)
        if not dsim_set_success:
            Aqf.failed('Failed to configure digitise simulator levels')
            return False

        try:
            labels = parameters(self)['input_labels']
            assert labels
        except AssertionError:
            Aqf.failed('Failed to retrieve input labels via CAM interface')

        try:
            Aqf.step('Confirm that the `Transient Buffer ready` is implemented.')
            reply, informs = self.corr_fix.katcp_rct.req.transient_buffer_trigger()
            Aqf.passed('Transient buffer trigger present.')
        except Exception:
            Aqf.failed('Transient buffer trigger failed.')

        label = labels[0]
        try:
            Aqf.step('Capture an ADC snapshot and confirm the fft length')
            reply, informs = self.corr_fix.katcp_rct.req.adc_snapshot(label)
        except Exception:
            Aqf.failed('Failed to grab adc snapshot.')
        fpga = self.correlator.fhosts[0]
        adc_data = fpga.get_adc_snapshots()['p0'].data
        fft_len = len(adc_data)
        Aqf.progress('ADC capture length: {}'.format(fft_len))
        fft_real = np.abs(np.fft.fft(adc_data))
        fft_pos = fft_real[0:int(fft_len / 2)]
        cw_chan = np.argmax(fft_pos)
        cw_freq_found = cw_chan / (fft_len / 2) * bw
        msg = ('Confirm that the expected frequency: {}Hz and measured frequency: '
               '{}Hz matches to within a channel bandwidth: {:.3f}Hz'.format(cw_freq_found,
                    cw_freq, ch_bw))
        Aqf.almost_equals(cw_freq_found, cw_freq, ch_bw, msg)
        aqf_plot_channels(np.log10(fft_pos),
                          plot_filename='{}/{}_fft_{}.png'.format(self.logs_path,
                                self._testMethodName, label),
                          plot_title=('Input Frequency = {} Hz\nMeasured Frequency at FFT bin {} '
                                      '= {}Hz'.format(cw_freq, cw_chan, cw_freq_found)),
                          log_dynamic_range=None,
                          caption=('FFT of captured small voltage buffer. {} voltage points captured '
                                   'on input {}. Input bandwidth = {}Hz'.format(fft_len, label, bw)),
                          xlabel='FFT bins')
