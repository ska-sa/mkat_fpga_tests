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
import json
import os
import Queue
import random
import socket
import struct
import subprocess
import sys
import time
import unittest
import re
import math
import multiprocessing
import hashlib
from ast import literal_eval as evaluate
from datetime import datetime

from nose.plugins.attrib import get_method_attr

import casperfpga
import corr2
import katcp
import h5py
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
    if os.path.exists("new_sensor.log"):
        os.remove("new_sensor.log")

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
            self.data_retries = int(self.conf_file["instrument_params"]["data_retries"])
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
 
        f_bitstream = self.cam_sensors.get_value('fengine_bitstream')
        x_bitstream = self.cam_sensors.get_value('xengine_bitstream') 
        feng_param = self.fpg_compile_parameter_check(f_bitstream)
        xeng_param = self.fpg_compile_parameter_check(x_bitstream)
        try:
            assert feng_param and xeng_param
        except AssertionError:
            errmsg = "FPG compile parameters incorrect."
            Aqf.failed(errmsg)
            sys.exit(errmsg)

        errmsg = "Failed to instantiate the dsim, investigate"
        try:
            self.dhost = self.corr_fix.dhost
            if not isinstance(self.dhost, corr2.dsimhost_fpga.FpgaDsimHost):
                raise AssertionError(errmsg)
            elif ("856M" in self.corr_fix.instrument) or ("107M" in self.corr_fix.instrument) or ("54M" in self.corr_fix.instrument):
                nominal_sample_freq = float(self.conf_file["instrument_params"]["sample_freq_l"])
            elif ("875M" in self.corr_fix.instrument) or ("109M" in self.corr_fix.instrument) or ("55M" in self.corr_fix.instrument):
                nominal_sample_freq = float(self.conf_file["instrument_params"]["sample_freq_s"])
            elif ("544M" in self.corr_fix.instrument) or ("68M" in self.corr_fix.instrument) or ("34M" in self.corr_fix.instrument):
                nominal_sample_freq = float(self.conf_file["instrument_params"]["sample_freq_u"])
            self.dsim_factor = (nominal_sample_freq 
                / self.cam_sensors.get_value("scale_factor_timestamp"))
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
                    # This is only setting the value that was just read
                    #reply, informs = self.katcp_req.sync_epoch(sync_time)
                    #self.assertTrue(reply.reply_ok(), msg="Failed to set digitiser sync epoch")
                    #self.logger.info("Digitiser sync epoch set successfully")
                    SET_DSIM_EPOCH = self._dsim_set = True
                except AssertionError as e:
                    self.Error(e, exe_info=True)
                except Exception as e:
                    self.Error('{}'.format(e), exc_info=True)
        self.start_time = str(datetime.now()) # record start time of test method
        #print '**self.start_time = ', self.start_time 

    # This needs proper testing
    def tearDown(self):
        try:
            self.katcp_req = None
            assert not self.receiver
        #except AssertionError:
        #    self.logger.info("Cleaning up the receiver!!!!")
        #    assert not self.receiver
        #except AssertionError:
        #    self.logger.info("Cleaning up the receiver!!!!")
        #    assert not self.receiver
        except AssertionError:
            self.logger.info("Cleaning up the receiver!!!!")
            #add_cleanup(self.receiver.stop)
            self.Step("Waiting for receiver to stop.")
            self.receiver.stop()
            #TODO Investigate why this is need
            try:
                #dump = self.receiver.get_clean_dump(dump_timeout = 10)
                self.get_real_clean_dump(quiet=True)
                self.logger.info("Got a clean dump while trying to stop the receiver.")
            except Exception as e:
                self.logger.info("Getting dump while stopping failed with exception: {}".format(e))
            #retry_count = 10
            #while retry_count > 0:
            #    try:
            #        _test_dump = self.receiver.get_clean_dump()
            #        break
            #    except Exception as e:
            #        if retry_count == 0:
            #            raise(e)
            #        self.logger.info("Got exception during get_clean_dump "
            #                         "(sleeping for 10 seconds then trying {} more times): "
            #                         "{}".format(retry_count, e))
            #        time.sleep(10)
            #        retry_count -= 1
            #TODO Join timeout to be investigated
            self.receiver.join(timeout = 1)
            if self.receiver.stopped():
                self.Step("SPEAD receiver has been stopped.")
            else:
                self.Error("Could not stop the receiver, memory leaks might occur.")
            del self.receiver
            #self.logger.info("Sleeping for 60 seconds to clean up memory.")
            #time.sleep(60)
        self.end_time = str(datetime.now()) # record end time of test method
        #print '**self.end_time = ', self.end_time
        #print '**self.id = ', self.id
        try:
            assert evaluate(os.getenv("SENSOR_LOGS", "False"))
            print("Sensor logs enabled.")
            self.get_sensor_logs() # call method for parsing sensor logs
        except AssertionError: 
            print("Sensor logs disabled.")

    def set_instrument(self, acc_time=None, start_channel=None, stop_channel=None, start_receiver=True, **kwargs):
        #self.receiver = None
        acc_timeout = 60
        self.errmsg = None
        # Reset digitiser simulator to all Zeros
        init_dsim_sources(self.dhost)
        self.addCleanup(init_dsim_sources, self.dhost)
        n_ants = int(self.cam_sensors.get_value("n_ants"))
        n_chans = int(self.cam_sensors.get_value("antenna_channelised_voltage_n_chans"))

        try:
            self.Step("Confirm running instrument, else start a new instrument")
            self.instrument = self.cam_sensors.get_value("instrument_state").split("_")[0]
            self.Progress(
                "Currently running instrument %s-%s as per /etc/corr" % (
                    self.corr_fix.array_name,
                    self.instrument))
            #self.Progress("Test configuration as per %s" % (self.conf_file["instrument_params"]["conf_file"]))
            #TODO: Add receiver back in
            if start_receiver:
                #self._systems_tests()
                pass
        except Exception:
            errmsg = "No running instrument on array: %s, Exiting...." % self.corr_fix.array_name
            self.Error(errmsg, exc_info=True)
            Aqf.end(message=errmsg)
            sys.exit(errmsg)

        if self._dsim_set:
            self.Step("Configure a digitiser simulator to be used as input source to F-Engines.")
            self.Progress("Digitiser Simulator running on host: %s" % self.dhost.host)
        #Stop all data streams    
        try:
            reply, informs = self.katcp_req.capture_list()
            self.assertTrue(reply.reply_ok())
            all_streams = []
            for msg in informs:
                if ('tied' or 'baseline') in msg.arguments[0]:
                    all_streams.append(msg.arguments[0])
            for stream in all_streams:
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(stream)
                self.assertTrue(reply.reply_ok())
        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False

        #Set accumulation time
        if acc_time:
            pass
        elif ("bc8" in self.instrument) or ("bc16" in self.instrument) or ("bc32" in self.instrument):
                acc_time = float(self.conf_file["instrument_params"]["accumulation_time_4-16ant"])
        #elif ("bc128n107M" in self.instrument) or ("bc128n54M" in self.instrument):
        #    acc_time = float(self.conf_file["instrument_params"]["accumulation_time_64ant_nb"])
        elif ("bc128" in self.instrument) and ("32k" in self.instrument):
                acc_time = float(self.conf_file["instrument_params"]["accumulation_time_64ant_32k"])
        elif ("bc64" in self.instrument) or ("bc128" in self.instrument):
                acc_time = float(self.conf_file["instrument_params"]["accumulation_time_32-64ant"])
        else:
            acc_time = 0.5

        for i in range(self.data_retries):  
            reply, informs = self.katcp_req.accumulation_length(acc_time, timeout=acc_timeout)
            if reply.reply_ok() == True:
                acc_time = float(reply.arguments[-1])
                self.Step("Set and confirm accumulation period via CAM interface.")
                self.Progress("Accumulation time set to {:.3f} seconds".format(acc_time))
                break
        try:
            self.assertTrue(reply.reply_ok())
        except AssertionError:
            try:
                if informs: pass
            except UnboundLocalError:
                informs = ''
            self.Error("Failed to set accumulation time: {},{}".format(reply, informs), exc_info=True)

        if start_receiver:
            init_receiver = False
            if 'self.receiver' not in locals():
                init_receiver = True
                self.logger.info('Receiver not in locals')
            elif self.receiver == None:
                init_receiver = True
                self.logger.info('Receiver is apparently None: {}'.format(self.receiver))
            try:
                if init_receiver:
                    data_output_ip, data_output_port = self.cam_sensors.get_value(
                        self.corr_fix.xeng_product_name.replace("-", "_") + "_destination"
                    ).split(":")
                    self.logger.info(
                        "Starting SPEAD receiver listening on %s:%s, CBF output product: %s"
                        % (data_output_ip, data_output_port, self.corr_fix.xeng_product_name)
                    )
                    katcp_ip = self.corr_fix.katcp_client
                    katcp_port = int(self.corr_fix.katcp_rct.port)
                    self.Step("Connected to katcp on %s" % katcp_ip)
                    if not(start_channel):
                        #if ("bc128n107M" in self.instrument) or ("bc128n54M" in self.instrument):
                        #    start_channel = int(self.conf_file["instrument_params"].get("start_channel_64ant_nb", 0))
                        if ("bc128" in self.instrument) and ("32k" in self.instrument):
                            start_channel = int(self.conf_file["instrument_params"].get("start_channel_64ant_32k", 0))
                        else:
                            start_channel = int(self.conf_file["instrument_params"].get("start_channel", 0))
                    if not(stop_channel):
                        if ("bc128" in self.instrument) and ("32k" in self.instrument):
                            stop_channel = int(self.conf_file["instrument_params"].get("stop_channel_64ant_32k", 32767))
                        elif ("1k" in self.instrument):
                            stop_channel = int(self.conf_file["instrument_params"].get("stop_channel_1k", 1024))
                        elif ("4k" in self.instrument):
                            stop_channel = int(self.conf_file["instrument_params"].get("stop_channel_4k", 4096))
                        elif ("32k" in self.instrument):
                            stop_channel = int(self.conf_file["instrument_params"].get("stop_channel_32k", 32767))
                    if stop_channel > n_chans:
                        self.logger.warn('Stop channels in config file is higher that available '
                                            'for this instrument. Setting to {}'.format(n_chans))
                        stop_channel = n_chans-1
                        
                    self.Note(
                        "Requesting SPEAD receiver to capture %s channels from %s to %s on port %s."
                        % (stop_channel - start_channel + 1, start_channel, stop_channel, data_output_port)
                    )
                    self.receiver = CorrRx(
                        product_name=self.corr_fix.xeng_product_name,
                        katcp_ip=katcp_ip,
                        katcp_port=katcp_port,
                        port=data_output_port,
                        channels=(start_channel, stop_channel),
                    )
                    self.receiver.setName("CorrRx Thread")
                    self.errmsg = "Failed to create SPEAD data receiver"
                    self.assertIsInstance(self.receiver, CorrRx), self.errmsg
                    #start_thread_with_cleanup(self, self.receiver, timeout=10, start_timeout=1)
                    self.logger.info("Starting receiver.")
                    self.receiver.start()
                    self.logger.info("Receiver start method returned.")
                    retry_count = 10
                    while retry_count > 0:
                        if self.receiver.isAlive() == False:
                            self.logger.info("Receiver is not alive yet, waiting 10 seconds and trying again.")
                            time.sleep(10)
                            retry_count -= 1
                        else:
                            break
                    self.errmsg = "Spead Receiver not Running, possible "
                    self.assertTrue(self.receiver.isAlive(), msg=self.errmsg)
                    self.corr_fix.start_x_data
                    self.logger.info(
                        "Getting a test dump to confirm number of channels else, test fails "
                        "if cannot retrieve dump"
                    )
                    retry_count = 10
                    while retry_count > 0:
                        try:
                            _test_dump = self.receiver.get_clean_dump()
                            self.assertIsInstance(_test_dump, dict)
                            break
                        except Exception as e:
                            if retry_count == 0:
                                self.errmsg = "No data received, data que empty."
                                self.Failed(self.errmsg)
                                raise(e)
                            self.logger.info("Got exception during get_clean_dump "
                                             "(sleeping for 10 seconds then trying {} more times): "
                                             "{}".format(retry_count, e))
                            time.sleep(10)
                            retry_count -= 1
                    self.n_chans_selected = int(_test_dump.get("n_chans_selected",
                        self.cam_sensors.get_value("antenna_channelised_voltage_n_chans"))
                    )
                    self.start_channel = int(_test_dump.get("start_channel",0))
                    self.stop_channel  = int(_test_dump.get("stop_channel",
                        self.cam_sensors.get_value("antenna_channelised_voltage_n_chans"))
                    )
                    self.Note(
                            "Actual number of channels captured (channels are captured in partitions): %s." % self.n_chans_selected
                    )
                    self.Note(
                            "Capturing from channel {} to {}.".format(self.start_channel, self.stop_channel)
                    )
            except Exception as e:
                self.Error(self.errmsg)
                self.Error("Exception: {}".format(e), exc_info=True)
                return False
            else:
                # Run system tests before each test is ran
                #TODO: Add systems test back in
                #self.addCleanup(self._systems_tests)
                self.addCleanup(self.corr_fix.stop_x_data)
                #if init_receiver:
                #    self.addCleanup(self.receiver.stop)
        self.addCleanup(executed_by)
        self.addCleanup(gc.collect)
        return True

    def fpg_compile_parameter_check(self, bitstream_path):
        """Verify the compile parameters in fpg/bitstream file are correct. Parameters are verified against
        values in json file.
        
        Parameters
        ----------
        bitstream_path : str
            Absolute path of fpg/bitstream file
        
        Returns
        -------
        bool
            Returns True if compile parameters are correct and returns False if incorrect.
        """
        with open('compile_parameters.json', 'r') as json_file:
            json_dict = json.load(json_file)
        bitstream_filename = os.path.basename(bitstream_path)
        fpg_param = casperfpga.casperfpga.parse_fpg(bitstream_path)[0]['77777'] 

        if re.search('s_c', bitstream_filename): #f-engine bitstreams
            try:
                fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs'])
                    
                fft_stages_fpg = int(fpg_param['fft_stages'])
                n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])

                self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                return True
            except AssertionError:
                msg1 = '''Compile parameters of {} are incorrect.
                          fft_stages = {} (expected: {}).
                          n_bits_xengs = {} (expected: {}).
                       '''.format(bitstream_filename, fft_stages_fpg, fft_stages_default,n_bits_xengs_fpg, n_bits_xengs_default)
                self.Failed(msg1)
                return False
        elif re.search('s_b', bitstream_filename): #x/b-engine bitstreams 
            if re.search('_nb|_54', bitstream_filename): #narrowband x/b-engine bitstreams
                try:
                    fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                    n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs']) 
                    n_bits_ants_default = int(json_dict[bitstream_filename]['n_bits_ants'])
                        
                    fft_stages_fpg = int(fpg_param['fft_stages'])
                    n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])
                    n_bits_ants_fpg = int(fpg_param['n_bits_ants'])
                        
                    self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                    self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                    self.assertEqual(n_bits_ants_fpg, n_bits_ants_default, 'n_bits_ants parameter incorrect.')
                    return True
                except AssertionError:
                    msg2 = '''Compile parameters of {} are incorrect.
                              fft_stages = {} (expected: {}).
                              n_bits_xengs = {} (expected: {}).
                              n_bits_ants = {} (expected: {}).
                           '''.format(bitstream_filename, fft_stages_fpg, fft_stages_default,n_bits_xengs_fpg, n_bits_xengs_default,n_bits_ants_fpg,n_bits_ants_default)
                    self.Failed(msg2)
                    return False
            elif not re.search('_nb|_54', bitstream_filename): #wideband x/b-engine bitstreams
                try:
                    fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                    n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs'])
                    n_bits_ants_default = int(json_dict[bitstream_filename]['n_bits_ants'])
                    n_bits_beams_default = int(json_dict[bitstream_filename]['n_bits_beams'])
                        
                    fft_stages_fpg = int(fpg_param['fft_stages'])
                    n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])
                    n_bits_ants_fpg = int(fpg_param['n_bits_ants'])
                    n_bits_beams_fpg = int(fpg_param['n_bits_beams'])

                    self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                    self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                    self.assertEqual(n_bits_ants_fpg, n_bits_ants_default, 'n_bits_ants parameter incorrect.')
                    self.assertEqual(n_bits_beams_fpg, n_bits_beams_default, 'n_bits_beams parameter incorrect.')
                    return True
                except AssertionError:
                    msg3 = '''Compile parameters of {} are incorrect.
                              fft_stages = {} (expected: {}).
                              n_bits_xengs = {} (expected: {}).
                              n_bits_ants = {} (expected: {}).
                              n_bits_beams = {} (expected: {}).
                           '''.format(bitstream_filename , fft_stages_fpg, fft_stages_default,n_bits_xengs_fpg, n_bits_xengs_default,n_bits_ants_fpg,n_bits_ants_default,n_bits_beams_fpg,n_bits_beams_default)
                    self.Failed(msg3)
                    return False

    #################################################
    #@ijtesting
    def dummy_test(self):
        start_time = str(datetime.now()) 
        my_procedure = '''
          **Dummy Test Procedure**
          1: Step one
          2: Step two
          3: Step three
          4: Step four
          '''

        Aqf.procedure(my_procedure)
        try:
            assert evaluate(os.getenv("DRY_RUN", "False"))
        except AssertionError:
            instrument_success = self.set_instrument()
            if True:
                Aqf.note("Instrument success for dummy test.")
                # reply, informs = self.katcp_req.sensor_list(timed=60)
        finally:
            #self.get_sensor_logs(start_time)
            pass
        Aqf.end(passed=True, message="End of dummy test.")

    #################################################    

    #@tbd
    #@skipped_test
    #@subset
    @array_release_x
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053", "CBF-REQ-0226", "CBF-REQ-0227", "CBF-REQ-0236" )
    def test_channelisation(self):
        Aqf.procedure(TestProcedure.Channelisation)
        if 'skipped_test' in test_CBF.__dict__['test_channelisation'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_channelisation'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                    if ((("107M32k" in self.instrument) or ("54M32k" in self.instrument) or ("68M32k" in self.instrument) or ("34M32k" in self.instrument)) and
                        (self.start_channel == 0)):
                        check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
                        check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
                        test_chan = random.choice(range(n_chans)[check_strt_ch:check_stop_ch])
                    else:
                        test_chan = random.choice(range(self.start_channel,
                            self.start_channel+self.n_chans_selected))
                    heading("CBF Channelisation")
                    # Figure out what this value should really be for different integrations
                    # 3 worked for CMC1 june 2019
                    # TODO: automate this by checking how long data takes to travel through integrations
                    num_discards = int(self.conf_file["instrument_params"]["num_discards"])
                    smpl_per_ch  = int(self.conf_file["instrument_params"]["num_channelisation_samples"])
                    if "107M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=3265.38, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch,
                            narrow_band="full", freq_band='lband'
                        )
                    elif "54M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=1632.69, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch,
                            narrow_band="half", freq_band='lband'
                        )
                    elif "856M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=30000, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='lband'
                        )
                    elif "856M4k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=250e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='lband'
                        )
                    elif "856M1k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=1000e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='lband'
                        )
###########################
                    elif "68M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=2075.2, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch,
                            narrow_band="full", freq_band='uhf'
                        )
                    elif "34M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=1037.6, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch,
                            narrow_band="half", freq_band='uhf'
                        )
                    elif "544M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=19350, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='uhf'
                        )
                    elif "544M4k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=250e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='uhf'
                        )
                    elif "544M1k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=1000e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='uhf'
                        )
##########################
                    # req_chan_spacing chosen to be the same as L-band
                    elif "875M32k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=30000, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='sband'
                        )
                    elif "875M4k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=250e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='sband'
                        )
                    elif "875M1k" in self.instrument:
                        self._test_channelisation(
                            test_chan,
                            req_chan_spacing=1000e3, num_discards=num_discards,
                            samples_per_chan=smpl_per_ch, freq_band='sband'
                        )
#########################
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @instrument_32k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine(self):
        # Aqf.procedure(TestProcedure.Channelisation)
        if 'skipped_test' in test_CBF.__dict__['test_channelisation_wideband_fine'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_channelisation_wideband_fine'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success and self.cam_sensors.get_value("antenna_channelised_voltage_n_chans") >= 32768:
                    n_chans = self.n_chans_selected
                    test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
                    heading("CBF Channelisation Wideband Fine L-band")
                    self._test_channelisation(test_chan, no_channels=32768, req_chan_spacing=30e3)
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @slow
    @array_release_x
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_sfdr_peaks(self):
        Aqf.procedure(TestProcedure.ChannelisationSFDR)
        if 'skipped_test' in test_CBF.__dict__['test_channelisation_sfdr_peaks'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_channelisation_sfdr_peaks'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    heading("CBF Channelisation SFDR")
                    # If sfdr_ch_to_test is specified in the config file use that
                    # otherwise test all the selected channels
                    num_discard = int(self.conf_file["instrument_params"]["num_discards"])
                    n_ch_to_test = int(self.conf_file["instrument_params"].get("sfdr_ch_to_test",
                        None))
                    if "107M32k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=3265.38, no_channels=n_ch_to_test,
                                num_discard=num_discard)
                    elif "54M32k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=1632.69, no_channels=n_ch_to_test,
                                num_discard=num_discard)
                    elif "68M32k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=2075.2, no_channels=n_ch_to_test,
                                num_discard=num_discard)
                    elif "34M32k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=1037.6, no_channels=n_ch_to_test,
                                num_discard=num_discard)
                    elif "32k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=31250, no_channels=n_ch_to_test,  # Hz
                                num_discard=num_discard)
                    elif "4k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=250e3, no_channels=n_ch_to_test,  # Hz
                                num_discard=num_discard)
                    elif "1k" in self.instrument:
                        self._test_sfdr_peaks(req_chan_spacing=1000e3, no_channels=n_ch_to_test,  # Hz
                                num_discard=num_discard)
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @slow
    @instrument_32k
    @aqf_vr("CBF.V.3.30")
    @aqf_requirements("CBF-REQ-0126", "CBF-REQ-0047", "CBF-REQ-0046", "CBF-REQ-0043", "CBF-REQ-0053")
    def test_channelisation_wideband_fine_sfdr_peaks(self):
        # Aqf.procedure(TestProcedure.ChannelisationSFDR)
        if 'skipped_test' in test_CBF.__dict__['test_channelisation_wideband_fine_sfdr_peaks'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_channelisation_wideband_fine_sfdr_peaks'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success and self.cam_sensors.get_value("antenna_channelised_voltage_n_chans") >= 32768:
                    heading("CBF Channelisation Wideband Fine SFDR L-band")
                    n_ch_to_test = int(self.conf_file["instrument_params"].get("sfdr_ch_to_test",
                        self.n_chans_selected))
                    self._test_sfdr_peaks(required_chan_spacing=30e3, no_channels=n_ch_to_test)  # Hz
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@skipped_test
    #@subset
    @generic_test
    @aqf_vr("CBF.V.3.46")
    @aqf_requirements("CBF-REQ-0164", "CBF-REQ-0191")
    def test_power_consumption(self):
        Aqf.procedure(TestProcedure.PowerConsumption)
        if 'skipped_test' in test_CBF.__dict__['test_power_consumption'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_power_consumption'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                self.Step("Test is being qualified by CBF.V.3.30")

    #@subset
    #@array_release_x
    #@generic_test
    #@aqf_vr("CBF.V.4.10")
    #@aqf_requirements("CBF-REQ-0127")
    #def test_lband_efficiency(self):
        #start_time = str(datetime.now())
        #Aqf.procedure(TestProcedure.LBandEfficiency)
        #try:
            #assert evaluate(os.getenv("DRY_RUN", "False"))
        #except AssertionError:
            #instrument_success = self.set_instrument(start_receiver=False)
            #if instrument_success:
                #self._test_efficiency()
            #else:
                #self.Failed(self.errmsg)
        #finally:
            ##self.get_sensor_logs(start_time)
            #pass

    #@tbd
    #@skipped_test
    #@subset
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.10")
    @aqf_requirements("CBF-REQ-0127")
    def test_lband_efficiency(self):
    	start_time = str(datetime.now())
        Aqf.procedure(TestProcedure.LBandEfficiency)
        if 'skipped_test' in test_CBF.__dict__['test_lband_efficiency'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_lband_efficiency'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert eval(os.getenv("DRY_RUN", "False")) 
                self.Note('This is a dry run.')
            except AssertionError:
                self.Note('Set instrument.')
                self.Note('Carry out test method.')
                instrument_success = self.set_instrument(start_receiver=False)
                if instrument_success:
                    self._test_efficiency()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.A.IF")
    @aqf_requirements("TBD")
    def test_linearity(self):
        start_time = str(datetime.now())
        Aqf.procedure(TestProcedure.Linearity)
        if 'skipped_test' in test_CBF.__dict__['test_linearity'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_linearity'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                heading("CBF Linearity")
                instrument_success = self.set_instrument()
                if instrument_success:
                    n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                    test_chan = random.choice(range(self.start_channel, self.start_channel+self.n_chans_selected))
                    if ((("107M32k" in self.instrument) or ("54M32k" in self.instrument) or ("68M32k" in self.instrument) or ("34M32k" in self.instrument)) and (self.start_channel == 0)):
                        check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
                        check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
                        test_chan = random.choice(range(n_chans)[check_strt_ch:check_stop_ch])
                    self._test_linearity(test_channel=test_chan, max_steps=20)
                else:
                    self.Failed(self.errmsg)
       # 	finally:
       #     	#self.get_sensor_logs(start_time)
       #     	pass

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.4")
    @aqf_requirements("CBF-REQ-0087", "CBF-REQ-0225", "CBF-REQ-0104")
    def test_c_baseline_correlation_product(self):
        Aqf.procedure(TestProcedure.BaselineCorrelation)
        if 'skipped_test' in test_CBF.__dict__['test_c_baseline_correlation_product'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_c_baseline_correlation_product'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    num_discard = int(self.conf_file["instrument_params"]["num_discards"])
                    num_tst_chs = int(self.conf_file["instrument_params"]["num_ch_to_test"])
                    n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                    chan_sel = self.n_chans_selected
                    check_strt_ch = None
                    check_stop_ch = None
                    if ((("107M32k" in self.instrument) or ("54M32k" in self.instrument) or ("68M32k" in self.instrument) or ("34M32k" in self.instrument))
                            and (self.start_channel == 0)):
                        check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
                        check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
                        test_channels = random.sample(range(n_chans)[check_strt_ch:check_strt_ch+chan_sel], num_tst_chs)
                        test_chan = random.choice(range(n_chans)[check_strt_ch:check_stop_ch])
                    else:
                        test_channels = random.sample(range(n_chans)[self.start_channel:self.start_channel+chan_sel], num_tst_chs)
                        test_chan = random.choice(range(self.start_channel,
                                self.start_channel+self.n_chans_selected))
                    test_channels = sorted(test_channels)
                    # Remove chan 0 to avoid DC issues
                    if test_channels[0] == 0:
                        test_channels = test_channels[1:]
                else:
                    test_channels = random.sample(range(n_chans)[self.start_channel:self.start_channel+chan_sel], num_tst_chs)
                    test_chan = random.choice(range(self.start_channel, 
                            self.start_channel+self.n_chans_selected))
                test_channels = sorted(test_channels)
                # Remove chan 0 to avoid DC issues
                if test_channels[0] == 0:
                    test_channels = test_channels[1:]
                #self.check_dsim_acc_offset()

                self._test_product_baselines(check_strt_ch, check_stop_ch, num_discard)
                self._test_back2back_consistency(test_channels, num_discard)
                self._test_freq_scan_consistency(test_chan, num_discard)
                #self._test_spead_verify()
                #self._test_product_baseline_leakage()
            else:
                self.Failed(self.errmsg)


    #@tbd
    #@subset
    #@skipped_test
    @generic_test
    @aqf_vr("CBF.V.3.62")
    @aqf_requirements("CBF-REQ-0238")
    def test_imaging_data_product_set(self):
        Aqf.procedure(TestProcedure.ImagingDataProductSet)
        if 'skipped_test' in test_CBF.__dict__['test_imaging_data_product_set'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_imaging_data_product_set'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_data_product(_baseline=True)
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @generic_test
    @aqf_vr("CBF.V.3.67")
    @aqf_requirements("CBF-REQ-0120")
    def test_tied_array_aux_baseline_correlation_products(self):
        Aqf.procedure(TestProcedure.TiedArrayAuxBaselineCorrelationProducts)
        if 'skipped_test' in test_CBF.__dict__['test_tied_array_aux_baseline_correlation_products'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_tied_array_aux_baseline_correlation_products'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_data_product(_baseline=True, _tiedarray=True)
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@skipped_test
    #@subset
    @generic_test
    @aqf_vr("CBF.V.3.64")
    @aqf_requirements("CBF-REQ-0242")
    def test_tied_array_voltage_data_product_set(self):
        Aqf.procedure(TestProcedure.TiedArrayVoltageDataProductSet)
        if 'skipped_test' in test_CBF.__dict__['test_tied_array_voltage_data_product_set'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_tied_array_voltage_data_product_set'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_data_product(_tiedarray=True)
                else:
                    self.Failed(self.errmsg)

    # @tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.7")
    @aqf_requirements("CBF-REQ-0096")
    def test_accumulation_length(self):
        Aqf.procedure(TestProcedure.VectorAcc)
        if 'skipped_test' in test_CBF.__dict__['test_accumulation_length'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_accumulation_length'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                    center_ch = int(n_chans/2)
                    if ((("107M32k" in self.instrument) or ("54M32k" in self.instrument) or ("68M32k" in self.instrument) or ("34M32k" in self.instrument))
                            and (self.start_channel == 0)):
                        check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
                        check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
                        test_chan = random.choice(range(n_chans)[check_strt_ch:check_stop_ch])
                        test_chan = 6000
                    else:
                        test_chan = random.choice(range(self.start_channel, self.start_channel+self.n_chans_selected))
                    n_ants = int(self.cam_sensors.get_value("n_ants"))
                    #TODO: figure out why this fails if not using 1 second
                    self._test_vacc(
                        test_chan,
                        acc_time=(0.998 if self.cam_sensors.get_value("n_ants") == 4
                                else 2 * n_ants / 32.0))
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.4.9")
    @aqf_requirements("CBF-REQ-0119")
    def test_gain_correction(self):
        Aqf.procedure(TestProcedure.GainCorr)
        if 'skipped_test' in test_CBF.__dict__['test_gain_correction'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_gain_correction'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_gain_correction()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @generic_test
    @aqf_vr("CBF.V.4.23")
    @aqf_requirements("CBF-REQ-0013")
    def test_product_switch(self):
        Aqf.procedure(TestProcedure.ProductSwitching)
        if 'skipped_test' in test_CBF.__dict__['test_product_switch'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_product_switch'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
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

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.31")
    @aqf_requirements("CBF-REQ-0066", "CBF-REQ-0072", "CBF-REQ-0077", "CBF-REQ-0110", "CBF-REQ-0200")
    def test_a_delay_phase_compensation_control(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation_Control)
        if 'skipped_test' in test_CBF.__dict__['test_a_delay_phase_compensation_control'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_a_delay_phase_compensation_control'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_delays_control()
                    self.clear_all_delays()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.32")
    @aqf_requirements("CBF-REQ-0112", "CBF-REQ-0128", "CBF-REQ-0185", "CBF-REQ-0187", "CBF-REQ-0188")
    def test_a_delay_phase_compensation_functional(self):
        Aqf.procedure(TestProcedure.CBF_Delay_Phase_Compensation)
        if 'skipped_test' in test_CBF.__dict__['test_a_delay_phase_compensation_functional'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_a_delay_phase_compensation_functional'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                #instrument_success = self.set_instrument(acc_time=(0.5
                #    if self.cam_sensors.sensors.n_ants.get_value() == 4
                #    else int(self.conf_file["instrument_params"]["delay_test_acc_time"])))
                inst = self.cam_sensors.get_value("instrument_state").split("_")[0]
                check_strt_ch = None
                check_stop_ch = None
                if ("107M32k" in inst) or ("54M32k" in inst) or ("68M32k" in inst) or ("34M32k" in inst):
                    instrument_success = self.set_instrument(2)
                    # If the full band is capture, set the part of band that should be checked
                    if self.start_channel == 0 and self.stop_channel == 32768:
                        check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
                        check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
                elif "32k" in inst:
                    instrument_success = self.set_instrument(4)
                elif "4k" in inst and "128" in inst:
                    instrument_success = self.set_instrument(2, start_channel = 1024, stop_channel = 3071)
                elif "4k" in inst:
                    instrument_success = self.set_instrument(1)
                elif "1k" in inst and "128" in inst:
                    instrument_success = self.set_instrument(1, start_channel = 256, stop_channel = 767)
                else:
                    instrument_success = self.set_instrument()
                # Remove previous delay tracking data files
                dl_fn = "/".join([self._katreport_dir, r"delay_*"])
                ph_fn = "/".join([self._katreport_dir, r"phase_*"])
                del_files = glob.glob(dl_fn)+glob.glob(ph_fn)
                try: 
                    for f in del_files:
                        os.remove(f)
                except OSError:
                    pass
                if instrument_success:
                    delay_mult = self.corr_fix._test_config_file["delay_req"]["test_delay_multipliers"].split(",")
                    delays = [float(x)* self.cam_sensors.sample_period for x in delay_mult]
                    self._test_delay_tracking(check_strt_ch,check_stop_ch, delays)
                    delay_resolution = float(self.conf_file["delay_req"].get("delay_resolution"))
                    self._test_delay_tracking(check_strt_ch,check_stop_ch, [0, delay_resolution, delay_resolution*2])
                    rate_mult = self.corr_fix._test_config_file["delay_req"]["test_delay_rate_multipliers"].split(",")
                    delay_rates = ([float(x)* (self.cam_sensors.sample_period / self.cam_sensors.get_value("int_time"))
                        for x in rate_mult])
                    self._test_delay_rate(check_strt_ch,check_stop_ch, delay_rates)
                    delay_rate_resolution = float(self.conf_file["delay_req"].get("delay_rate_resolution"))
                    self._test_delay_rate(check_strt_ch, check_stop_ch, delay_rates=[delay_rate_resolution])
                    #self._test_delay_rate(check_strt_ch, check_stop_ch, delay_rate_mult=[16], awgn_scale=0.01, gain=500)
                    self._test_phase_rate(check_strt_ch, check_stop_ch)
                    self._test_phase_offset(check_strt_ch, check_stop_ch, gain_multiplier=2)
                    self._test_delay_inputs(check_strt_ch, check_stop_ch)
                    self.clear_all_delays()
                else:
                    self.Failed(self.errmsg)
                self.Note("Refer to Appendix for analysis of delay tracking performance.")

    #@tbd
    #@skipped_test
    @subset
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.27")
    @aqf_requirements("CBF-REQ-0178")
    def test_report_configuration(self):
        Aqf.procedure(TestProcedure.ReportConfiguration)
        if 'skipped_test' in test_CBF.__dict__['test_report_configuration'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_report_configuration'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument(start_receiver=False)
                if instrument_success:
                    self._test_report_config()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.29")
    @aqf_requirements("CBF-REQ-0067")
    def test_systematic_error_reporting(self):
        Aqf.procedure(TestProcedure.PFBFaultDetection)
        if 'skipped_test' in test_CBF.__dict__['test_systematic_error_reporting'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_systematic_error_reporting'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument(start_receiver=False)
                if instrument_success:
                    self._test_fft_overflow()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.28")
    @aqf_requirements("CBF-REQ-0157")
    def test_fault_detection(self):
        Aqf.procedure(TestProcedure.ProcessingPipelineFaultDetection)
        Aqf.procedure(TestProcedure.MemoryFaultDetection)
        Aqf.procedure(TestProcedure.LinkFaultDetection)
        if 'skipped_test' in test_CBF.__dict__['test_fault_detection'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_fault_detection'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument(start_receiver=False)
                if instrument_success:
                    self._test_network_link_error()
                    #self._test_memory_error()
                    #heading("Processing Pipeline Failures")
                    #self.Note("Test is being qualified by CBF.V.3.29")
                    #heading("HMC Memory errors")
                    #self.Note("See waiver")
                    #heading("Network Link errors")
                    #self.Note("See waiver")
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @generic_test
    @aqf_vr("CBF.V.3.26")
    @aqf_requirements("CBF-REQ-0056", "CBF-REQ-0068", "CBF-REQ-0069")
    def test_monitor_sensors(self):
        Aqf.procedure(TestProcedure.MonitorSensors)
        if 'skipped_test' in test_CBF.__dict__['test_monitor_sensors'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_monitor_sensors'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._test_sensor_values()
                    # self._test_host_sensors_status()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @generic_test
    @aqf_vr("CBF.V.3.38")
    @aqf_requirements("CBF-REQ-0203")
    def test_time_synchronisation(self):
        Aqf.procedure(TestProcedure.TimeSync)
        if 'skipped_test' in test_CBF.__dict__['test_time_synchronisation'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_time_synchronisation'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                self._test_time_sync()

    #@tbd
    #@subset
    #@skipped_test
    @generic_test
    @aqf_vr("CBF.V.4.26")
    @aqf_requirements("CBF-REQ-0083", "CBF-REQ-0084", "CBF-REQ-0085", "CBF-REQ-0086", "CBF-REQ-0221")
    def test_antenna_voltage_buffer(self):
        Aqf.procedure(TestProcedure.VoltageBuffer)
        if 'skipped_test' in test_CBF.__dict__['test_antenna_voltage_buffer'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_antenna_voltage_buffer'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument()
                if instrument_success:
                    self._small_voltage_buffer()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    @array_release_x
    @beamforming
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.3.34")
    @aqf_requirements("CBF-REQ-0076", "CBF-REQ-0094", "CBF-REQ-0117", "CBF-REQ-0118", "CBF-REQ-0122", "CBF-REQ-0123", "CBF-REQ-0183", "CBF-REQ-0220")
    def test_beamforming(self):
        Aqf.procedure(TestProcedure.Beamformer)
        if 'skipped_test' in test_CBF.__dict__['test_beamforming'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_beamforming'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                heading("Beamformer Functionality")
                instrument_success = self.set_instrument(start_receiver = False)
                if instrument_success:
                    self._test_beamforming()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    #@array_release_x
    @beamforming
    @instrument_1k
    @instrument_4k
    @aqf_vr('CBF.V.3.35')
    @aqf_requirements("CBF-REQ-0124")
    def test_beamformer_efficiency(self):
        Aqf.procedure(TestProcedure.BeamformerEfficiency)
        if 'skipped_test' in test_CBF.__dict__['test_beamformer_efficiency'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_beamformer_efficiency'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert eval(os.getenv('DRY_RUN', 'False'))
            except AssertionError:
                self.Note("Test not implemented.")
                #instrument_success = self.set_instrument()
                #if instrument_success:
                #    self._bf_efficiency()
                #else:
                #    Aqf.failed(self.errmsg)

    #@tbd
    #@skipped_test
    #@array_release_x
    @beamforming
    # @wipd  # Test still under development, Alec will put it under test_informal
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.A.IF")
    def test_y_beamforming_timeseries(self):
        Aqf.procedure(TestProcedure.TimeSeries)
        if 'skipped_test' in test_CBF.__dict__['test_y_beamforming_timeseries'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_y_beamforming_timeseries'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                instrument_success = self.set_instrument(start_receiver = False)
                if instrument_success:
                    self._test_beamforming_timeseries()
                else:
                    self.Failed(self.errmsg)

    #@tbd
    #@subset
    #@skipped_test
    #@array_release_x
    @beamforming
    @instrument_1k
    @instrument_4k
    @aqf_vr("CBF.V.A.IF")
    def test_z_group_delay(self):
        Aqf.procedure(TestProcedure.GroupDelay)
        if 'skipped_test' in test_CBF.__dict__['test_z_group_delay'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_z_group_delay'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
            except AssertionError:
                heading("Group Delay")
                instrument_success = self.set_instrument(start_receiver = False)
                if instrument_success:
                    self._test_group_delay()
                else:
                    self.Failed(self.errmsg)

    #@skipped_test
    @array_release_x
    #@subset
    @beamforming
    @aqf_vr("CBF.V.3.34")
    @aqf_requirements("CBF-REQ-0076", "CBF-REQ-0094", "CBF-REQ-0117", "CBF-REQ-0118", "CBF-REQ-0122", "CBF-REQ-0123", "CBF-REQ-0183", "CBF-REQ-0220")
    def test_beam_delay(self):
        Aqf.procedure(TestProcedure.BeamDelay)
        try:
            reply, informs = self.katcp_req.capture_list()
            self.assertTrue(reply.reply_ok())
            beams = []
            for msg in informs:
                if 'tied' in msg.arguments[0]:
                    beams.append(msg.arguments[0])
        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False
        if 'skipped_test' in test_CBF.__dict__['test_beam_delay'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test_beam_delay'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        elif len(beams) < 3:
            self.Note('Current running instrument does not contain multiple beams. Skipping beam delay test.')
        else:
            try:
                assert evaluate(os.getenv("DRY_RUN", "False"))
                
            except AssertionError:
                heading("Beamformer Steering Functionality")
                instrument_success = self.set_instrument(start_receiver = False)
                if instrument_success:
                    self._test_beam_delay()
                else:
                    self.Failed(self.errmsg)
                self.Note("Refer to Appendix for analysis of beam steering performance.")

    #def test_beam_steering(self):
    #    Aqf.procedure(TestProcedure.GroupDelay)
    #    if 'skipped_test' in test_CBF.__dict__['test_beam_delay'].__dict__:
    #        self.Note('Mark test as skipped.')
    #        Aqf.skipped('Test skipped')
    #    elif 'tbd' in test_CBF.__dict__['test_beam_delay'].__dict__:
    #        self.Note('Mark test as tbd.')
    #        Aqf.tbd('Test tbd')
    #    else:
    #        try:
    #            assert evaluate(os.getenv("DRY_RUN", "False"))
    #        except AssertionError:
    #            instrument_success = self.set_instrument(start_receiver = False)
    #            if instrument_success:
    #                self._test_beam_steering()
    #            else:
    #                self.Failed(self.errmsg)


    # ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------MANUAL TESTS-----------------------------------------
    # ---------------------------------------------------------------------------------------------------

    # Perhaps, enlist all manual tests here with VE & REQ

    #@tbd
    #@skipped_test
    @manual_test
    @aqf_vr("CBF.V.3.56")
    @aqf_requirements("CBF-REQ-0228")
    def test__subarray(self):
        if 'skipped_test' in test_CBF.__dict__['test__subarray'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__subarray'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.56")

    #@tbd
    #@skipped_test
    @array_release_x
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.37")
    @aqf_requirements("CBF-REQ-0071", "CBF-REQ-0204")
    def test__control(self):
        if 'skipped_test' in test_CBF.__dict__['test__control'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__control'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.37")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.37*"))
            caption_list = ["Screenshot of the command executed and reply: CAM interface"]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.1.11")
    @aqf_requirements("CBF-REQ-0137")
    def test__procured_items_emc_certification(self):
        if 'skipped_test' in test_CBF.__dict__['test__procured_items_emc_certification'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__procured_items_emc_certification'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.1.11")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.3")
    @aqf_requirements("CBF-REQ-0018", "CBF-REQ-0019", "CBF-REQ-0022", "CBF-REQ-0024")
    @aqf_requirements("CBF-REQ-0011", "CBF-REQ-0012", "CBF-REQ-0014", "CBF-REQ-0016", "CBF-REQ-0017")
    @aqf_requirements("CBF-REQ-0027", "CBF-REQ-0064")
    def test__states_and_modes_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__states_and_modes_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__states_and_modes_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.3")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.77")
    @aqf_requirements("CBF-REQ-0021")
    def test__full_functional_mode_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__full_functional_mode_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__full_functional_mode_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.77")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.15")
    @aqf_requirements("CBF-REQ-0131", "CBF-REQ-0132", "CBF-REQ-0133")
    def test__power_supply_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__power_supply_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__power_supply_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.15")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.16")
    @aqf_requirements("CBF-REQ-0199")
    def test__safe_design_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__safe_design_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__safe_design_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.16")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.17")
    @aqf_requirements("CBF-REQ-0061")
    def test__lru_status_and_display_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__lru_status_and_display_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__lru_status_and_display_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.17")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.18")
    @aqf_requirements("CBF-REQ-0197")
    def test__cots_lru_status_and_display_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__cots_lru_status_and_display_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__cots_lru_status_and_display_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.18")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.18*"))
            caption_list = [
                "Mellanox SX1710 switches and status LEDs visible from front of rack.",
                "Dell PowerEdge servers and status via front panel display visible.",
                "AP8981 PDUs have status LEDs visible from the back of the rack.",
            ]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.19")
    @aqf_requirements("CBF-REQ-0182")
    def test__interchangeability_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__interchangeability_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__interchangeability_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.19")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.20")
    @aqf_requirements("CBF-REQ-0168", "CBF-REQ-0171")
    def test__periodic_maintenance_lru_storage_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__periodic_maintenance_lru_storage_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__periodic_maintenance_lru_storage_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.20")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.21")
    @aqf_requirements("CBF-REQ-0169", "CBF-REQ-0170", "CBF-REQ-0172", "CBF-REQ-0173")
    def test__lru_storage_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__lru_storage_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__lru_storage_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.21")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.22")
    @aqf_requirements("CBF-REQ-0147", " CBF-REQ-0148")
    def test__item_handling_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__item_handling_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__item_handling_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.22")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.23")
    @aqf_requirements("CBF-REQ-0152", "CBF-REQ-0153", "CBF-REQ-0154", "CBF-REQ-0155", "CBF-REQ-0184")
    def test__item_marking_and_labelling_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__item_marking_and_labelling_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__item_marking_and_labelling_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
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

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.24")
    @aqf_requirements("CBF-REQ-0162")
    def test__use_of_cots_equipment_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__use_of_cots_equipment_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__use_of_cots_equipment_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.24")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.25")
    @aqf_requirements("CBF-REQ-0060", "CBF-REQ-0177", "CBF-REQ-0196")
    def test__logging_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__logging_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__logging_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.25")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.25*"))
            caption_list = [
                "Screenshot of the command executed via CAM interface (log-level)"] * len(image_files)
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.33")
    @aqf_requirements("CBF-REQ-0103")
    def test__accumulator_dynamic_range_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__accumulator_dynamic_range_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__accumulator_dynamic_range_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.33")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.36")
    @aqf_requirements("CBF-REQ-0001")
    def test__data_products_available_for_all_receivers_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__data_products_available_for_all_receivers_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__data_products_available_for_all_receivers_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.36")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.39")
    @aqf_requirements("CBF-REQ-0140")
    def test__cooling_method_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__cooling_method_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__cooling_method_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.39")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.39*"))
            caption_list = [
                "Rear doors of all CBF racks are perforated",
                "Front doors of all CBF racks are perforated"
                ]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.40")
    @aqf_requirements("CBF-REQ-0142", "CBF-REQ-0143")
    def test__humidity_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__humidity_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__humidity_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.40")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.41")
    @aqf_requirements("CBF-REQ-0145")
    def test__storage_environment_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__storage_environment_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__storage_environment_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.41")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.42")
    @aqf_requirements("CBF-REQ-0141")
    def test__temperature_range_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__temperature_range_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__[''].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.42")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.43")
    @aqf_requirements("CBF-REQ-0146")
    def test__transportation_of_components_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__transportation_of_components_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__transportation_of_components_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.43")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.44")
    @aqf_requirements("CBF-REQ-0156")
    def test__product_marking_environmentals_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__product_marking_environmentals_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__product_marking_environmentals_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.44")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.44*"))
            caption_list = [
                "All equipment labels are still attached on {}".format(i.split("/")[-1].split(".jpg")[0])
                for i in image_files
            ]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.45")
    @aqf_requirements("CBF-REQ-0158", "CBF-REQ-0160")
    def test__fail_safe_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__fail_safe_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__fail_safe_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.45")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.47")
    @aqf_requirements("CBF-REQ-0161", "CBF-REQ-0186")
    def test__safe_physical_design_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__safe_physical_design_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__safe_physical_design_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.47")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.48")
    @aqf_requirements("CBF-REQ-0107")
    def test__digitiser_cam_data_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__digitiser_cam_data_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__digitiser_cam_data_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.48")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.50")
    @aqf_requirements("CBF-REQ-0149")
    def test__mtbf_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__mtbf_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__mtbf_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.50")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.52")
    @aqf_requirements("CBF-REQ-0179", "CBF-REQ-0180", "CBF-REQ-0190", " CBF-REQ-0194")
    @aqf_requirements("CBF-REQ-0201", "CBF-REQ-0202")
    def test__internal_interfaces_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__internal_interfaces_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__internal_interfaces_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.52")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.53")
    @aqf_requirements("CBF-REQ-0136", "CBF-REQ-0166")
    def test__external_interfaces_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__external_interfaces_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__external_interfaces_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.53")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.54")
    @aqf_requirements("CBF-REQ-0150", "CBF-REQ-0151")
    def test__lru_replacement_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__lru_replacement_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__lru_replacement_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.54")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.3.54*"))
            caption_list = [
                "LRU replacement: {}".format(i.split("/")[-1].split(".jpg")[0])
                for i in image_files
            ]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @untested
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.57")
    @aqf_requirements("CBF-REQ-0193")
    # @aqf_requirements("CBF-REQ-0195", "CBF-REQ-0230", "CBF-REQ-0231", "CBF-REQ-0232",)
    # @aqf_requirements("CBF-REQ-0233", "CBF-REQ-0235")
    def test__data_subscribers_link_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__data_subscribers_link_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__data_subscribers_link_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.57")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.6.9")
    @aqf_requirements("CBF-REQ-0138")
    def test__design_to_emc_sans_standard_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__design_to_emc_sans_standard_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__design_to_emc_sans_standard_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.6.9")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.6.9*"))
            caption_list = [
                "Cables are bundled separately but the separation distance is not more than "
                "500mm due to space constraints in the racks."
            ] * len(image_files)
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.6.10")
    @aqf_requirements("CBF-REQ-0139")
    def test__design_standards_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__design_standards_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__design_standards_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.6.10")
            image_files = sorted(glob.glob(self._images_dir + "/CBF.V.6.10*"))
            caption_list = ["CBF processing nodes contains an integrated power filter."]
            Report_Images(image_files, caption_list)

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.66")
    @aqf_requirements("CBF-REQ-0223")
    def test__channelised_voltage_data_transfer_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__channelised_voltage_data_transfer_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__channelised_voltage_data_transfer_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.66")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.49")
    @aqf_requirements("CBF-REQ-0224")
    def test__route_basic_spectrometer_data_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__route_basic_spectrometer_data_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__route_basic_spectrometer_data_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
            self._test_global_manual("CBF.V.3.49")

    #@tbd
    #@skipped_test
    @manual_test
    @generic_test
    @aqf_vr("CBF.V.3.58")
    @aqf_requirements("CBF-REQ-0237")
    def test__subarray_data_product_set_ve(self):
        if 'skipped_test' in test_CBF.__dict__['test__subarray_data_product_set_ve'].__dict__:
            self.Note('Mark test as skipped.')
            Aqf.skipped('Test skipped')
        elif 'tbd' in test_CBF.__dict__['test__subarray_data_product_set_ve'].__dict__:
            self.Note('Mark test as tbd.')
            Aqf.tbd('Test tbd')
        else:
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

    #@array_release_x
    #@generic_test
    #@manual_test
    #@aqf_vr("CBF.V.A.IF")
    #def test__informal(self):
    #    Aqf.procedure(
    #        "This verification event pertains to tests that are executed, "
    #        "but do not verify any formal requirements."
    #    )
    #    # self._test_informal()

    # -----------------------------------------------------------------------------------------------------

    #################################################################
    #                       Test Methods                            #
    #################################################################

    def _test_channelisation(self, test_chan=1500, req_chan_spacing=None, 
            num_discards=5, samples_per_chan=60, narrow_band = None, freq_band ='lband'):
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        if narrow_band == 'full' and freq_band == 'lband':
            # [CBF-REQ-0236]
            min_bandwidth_req = 107e6
        elif narrow_band == 'half' and freq_band == 'lband':
            # [CBF-REQ-0236]
            min_bandwidth_req = 53.5e6
        elif narrow_band == 'full' and freq_band == 'uhf':
            # [CBF-REQ-0243]
            min_bandwidth_req = 68e6
        elif narrow_band == 'half' and freq_band == 'uhf':
            # [CBF-REQ-0236]
            min_bandwidth_req = 34e6
        elif freq_band == 'uhf':
            # [CBF-REQ-0050]
            min_bandwidth_req = 435e6
            # sband REQ TBD
        elif freq_band == 'sband':
            min_bandwidth_req = 790e6 #(for now use -10% of total bandwidth)
        else:
            # [CBF-REQ-0053]
            min_bandwidth_req = 770e6
        nominal_bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") * self.dsim_factor
        # [CBF-REQ-0126] CBF channel isolation
        cutoff = 53  # dB
        # Placeholder of actual frequencies that the signal generator produces
        actual_test_freqs = []
        # Channel magnitude responses for each frequency
        chan_responses = []
        last_source_freq = None
        # Test channel relative to selected channels
        test_chan_rel = test_chan - self.start_channel

        print_counts = 3
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        requested_test_freqs = self.cam_sensors.calc_freq_samples(self.dhost, 
                test_chan, samples_per_chan=samples_per_chan, chans_around=2)
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
        else:
            curr_mcount = self.current_dsim_mcount() #dump_after_mcount
        n_accs = self.cam_sensors.get_value("n_accs")
        bls_to_test = evaluate(self.cam_sensors.get_value("bls_ordering"))[test_baseline]
        self.Progress(
            "Randomly selected frequency channel to test: {} and "
            "selected baseline {} / {} to test.".format(test_chan, test_baseline, bls_to_test)
        )
        self.Step(
            "The CBF, when configured to produce the Imaging data product set "
            "shall channelise a total bandwidth of >= %s" % (
                min_bandwidth_req)
        )
        Aqf.is_true(
            nominal_bw >= min_bandwidth_req,
            "Channelise total bandwidth {}Hz shall be >= {}Hz.".format(
                nominal_bw, min_bandwidth_req
            ),
        )
        # TODO (MM) 2016-10-27, As per JM
        # Channel spacing is reported as 209.266kHz. This is probably spot-on, considering we're
        # using a dsim that's not actually sampling at 1712MHz. But this is problematic for the
        # test report. We would be getting 1712MHz/8192=208.984375kHz on site.
        # Maybe we should be reporting this as a fraction of total sampling rate rather than
        # an absolute value? ie 1/4096=2.44140625e-4 I will speak to TA about how to handle this.
        # chan_spacing = 856e6 / np.shape(initial_dump['xeng_raw'])[0]
        chan_spacing = round(nominal_bw / self.cam_sensors.get_value("antenna_channelised_voltage_n_chans"),2)
        chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100), chan_spacing + (chan_spacing * 1 / 100)]
        self.Step("CBF-REQ-0043, 0053, 0226, 0227 and 0236 Confirm channel spacing.")
        msg = ("Verify that the calculated channel frequency step ({:.3f} kHz) is equal or less than {} kHz"
               "".format(chan_spacing/1000., req_chan_spacing/1000.))
        if chan_spacing <= req_chan_spacing:
            ch_spacing_res = True
        else:
            ch_spacing_res = False
        Aqf.is_true(ch_spacing_res, msg)

        self.Step(
            "CBF-REQ-0046 and CBF-REQ-0047 Confirm channelisation spacing and "
            "confirm that it is within the maximum tolerance."
        )
        msg = "Channelisation frequency is within maximum tolerance of 1% of the channel spacing."
        Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)
        for i in range(self.data_retries):  
            #initial_dump = self.get_real_clean_dump(discard=num_discards)
            initial_dump = self.get_dump_after_mcount(curr_mcount) #dump_after_mcount
            if initial_dump is not False:
                initial_freq_response = normalised_magnitude(initial_dump["xeng_raw"][:, test_baseline, :])
                where_is_the_tone = np.argmax(initial_freq_response) + self.start_channel
                max_tone_val = np.max(initial_freq_response)
                if where_is_the_tone == test_chan: break
            self.logger.warn("CW not found, retrying capture.")
        # 1) I think the channelisation tests might still be saturating.
        # Could you include a dBFS peak value in the output?
        # (take the peak auto correlation output value and divide it by the number of accumulations;
        # you should get a value in the range 0-16129).
        # TODO: check for saturation
        value = np.max(magnetise(initial_dump["xeng_raw"][:, test_baseline, :]))
        valuedBFS = 20*np.log10(abs(value)/n_accs)
        self.Note(
            "Single peak found at channel %s, with max power of %.5f (%.5f dB)"
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
        aqf_plot_channels(initial_freq_response, plt_filename, plt_title, caption=caption, ylimits=(-100, 1), 
                          start_channel=self.start_channel, 
                          ylabel = "dbFS relative to VACC max")
        self.Step(
            "Sweep the digitiser simulator over the centre frequencies of at "
            "least all the channels that fall within the complete L-band"
        )
        failure_count = 0
        int_time = self.cam_sensors.get_value("int_time")

        
        #TODO: Hack, rather move this to the requested freq method
        middle_idx = int(len(requested_test_freqs)/2)
        start = requested_test_freqs[middle_idx]
        self.dhost.sine_sources.sin_0.set(frequency=start, scale=cw_scale)
        set_middle_freq = self.dhost.sine_sources.sin_0.frequency
        curr_setf = set_middle_freq
        #Find the dsim set step size
        for i in range(20):
            currf = start+i
            self.dhost.sine_sources.sin_0.set(frequency=currf, scale=cw_scale)
            setf = self.dhost.sine_sources.sin_0.frequency
            if setf != curr_setf:
                dsim_step = setf-curr_setf
                curr_setf = setf
        ave_f_step = np.average(np.diff(requested_test_freqs))
        real_f_step = dsim_step*round(ave_f_step/dsim_step)
        half_band = real_f_step*middle_idx
        strt_band = set_middle_freq - half_band
        stop_band = set_middle_freq + half_band
        requested_test_freqs = np.arange(strt_band, stop_band, real_f_step)

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
            curr_mcount = self.current_dsim_mcount() #dump_after_mcount
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
                #this_freq_dump = self.receiver.get_clean_dump(discard=num_discards)
                this_freq_dump = self.get_dump_after_mcount(curr_mcount) #dump_after_mcount
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
                ## No of spead heap discards relevant to vacc
                #discards = 0
                #max_wait_dumps = 10
                #while True:
                #    try:
                #        a = time.time()
                #        queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                #        print (time.time()-a)
                #        self.assertIsInstance(queued_dump, dict)
                #        deng_timestamp = float(self.dhost.registers.sys_clkcounter.read().get("timestamp"))
                #        self.assertIsInstance(deng_timestamp, float)
                #    except Exception:
                #        errmsg = "Could not retrieve clean queued accumulation for freq(%s @ %s: " "%s MHz)." % (
                #            i + 1,
                #            len(requested_test_freqs),
                #            freq / 1e6,
                #        )
                #        self.Error(errmsg, exc_info=True)
                #        break
                #    else:
                #        timestamp_diff = np.abs(queued_dump["dump_timestamp"] - deng_timestamp)
                #        print colored(timestamp_diff, 'red')
                #        if timestamp_diff < (num_discards*int_time)*2:
                #            msg = (
                #                "Received correct accumulation timestamp: %s, relevant to "
                #                "DEngine timestamp: %s (Difference %.2f)"
                #                % (queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                #            )
                #            self.logger.info(_msg)
                #            self.logger.info(msg)
                #            break

                #        if discards > max_wait_dumps:
                #            errmsg = (
                #                "Could not get accumulation with correct timestamp within %s "
                #                "accumulation periods." % max_wait_dumps
                #            )
                #            self.Failed(errmsg)
                #            if discards > 10:
                #                return
                #            break
                #        else:
                #            msg = (
                #                "Discarding subsequent dumps (%s) with dump timestamp (%s) "
                #                "and DEngine timestamp (%s) with difference of %s."
                #                % (discards, queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
                #            )
                #            self.logger.info(msg)
                #        deng_timestamp = None
                #    discards += 1
                try:
                    #queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
                    queued_dump = self.receiver.get_clean_dump(discard=0)
                    self.assertIsInstance(queued_dump, dict)
                    #deng_timestamp = float(self.dhost.registers.sys_clkcounter.read().get("timestamp"))
                    #self.assertIsInstance(deng_timestamp, float)
                except Exception:
                    errmsg = "Could not retrieve clean queued accumulation for freq(%s @ %s: " "%s MHz)." % (
                        i + 1,
                        len(requested_test_freqs),
                        freq / 1e6,
                    )
                    self.Error(errmsg, exc_info=True)
                    return
                self.logger.info(_msg)

                this_freq_data = queued_dump["xeng_raw"]
                this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                # print("{} {} ".format(np.max(this_freq_response), np.argmax(this_freq_response)))
                actual_test_freqs.append(this_source_freq)
                chan_responses.append(this_freq_response)

            ## Plot an overall frequency response at the centre frequency just as
            ## a sanity check

            #if np.abs(freq - expected_fc) < 0.1:
            #    plt_filename = "{}/{}_overall_channel_resolution.png".format(self.logs_path, self._testMethodName)
            #    plt_title = "Overall frequency response at {} at {:.3f}MHz.".format(test_chan, this_source_freq / 1e6)
            #    max_peak = np.max(loggerise(this_freq_response))
            #    self.Note(
            #        "Single peak found at channel %s, with max power of %s (%fdB) midway "
            #        "channelisation, to confirm if there is no offset."
            #        % (np.argmax(this_freq_response), np.max(this_freq_response), max_peak)
            #    )
            #    new_cutoff = max_peak - cutoff
            #    y_axis_limits = (-100, 1)
            #    caption = (
            #        "An overall frequency response at the centre frequency, and ({:.3f}dB) "
            #        "and selected baseline {} / {} to test. CBF channel isolation [max channel"
            #        " peak ({:.3f}dB) - ({}dB) cut-off] when "
            #        "digitiser simulator is configured to generate a continuous wave, with "
            #        "cw scale: {}, awgn scale: {}, Eq gain: {} and FFT shift: {}".format(
            #            new_cutoff, test_baseline, bls_to_test, max_peak, cutoff, cw_scale, awgn_scale, gain, fft_shift
            #        )
            #    )
            #    aqf_plot_channels(
            #        this_freq_response,
            #        plt_filename,
            #        plt_title,
            #        caption=caption,
            #        ylimits=y_axis_limits,
            #        cutoff=new_cutoff,
            #    )

        if not where_is_the_tone == test_chan:
            self.Note(
                "We expect the channel response at %s, but in essence it is in channel %s, ie "
                "There's a channel offset of %s" % (test_chan, where_is_the_tone, np.abs(test_chan - where_is_the_tone))
            )
            test_chan += np.abs(test_chan - where_is_the_tone)

        # Convert the lists to numpy arrays for easier working
        actual_test_freqs = np.array(actual_test_freqs)
        chan_responses = np.array(chan_responses)
        fn = "/".join([self._katreport_dir, r"channelisation_raw_data.npz"])
        with open(fn, 'w') as f:
            np.savez_compressed(f, test_freqs=actual_test_freqs, response=chan_responses)
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
            np.savetxt(csv_filename, zip(chan_responses[:, test_chan_rel], requested_test_freqs), delimiter=",")
            plt_filename = "{}/{}_Channel_Response.png".format(self.logs_path, self._testMethodName)
            plot_data = loggerise(chan_responses[:, test_chan_rel], dynamic_range=90, normalise=True, no_clip=True)
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
                no_of_responses = 3
                channel_response_list = [chan_responses[:, test_chan_rel + i - 1] for i in range(no_of_responses)]
                def find_channel_crossing(chs=channel_response_list, side='low'):
                    if side == 'low':
                        high_idx = 1
                        low_idx = 0
                    else:
                        high_idx = 2
                        low_idx = 1
                    band_diff = chs[high_idx] - chs[low_idx]
                    search_strt = np.argmin(band_diff)
                    search_stop = np.argmax(band_diff)
                    offset = np.argmin(np.abs(band_diff[search_strt:search_stop]))
                    return offset+search_strt 
                low_cross_idx = find_channel_crossing(channel_response_list, "low")
                hig_cross_idx = find_channel_crossing(channel_response_list, "high")
                # Find real channel spacing, if center falls between indexes calc accordingly
                cent_to_cross_idx = (hig_cross_idx - low_cross_idx)/2
                if isinstance(cent_to_cross_idx, int):
                    cent_on_idx = True
                else:
                    cent_on_idx = False
                cent_to_cross_idx = int(cent_to_cross_idx)
                cent_idx = low_cross_idx + cent_to_cross_idx
                left_idx = low_cross_idx - cent_to_cross_idx
                rght_idx = hig_cross_idx + cent_to_cross_idx
                if cent_on_idx:
                    cent_freq = actual_test_freqs[cent_idx]
                    left_freq = actual_test_freqs[left_idx]
                    rght_freq = actual_test_freqs[rght_idx]
                else:
                    cent_freq = (actual_test_freqs[cent_idx+1] + actual_test_freqs[cent_idx])/2
                    left_freq = (actual_test_freqs[left_idx-1] + actual_test_freqs[left_idx])/2
                    rght_freq = (actual_test_freqs[rght_idx+1] + actual_test_freqs[rght_idx])/2
                left_ch_spacing = cent_freq - left_freq
                rght_ch_spacing = rght_freq - cent_freq
                if round(left_ch_spacing,2) != round(rght_ch_spacing,2):
                    self.Note('Channel spacing between 3 test channels are not equal. '
                            'Centre to low = {:.1f} Hz and centre to high = {:.1f} Hz.'
                            ''.format(left_ch_spacing, rght_ch_spacing))
                measured_ch_spacing = left_ch_spacing
                self.Note("Measured channel spacing = {:.3f} kHz, "
                          "expected channel spacing from sensors = {:.3f} kHz."
                          "".format(measured_ch_spacing/1000, chan_spacing/1000))
                crossover_ch_bw = actual_test_freqs[hig_cross_idx] - actual_test_freqs[low_cross_idx]
                self.Note("Measured cross-over point bandwidth = {:.3f} kHz"
                          "".format(crossover_ch_bw/1000))

                low_cross_dbfs = plot_data[low_cross_idx]
                high_cross_dbfs = plot_data[hig_cross_idx]
                self.Note("Low  band channel cross-over point at {:.3f} dBfs.".format(low_cross_dbfs))
                self.Note("High band channel cross-over point at {:.3f} dBfs.".format(high_cross_dbfs))

                # CBF-REQ-0126
                #pass_bw_min_max = np.argwhere((np.abs(plot_data) >= 3.0) & (np.abs(plot_data) <= 3.3))
                #pass_bw = float(np.abs(actual_test_freqs[pass_bw_min_max[0]] - actual_test_freqs[pass_bw_min_max[-1]]))
                pass_bw = crossover_ch_bw


                att_bw_min_max = np.where(np.abs(plot_data) < cutoff)[0]
                # Find center of response, ignore possible sidelobes
                try:
                    idxs = np.where(np.diff(att_bw_min_max)>1)[0]
                    start_main = np.argmax(np.diff(idxs))
                    main_idxs = idxs[start_main:start_main+2]
                    att_bw_min_max = att_bw_min_max[main_idxs[0]+1:main_idxs[-1]+1]
                except:
                    # Array was contiguous
                    pass
                low_idx = att_bw_min_max[0]-1
                high_idx = att_bw_min_max[-1]+1
                if low_idx < 0:
                    low_idx = 0
                if high_idx >= len(actual_test_freqs):
                    high_idx = len(actual_test_freqs) - 1

                att_bw = actual_test_freqs[high_idx] - actual_test_freqs[low_idx]
                self.Note("-{} dB bandwith calculated at {:.3f} dBfs (low side) and {:.3f} dBfs (high side)."
                          "".format(cutoff, plot_data[low_idx], plot_data[high_idx]))

            except Exception as e:
                msg = ("Could not compute cross-over point bandwith or -{}dB attenuation bandwith. "
                        "CBF-REQ-0126 could not be verified. Exception: {}".format(cutoff))
                self.Failed(msg, exc_info=True)
            else:
                msg = (
                        "The CBF shall perform channelisation such that the 53dB attenuation bandwidth {:.3f} kHz "
                        "is less/equal to 2x the pass bandwidth {:.3f} kHz".format(att_bw/1000, pass_bw/1000)
                )
                Aqf.is_true(att_bw <= pass_bw*2, msg)

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
                central_chan_responses[:, test_chan_rel], dynamic_range=90, normalise=True, no_clip=True
            )

            caption = (
                "Channel {} central response vs source frequency for "
                "selected baseline {} / {} to test."
                "".format(test_chan, test_baseline, bls_to_test)
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
                max_chan = np.argmax(np.abs(central_chan_responses[i])) + self.start_channel
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
                np.max(np.abs(central_chan_responses[:, test_chan_rel])),
                0.99,
                "Confirm that the VACC output is at < 99% of maximum value, if fails "
                "then it is probably over-ranging.",
            )

            max_central_chan_response = np.max(10 * np.log10(central_chan_responses[:, test_chan_rel]))
            min_central_chan_response = np.min(10 * np.log10(central_chan_responses[:, test_chan_rel]))
            chan_ripple = max_central_chan_response - min_central_chan_response
            acceptable_ripple_lt = 1.5
            Aqf.hop(
                "80% channel cut-off ripple at {:.2f} dB, should be less than {} dB".format(
                    chan_ripple, acceptable_ripple_lt
                )
            )

            # The following is replaced by crossover point calculation and reporting
            ## Get frequency samples closest channel fc and crossover points
            #co_low_freq = expected_fc - df / 2
            #co_high_freq = expected_fc + df / 2

            #def get_close_result(freq):
            #    ind = np.argmin(np.abs(actual_test_freqs - freq))
            #    source_freq = actual_test_freqs[ind]
            #    response = chan_responses[ind, test_chan]
            #    return ind, source_freq, response

            #fc_ind, fc_src_freq, fc_resp = get_close_result(expected_fc)
            #co_low_ind, co_low_src_freq, co_low_resp = get_close_result(co_low_freq)
            #co_high_ind, co_high_src_freq, co_high_resp = get_close_result(co_high_freq)
            ## [CBF-REQ-0047] CBF channelisation frequency resolution requirement
            #self.Step(
            #    "Confirm that the response at channel-edges are -3 dB "
            #    "relative to the channel centre at {:.3f} Hz, actual source freq "
            #    "{:.3f} Hz".format(expected_fc, fc_src_freq)
            #)

            #desired_cutoff_resp = -6  # dB
            #acceptable_co_var = 0.1  # dB, TODO 2015-12-09 NM: thumbsuck number
            #co_mid_rel_resp = 10 * np.log10(fc_resp)
            #co_low_rel_resp = 10 * np.log10(co_low_resp)
            #co_high_rel_resp = 10 * np.log10(co_high_resp)

            #co_lo_band_edge_rel_resp = co_mid_rel_resp - co_low_rel_resp
            #co_hi_band_edge_rel_resp = co_mid_rel_resp - co_high_rel_resp

            #low_rel_resp_accept = np.abs(desired_cutoff_resp + acceptable_co_var)
            #hi_rel_resp_accept = np.abs(desired_cutoff_resp - acceptable_co_var)
            # Not a requirement anymore, just report
            #low_rel_resp_accept <= co_lo_band_edge_rel_resp <= hi_rel_resp_accept,
            #Aqf.step(
            #    "The relative response at the low band-edge: "
            #    "(-{co_lo_band_edge_rel_resp} dB @ {co_low_freq} Hz, actual source freq "
            #    "{co_low_src_freq}) relative to channel centre response.".format(**locals()),
            #)
            ##low_rel_resp_accept <= co_hi_band_edge_rel_resp <= hi_rel_resp_accept,
            #Aqf.step(
            #    "The relative response at the high band-edge: "
            #    "(-{co_hi_band_edge_rel_resp} dB @ {co_high_freq} Hz, actual source freq "
            #    "{co_high_src_freq}) relative to channel centre response.".format(**locals()),
            #)
            #cutoff_edge = np.abs((co_lo_band_edge_rel_resp + co_hi_band_edge_rel_resp) / 2)
            crossover = (low_cross_dbfs + high_cross_dbfs)/2

            center_bin = [left_idx, cent_idx, rght_idx]
            y_axis_limits = (-90, 1)
            legends = [
                "Channel {} / Sample {} \n@ {:.3f} MHz".format(
                    ((test_chan + i) - 1), v, self.cam_sensors.ch_center_freqs[test_chan + i] / 1e6
                )
                for i, v in zip(range(no_of_responses), center_bin)
            ]
            # center_bin.append('Channel spacing: {:.3f}kHz'.format(856e6 / self.n_chans_selected / 1e3))
            center_bin.append("Channel spacing: {:.3f}kHz".format(measured_ch_spacing / 1e3))

            csv_filename = "/".join([self._katreport_dir, r"channelisation_response_data.csv"])
            np.savetxt(csv_filename, channel_response_list, delimiter=",")
            plot_title = "PFB Channel Response"
            plot_filename = "{}/{}_adjacent_channels.png".format(self.logs_path, self._testMethodName)

            caption = (
                "Sample PFB central channel response between channel {} and selected baseline "
                "{}/{},with channelisation spacing of {:.3f}kHz within tolerance of 1%, with "
                "the digitiser simulator configured to generate a continuous wave, with cw scale:"
                " {}, awgn scale: {}, Eq gain: {} and FFT shift: {}".format(
                    test_chan, test_baseline, bls_to_test, measured_ch_spacing / 1e3, cw_scale, awgn_scale, gain, fft_shift
                )
            )

            aqf_plot_channels(
                zip(channel_response_list, legends),
                plot_filename,
                plot_title,
                normalise=True,
                caption=caption,
                crossover=crossover,
                vlines=center_bin,
                xlabel="Sample Steps",
                ylimits=y_axis_limits,
            )

            self.Step(
                "Measure the attenuation between the middle of the center bin and the middle of "
                "adjacent bins and confirm that is > %sdB" % cutoff
            )
            if cent_on_idx:
                cent_pwr  = plot_data[cent_idx]
                left_pwr  = plot_data[left_idx]
                rght_pwr  = plot_data[rght_idx]
            else:
                cent_pwr  = (plot_data[cent_idx+1] + plot_data[cent_idx])/2
                left_pwr  = (plot_data[left_idx-1] + plot_data[left_idx])/2
                rght_pwr  = (plot_data[rght_idx+1] + plot_data[rght_idx])/2
            power_diff = np.abs(cent_pwr - left_pwr)
            msg = ("Attenuation between channel centre and at centre of low channnel is {:.1f} dB"
                   "".format(power_diff))
            #Aqf.more(power_diff, cutoff, msg)
            self.Note(msg)
            power_diff = np.abs(cent_pwr - rght_pwr)
            msg = ("Attenuation between channel centre and at centre of high channnel is {:.1f} dB"
                   "".format(power_diff))
            self.Note(msg)
            #Aqf.more(power_diff, cutoff, msg)
            

            #TODO This does not make a useful measurement?
            #for bin_num, chan_resp in enumerate(channel_response_list, 1):
            #    power_diff = np.max(loggerise(chan_resp)) - cutoff
            #    msg = "Confirm that the power difference (%.2fdB) in bin %s is more than %sdB" % (
            #        power_diff,
            #        bin_num,
            #        -cutoff,
            #    )
            #    Aqf.less(power_diff, -cutoff, msg)

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
            

    def _test_sfdr_peaks(self, req_chan_spacing, no_channels=None, num_discard=5,
                         cutoff=53, plots_debug=False, log_power=True):

        """Test channel spacing and out-of-channel response

        Check that the correct channels have the peak response to each
        frequency and that no other channels have significant relative power.

        Will loop over all the channels, placing the source frequency as close to the
        centre frequency of that channel as possible.

        Parameters
        ----------
        required_chan_spacing: float
        no_channels: int
            if no_channels is None all channels will be used
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
        max_channels_errors = []
        # Channel responses higher than -cutoff dB relative to expected channel
        extra_peaks = []
        # Band Shape Sweep
        band_shape_sweep_vals = []
        band_shape_chans = []
        band_shape_ch_freq = []
        band_shape_resp = []
        # Checking for all channels.
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        full_bw = self.cam_sensors.get_value("bandwidth", exact=True)
        volt_bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")

        # Get test frequencies, if all channels are not being received a band shape sweep 
        # will not be done. If narrowband is being tested select frequencies across the 
        # full band to do a band shape sweep. If a band shape sweep is to be done and 
        # a narrow band instrument is selected frequencies must be chosen to fall across 
        # the full band and not just the channelised voltage band.
        # All channels not selected, do not perform a band shape sweep:
        if n_chans != self.n_chans_selected:
            _msg = ("Due to performance constraints the test will sweep through {} channels."
                    "".format(no_channels)
            )
            self.Note(_msg)
            band_shape_sweep = False
            if no_channels:
                sel_chs = np.linspace(self.start_channel, self.stop_channel, 
                        no_channels, endpoint=False, dtype='int16')
            else:
                sel_chs = np.arange(self.start_channel,self.stop_channel)
            channel_freqs = np.asarray(
                [self.cam_sensors.ch_center_freqs[i] for i in sel_chs[1:]])
            test_ch_and_freqs = zip(sel_chs[1:], channel_freqs)
            # This variable is for band shape sweep but put a copy
            # here so it exists when called
            bss_inc_c_and_f = np.asarray(test_ch_and_freqs[:])
        else:
            if no_channels:
                _msg = ("Due to resource time constraints the test will sweep through {} channels."
                        "".format(no_channels)
                )
                self.Note(_msg)
            # Perform a band shape sweep in addition to normal sweep
            # If full_bw is more than the channelised voltage bandwidth a narrowband
            # instrument is under test and frequencies outside the channelised voltage
            # band must be chosen
            band_shape_sweep = True
            # narrow band
            if full_bw != volt_bw:
                ch_bw = self.cam_sensors.ch_center_freqs[1]-self.cam_sensors.ch_center_freqs[0]
                center_f = self.cam_sensors.get_value("antenna_channelised_voltage_center_freq")
                n_ch_nb_fullbw = int(full_bw/ch_bw)
                f_start = center_f - (full_bw/2.) # Center freq of the first channel
                ch_center_freqs_nbfull = f_start + np.arange(n_ch_nb_fullbw) * ch_bw
                if no_channels:
                    sel_chs = np.linspace(0,len(ch_center_freqs_nbfull), 
                            no_channels, endpoint=False, dtype='int32')
                    channel_freqs = np.asarray([ch_center_freqs_nbfull[i] for i in sel_chs[1:]])
                    test_ch_and_freqs = zip(sel_chs[1:], channel_freqs)
                else:
                    channel_freqs = ch_center_freqs_nbfull[:]
                    test_ch_and_freqs = zip(np.arange(len(channel_freqs)), channel_freqs)
                # Band shape sweep channels that must also be used for channel
                # frequency SFDR test, for narrowband use channels specified in
                # config file
                check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel"))
                check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel"))
                chk_strt_f = self.cam_sensors.ch_center_freqs[check_strt_ch]
                chk_stop_f = self.cam_sensors.ch_center_freqs[check_stop_ch]
                bss_inc_freqs = channel_freqs[((channel_freqs >= chk_strt_f) & 
                                               (channel_freqs <= chk_stop_f))]
                # Find narrowband channels numbers for frequencies that will be used 
                # during channel frequency SFDR test
                nb_test_chans = [np.argwhere(x == self.cam_sensors.ch_center_freqs)[0][0] 
                        for x in bss_inc_freqs]
                bss_inc_c_and_f = np.asarray(zip(nb_test_chans, bss_inc_freqs))
                if len(nb_test_chans) != len(bss_inc_freqs):
                    Aqf.failed("Frequency and channel calculations do not make sense. "
                               "Bandwidth or channel sensor or coding error.")
                    return     
            else:
                channel_freqs = self.cam_sensors.ch_center_freqs
                if no_channels:
                    sel_chs = np.linspace(0,len(channel_freqs), 
                            no_channels, endpoint=False, dtype='int32')
                    channel_freqs = [channel_freqs[i] for i in sel_chs[1:]]
                    test_ch_and_freqs = zip(sel_chs[1:], channel_freqs)
                else:
                    test_ch_and_freqs = zip(np.arange(n_chans), channel_freqs)
                # For wideband include all channels in channel frequency SFDR test
                bss_inc_c_and_f = np.asarray(test_ch_and_freqs[:])
        if test_ch_and_freqs[0][1] == 0:
            # skip DC channel since dsim puts out zeros for freq=0
            test_ch_and_freqs = test_ch_and_freqs[1:]

        msg = (
            "This tests confirms that the correct channels have the peak response to each"
            " frequency and that no other channels have significant relative power, while logging "
            "the power usage of the CBF in the background."
        )
        self.Step(msg)
        if log_power:
            self.Progress("Logging power usage in the background.")

        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        self.Step(
            "Dsim cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                cw_scale, awgn_scale, gain, fft_shift
            )
        )

        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale,
            cw_scale=cw_scale,
            freq=volt_bw/2.0,
            fft_shift=fft_shift,
            gain=gain,
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        else:
            curr_mcount = self.current_dsim_mcount()  #dump_after_mcount

        self.logger.info(
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
            # [CBF-REQ-0043]
            nominal_bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") * self.dsim_factor
            chan_spacing = round(nominal_bw / n_chans, 2)
            chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100), chan_spacing + (chan_spacing * 1 / 100)]
            self.Step("CBF-REQ-0043, 0053, 0226, 0227 and 0236 Confirm channel spacing.")
            msg = ("Verify that the calculated channel frequency step ({:.3f} kHz) is equal or less than {} kHz"
                   "".format(chan_spacing/1000., req_chan_spacing/1000.))
            if chan_spacing <= req_chan_spacing:
                ch_spacing_res = True
            else:
                ch_spacing_res = False
            Aqf.is_true(ch_spacing_res, msg)

        self.Step(
            "Sweep a digitiser simulator tone over channels that fall within the band.")
        channel_response_lst = []
        num_prints = 3
        # List of channels to be plotted
        try:
            nctst = len(nb_test_chans)
            ctst  = nb_test_chans[:]
        except NameError:
            nctst = len(test_ch_and_freqs)
            ctst  = np.asarray(test_ch_and_freqs)[:,0]
        try:
            middle = int(ctst[nctst//2])
            start =  int(ctst[nctst//4])
            end =    int(ctst[nctst - nctst//4])
            chans_to_plot = (start,middle,end)
        except IndexError:
            chans_to_plot = ()
        print_cnt = 0
        # Clear the que
        #this_freq_dump = self.get_real_clean_dump(discard=num_discard)
        this_freq_dump = self.get_dump_after_mcount(curr_mcount)  #dump_after_mcount
        for channel, channel_f0 in test_ch_and_freqs:
            if print_cnt < num_prints:
                self.Progress(
                    "Getting channel response for freq %s @ %s: %.3f MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )
            elif print_cnt == num_prints:
                self.Progress("...")
            elif print_cnt >= (len(test_ch_and_freqs)-num_prints):
                self.Progress(
                    "Getting channel response for freq %s @ %s: %.3f MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )
            else:
                self.logger.info(
                    "Getting channel response for freq %s @ %s: %s MHz."
                    % (channel, len(channel_freqs), channel_f0 / 1e6)
                )
            print_cnt+=1

            self.dhost.sine_sources.sin_0.set(frequency=channel_f0, scale=cw_scale)
            self.dhost.sine_sources.sin_1.set(frequency=0, scale=0)
            # self.dhost.sine_sources.sin_corr.set(frequency=0, scale=0)
            curr_mcount = self.current_dsim_mcount()

            this_source_freq = self.dhost.sine_sources.sin_0.frequency
            actual_test_freqs.append(this_source_freq)
            for i in range(self.data_retries):  
                #this_freq_dump = self.get_real_clean_dump(discard=num_discard)
                this_freq_dump = self.get_dump_after_mcount(curr_mcount)  #dump_after_mcount
                if this_freq_dump is not False:
                    break
                self.Error("Could not retrieve clean SPEAD accumulation", exc_info=True)
                return False
            this_freq_data = this_freq_dump["xeng_raw"]
            this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])

            if len(bss_inc_c_and_f) != 0:
                if channel_f0 in bss_inc_c_and_f[:,1]:
                    # get channel out of band shape sweep include array
                    ch_idx = np.argwhere(channel_f0 == bss_inc_c_and_f[:,1])[0][0]
                    ch = int(bss_inc_c_and_f[ch_idx][0])
                    if ch in chans_to_plot:
                        channel_response_lst.append(this_freq_response)

                    max_chan_rel = np.argmax(this_freq_response)
                    max_chan     = max_chan_rel + self.start_channel
                    # TODO: figure out if pipelining test could work
                    #print max_chan
                    if max_chan != ch:
                        max_chan_val = this_freq_response[max_chan_rel]
                        ch_val = this_freq_response[ch - self.start_channel]
                        max_channels_errors.append((ch,ch_val,max_chan,max_chan_val))

                    # Find responses that are more than -cutoff relative to max
                    new_cutoff = np.max(loggerise(this_freq_response)) - cutoff
                    # TODO: Figure out what this was all about
                    # unwanted_cutoff = this_freq_response[max_chan] / 10 ** (new_cutoff / 100.0)
                    extra_responses = [
                        (i + self.start_channel) for i, resp in enumerate(loggerise(this_freq_response))
                        #if i != max_chan and resp >= unwanted_cutoff
                        if (i != (ch - self.start_channel)) and (resp >= new_cutoff)
                    ]
                    if len(extra_responses) != 0:
                        extra_peaks.append((ch,extra_responses,this_freq_response))

                    # TODO: add plots back in if spurious channels or channels above cutoff found
                    #plt_title = "Frequency response at {}".format(channel)
                    #plt_filename = "{}/{}_channel_{}_resp.png".format(self.logs_path,
                    #    self._testMethodName, channel)
                    #if extra_responses:
                    #    msg = "Weirdly found an extra responses on channel %s" % (channel)
                    #    self.Note(msg)
                    #    plt_title = "Extra responses found around {}".format(channel)
                    #    plt_filename = "{}_extra_responses.png".format(self._testMethodName)
                    #    plots_debug = True

                    #if plots_debug:
                    #    plots_debug = False
                    #    new_cutoff = np.max(loggerise(this_freq_response)) - cutoff
                    #    aqf_plot_channels(
                    #        this_freq_response, plt_filename, plt_title, log_dynamic_range=90,
                    #        hlines=new_cutoff
                    #    )

            if band_shape_sweep:
                band_shape_sweep_vals.append(np.max(loggerise(this_freq_response)))
                band_shape_chans.append(channel)
                band_shape_ch_freq.append(channel_f0)
                band_shape_resp.append(this_freq_response)
            

        for channel, channel_resp in zip(chans_to_plot, channel_response_lst):
            plt_filename = "{}/{}_channel_{}_resp.png".format(self.logs_path, self._testMethodName, channel)
            test_freq_mega = self.cam_sensors.ch_center_freqs[channel] / 1e6
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
                channel_resp, plt_filename, plt_title, log_dynamic_range=90, caption=caption, 
                hlines=new_cutoff,
                ylabel="dBFS relative to VACC max",
                start_channel=self.start_channel
            )
        if band_shape_sweep:
            fn = "/".join([self._katreport_dir, r"band_shape_sweep.npz"])
            with open(fn, 'w') as f:
                np.savez_compressed(f, 
                        channels = band_shape_chans,
                        freqs    = band_shape_ch_freq,
                        response = band_shape_resp
                )
            plt_filename = "{}/{}_band_shape_sweep.png".format(self.logs_path, self._testMethodName)
            plt_title = "Peak responses for CW swept across band"
            caption = ("CW sweep across the entire band using {} points. The maximum value found in "
                       "the band response is plotted at each sample point."
                       "".format(len(band_shape_chans)))

            aqf_plot_band_sweep(
                band_shape_ch_freq, band_shape_sweep_vals, plt_filename, plt_title, caption=caption, 
            )

        if max_channels_errors == []:
            msg = ("The correct channels have peak responses for each frequency.")
            self.Passed(msg)
        else:
            msg = ("The following channels do not have peak responses where expected [Channel under test, value, Channel where max val found, value]: "
                   "{}".format(max_channels_errors))
            self.Failed(msg)

        #msg = "Confirm that no other channels response more than -%s dB.\n" % cutoff
        
        if extra_peaks == []:
            self.Note("No channels found with power greater that {} db from peak values."
                      "".format(cutoff))
        else:
            print_count = 5
            for channel, peaks, response in extra_peaks:
                if print_count > 1:
                    rel_ch = channel - self.start_channel
                    blanked_resp = loggerise(response)
                    ch_max_val = blanked_resp[rel_ch]
                    blanked_resp[rel_ch] = np.average(blanked_resp)
                    max_spurious_ch = np.argmax(blanked_resp) + self.start_channel
                    max_spurious_val = blanked_resp[max_spurious_ch - self.start_channel] - ch_max_val
                    new_cutoff = ch_max_val - cutoff
                    self.Note("Found {} channels with power more than -{} dB "
                              "from peak in channel {}.".format(len(peaks), cutoff, channel))
                    self.Note("Maximum spurious channel: {} at {:.1f} db from CW max."
                              "".format(max_spurious_ch, max_spurious_val))
                    print_count -= 1
                    plt_filename = ("{}/{}_channel_{}_err_resp.png"
                                    ''.format(self.logs_path, self._testMethodName, channel)
                    )
                    test_freq_mega = self.cam_sensors.ch_center_freqs[channel] / 1e6
                    plt_title = "Frequency response at {} @ {:.3f} MHz".format(channel, test_freq_mega)
                    caption = (
                        "Frequency response around channel {} @ {:.3f}MHz, "
                        "when digitiser simulator is configured to generate a continuous wave, "
                        "with cw scale: {}. awgn scale: {}, eq gain: {}, fft shift: {}".format(
                            channel, test_freq_mega, cw_scale, awgn_scale, gain, fft_shift
                        )
                    )
                    aqf_plot_channels(
                        response[rel_ch-4:rel_ch+5], plt_filename, plt_title, log_dynamic_range=90, 
                        caption=caption, 
                        hlines=new_cutoff,
                    )

                elif print_count == 1:
                    self.Note('More channels found with spurious peaks.')
                    print_count -= 1
                else:
                    pass
            #self.logger.debug("Expected: %s\n\nGot: %s" % (extra_peaks, [[]] * len(max_channels)))
            #self.Failed(msg)
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
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        #cw_scale = 0.035
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
            self.dhost.sine_sources.sin_0.set(scale=0, frequency=freq)
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

    def _test_product_baselines(self, check_strt_ch=None, check_stop_ch=None, num_discard=5):
        heading("CBF Baseline Correlation Products")
        # Setting DSIM to generate noise
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
        self.Step(
            "Digitiser simulator configured to generate Gaussian noise, "
            "with scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
        )
        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale,
            freq=self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") / 2.0,
            fft_shift=fft_shift,
            gain=gain,
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        else:
            curr_mcount = self.current_dsim_mcount()

        # Does not make sense to change input labels anymore. These should be set at instrument initialisation
        #try:
        #    #self.Step("Change CBF input labels and confirm via CAM interface.")
        #    #reply_, _ = self.katcp_req.input_labels()
        #    #self.assertTrue(reply_.reply_ok())
        #    #ori_source_name = reply_.arguments[1:]
        #    ori_source_name = self.cam_sensors.input_labels
        #    self.Progress("Original source names: {}".format(", ".join(ori_source_name)))
        #except Exception:
        #    self.Error("Failed to retrieve input labels via CAM interface", exc_info=True)
        #try:
        #    local_src_names = self.cam_sensors.custom_input_labels
        #    reply, _ = self.katcp_req.input_labels(*local_src_names)
        #    self.assertTrue(reply.reply_ok())
        #except Exception:
        #    self.Error("Could not retrieve new source names via CAM interface:\n %s" % (str(reply)))
        #else:
        #    #source_names = reply.arguments[1:]
        #    source_names = self.cam_sensors.input_labels
        #    msg = "Source names changed to: {}".format(", ".join(source_names))
        #    self.Passed(msg)

        # Moving away from discards. Going to retry captures until it works or times out.
        #try:
        #    if self.cam_sensors.sensors.n_ants.value > 16:
        #        #_discards = 60
        #        _discards = 10
        #    else:
        #        #_discards = 30
        #        _discards = 5

        #try:
        #    self.Step(
        #        "Capture an initial correlator SPEAD accumulation, "
        #        "and retrieve list of all the correlator input labels via "
        #        "Cam interface.".format(_discards)
        #    )
        #    #test_dump = self.receiver.get_clean_dump(discard=_discards)
        #    test_dump = self.get_real_clean_dump()
        #    self.assertIsInstance(test_dump, dict)
        #except AssertionError:
        #    errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
        #    self.Error(errmsg, exc_info=True)
        #else:
            # Get bls ordering from get dump
        self.Step(
            "Get list of all possible baselines (including redundant baselines) present "
            "in the correlator output."
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
        for i in range(self.data_retries):  
            # Dump till mcount change
            #test_data = self.get_real_clean_dump(discard=num_discard)
            test_data = self.get_dump_after_mcount(curr_mcount)
            if check_strt_ch and check_stop_ch:
                test_data["xeng_raw"] = test_data["xeng_raw"][check_strt_ch:check_stop_ch,:,:]
            if test_data is not False:
                z_baselines = zero_baselines(test_data["xeng_raw"])
                if not(z_baselines): break
            self.logger.warn("Baseslines with all-zero visibilites found, retrying capture.")
        #self.Step(
        #    "Expect all baselines and all channels to be " "non-zero with Digitiser Simulator set to output AWGN."
        #)
        if test_data is not False:
            msg = "Confirm that no baselines have all-zero visibilities."
            Aqf.is_false(z_baselines, msg)
            msg = "Confirm that all baseline visibilities are non-zero across all channels"
            if not nonzero_baselines(test_data["xeng_raw"]) == all_nonzero_baselines(test_data["xeng_raw"]):
                self.Failed(msg)
            else:
                self.Passed(msg)
        #self.Step("Save initial f-engine equalisations, and ensure they are " "restored at the end of the test")
        #initial_equalisations = self.get_gain_all()
        #self.Progress("Stored original F-engine equalisations.")

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
            retries = 5
            while True:
                try:
                    for _input in input_labels:
                        reply, _ = self.katcp_req.gain(_input)
                        self.assertTrue(reply.reply_ok())
                        eq_values = reply.arguments[1:]
                        eq_values = [complex(x) for x in eq_values]
                        eq_values = np.asarray(eq_values)
                        eq_sum = eq_values.sum()
                except Exception as e:
                    self.Failed("Failed to retrieve gains/equalisations: {}".format(e))
                    break
                else:
                    msg = "Confirm that all the inputs equalisations have been set to 'Zero'."
                    if (eq_sum == complex(0)) or (retries == 0):
                        Aqf.equals(eq_sum, complex(0), msg)
                        break
                    else:
                        self.logger.info("Sum of eq values read back = {}, retyring".format(eq_sum))
                        retries -= 1
                        time.sleep(5)

        self.Step("Set all inputs gains to 'Zero', and confirm that output product is all-zero")
        set_zero_gains()
        read_zero_gains()
        curr_mcount = self.current_dsim_mcount()
        for i in range(self.data_retries):  
            # Dump till mcount change
            #test_data = self.get_real_clean_dump()
            test_data = self.get_dump_after_mcount(curr_mcount)
            if check_strt_ch and check_stop_ch:
                test_data["xeng_raw"] = test_data["xeng_raw"][check_strt_ch:check_stop_ch,:,:]
            if test_data is not False:
                nz_baselines = nonzero_baselines(test_data["xeng_raw"])
                if not(nz_baselines): break
            self.logger.warn("Baseslines with non-zero visibilites found, retrying capture.")

        if test_data is not False:
            Aqf.is_false(
                nz_baselines,
                "Confirm that all baseline visibilities are 'Zero'."
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
            "the correct output appears in the correct baseline product."
        )
        self.Step(bls_msg)
        self.Step(
            "Confirm that gain/equalisation correction has been applied."
                )
        # dataFrame = pd.DataFrame(index=sorted(input_labels),
        #                          columns=list(sorted(present_baselines)))

        for count, inp in enumerate(input_labels, start=1):
            if count > 10:
                break
            #old_eq = complex(initial_equalisations)
            old_eq = gain
            self.logger.info(
                "Gain/equalisation correction on input %s set to %s." % (inp, old_eq)
            )
            try:
                reply, _ = self.katcp_req.gain(inp, old_eq)
                self.assertTrue(reply.reply_ok())
                curr_mcount = self.current_dsim_mcount()
            except AssertionError:
                errmsg = "%s: Failed to set gain/eq of %s for input %s" % (str(reply), old_eq, inp)
                self.Error(errmsg, exc_info=True)
            else:
                msg = "Gain/Equalisation correction on input %s set to %s." % (inp, old_eq)
                self.Passed(msg)
                zero_inputs.remove(inp)
                nonzero_inputs.add(inp)
                expected_z_bls, expected_nz_bls = calc_zero_and_nonzero_baselines(nonzero_inputs)
                for i in range(self.data_retries):  
                    #test_dump = self.get_real_clean_dump(discard=num_discard)
                    test_dump = self.get_dump_after_mcount(curr_mcount)
                    if check_strt_ch and check_stop_ch:
                        test_dump["xeng_raw"] = (test_dump["xeng_raw"]
                                [check_strt_ch:check_stop_ch,:,:])
                    if test_dump is not False:
                        test_data = test_dump["xeng_raw"]
                        actual_nz_bls_indices = all_nonzero_baselines(test_data)
                        actual_nz_bls = set([tuple(bls_ordering[i]) for i in actual_nz_bls_indices])
                        actual_z_bls_indices = zero_baselines(test_data)
                        actual_z_bls = set([tuple(bls_ordering[i]) for i in actual_z_bls_indices])
                        if (actual_nz_bls == expected_nz_bls) and (actual_z_bls == expected_z_bls): break
                    self.logger.warn("Correct baselines with non-zero and zero visibilites not found, retrying capture.")

                if test_dump is not False:
                    msg = "Baseline visibilities are non-zero with non-zero inputs"
                    msg = msg + " (%s)" % (sorted(nonzero_inputs))
                    if actual_nz_bls == expected_nz_bls:
                        self.Passed(msg)
                    else:
                        self.Failed(msg)

                    msg = "All other baseline visibilities are zero."
                    if actual_z_bls == expected_z_bls:
                        self.Passed(msg)
                    else:
                        self.Failed(msg)
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
                            start_channel=self.start_channel
                        )

                    # Sum of all baselines powers expected to be non zeros
                    # sum_of_bl_powers = (
                    #     [normalised_magnitude(test_data[:, expected_bl, :])
                    #      for expected_bl in [baselines_lookup[expected_nz_bl_ind]
                    #                          for expected_nz_bl_ind in sorted(expected_nz_bls)]])
                    test_data = None
                    # dataFrame.loc[inp][sorted(
                    #     [i for i in expected_nz_bls])[-1]] = np.sum(sum_of_bl_powers)

        # dataFrame.T.to_csv('{}.csv'.format(self._testMethodName), encoding='utf-8')

    def _test_back2back_consistency(self, test_channels, num_discard):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect.
        """

        def get_expected_acc_val(test_chan):
            try:
                test_input = self.cam_sensors.input_labels[0]
                reply, informs = self.katcp_req.quantiser_snapshot(test_input)
                self.assertTrue(reply.reply_ok())
                informs = informs[0]
            except Exception:
                errmsg = (
                    "Failed to retrieve quantiser snapshot of input %s via "
                    "CAM Interface: \nReply %s" % (test_input, str(reply).replace("_", " "),))
                self.logger.info(errmsg)
                # Quantiser value not found, take expected value as found in baseline 0
                return 0
            else:
                n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                center_ch = int(n_chans/2)
                quantiser_spectrum = np.array(evaluate(informs.arguments[-1]))
                quant_check = np.where(np.abs(quantiser_spectrum) > 0)[0]
                if (quant_check.shape[0] == 1):
                    if quant_check[0] != test_chan:
                        Aqf.failed("Tone not in correct channel: {}".format(quant_check[0]))
                        return False
                elif (quant_check.shape[0] == 0):
                    self.logger.warn("No tone found in quantiser output.")
                    return 0
                else:
                    Aqf.failed("More than one value found in quantiser "
                               "@ channels: {}".format(quant_check))
                    return False

                # The re-quantiser outputs signed int (8bit), but the snapshot code
                # normalises it using binary 8.7 scaling. Since we want to calculate the
                # output of the vacc which sums integers, denormalise the snapshot
                # output back to ints. Then convert to power
                quant_value = np.abs(quantiser_spectrum[test_chan])
                quant_power = (quant_value*2**7)**2
                #self.Note("Quantiser test channel voltage magnitude (represented in binary "
                #          "8.7): {:.4f}".format(quant_value))
                #self.Note("Converted to power and scaled by 2**7: {}".format(quant_power))
                no_accs = self.cam_sensors.get_value("baseline_correlation_products_n_accs")
                return quant_power * no_accs

        heading("Spead Accumulation Back-to-Back Consistency")
        self.Progress("Randomly selected test channels: %s" % (test_channels))
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
        source_period_in_samples = n_chans * 2 * decimation_factor
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        #if decimation_factor != 1:
        #    cw_scale = cw_scale/decimation_factor
        #else:
        cw_scale = cw_scale/2
        #cw_scale = cw_scale/8
        # Reset the dsim and set fft_shifts and gains
        dsim_set_success = self.set_input_levels(
            awgn_scale=0, cw_scale=0, freq=0, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        self.Step(
            "Digitiser simulator configured to generate periodic CW "
            "with repeating samples equal to FFT-length {}) in order for each FFT to be "
            "identical.".format(source_period_in_samples)
        )
        self.Step(
            "Get a reference SPEAD accumulation and confirm that the following accumulation is identical."
        )
        ch_list = self.cam_sensors.ch_center_freqs
        ch_bw = ch_list[1]-ch_list[0]
        num_prints = 3
        print_cnt = 0
        # Clear cue
        # Dump till mcount change:
        #dump = self.get_real_clean_dump(discard=num_discard, quiet=True)
        for chan in test_channels:
            if print_cnt < num_prints:
                print_output = True
            elif print_cnt == num_prints:
                self.Progress("...")
                print_output = False
            elif print_cnt >= (len(test_channels)-num_prints):
                print_output = True
            else:
                print_output = False
            print_cnt+=1
            freq = ch_list[chan]
            try:
                # Make dsim output periodic in FFT-length so that each FFT is identical
                self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=cw_scale, 
                        repeat_n=source_period_in_samples)
                assert self.dhost.sine_sources.sin_corr.repeat == source_period_in_samples
                this_source_freq = self.dhost.sine_sources.sin_corr.frequency
                # Dump till mcount change
                curr_mcount = self.current_dsim_mcount()
            except AssertionError:
                errmsg = ("Failed to make the DEng output periodic in FFT-length so "
                          "that each FFT is identical, or cw0 does not equal cw1 freq.")
                self.Error(errmsg, exc_info=True)
                return False
            if print_output:
                Aqf.hop("Getting response for channel {} @ {:.3f} MHz."
                        .format(chan, (this_source_freq / 1e6)))
            else:
                self.logger.info("Getting response for channel {} @ {:.3f} MHz."
                        .format(chan, (this_source_freq / 1e6)))

            # Retry if correct data not received. This may be due to congestion on the receiver que
            # Clear cue
            # Dump till mcount change
            #dump = self.get_real_clean_dump(discard=num_discard, quiet=True)
            for i in range(self.data_retries):
                dumps_data = []
                #chan_responses = []
                # Dump till mcount change
                #if dump is not False:
                for dump_no in range(3):
                    # Dump till mcount change
                    #this_freq_dump = self.get_real_clean_dump(quiet=True)
                    this_freq_dump = self.get_dump_after_mcount(curr_mcount)
                    if this_freq_dump is not False:
                        this_freq_data = this_freq_dump["xeng_raw"]
                        dumps_data.append(this_freq_data)
                        #this_freq_response = normalised_magnitude(this_freq_data[:, test_baseline, :])
                        #chan_responses.append(this_freq_response)
                    else:
                        break
                try:
                    dumps_comp = np.diff(dumps_data, axis=0)
                    dumps_comp_max = np.max(dumps_comp)
                    if dumps_comp_max == 0: break
                    Aqf.hop('Dumps found to not be equal for channel {}, trying again'.format(chan))
                except:
                    pass

            if (i == self.data_retries-1) and ("dumps_comp" not in locals()):
                errmsg = "SPEAD data not received."
                self.Error(errmsg, exc_info=True)
                return False

            if dumps_comp_max == 0: 
                msg = ("Subsequent SPEAD accumulations are identical.")
                if print_output:
                    self.Passed(msg)
                else:
                    self.logger.info(msg)
            else:
                Aqf.failed("Channels and baseline indexes where subsequent "
                        "accumulations are not identical: {}"
                        .format(np.where(np.sum(dumps_comp, axis=0))[0:2]))
            #    legends = ["dump #{}".format(x) for x in range(len(chan_responses))]
            #    plot_filename = "{}/{}_chan_resp_{}.png".format(self.logs_path, self._testMethodName, i + 1)
            #    plot_title = "Frequency Response {} @ {:.3f}MHz".format(chan, this_source_freq / 1e6)
            #    caption = (
            #        "Comparison of back-to-back SPEAD accumulations with digitiser simulator "
            #        "configured to generate periodic wave ({:.3f}Hz with FFT-length {}) "
            #        "in order for each FFT to be identical".format(this_source_freq, source_period_in_samples)
            #    )
            #    aqf_plot_channels(
            #        zip(chan_responses, legends),
            #        plot_filename,
            #        plot_title,
            #        log_dynamic_range=90,
            #        log_normalise_to=1,
            #        normalise=False,
            #        caption=caption,
            #    )
            expected_val = get_expected_acc_val(chan)
            if expected_val == 0:
                # quantiser snapshot did not work, take expected as value in baseline 0
                # expected_val = np.max(magnetise(dumps_data[0][:,0,:]))
                # Quantiser fixed, this is an error
                return False
            elif expected_val == False:
                return False
            failed = False
            num_err_prints = 2
            for bline, idx in dict.items(self.get_baselines_lookup()):
                baseline_dumps_mag = []
                _dummy = [baseline_dumps_mag.append(magnetise(x[:,idx,:])) for x in dumps_data]
                leak_check = np.where(np.sum(baseline_dumps_mag, axis=0) > 0)[0]
                if len(leak_check) == 0:
                    if num_err_prints != 0:
                        Aqf.note("No tone found for baseline {} @ channel: {}".format(bline, chan))
                        num_err_prints -= 1
                    else:
                        self.logger.error("No tone found for baseline {} @ channel: {}".format(bline, chan))
                    failed = True
                else:
                    baseline_dumps_chval = np.asarray([np.max(x) for x in baseline_dumps_mag])
                    leak_check = leak_check + self.start_channel
                    if (leak_check.shape[0] != 1):
                        if num_err_prints != 0:
                            Aqf.failed("More than one value found in baseline {} "
                                    "@ channels: {}".format(bline,leak_check))
                            num_err_prints -= 1
                        else:
                            self.logger.error("More than one value found in baseline {} "
                                    "@ channels: {}".format(bline,leak_check))
                        failed = True
                    elif leak_check[0] != chan:
                        if num_err_prints != 0:
                            Aqf.failed("CW found in channel {} for baseline {}, "
                                    "but was expected in channel: {}."
                                    .format(leak_check[0], bline, chan))
                            num_err_prints -= 1
                        else:
                            self.logger.error("CW found in channel {} for baseline {}, "
                                    "but was expected in channel: {}."
                                    .format(leak_check[0], bline, chan))
                        failed = True
                    check_vacc = np.where(np.round(baseline_dumps_chval,5) != np.round(expected_val,5))[0]
                    if len(check_vacc) != 0:
                        if num_err_prints != 0:
                            self.logger.error("Expected VACC value ({}) is not equal to "
                                    "measured values for captured accumulations ({}) "
                                    "for baseline {}, channel {}."
                                    .format(expected_val, baseline_dumps_chval, bline, chan))
                            num_err_prints -= 1
                        else:
                            self.logger.error("Expected VACC value ({}) is not equal to "
                                    "measured values for captured accumulations ({}) "
                                    "for baseline {}, channel {}."
                                    .format(expected_val, baseline_dumps_chval, bline, chan))
                        #failed = True
                        #import IPython;IPyhton.embed()
            if num_err_prints < 1:
                Aqf.failed('More failures occured, but not printed, check log for output.')
            if not failed:
                if print_output:
                    self.Passed("CW magnitude and location correct for all baselines, "
                            "no leakage found.")
                else:
                    self.logger.info("CW magnitude and location correct for all baselines, "
                            "no leakage found.")
                    

    def _test_freq_scan_consistency(self, test_chan, num_discard=4, threshold=1e-1):
        """This test confirms if the identical frequency scans produce equal results."""
        heading("Spead Accumulation Frequency Consistency")
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        self.Step(
            "Randomly selected Frequency channel {} @ {:.3f}MHz for testing."
            "".format(test_chan, expected_fc / 1e6)
        )
        requested_test_freqs = self.cam_sensors.calc_freq_samples(self.dhost, 
                test_chan, samples_per_chan=3, chans_around=1)
        # Get baseline 0 data, i.e. auto-corr of m000h
        test_baseline = 0
        chan_responses = []
        scans = []
        initial_max_freq_list = []
        decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
        source_period_in_samples = n_chans * 2 * decimation_factor

        try:
            test_dump = self.get_real_clean_dump(discard = num_discard)
            assert isinstance(test_dump, dict)
        except Exception:
            errmsg = "Could not retrieve clean SPEAD accumulation, as Queue is Empty."
            self.Error(errmsg, exc_info=True)
            return
        else:
            awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
            awgn_scale = 0
            dsim_set_success = self.set_input_levels(
                awgn_scale=awgn_scale, cw_scale=cw_scale, freq=expected_fc, fft_shift=fft_shift, gain=gain
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False
            #else:
            #    curr_mcount = self.current_dsim_mcount() #dump_after_mcount
            self.Step("Digitiser simulator configured to generate continuous wave")
            #TODO: this test does not sweep across the full l-band, should it?
            #self.Step(
            #    "Sweeping the digitiser simulator over the centre frequencies of at "
            #    "least all channels that fall within the complete L-band: {} Hz".format(expected_fc)
            #)
            for scan_i in range(3):
                scan_dumps = []
                frequencies = []
                scans.append(scan_dumps)
                for i, freq in enumerate(requested_test_freqs):
                    if scan_i == 0:
                        Aqf.hop(
                            "Getting reference channel response for freq {} @ {}: {} MHz.".format(
                                i + 1, len(requested_test_freqs), freq / 1e6
                            )
                        )
                        self.dhost.sine_sources.sin_corr.set(
                            frequency=freq, scale=cw_scale, repeat_n=source_period_in_samples
                        )
                        curr_mcount = self.current_dsim_mcount() #dump_after_mcount
                        freq_val = self.dhost.sine_sources.sin_corr.frequency
                        try:
                            # this_freq_dump = self.receiver.get_clean_dump()
                            #TODO Check if 4 discards are enough.
                            #this_freq_dump = self.get_real_clean_dump(discard = num_discard)
                            this_freq_dump = self.get_dump_after_mcount(curr_mcount)  #dump_after_mcount
                            assert isinstance(this_freq_dump, dict)
                        except Exception:
                            errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                            self.Error(errmsg, exc_info=True)
                        else:
                            initial_max_freq = np.max(this_freq_dump["xeng_raw"])
                            this_freq_data = this_freq_dump["xeng_raw"]
                            initial_max_freq_list.append(initial_max_freq)
                    else:
                        Aqf.hop(
                            "Getting comparison {} channel response for freq {} @ {}: {} MHz.".format(
                                scan_i, i + 1, len(requested_test_freqs), freq / 1e6
                            )
                        )
                        self.dhost.sine_sources.sin_corr.set(
                            frequency=freq, scale=cw_scale, repeat_n=source_period_in_samples
                        )
                        freq_val = self.dhost.sine_sources.sin_corr.frequency
                        try:
                            # this_freq_dump = self.receiver.get_clean_dump()
                            this_freq_dump = self.get_real_clean_dump(discard = num_discard)
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

                    msg = ("Frequency scans identical between SPEAD "
                           "accumulations at {:.3f} MHz".format(freq_x / 1e6))
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
                            start_channel=self.start_channel,
                        )

    def _test_restart_consistency(self, instrument, no_channels):
        """
        This test confirms that back-to-back SPEAD accumulations with same frequency input are
        identical/bit-perfect on CBF restart.
        """
        self.Step(self._testMethodDoc)
        threshold = 1.0e1  #
        test_baseline = 0
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
        requested_test_freqs = self.cam_sensors.calc_freq_samples(
                self.dhost, test_chan, samples_per_chan=3, chans_around=1)
        expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        self.Step(
            "Sweeping the digitiser simulator over {:.3f}MHz of the channels that "
            "fall within {} complete L-band".format(np.max(requested_test_freqs) / 1e6, test_chan)
        )

        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
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

    def _test_delay_tracking(self, check_strt_ch=None, check_stop_ch=None, test_delays=None):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Delay tracking"
        heading(msg)
        num_inputs = len(self.cam_sensors.input_labels)
        tst_idx = random.choice(range(1,num_inputs))
        #ref_idx = random.choice(range(0,tst_idx) + range(tst_idx+1, num_inputs))
        setup_data = self._delays_setup(test_source_idx=(tst_idx,0), determine_start_time=False)
        if setup_data:
            delay_load_lead_time = float(self.conf_file['instrument_params']['delay_load_lead_time'])
            int_time = self.cam_sensors.get_value("int_time")
            delay_load_lead_intg = math.ceil(delay_load_lead_time / int_time)
            # katcp_port = self.cam_sensors.get_value('katcp_port')
            no_chans = range(self.n_chans_selected)
            sampling_period = self.cam_sensors.sample_period
            dump_counts = len(test_delays)
            test_delays_ps = map(lambda delay: delay * 1e12, test_delays)
            # num_inputs = len(self.cam_sensors.input_labels)
            delays = [0] * setup_data["num_inputs"]
            self.Step("Delays to be set (iteratively) %s for testing purposes\n" % (test_delays))

            def get_expected_phases():
                expected_phases = []
                for delay in test_delays:
                    phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                    # For Narrowband remove the phase offset (because the center of the band is selected)
                    phases = phases - phases[0]
                    phases -= np.max(phases) / 2.0
                    phases = phases[self.start_channel:self.stop_channel]
                    expected_phases.append(phases)
                return zip(test_delays_ps, expected_phases)

            def get_actual_phases():
                actual_phases_list = []
                raw_data = []
                # chan_responses = []
                for count, delay in enumerate(test_delays, 1):
                    delays[setup_data["test_source_ind"]] = delay
                    delay_coefficients = ["{},0:0,0".format(dv) for dv in delays]
                    try:
                        #errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        #this_freq_dump = self.receiver.get_clean_dump(discard=0)
                        #self.assertIsInstance(this_freq_dump, dict), errmsg
                        #t_apply = this_freq_dump["dump_timestamp"] + (num_int * int_time)
                        #t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
                        curr_time = time.time()
                        curr_time_readable = datetime.fromtimestamp(curr_time).strftime("%H:%M:%S")
                        t_apply = curr_time + delay_load_lead_time
                        t_apply_readable = datetime.fromtimestamp(t_apply).strftime("%H:%M:%S")
                        self.Step("Delay #%s will be applied with the following parameters:" % count)
                        msg = (
                            "On baseline %s and input %s, Current cmc time: %s (%s), "
                            "Delay(s) will be applied @ %s (%s), Delay to be applied: %s"
                            % (
                                setup_data["baseline_index"],
                                setup_data["test_source"],
                                curr_time,
                                curr_time_readable,
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
                        reply, _informs = self.katcp_req.delays(self.corr_fix.feng_product_name, t_apply, *delay_coefficients)
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
                    except Exception as e:
                        self.Error('Error occured during delay tracking test: {}'.format(e), exc_info=True)

                    try:
                        _num_discards = delay_load_lead_intg + 4
                        self.Step(
                            "Getting SPEAD accumulation(while discarding %s dumps) containing "
                            "the change in delay(s) on input: %s baseline: %s."
                            % (_num_discards, setup_data["test_source"], setup_data["baseline_index"])
                        )
                        self.logger.info("Getting dump...")
                        dump = self.receiver.get_clean_dump(discard=_num_discards)
                        if not(self._confirm_delays(delay_coefficients, 
                                                    err_margin = float(self.conf_file["delay_req"]["delay_resolution"]))[0]):
                            self.Error('Requested delay was not set, check output of logfile.')
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
                        raw_data.append(dump)
                        data = complexise(dump["xeng_raw"][:, setup_data["baseline_index"], :])
                        phases = np.angle(data)
                        # # actual_channel_responses = zip(test_delays, chan_responses)
                        # # return zip(actual_phases_list, actual_channel_responses)
                        actual_phases_list.append(phases)
                fn = "/".join([self._katreport_dir, r"delay_tracking_bl_{}.npz".format(setup_data["baseline_index"])])
                with open(fn, 'w') as f:
                    np.savez_compressed(f, raw_data=raw_data)
                return actual_phases_list

            expected_phases = get_expected_phases()
            for i in range(self.data_retries):
                actual_phases = get_actual_phases()
                if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                    self.logger.error("Phases are all zero, retrying capture. TODO debug why this is neccessary.")
                elif not actual_phases:
                    self.logger.error("Phases not captured, retrying capture. TODO debug why this is neccessary.")
                else:
                    break
            try:
                if set([float(0)]) in [set(i) for i in actual_phases[1:]]:
                    self.Failed("Phases are all zero")
                elif not actual_phases:
                    self.Failed("Phases were not captured")
                else:
                    # actual_phases = [phases for phases, response in actual_data]
                    # actual_response = [response for phases, response in actual_data]
                    plot_title = "CBF Delay Compensation"
                    caption = (
                        "Actual and expected Unwrapped Correlation Phase [Delay tracking].\n"
                        "Note: Dashed line indicates expected value and solid line "
                        "indicates measured value."
                    )
                    plot_filename = "{}/{}_test_delay_tracking.png".format(self.logs_path, self._testMethodName)
                    plot_units = "ps/s"
                    if self.start_channel != 0:
                        start_channel = self.start_channel
                    else:
                        start_channel = None
                    aqf_plot_phase_results(
                        no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption, 
                        dump_counts, start_channel=start_channel
                        )

                    nc_sel = self.n_chans_selected
                    expected_phases_ = [phase[:nc_sel] for _rads, phase in expected_phases]

                    degree = 1.0
                    decimal = len(str(degree).split(".")[-1])
                    try:
                        for i, delay in enumerate(test_delays):
                            # This checking does not really make sense:
                            # TODO: But difference between integrations should possibly be checked
                            #delta_actual = np.max(actual_phases[i]) - np.min(actual_phases[i])
                            #delta_expected = np.max(expected_phases_[i]) - np.min(expected_phases_[i])
                            #abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                            ## abs_diff = np.abs(delta_expected - delta_actual)
                            #msg = (
                            #    "Confirm that difference expected({:.5f}) "
                            #    "and actual({:.5f}) phases are equal at delay {:.5f}ns within "
                            #    "{} degree.".format(delta_expected, delta_actual, delay * 1e9, degree)
                            #)
                            #Aqf.almost_equals(delta_expected, delta_actual, degree, msg)

                            #Aqf.less(
                            #    abs_diff,
                            #    degree,
                            #    "Confirm that the maximum difference ({:.3f} degree/"
                            #    " {:.3f} rad) between expected phase and actual phase between "
                            #    "integrations is less than {} degree.\n".format(abs_diff, np.deg2rad(abs_diff), degree),
                            #)
                            delta_phase = actual_phases[i] - expected_phases_[i]
                            # Cut slice to check if check_strt_ch set
                            plot_start_ch = self.start_channel
                            if check_strt_ch and check_stop_ch:
                                delta_phase = delta_phase[check_strt_ch:check_stop_ch]
                                plot_start_ch = check_strt_ch
                            # Replace first value with average as DC component might skew results
                            delta_phase = [np.average(delta_phase)] + delta_phase[1:]
                            max_diff     = np.max(np.abs(delta_phase))
                            max_diff_deg = np.rad2deg(max_diff)

                            Aqf.less(
                                max_diff_deg,
                                degree,
                                "Maximum difference ({:.3f} degrees "
                                "{:.3f} rad) between expected phase "
                                "and actual phase less than {} degree."
                                "".format(max_diff_deg, max_diff, degree),
                            )
                            if i > 0:
                                plot_filename="{}/{}_acc_{}_delay_tracking_error_vector.png".format(
                                    self.logs_path, self._testMethodName, i
                                )
                                caption = ("Offset vector between expected and measured phase (error vector). "
                                           "This plot is generated by subtracting the measured phase from the "
                                           "expected phase for a delay rate of {:1.2e} ns/s".format(delay))
                                aqf_plot_channels(np.rad2deg(delta_phase), plot_filename, caption=caption, 
                                                  log_dynamic_range=None, plot_type="error_vector",
                                                  start_channel=plot_start_ch)
                            #TODO: Perhaps add this back in:
                            #try:
                            #    delta_actual_s = delta_actual - (delta_actual % degree)
                            #    delta_expected_s = delta_expected - (delta_expected % degree)
                            #    np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
                            #except AssertionError:
                            #    msg = (
                            #        "Difference expected({:.5f}) phases"
                            #        " and actual({:.5f}) phases are 'Not almost equal' "
                            #        "within {} degree when delay of {}ns is applied.".format(
                            #            delta_expected, delta_actual, degree, delay * 1e9
                            #        )
                            #    )
                            #    self.Step(msg)

                            #    caption = (
                            #        "The figure above shows, The difference between expected({:.5f}) "
                            #        "phases and actual({:.5f}) phases are 'Not almost equal' within {} "
                            #        "degree when a delay of {:.5f}s is applied. Therefore CBF-REQ-0128 and"
                            #        ", CBF-REQ-0187 are not verified.".format(
                            #            delta_expected, delta_actual, degree, delay
                            #        )
                            #    )

                            #    actual_phases_i = (delta_actual, actual_phases[i])
                            #    if len(expected_phases[i]) == 2:
                            #        expected_phases_i = (delta_expected, expected_phases[i][-1])
                            #    else:
                            #        expected_phases_i = (delta_expected, expected_phases[i])
                            #    aqf_plot_phase_results(
                            #        no_chans,
                            #        actual_phases_i,
                            #        expected_phases_i,
                            #        plot_filename="{}/{}_{}_delay_tracking.png".format(
                            #            self.logs_path, self._testMethodName, i
                            #        ),
                            #        plot_title=("Delay offset:\n" "Actual vs Expected Phase Response"),
                            #        plot_units=plot_units,
                            #        caption=caption,
                            #    )

                        # Cut slice to check if check_strt_ch set
                        strt_idx = 5
                        stop_idx = -5
                        if check_strt_ch and check_stop_ch:
                            strt_idx = check_strt_ch
                            stop_idx = check_stop_ch
                        for delay, count in zip(test_delays[1:], range(1, len(expected_phases))):
                            msg = (
                                "Confirm that when a delay of {} clock "
                                "cycle ({:.5f} ns) is introduced there is a phase change "
                                "of {:.3f} degrees as expected to within {} degree.".format(
                                    (count + 1) * 0.5, delay * 1e9, np.rad2deg(np.pi) * (count + 1) * 0.5, degree
                                )
                            )
                            try:
                                Aqf.array_abs_error(
                                    actual_phases[count][strt_idx:stop_idx], expected_phases_[count][strt_idx:stop_idx], msg, degree
                                )
                            except Exception:
                                Aqf.array_abs_error(
                                    actual_phases[count][start_idx:stop_idx],
                                    expected_phases_[count][start_idx : stop_idx + len(actual_phases[count])],
                                    msg,
                                    degree,
                                )
                    except Exception as e:
                        self.Error("Error occurred: {}".format(e), exc_info=True)
                        return
            except Exception as e:
                self.Error("Error occurred: {}".format(e), exc_info=True)
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
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        cw_freq = ch_list[int(n_chans / 2)]

        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
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
            retries = 4
            while True:
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value(timeout=60)
                except BaseException:
                    reply, informs = self.katcp_req.sensor_value(timeout=60)
                self.assertTrue(reply.reply_ok())
                pfb_status = list(set([i.arguments[-2] for i in informs if "pfb.or0-err-cnt" in i.arguments[2]]))[0]
                if (pfb_status == 'nominal') or (retries == 0):
                    break
                else:
                    retries -= 1
        except Exception:
            msg = "Failed to retrieve sensor values via CAM interface"
            self.Error(msg, exc_info=True)
            return
        else:
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
            for i in range(self.data_retries):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                    self.assertTrue(reply.reply_ok())
                    break
                except AssertionError:
                    time.sleep(20)
                    pass
                #except BaseException:
                #    reply, informs = self.katcp_req.sensor_value()
            self.assertTrue(reply.reply_ok())
        except Exception as e:
            msg = "Failed to retrieve sensor values via CAM interface: {}".format(e)
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

        def get_spead_data(self):
            try:
                self.receiver.get_clean_dump()
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
            else:
                msg = (
                    "Confirm that the SPEAD accumulation is being produced by "
                    "instrument but not verified.")
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

        def get_xeng_status(self, status='warn', sensor='network-reorder', timeout = 360):

            curr_time = time.time()
            timeout_t = curr_time + timeout
            while (time.time() < timeout_t):
                try:
                    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
                    self.assertTrue(reply.reply_ok())
                except Exception:
                    msg = "Failed to retrieve sensor values via CAM interface"
                    self.Error(msg, exc_info=True)
                    return False
                else:
                    x_device_status = list(set([i.arguments[-2] for i in informs 
                        if re.match(r'xhost[0-9]{2}.'+'{}'.format(sensor),
                                i.arguments[2])]))
                    if len(x_device_status) == 1:
                        if x_device_status[0] == status:
                            msg = ("All X-Engines report {} {}."
                                   "".format(sensor, status))
                            Aqf.equals(x_device_status[0], status, msg)
                            return True
                        else:
                            time.sleep(10)
                    else:
                        time.sleep(10)
            Aqf.failed('All X-Engines are not reporting network-reorder {}'.format(status))
            return False
            
            #try:
            #    reply, informs = self.corr_fix.katcp_rct_sensor.req.sensor_value()
            #    self.assertTrue(reply.reply_ok())
            #except Exception:
            #    msg = "Failed to retrieve sensor values via CAM interface"
            #    self.Error(msg, exc_info=True)
            #    return False
            #else:
            #    x_device_status = list(set([i.arguments[-2] for i in informs 
            #        if re.match(r'xhost[0-9]{2}.device-status',i.arguments[2])]))

            #    if len(x_device_status) == 1:
            #        msg = ("Confirm that all X-Engines report device-status {}."
            #               "".format(status))
            #        Aqf.equals(x_device_status[0], status, msg)
            #    else:
            #        Aqf.failed('All X-Engines are not reporting device-status {}'.format(status))

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
                    "Multicast dest addr for %s changed "
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

        # TODO: This code connects directly to the SKARAB using casperfpga.
        # change to use kcpcmd
        self.fhosts, self.xhosts = (self.get_hosts("fhost"), self.get_hosts("xhost"))
        fhosts = self.fhosts
        xhosts = self.xhosts
        fhost = fhosts[random.randrange(len(fhosts))]
        xhost = xhosts[random.randrange(len(xhosts))]
        bitstream = self.corr_fix.corr_config['fengine']['bitstream']
        import casperfpga, logging
        dummy_logger = logging.getLogger()
        dummy_logger.level = logging.ERROR
        try:
            fhost_fpga = casperfpga.CasperFpga(fhost, logger=dummy_logger)
            fhost_fpga.get_system_information(bitstream)
        except Exception as e:
            self.Failed('Could not connect to fhost: {}'.format(e))
            return False
        ip_new = "239.101.2.250"
        self.Step(
            "Randomly selected f-host %s to trigger a link error." % (fhost_fpga.host)
        )
        current_ip = get_host_ip(fhost_fpga)
        if not current_ip:
            self.Failed("Failed to retrieve multicast destination address of %s" % fhost_fpga.host)
        elif current_ip != ip_new:
            self.Passed("Current multicast destination address for %s: %s." % (fhost_fpga.host,
                current_ip))
        else:
            self.Failed("Multicast destination address of %s not as expected." % (fhost_fpga.host))

        # report_lru_status(self, xhost, get_lru_status)
        # This will check if any of the sensors are nominal
        # and then continue with the test using that sensor
        sensors = ('network-reorder.device-status', 'missing-pkts.device-status')
        for i, sensor in enumerate(sensors):
            self.Step('Check xengine {} sensors report NOMINAL.'.format(sensor))
            result = get_xeng_status(self, status='nominal', sensor=sensor)
            if not result:
                if i == len(sensors)-1:
                    aqf.Failed('Sensors are not nominal before test.')
                    return
            else:
                break
        # TODO Why get spead data?
        #get_spead_data(self)
        write_new_ip(fhost_fpga, ip_new, current_ip)
        self.Step('Waiting until {} sensors report WARN.'.format(sensor))
        start_time = time.time()
        result = get_xeng_status(self, status='warn', sensor=sensor)
        if result:
            self.Note('Sensors took {} seconds to change to WARN.'
                    ''.format(int(time.time() - start_time)))
        #get_spead_data(self)
        self.logger.info('Restoring the multicast destination from %s to the original %s' % (
                ip_new, current_ip))
        write_new_ip(fhost_fpga, current_ip, ip_new)
        self.Step('Waiting until {} sensors report NOMINAL.'.format(sensor))
        start_time = time.time()
        result = get_xeng_status(self, status='nominal', sensor=sensor)
        if result:
            self.Note('Sensors took {} seconds to change back to NOMINAL.'
                      ''.format(int(time.time() - start_time)))

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

    def _test_vacc(self, test_chan, acc_time=0.998):
        """Test vector accumulator"""
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        awgn_scale=0
        cw_scale = cw_scale/2
        dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, 
                                                 fft_shift=fft_shift, gain=gain)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        #else:
        #    curr_mcount = self.current_dsim_mcount() #dump_after_mcount
        test_input = self.cam_sensors.input_labels[0]
        eq_scaling = complex(gain)
        acc_times = [acc_time / 2, acc_time]
        # acc_times = [acc_time/2, acc_time, acc_time*2]
        n_chans_selected = self.n_chans_selected
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
        source_period_in_samples = n_chans * 2 * decimation_factor
        center_ch = int(n_chans/2)
        try:
            #TODO: Why is this not a sensor anymore?
            #internal_accumulations = int(self.cam_sensors.get_value("xeng_acc_len"))
            internal_accumulations = int(self.corr_fix.corr_config["xengine"]["xeng_accumulation_len"])
        except Exception:
            self.Error("Failed to retrieve X-engine accumulation length", exc_info=True)
        try:
            initial_dump = self.receiver.get_clean_dump()
            assert isinstance(initial_dump, dict)
        except Exception:
            self.Error("Could not retrieve clean SPEAD accumulation: Queue is Empty.",
                exc_info=True)
            return

        decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
        delta_acc_t = self.cam_sensors.fft_period * internal_accumulations * decimation_factor
        test_acc_lens = [np.ceil(t / delta_acc_t) for t in acc_times]
        test_freq = self.cam_sensors.ch_center_freqs[test_chan]
        self.Step("Selected test input {} and test channel {} @ {:.3f} Mhz"
                  "".format(test_input, test_chan, test_freq/1e6))
        eqs = np.zeros((n_chans), dtype=np.complex)
        eqs[test_chan] = eq_scaling
        self.restore_initial_equalisations()
        try:
            reply, _informs = self.katcp_req.gain(test_input, *list(eqs))
            self.assertTrue(reply.reply_ok())
            Aqf.hop("Gain set to {} with an input cw scale of {}.".format(gain, cw_scale))
        except Exception:
            errmsg = "Gains/Eq could not be set on input %s via CAM interface" % test_input
            self.Error(errmsg, exc_info=True)

        self.Step(
            "Configure Dsim output to be periodic to FFT-length: {}.".format(source_period_in_samples)
        )
        self.Note("Each FFT window will be identical.")

        try:
            # Make dsim output periodic in FFT-length so that each FFT is identical
            self.dhost.sine_sources.sin_corr.set(frequency=test_freq, scale=cw_scale, 
                    repeat_n=source_period_in_samples)
            curr_mcount = self.current_dsim_mcount() #dump_after_mcount
            assert self.dhost.sine_sources.sin_corr.repeat == source_period_in_samples
            time.sleep(1)
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
            quant_check = np.where(np.abs(quantiser_spectrum) > 0)[0]
            if (quant_check.shape[0] == 1):
                Aqf.equals(quant_check[0], test_chan, "Tone found in correct channel.")
            elif (quant_check.shape[0] == 0):
                Aqf.failed("No tone found in quantiser output, test cannot continue.")
                return
            else:
                Aqf.failed("More than one value found in quantiser "
                           "@ channels: {}".format(quant_check))
                return

            # The re-quantiser outputs signed int (8bit), but the snapshot code
            # normalises it using binary 8.7 scaling. Since we want to calculate the
            # output of the vacc which sums integers, denormalise the snapshot
            # output back to ints. Then convert to power
            quant_value = np.abs(quantiser_spectrum[test_chan])
            quant_power = (quant_value*2**7)**2
            self.Note("Quantiser test channel voltage magnitude (represented in binary "
                      "8.7): {:.4f}".format(quant_value))
            self.Note("Converted to power and scaled by 2**7: {}".format(quant_power))

            self.Note("One FFT Window ({} samples) takes {:.3f} micro seconds."
                      "".format(n_chans*2, self.cam_sensors.fft_period*1e6)
            )
            self.Note("After {} internal accumulations one VACC accumulation takes "
                      "{:.3f} ms".format(internal_accumulations, delta_acc_t*1e3)
            )
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
                    no_accs = self.cam_sensors.get_value("baseline_correlation_products_n_accs")
                    exp_no_accs= internal_accumulations * vacc_accumulations
                    Aqf.almost_equals(
                        no_accs,
                        exp_no_accs,
                        300,
                        ("VACC length set to {}, equals an accumulation time of {:.3f}s"
                         .format(no_accs, no_accs * self.cam_sensors.fft_period))
                    )
                    expected_response = quant_power * no_accs
                    try:
                        #dump = self.get_real_clean_dump(discard=2)
                        dump = self.get_dump_after_mcount(curr_mcount) #dump_after_mcount
                        baselines = self.get_baselines_lookup()
                        bl_idx = baselines[test_input,test_input]
                        assert isinstance(dump, dict)
                        #data = (dump["xeng_raw"][:, bl_idx, :])[self.start_channel:self.stop_channel]
                        data = (dump["xeng_raw"][:, bl_idx, :])
                    except Exception:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        actual_response = np.abs(complexise(data)[test_chan-self.start_channel])
                        self.Note('Received channel magnitude: {}'.format(actual_response))
                        # Check that the accumulator response is equal to the expected response
                        msg = ("Quantiser value matches accumulated output after scaling.")
                        tol = 0.0001
                        Aqf.almost_equals(actual_response, expected_response, tol, msg)


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

    def _test_delay_rate(self, check_strt_ch=None, check_stop_ch=None,
                         delay_rates=None,
                         awgn_scale=None, 
                         gain=None):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Delay Rate"
        heading(msg)
        num_inputs = len(self.cam_sensors.input_labels)
        tst_idx = random.choice(range(1,num_inputs))
        #ref_idx = random.choice(range(0,tst_idx) + range(tst_idx+1, num_inputs))
        #for mult in [-0.1, -0.5, -1, -1.5, -2, -2.5, -3]:
        for delay_rate in delay_rates:
            setup_data = self._delays_setup(test_source_idx=(tst_idx,0), determine_start_time=False,
                                            awgn_scale_override=awgn_scale,
                                            gain_override=gain)
            if setup_data:
                dump_counts = 4
                # delay_rate = ((setup_data['sample_period'] / self.cam_sensors.get_value('int_time']) *
                # np.random.rand() * (dump_counts - 3))
                # delay_rate = 3.98195128768e-09
                # _rate = get_delay_bounds(self.corr_fix.correlator).get('min_delay_rate')
                sampling_period = self.cam_sensors.sample_period
                #delay_value = 3 * sampling_period
                delay_value = 0
                phase_offset = 0
                phase_rate = 0
                load_time = setup_data["t_apply"]
                delay_rates = [0] * setup_data["num_inputs"]
                delay_rates[setup_data["test_source_ind"]] = delay_rate
                delay_coefficients = ["{},{}:0,0".format(delay_value,fr) for fr in delay_rates]
                self.Progress(
                    "Delay Rate: %s, Delay Value: %s, Phase Offset: %s, Phase Rate: %s "
                    % (delay_rate, delay_value, phase_offset, phase_rate)
                )

                try:
                    fn = "/".join([self._katreport_dir, r"delay_rate_bl_{}_{}.npz".format(setup_data["baseline_index"], str(delay_rate))])
                    actual_data, raw_captures = self._get_actual_data(setup_data, dump_counts, delay_coefficients, save_filename=fn)
                    actual_phases = [phases for phases, response in actual_data]
                except TypeError:
                    errmsg = "Could not retrieve actual delay rate data. Aborting test"
                    self.Error(errmsg, exc_info=True)
                    return
                else:
                    expected_phases = self._get_expected_data(
                        setup_data, dump_counts, delay_coefficients, actual_phases, save_filename=fn)

                    no_chans = range(self.n_chans_selected)

                    msg = "Observe the change in the phase slope, and confirm the phase change is as expected."
                    self.Step(msg)

                    expected_phases_ = np.asarray([phase for label, phase in expected_phases])
                    actual_phases_   = np.asarray(actual_phases)
                    # Cut slice to check if check_strt_ch set
                    plot_start_ch = self.start_channel
                    if check_strt_ch and check_stop_ch:
                        actual_phases_   = actual_phases_[:,check_strt_ch:check_stop_ch]
                        expected_phases_ = expected_phases_[:,check_strt_ch:check_stop_ch]
                        plot_start_ch = check_strt_ch
                    actual_phases_ = np.unwrap(actual_phases_)
                    expected_phases_ = np.unwrap(expected_phases_)
                    degree = 1.0
                    decimal = len(str(degree).split(".")[-1])
                    for i in range(0, len(expected_phases_)):
                        delta_phase = actual_phases_[i] - expected_phases_[i]
                        # Replace first value with average as DC component might skew results
                        delta_phase = [np.average(delta_phase)] + delta_phase[1:]
                        max_diff     = np.max(np.abs(delta_phase))
                        max_diff_deg = np.rad2deg(max_diff)

                        Aqf.less(
                            max_diff_deg,
                            degree,
                            "Maximum difference ({:.3f} degrees "
                            "{:.3f} rad) between expected phase "
                            "and actual phase less than {} degree."
                            "".format(max_diff_deg, max_diff, degree),
                        )
                        plot_filename="{}/{}_acc_{}_delay_rate_{:1.2e}_error_vector.png".format(
                            self.logs_path, self._testMethodName, i, delay_rate
                        )
                        caption = ("Offset vector between expected and measured phase (error vector). "
                                   "This plot is generated by subtacting the measured phase from the "
                                   "expected phase for accumulation {} with a delay rate of {:1.1f} ps/s".format(i, delay_rate*1e12))
                        aqf_plot_channels(np.rad2deg(delta_phase), plot_filename, caption=caption, log_dynamic_range=None, 
                                          plot_type="error_vector",
                                          start_channel=plot_start_ch)
                        # Old method of checking
                        #delta_expected = np.abs(np.max(expected_phases_[i + 1] - expected_phases_[i]))
                        #delta_actual = np.abs(np.max(actual_phases_[i + 1] - actual_phases_[i]))
                        ## abs_diff = np.rad2deg(np.abs(delta_expected - delta_actual))
                        #abs_diff = np.abs(delta_expected - delta_actual)
                        #msg = (
                        #    "Confirm that if difference (radians) between expected({:.3f}) "
                        #    "phases and actual({:.3f}) phases are 'Almost Equal' "
                        #    "within {} degree when delay rate of {} is applied.".format(
                        #        delta_expected, delta_actual, degree, delay_rate
                        #    )
                        #)
                        #Aqf.almost_equals(delta_expected, delta_actual, radians, msg)

                        #msg = (
                        #    "Confirm that the maximum difference ({:.3f} "
                        #    "degree/{:.3f} rad) between expected phase and actual phase "
                        #    "between integrations is less than {} degree.".format(
                        #        np.rad2deg(abs_diff), abs_diff, degree)
                        #)
                        #Aqf.less(abs_diff, radians, msg)

                        #try:
                        #    abs_error = np.max(actual_phases_[i] - expected_phases_[i])
                        #except ValueError:
                        #    abs_error = np.max(actual_phases_[i] - expected_phases_[i][: len(actual_phases_[i])])
                        #msg = (
                        #    "Confirm that the absolute maximum difference ({:.3f} "
                        #    "degree/{:.3f} rad) between expected phase and actual phase "
                        #    "is less than {} degree.".format(np.rad2deg(abs_error), abs_error, degree)
                        #)
                        #Aqf.less(abs_error, radians, msg)

                        #try:
                        #    delta_actual_s = delta_actual - (delta_actual % degree)
                        #    delta_expected_s = delta_expected - (delta_expected % degree)
                        #    np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

                        #except AssertionError:
                        #    self.Step(
                        #        "Difference  between expected({:.3f}) "
                        #        "phases and actual({:.3f}) phases are "
                        #        "'Not almost equal' within {} degree when delay rate "
                        #        "of {} is applied.".format(delta_expected, delta_actual, degree, delay_rate)
                        #    )
                        #    caption = (
                        #        "Difference expected({:.3f}) and actual({:.3f})"
                        #        " phases are not equal within {} degree when delay rate of {} "
                        #        "is applied.".format(delta_expected, delta_actual, degree, delay_rate)
                        #    )

                        #    actual_phases_i = (delta_actual, actual_phases[i])
                        #    if len(expected_phases[i]) == 2:
                        #        expected_phases_i = (delta_expected, expected_phases[i][-1])
                        #    else:
                        #        expected_phases_i = (delta_expected, expected_phases[i])
                        #    aqf_plot_phase_results(
                        #        no_chans,
                        #        actual_phases_i,
                        #        expected_phases_i,
                        #        plot_filename="{}/{}_{}_delay_rate.png".format(self.logs_path, self._testMethodName, i),
                        #        plot_title="Delay Rate:\nActual vs Expected Phase Response",
                        #        plot_units=plot_units,
                        #        caption=caption,
                        #    )
                    plot_units = "rads"
                    plot_title = "Delay rate {:1.1f} ps/s ".format(
                        delay_rate * 1e12)
                    plot_filename = "{}/{}_delay_rate.png".format(self.logs_path, self._testMethodName)
                    caption = (
                        "Actual vs Expected Unwrapped Correlation Phase [Delay Rate].\n"
                        "Note: Dashed line indicates expected value and solid line indicates "
                        "measured value."
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
                        start_channel=self.start_channel,
                    )
                    mag_plots = []
                    caption = "Response of accumulations to which delays were applied."
                    plot_filename="{}/{}_delay_rate_response.png".format(
                        self.logs_path, self._testMethodName
                    )
                    for dump in raw_captures:
                        mag_plots.append((normalised_magnitude(dump['xeng_raw'][:,setup_data["baseline_index"],:]), None))
                    aqf_plot_channels(mag_plots, plot_filename, log_dynamic_range=90, caption=caption, start_channel=self.start_channel)

    def _test_phase_rate(self, check_strt_ch=None, check_stop_ch=None):
        msg = "CBF Delay and Phase Compensation Functional VR: -- Phase rate"
        heading(msg)
        num_inputs = len(self.cam_sensors.input_labels)
        tst_idx = random.choice(range(1,num_inputs))
        #ref_idx = random.choice(range(0,tst_idx) + range(tst_idx+1, num_inputs))
        setup_data = self._delays_setup(test_source_idx=(tst_idx,0), determine_start_time=False)
        #for i in np.arange(0.1,0.5,0.001):
        #    phase_rate = i
        #    phase_rates = [0] * setup_data["num_inputs"]
        #    phase_rates[setup_data["test_source_ind"]] = phase_rate
        #    delay_coefficients = ["0,0:0,{}".format(fr) for fr in phase_rates]
        #    self._test_coeff(setup_data, delay_coefficients)
        #return

        if setup_data:
            dump_counts = 4
            #_rand_gen = self.cam_sensors.get_value("int_time") * np.random.rand() * dump_counts
            #phase_rate = (np.pi / 8.0) / _rand_gen
            phase_rate = 0.15
            delay_value = 0
            delay_rate = 0
            phase_offset = 0
            load_time = setup_data["t_apply"]
            phase_rates = [0] * setup_data["num_inputs"]
            phase_rates[setup_data["test_source_ind"]] = phase_rate
            delay_coefficients = ["0,0:0,{}".format(fr) for fr in phase_rates]

            self.Progress(
                "Delay Rate: %s seconds/second, Delay Value: %s radians, "
                "Phase Offset: %s radians, Phase Rate: %s radians/second"
                % (delay_rate, delay_value, phase_offset, phase_rate)
            )
            try:
                fn = "/".join([self._katreport_dir, r"phase_rate_bl_{}.npz".format(setup_data["baseline_index"])])
                actual_data, raw_captures = self._get_actual_data(setup_data, dump_counts, delay_coefficients, save_filename=fn)
                actual_phases = [phases for phases, response in actual_data]

            except TypeError:
                errmsg = "Could not retrieve actual delay rate data. Aborting test"
                self.Error(errmsg, exc_info=True)
                return
            else:
                phase_rad_per_sec_req = float(self.conf_file["delay_req"]["phase_rate_resolution"])
                phase_resolution_req = float(self.conf_file["delay_req"]["phase_resolution"])
                phase_rate_err, act_delay_coeff = self._confirm_delays(delay_coefficients, err_margin = phase_rad_per_sec_req)
                Aqf.is_true(phase_rate_err, "Confirm that the phase rate set is within {} radians/second.".format(phase_rad_per_sec_req))
                expected_phases = self._get_expected_data(setup_data, dump_counts, act_delay_coeff, actual_phases)

                no_chans = range(self.n_chans_selected)
                
                # TODO: Phased don't need unwrapping do they?
                #actual_phases_ = np.unwrap(actual_phases)
                #expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                expected_phases_ = np.asarray([phase for label, phase in expected_phases])
                actual_phases_ = np.asarray(actual_phases)
                # Replace channel 0 with channel 1 to avoid dc channel issues
                actual_phases_[:,[0]] = actual_phases_[:,[1]]
                # Cut slice to check if check_strt_ch set
                plot_start_ch = self.start_channel
                if check_strt_ch and check_stop_ch:
                    actual_phases_ = actual_phases_[:,check_strt_ch:check_stop_ch]
                    expected_phases_ = expected_phases_[:,check_strt_ch:check_stop_ch]
                    plot_start_ch = check_strt_ch
                msg = "Observe the change in the phase, and confirm the phase change is as expected."
                self.Step(msg)
                try:
                    #expected_phases_slice = expected_phases_[:,:actual_phases_.shape[1]]
                    phase_err     = actual_phases_ - expected_phases_
                    phase_err_max = np.max(phase_err, axis=1)
                    # This measurement does not really test the step and phase noise give false fails
                    #phase_step_act = np.diff(actual_phases_, axis=0)
                    #phase_step_exp = np.diff(expected_phases_, axis=0)
                    #phase_step_err = phase_step_act - phase_step_exp
                    #phase_err_max      = np.max(phase_err, axis=1)
                    #phase_step_err_max = np.max(phase_step_err, axis=1)
                    phase_step_act = np.average(np.diff(actual_phases_, axis=0), axis=1) 
                    phase_step_exp = np.diff(expected_phases_, axis=0)[:,0]
                    phase_step_err = phase_step_act - phase_step_exp
                    Aqf.step('Check that the phase step per accumulation is within '
                            '{:.3f} deg / {} radians of the expected step ({:.3f} radians) '
                             'at a rate of {:.3f} radians/second.'
                             ''.format(np.rad2deg(phase_resolution_req), phase_resolution_req, 
                                       phase_step_exp[0],
                                       phase_rate))
                    for i, err in enumerate(phase_step_err):
                        msg = ('Expected vs measured phase offset between accumulations '
                               '{} and {} = {:.3f} radians.'.format(i, i+1, err))
                        Aqf.less(np.abs(err), phase_resolution_req, msg)
                    Aqf.step('Check that the absolute phase for each accumulation is within '
                            '{:.3f} deg / {:.3f} rad of the expected phase.'
                             ''.format(np.rad2deg(phase_resolution_req), phase_resolution_req ))
                    for i, err in enumerate(phase_err_max):
                        #msg = ('Accumulation {}: Mean phase: {:.3f}, expected phase: {:.3f}. '
                        #       'Offset: {:.3f} radians.'
                        #       ''.format(i, 
                        #                 actual_phases_[i].mean(), 
                        #                 expected_phases_slice[i].mean(),
                        #                 err))
                        msg = ('Accumulation {} maximum phase offset from '
                               'expected: {:.3f} radians.'
                               ''.format(i, err))
                        Aqf.less(round(np.abs(err),4), phase_resolution_req, msg)

                    phase_offset_values = [x[0] for x in expected_phases_]
                    for i, ph_offset in enumerate(phase_offset_values):
                        plot_filename="{}/{}_{}_phase_rate_error_vector.png".format(
                            self.logs_path, self._testMethodName, i
                        )
                        caption = ("Offset vector between expected and measured phase (error vector). "
                                   "This plot is generated by subtacting the measured phase from the "
                                   "expected phase for phase offset = {:.3f} radians".format(ph_offset))
                        aqf_plot_channels(phase_err[i], plot_filename, caption=caption, log_dynamic_range=None, 
                                          plot_type="error_vector_rad",
                                          start_channel=plot_start_ch)

                except Exception as e:
                    self.Error(e, exc_info=True)

                #for i in range(0, len(expected_phases_) - 1):
                #    try:
                #        delta_expected = np.max(expected_phases_[i + 1] - expected_phases_[i])
                #        delta_actual = np.max(actual_phases_[i + 1] - actual_phases_[i])
                #        delta_diff = delta_actual - delta_expected
                #        abs_diff = np.max(actual_phases_[i] - expected_phases_[i][:actual_phases_[i].size])
                #    except IndexError:
                #        errmsg = "Failed: Index is out of bounds"
                #        self.Error(errmsg, exc_info=True)
                #    else:
                #        #abs_diff = np.abs(delta_expected - delta_actual)
                #        #abs_diff_deg = np.rad2deg(abs_diff)
                #        Aqf.step(
                #            "Maximum phase change between accumulations {} and {} in radians, "
                #            "expected = {:.3f}, measured = {:.3f}."
                #            "".format(i, i+1, delta_expected, delta_actual))
                #        msg = ("Expected vs measured difference = {:.3f} radians "
                #               "which shall be within {} degree ({:.3f} radians)"
                #               "".format(delta_diff, degree, np.deg2rad(degree)))
                #        Aqf.almost_equals(delta_actual, delta_expected, np.deg2rad(degree), msg)

                #        Aqf.step("Accumulation {}: Mean expected phase: ({:.3f} rad), mean measured phase: ({:.3f} rad)."
                #                 "".format(i, expected_phases_[i].mean(), actual_phases_[i].mean()))
                #        msg = ("Confirm maximum difference between expected phase and measured phase ({:.3f} rad) is less than {:.3f} rad."
                #               "".format(abs_diff, np.deg2rad(degree)))
                #        Aqf.less(abs_diff, np.deg2rad(degree), msg)

                        #try:
                        #    delta_actual_s = delta_actual - (delta_actual % degree)
                        #    delta_expected_s = delta_expected - (delta_expected % degree)
                        #    np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)
                        #except AssertionError:
                        #    self.Step(
                        #        "Difference between expected({:.3f}) phases and actual({:.3f}) "
                        #        "phases are 'Not almost equal' within {} degree when phase rate "
                        #        "of {} is applied.".format(delta_expected, delta_actual, degree, phase_rate)
                        #    )

                        #    caption = (
                        #        "Difference expected({:.3f}) and "
                        #        "actual({:.3f}) phases are not equal within {} degree when "
                        #        "phase rate of {} is applied.".format(
                        #            delta_expected, delta_actual, degree, phase_rate
                        #        )
                        #    )

                        #    actual_phases_i = (delta_actual, actual_phases[i])
                        #    if len(expected_phases[i]) == 2:
                        #        expected_phases_i = (delta_expected, expected_phases[i][-1])
                        #    else:
                        #        expected_phases_i = (delta_expected, expected_phases[i])

                        #    aqf_plot_phase_results(
                        #        no_chans,
                        #        actual_phases_i,
                        #        expected_phases_i,
                        #        plot_filename="{}/{}_phase_rate_{}.png".format(
                        #            self.logs_path, self._testMethodName, i
                        #        ),
                        #        plot_title="Phase Rate: Actual vs Expected Phase Response",
                        #        plot_units=plot_units,
                        #        caption=caption,
                        #    )
                plot_units = "rads"
                plot_title = "Phase rate {:1.3f} rad/s".format(phase_rate)
                plot_filename = "{}/{}_phase_rate.png".format(self.logs_path, self._testMethodName)
                caption = (
                    "Actual vs Expected Unwrapped Correlation Phase [Phase Rate].\n"
                    "Note: Dashed line indicates expected value and solid line "
                    "indicates measured value."
                )

                aqf_plot_phase_results(
                    no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption,
                    dump_counts=dump_counts, start_channel=self.start_channel
                )

    def _test_phase_offset(self, check_strt_ch=None, check_stop_ch=None,
                           awgn_scale_override=None, 
                           gain_override=None, 
                           gain_multiplier=None):
        msg = "CBF Delay and Phase Compensation Functional VR: Phase offset"
        heading(msg)
        num_inputs = len(self.cam_sensors.input_labels)
        tst_idx = random.choice(range(1,num_inputs))
        #ref_idx = random.choice(range(0,tst_idx) + range(tst_idx+1, num_inputs))
        setup_data = self._delays_setup(test_source_idx=(tst_idx,0), determine_start_time=False,
                                        awgn_scale_override=awgn_scale_override,
                                        gain_override=gain_override,
                                        gain_multiplier=gain_multiplier)

        if setup_data:
            dump_counts = 1
            # phase_offset = (np.pi / 2.0) * np.random.rand() * dump_counts
            # phase_offset = 1.22796022444
            phase_offset = np.pi*random.random()*[1 if random.random() < 0.5 else -1][0]
            delay_value = 0
            delay_rate = 0
            phase_rate = 0
            load_time = setup_data["t_apply"]
            phase_offsets = [0] * setup_data["num_inputs"]
            phase_offsets[setup_data["test_source_ind"]] = phase_offset
            delay_coefficients = ["0,0:{},0".format(fo) for fo in phase_offsets]

            self.Progress(
                "Delay Rate: %s, Delay Value: %s, Phase Offset: %s, Phase Rate: %s "
                % (delay_rate, delay_value, phase_offset, phase_rate))

            try:
                fn = "/".join([self._katreport_dir, r"phase_offset_bl_{}.npz".format(setup_data["baseline_index"])])
                actual_data, raw_captures = self._get_actual_data(setup_data, dump_counts, delay_coefficients, save_filename=fn)
                actual_phases = [phases for phases, response in actual_data]
                #TODO Get set phase offset value and calculate expected accordingly
            except TypeError:
                self.Error("Could not retrieve actual delay rate data. Aborting test", exc_info=True)
                return
            else:
                expected_phases = self._get_expected_data(setup_data, dump_counts, delay_coefficients, actual_phases)
                no_chans = range(self.n_chans_selected)
                plot_units = "rads"
                plot_title = "Randomly generated phase offset {:.3f} {}".format(phase_offset, plot_units)
                plot_filename = "{}/{}_phase_offset.png".format(self.logs_path, self._testMethodName)
                caption = (
                    "Actual vs Expected Unwrapped Correlation Phase [Phase Offset].\n"
                    "Note: Dashed line indicates expected value and solid line "
                    "indicates measured value. "
                    "Values are rounded off to 3 decimals places"
                )

                # Ignoring first dump because the delays might not be set for full
                # integration - Note this is not done as the apply time should be at the start
                # of integration so first dump is used.
                phase_resolution_req = float(self.conf_file["delay_req"]["phase_resolution"])
                # Not needed for phase
                #actual_phases_ = np.unwrap(actual_phases)
                actual_phases_ = np.asarray(actual_phases)
                # Replace channel 0 with channel 1 to avoid dc channel issues
                actual_phases_[:,[0]] = actual_phases_[:,[1]]
                expected_phases_ = np.unwrap([phase for label, phase in expected_phases])
                if check_strt_ch and check_stop_ch:
                    actual_phases_   = actual_phases_[:,check_strt_ch:check_stop_ch]
                    expected_phases_ = expected_phases_[:,check_strt_ch:check_stop_ch]
                msg = "Observe a step change in the phase, and confirm the phase change is as expected."
                self.Step(msg)
                #for i in range(1, len(expected_phases) - 1):
                delta_expected = (np.max(expected_phases_[0]))
                delta_actual = (np.max(actual_phases_[0]))
                abs_diff = np.abs(delta_expected - delta_actual)
                msg = (
                    "Confirm that the difference between expected ({:.3f} radians)"
                    " phases and actual ({:.3f} radians) phases are 'Almost Equal' "
                    "within {:.3f} radians when phase offset of {:.3f} radians is "
                    "applied.".format(delta_expected, delta_actual, phase_resolution_req, phase_offset)
                )

                Aqf.almost_equals(delta_actual, delta_expected, phase_resolution_req, msg)

                Aqf.less(
                    abs_diff,
                    phase_resolution_req,
                    "Confirm that the maximum difference ("
                    "{:.3f} radians) between expected phase and actual phase "
                    "between integrations is less than {:.3f} radians\n".format(
                        abs_diff, phase_resolution_req
                    ),
                )
                #TODO Seems like supurflous testing, need to look at this code again.
                #import IPython;IPython.embed()
                #try:
                #    delta_actual_s = delta_actual - (delta_actual % degree)
                #    delta_expected_s = delta_expected - (delta_expected % degree)
                #    decimal = len(str(degree).split(".")[-1])
                #    np.testing.assert_almost_equal(delta_actual_s, delta_expected_s, decimal=decimal)

                #except AssertionError:
                #    self.Step(
                #        "Difference between expected({:.5f}) phases "
                #        "and actual({:.5f}) phases are 'Not almost equal' "
                #        "within {} degree when phase offset of {} is applied.".format(
                #            delta_expected, delta_actual, degree, phase_offset
                #        )
                #    )

                #    caption = (
                #        "Difference expected({:.3f}) and actual({:.3f}) "
                #        "phases are not equal within {:.3f} degree when phase offset "
                #        "of {:.3f} {} is applied.".format(
                #            delta_expected, delta_actual, degree, phase_offset, plot_units
                #        )
                #    )

                #    actual_phases_i = (delta_actual, actual_phases[i])
                #    if len(expected_phases[i]) == 2:
                #        expected_phases_i = (delta_expected, expected_phases[i][-1])
                #    else:
                #        expected_phases_i = (delta_expected, expected_phases[i])
                #    aqf_plot_phase_results(
                #        no_chans,
                #        actual_phases_i,
                #        expected_phases_i,
                #        plot_filename="{}/{}_{}_phase_offset.png".format(self.logs_path,
                #            self._testMethodName, i),
                #        plot_title=("Phase Offset:\nActual vs Expected Phase Response"),
                #        plot_units=plot_units,
                #        caption=caption,
                #    )

                #import IPython;IPython.embed()
                aqf_plot_phase_results(
                    no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption, 
                    dump_counts,
                    start_channel=self.start_channel
                )

    def _test_delay_inputs(self, check_strt_ch=None, check_stop_ch=None):
        """
        CBF Delay Compensation/LO Phase stopping polynomial:
        Delay applied to the correct input
        """
        msg = "CBF Delay and Phase Compensation Functional VR: Delays applied to the correct input"
        heading(msg)
        self.Step(
            "The test will sweep through four(4) randomly selected baselines, select and "
            "set a delay value, Confirm if the delay set is as expected. Only a few results will be printed, "
            "all errors will be printed."
        )
        setup_data = self._delays_setup()
        num_inputs = len(self.cam_sensors.input_labels)
        delay_resolution_req = float(self.conf_file["delay_req"]["delay_resolution"])
        phase_resolution_req = float(self.conf_file["delay_req"]["phase_resolution"])
        input_labels = self.cam_sensors.input_labels
        shuffled_labels = input_labels[:]
        random.shuffle(shuffled_labels)
        shuffled_labels = shuffled_labels[-4:]
        delay_load_lead_time = float(self.conf_file['instrument_params']['delay_load_lead_time'])
        #for delayed_input in shuffled_labels:
        label_len = len(input_labels)
        strt_prnt = 2
        stop_prnt = label_len - strt_prnt
        prnt_idx = 0
        for delayed_input in input_labels:
            test_delay_val = random.randrange(self.cam_sensors.sample_period, step=0.83e-10, int=float)
            # test_delay_val = self.cam_sensors.sample_period  # Pi
            expected_phases = self.cam_sensors.ch_center_freqs * 2 * np.pi * test_delay_val
            # For Narrowband remove the phase offset (because the center of the band is selected)
            expected_phases = expected_phases - expected_phases[0]
            expected_phases -= np.max(expected_phases) / 2.0
            expected_phases = expected_phases[self.start_channel:self.stop_channel]
            delays = [0] * num_inputs
            # Get index for input to delay
            test_source_idx = input_labels.index(delayed_input)
            if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                self.Step("Selected input to test: {}".format(delayed_input))
            else:
                self.logger.info("Selected input to test: {}".format(delayed_input))
            delays[test_source_idx] = test_delay_val
            if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                self.Step("Randomly selected delay value ({}) relative to sampling period".format(test_delay_val))
            else:
                self.logger.info("Randomly selected delay value ({}) relative to sampling period".format(test_delay_val))
            delay_coefficients = ["{},0:0,0".format(dv) for dv in delays]
            if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                self.Progress("Delay coefficients: %s" % delay_coefficients)
            else:
                self.logger.info("Delay coefficients: %s" % delay_coefficients)
            try:
                reply, _informs = self.katcp_req.delays(self.corr_fix.feng_product_name, 
                            time.time() + delay_load_lead_time, *delay_coefficients)
                self.assertTrue(reply.reply_ok())
                time.sleep(delay_load_lead_time)
                curr_mcount = self.current_dsim_mcount()
            except Exception as e:
                self.Failed("Error occured: {}".format(e))
                return
            else:
                timeout = 4
                while True:
                    # Get dumps until the que is empty and check that delays have been applied
                    dump = self.receiver.get_clean_dump()
                    dly_applied, act_dly_coeff = self._confirm_delays(delay_coefficients,
                                                                      err_margin = delay_resolution_req)
                    if dly_applied:
                        if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                            self.Passed("Delays where successfully applied on input: {}".format(delayed_input))
                        else:
                            self.logger.info("Delays where successfully applied on input: {}".format(delayed_input))
                        break
                    elif timeout == 0:
                        self.Error("Delays could not be applied to reqested input {}.".format(delayed_input))
                        break
                    else:
                        timeout -= 1
            try:
                if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                    self.Step(
                        "Getting SPEAD accumulation containing "
                        "the change in delay(s) on input: %s." % (test_source_idx)
                    )
                else:
                    self.logger.info(
                        "Getting SPEAD accumulation containing "
                        "the change in delay(s) on input: %s." % (test_source_idx)
                    )
                dump = self.get_dump_after_mcount(curr_mcount)
                #dump = self.receiver.get_clean_dump(discard=9)
            except Exception:
                self.Error("Could not retrieve clean SPEAD accumulation: Queue is Empty.",
                    exc_info=True)
            else:
                sorted_bls = self.get_baselines_lookup(dump, sorted_lookup=True)
                if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                    self.Step("Maximum expected delay: %s" % np.max(expected_phases))
                else:
                    self.logger.info("Maximum expected delay: %s" % np.max(expected_phases))
                for b_line in sorted_bls:
                    b_line_val = b_line[1]
                    b_line_dump = dump["xeng_raw"][:, b_line_val, :]
                    b_line_phase = np.angle(complexise(b_line_dump))
                    b_line_phase_max = np.max(b_line_phase)
                    strt_idx = 5
                    stop_idx = -5
                    if check_strt_ch and check_stop_ch:
                        strt_idx = check_strt_ch
                        stop_idx = check_stop_ch
                    if (delayed_input in b_line[0]) and b_line[0] != (delayed_input, delayed_input):
                        msg = "Confirm baseline(s) {} expected delay.".format(b_line[0])
                        if (prnt_idx < strt_prnt) or (prnt_idx > stop_prnt): 
                            Aqf.array_abs_error(
                                np.abs(b_line_phase[strt_idx:stop_idx]), np.abs(expected_phases[strt_idx:stop_idx]), msg, phase_resolution_req
                            )
                        else:
                            max_diff = np.max(np.abs(np.abs(b_line_phase[strt_idx:stop_idx]) - 
                                np.abs(expected_phases[strt_idx:stop_idx])))
                            if max_diff > phase_resolution_req:
                                Aqf.array_abs_error(
                                    np.abs(b_line_phase[strt_idx:stop_idx]), np.abs(expected_phases[strt_idx:stop_idx]), msg, phase_resolution_req
                                )
                    else:
                        # TODO What should the maximum expeced be here?
                        if b_line_phase_max > phase_resolution_req:
                            desc = (
                                "Checking baseline {}, index: {}, phase offset found, "
                                "maximum error value = {} rads".format(b_line[0], b_line_val, b_line_phase_max)
                            )
                            self.Failed(desc)
            if (prnt_idx == strt_prnt):
                self.Step("\n\nDelay values applied to correct inputs testing continuing...\n")
            prnt_idx += 1



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
                        actual_data, raw_captures = self._get_actual_data(setup_data, dump_counts, delay_coefficients)
                    except TypeError:
                        self.Error("Failed to set the delays/phases", exc_info=True)
                    else:
                        self.Step("Confirm that the %s where successfully set" % _new_name)
                        reply, informs = self.katcp_req.delays("antenna-channelised-voltage", )
                        msg = (
                            "%s where successfully set via CAM interface."
                            "\n\t\t\t    Reply: %s\n\n" % (_new_name, reply,))
                        Aqf.is_true(reply.reply_ok(), msg)


    def _test_delays_control(self):
        delays_cleared = self.clear_all_delays()
        int_time = self.cam_sensors.get_value("int_time")
        self.Step("Disable Delays and/or Phases for all inputs.")
        if not delays_cleared:
            self.Failed("Delays were not completely cleared, data might be corrupted.\n")
        else:
            self.Passed("Confirm that the user can disable Delays and/or Phase changes via CAM interface.")
        delay_load_lead_time = float(self.conf_file['instrument_params']['delay_load_lead_time'])
        num_inputs = len(self.cam_sensors.input_labels)
        no_inputs = [0] * num_inputs
        input_source = self.cam_sensors.input_labels[0]
        no_inputs[0] = self.cam_sensors.sample_period * 2
        delay_coefficients = ["{},0:0,0".format(dv) for dv in no_inputs]
        try:
            self.Step(
                "Request and enable Delays and/or Phases Corrections on input (%s) "
                "via CAM interface." % input_source
            )
            load_strt_time = time.time()
            t_apply = load_strt_time + delay_load_lead_time
            reply_, _informs = self.katcp_req.delays(self.corr_fix.feng_product_name, t_apply, *delay_coefficients)
            load_done_time = time.time()
            msg = "Delay/Phase(s) set via CAM interface reply : %s" % str(reply_)
            self.assertTrue(reply_.reply_ok())
            cmd_load_time = round(load_done_time - load_strt_time, 3)
            Aqf.is_true(reply_.reply_ok(), msg)
            self.Step("Phase/Delay load command took {} seconds".format(cmd_load_time))
            if not(self._confirm_delays(delay_coefficients, 
                   err_margin = float(self.conf_file["delay_req"]["delay_resolution"]))[0]):
                raise Exception
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

        except Exception as e:
            errmsg = (
                "Failed to set delays via CAM interface with load-time: %s, "
                "Delay coefficients: %s" % (t_apply, delay_coefficients,))
            self.Error(errmsg + "Exception = {}".format(e), exc_info=True)
            return
        else:
            cam_max_load_time = self.conf_file["instrument_params"]["cam_max_load_time"]
            msg = "Time it took to load delay/phase(s) %s is less than %ss" % (cmd_load_time,
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
                corr2_version_link = "".join([i for i in corr2_version.split(".") if len(i) == 7])
                corr2_link = "https://github.com/ska-sa/%s/commit/%s" % (corr2_name, corr2_version_link)
            except Exception:
                corr2_link = "https://github.com/ska-sa/%s/commit/%s" % (corr2_name, corr2_version)

            casper_name = casperfpga.__name__
            casper_version = casperfpga.__version__
            casper_pn = "M1200-0055"
            try:
                assert "dev" in casper_version
                casper_version_link = "".join([i for i in casper_version.split(".") if len(i) == 7])
                casper_link = "https://github.com/ska-sa/%s/commit/%s" % (casper_name, casper_version_link)
            except Exception:
                casper_link = "https://github.com/ska-sa/%s/commit/%s" % (casper_name, casper_version)

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
                bitstream_dir = self.corr_fix.corr_config.get("xengine").get("bitstream")
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

        def md5(fname):
            hash_md5 = hashlib.md5()
            try:
                with open(fname, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            except IOError:
                self.Failed("Could not find file: {}".format(fname))
                return ("File not found")

        def test_fpg_compile_parameters(bitstream_path):
            """Verify compile parameters in fpg/bitstream file are correct. Parameters are verified against 
            values in json file. Results are printed in QTR.

            Parameters
            ----------

            bitstream_path : str
                Absolute path of fpg/bitstream file

            """
            with open('compile_parameters.json', 'r') as json_file:
                json_dict = json.load(json_file)
            bitstream_filename = os.path.basename(bitstream_path)
            fpg_param = casperfpga.casperfpga.parse_fpg(bitstream_path)[0]['77777']

            if re.search('s_c', bitstream_filename): #f-engine bitstreams
                try:
                    self.Progress("Compile Parameters of %s:" % (bitstream_filename))
                    fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                    n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs'])
                    
                    fft_stages_fpg = int(fpg_param['fft_stages'])
                    n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])
                    
                    self.Progress("fft_stages = %s (expected: %s)" % (fft_stages_fpg, fft_stages_default))
                    self.Progress("n_bits_xengs = %s (expected: %s)" % (n_bits_xengs_fpg, n_bits_xengs_default))

                    self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                    self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                except AssertionError:
                    self.Failed("FPG runtime compile parameters are incorrect.")
            elif re.search('s_b', bitstream_filename): #x/b-engine bitstreams 
                if re.search('_nb|_54', bitstream_filename): #narrowband x/b-engine bitstreams
                    try:
                        self.Progress("Compile Parameters of %s:" % (bitstream_filename))
                        fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                        n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs']) 
                        n_bits_ants_default = int(json_dict[bitstream_filename]['n_bits_ants'])
                        
                        fft_stages_fpg = int(fpg_param['fft_stages'])
                        n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])
                        n_bits_ants_fpg = int(fpg_param['n_bits_ants'])
                        
                        self.Progress("fft_stages = %s (expected: %s)" % (fft_stages_fpg, fft_stages_default))
                        self.Progress("n_bits_xengs = %s (expected: %s)" % (n_bits_xengs_fpg, n_bits_xengs_default))
                        self.Progress("n_bits_ants = %s (expected: %s)" % (n_bits_ants_fpg, n_bits_ants_default))
                        
                        self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                        self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                        self.assertEqual(n_bits_ants_fpg, n_bits_ants_default, 'n_bits_ants parameter incorrect.')
                    except AssertionError:
                        self.Failed("FPG runtime compile parameters are incorrect.")
                elif not re.search('_nb|_54', bitstream_filename): #wideband x/b-engine bitstreams
                    try:
                        self.Progress("Compile Parameters of %s:" % (bitstream_filename))
                        fft_stages_default = int(json_dict[bitstream_filename]['fft_stages'])
                        n_bits_xengs_default = int(json_dict[bitstream_filename]['n_bits_xengs'])
                        n_bits_ants_default = int(json_dict[bitstream_filename]['n_bits_ants'])
                        n_bits_beams_default = int(json_dict[bitstream_filename]['n_bits_beams'])
                        
                        fft_stages_fpg = int(fpg_param['fft_stages'])
                        n_bits_xengs_fpg = int(fpg_param['n_bits_xengs'])
                        n_bits_ants_fpg = int(fpg_param['n_bits_ants'])
                        n_bits_beams_fpg = int(fpg_param['n_bits_beams'])

                        self.Progress("fft_stages = %s (expected: %s)" % (fft_stages_fpg, fft_stages_default))
                        self.Progress("n_bits_xengs = %s (expected: %s)" % (n_bits_xengs_fpg, n_bits_xengs_default))
                        self.Progress("n_bits_ants = %s (expected: %s)" % (n_bits_ants_fpg, n_bits_ants_default))
                        self.Progress("n_bits_beams = %s (expected: %s)" % (n_bits_beams_fpg, n_bits_beams_default))

                        self.assertEqual(fft_stages_fpg, fft_stages_default, 'fft_stages parameter incorrect.')
                        self.assertEqual(n_bits_xengs_fpg, n_bits_xengs_default, 'n_bits_xengs parameter incorrect.')
                        self.assertEqual(n_bits_ants_fpg, n_bits_ants_default, 'n_bits_ants parameter incorrect.')
                        self.assertEqual(n_bits_beams_fpg, n_bits_beams_default, 'n_bits_beams parameter incorrect.')
                    except AssertionError:
                        self.Failed("FPG runtime compile parameters are incorrect.")

        def get_gateware_info():
            f_bitstream = self.cam_sensors.get_value('fengine_bitstream') 
            x_bitstream = self.cam_sensors.get_value('xengine_bitstream') 
            self.Progress("F-ENGINE (CBF) - M1200-0064:")
            self.Progress("Bitstream filename: {}".format(f_bitstream))
            self.Progress("Bitstream md5sum:   {}".format(md5(f_bitstream)))
            test_fpg_compile_parameters(f_bitstream);
            self.Progress("X/B-ENGINE (CBF) - M1200-0067:")
            self.Progress("Bitstream filename: {}".format(x_bitstream))
            self.Progress("Bitstream md5sum:   {}".format(md5(x_bitstream)))
            test_fpg_compile_parameters(x_bitstream);
            #if feng_parameters and xeng_parameters:
            #    print "All parameters passed!"
            #    sys.exit("FPG runtime compile parameters are incorrect. Halting test.")
            #else:
            #    print "Parameters incorrect!"
            #    sys.exit("FPG runtime compile parameters are incorrect. Halting test.")
            try:
                reply, informs = self.katcp_req.version_list()
                self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Failed("Could not retrieve CBF Gate-ware Version Information")
            else:
                for inform in informs:
                    if [s for s in inform.arguments if "xengine-firmware" in s]:
                        #_hash = inform.arguments[-1].split(" ")
                        #_hash = "".join([i.replace("[", "").replace("]", "") for i in _hash if 40 < len(i) < 42])
                        #self.Progress("%s: %s" % (inform.arguments[0], _hash))
                        self.Progress("X/B-ENGINE (CBF) - M1200-0067:")
                        self.Progress(": ".join(inform.arguments))
                    elif [s for s in inform.arguments if "fengine-firmware" in s]:
                        #_hash = inform.arguments[-1].split(" ")
                        #_hash = "".join([i.replace("[", "").replace("]", "") for i in _hash if 40 < len(i) < 42])
                        #self.Progress("%s: %s" % (inform.arguments[0], _hash))
                        self.Progress("F-ENGINE (CBF) - M1200-0064:")
                        
                        self.Progress(": ".join(inform.arguments))
                    elif [s for s in inform.arguments if "bengine-firmware" in s]:
                        pass
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
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
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
                no_channels = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
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
                bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
                nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
                ants = self.cam_sensors.get_value("n_ants")
                ch_list = self.cam_sensors.ch_center_freqs
                ch_bw = ch_list[1]-ch_list[0]
                substreams = self.cam_sensors.get_value("n_xengs")
            except AssertionError:
                errmsg = "%s" % str(reply).replace("\_", " ")
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                self.Error("Error occurred", exc_info=True)
                return False

            self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
            self.Progress("Number of channels = {}".format(nr_ch))
            self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))

            beam = beams[0]
            try:
                beam_name = beam.replace("-", "_").replace(".", "_")
                beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
                beam_ip = beam_ip.split("+")[0]
                start_beam_ip = beam_ip
                #n_substrms_to_cap_m = int(frac_to_cap*substreams)
#################################################################################################
                if "bc8" in self.instrument:
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
                elif "bc16" in self.instrument:
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
                elif "bc32" in self.instrument:
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
                elif "bc64" in self.instrument:
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
                elif "bc128" in self.instrument:
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
                if "1k" in self.instrument:
                    n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
###################################################################################################
                #start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
                # Algorithm now just pics the center of the band and substreams around that.
                # This may lead to capturing issues. TODO: investigate
                start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
                if start_substream > (substreams - 1):
                    self.logger.warn = (
                        "Starting substream is larger than substreams available: {}. "
                        "Fix in test configuration file".format(substreams)
                    )
                    start_substream = substreams - 1
                if start_substream + n_substrms_to_cap_m > substreams:
                    self.logger.warn = (
                        "Substream start + substreams to process "
                        "is more than substreams available: {}. "
                        "Fix in test configuration file".format(substreams)
                    )
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
            strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
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
            host_ip = "10.97.64.1"
            ntp_offset = ntplib.NTPClient().request(host_ip, version=3).offset
        except ntplib.NTPException:
            host_ip = "katfs.kat.ac.za"
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
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
        nominal_gain = float(gain.split('+')[0])
        # previous 4k gain was 30 which was less than normal 113. Levels should be checked
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
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        if ((("107M32k" in self.instrument) or ("54M32k" in self.instrument) or ("68M32k" in self.instrument) or ("34M32k" in self.instrument))
                and (self.start_channel == 0)):
            check_strt_ch = int(self.conf_file["instrument_params"].get("check_start_channel", 0))
            check_stop_ch = int(self.conf_file["instrument_params"].get("check_stop_channel", 0))
            rand_ch = random.choice(range(n_chans)[check_strt_ch:check_stop_ch])
            rel_ch = rand_ch
        else:
            rand_ch = random.choice(range(self.start_channel, 
                    self.start_channel+self.n_chans_selected))
            rel_ch = rand_ch - self.start_channel
        gain_vector = [nominal_gain] * n_chans
        try:
            reply, informs = self.katcp_req.gain(test_input, nominal_gain)
            self.assertTrue(reply.reply_ok())
        except Exception:
            self.Failed("Gain correction on %s could not be set to %s.: "
                    "KATCP Reply: %s" % (test_input, nominal_gain, reply))
            return False

        _discards = 5
        try:
            initial_dump = self.receiver.get_clean_dump(discard=_discards)
            self.assertIsInstance(initial_dump, dict)
            if not np.any(initial_dump["xeng_raw"]):
                errmsg = "Captured data all zeros. Check DSIM input."
                self.Error(errmsg, exc_info=True)
                return
        except AssertionError:
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
                    "Gain correction on %s could not be set to %s.: " "KATCP Reply: %s" % (test_input, nominal_gain, reply)
                )
                return False
            gain_step = nominal_gain/4
            start_gain = gain_step
            max_chan_found = False
            upper_req_string = ""
            while not found:
                if not fnd_less_one:
                    target = 1
                    gain_step = int(gain_step/2)
                else:
                    target = 6
                    gain_step = int(gain_step*2)
                gain_set  = start_gain + gain_step
                gain_vector[rand_ch] = gain_set
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
                    self.Step(msg)
                    try:
                        dump = self.receiver.get_clean_dump(discard=_discards)
                        self.assertIsInstance(dump, dict)
                    except AssertionError:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                        self.Error(errmsg, exc_info=True)
                    else:
                        response = np.abs(complexise(dump["xeng_raw"][:, auto_corr_idx, :]))
                        response = 10 * np.log10(response)
                        #TODO Is there any value in checking which channel is maximum
                        #if not(max_chan_found):
                        #    max_chan_found = True
                        #    max_chan = np.argmax(response)
                        #    self.Progress("Maximum value found in channel {}".format(max_chan))
                        #elif np.argmax(response) != max_chan:
                        #    self.Error('Maximum value found in channel {} which is not the correct channel'.format(np.argmax(response)))

                        if fnd_less_one:
                            resp_diff = response[rel_ch] - min_response[rel_ch]
                        else:
                            resp_diff = response[rel_ch] - prev_resp[rel_ch]
                        prev_resp = response
                        if abs(resp_diff) < target:
                            msg = (
                                "Output power changed by less than {} dB "
                                "(actual = {:.2f} dB) {}with gain set to "
                                "{}.".format(target, resp_diff, upper_req_string, complex(gain_set))
                            )
                            if fnd_less_one:
                                self.Step(msg)
                            else:
                                self.Passed(msg)
                                fnd_less_one = True
                                min_response = response
                                upper_req_string = "from initial value of {:.2f} at gain {} ".format(
                                                       min_response[rel_ch], complex(gain_set))
                            chan_resp.append(response)
                            legends.append("Gain set to %s" % (complex(gain_set)))
                        elif abs(resp_diff) > target:
                            msg = (
                                "Output power changed by more than {} dB "
                                "(actual = {:.2f} dB) {}with gain set to "
                                "{}.".format(target, resp_diff, upper_req_string, complex(gain_set))
                            )
                            if not(fnd_less_one):
                                self.Step(msg)
                            else:
                                self.Passed(msg)
                                found = True
                            chan_resp.append(response)
                            legends.append("Gain set to %s" % (complex(gain_set)))
                        else:
                            pass
                count += 1
                if count == 20:
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
                    "all remaining channels are set to %s" % (rand_ch, complex(nominal_gain)),
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
            labels = self.cam_sensors.input_labels
            reply, informs = self.katcp_req.capture_list()
            self.assertTrue(reply.reply_ok())
            beams = []
            for msg in informs:
                if 'tied' in msg.arguments[0]:
                    beams.append(msg.arguments[0])
            num_beams_to_test = int(self.conf_file["beamformer"]["num_beams_to_test"])
            beams = random.sample(beams, num_beams_to_test)
            running_instrument = self.corr_fix.get_running_instrument()
            assert running_instrument is not False
            #msg = 'Running instrument currently does not have beamforming capabilities.'
            #assert running_instrument.endswith('4k'), msg
            # Get instrument parameters
            bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
            nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1] - ch_list[0]
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

        self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))

        beam = beams[0]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            # TODO: This is not optimal
            #if "1k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["1k_band_to_capture"])
            #elif "4k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["4k_band_to_capture"])
            #elif "32k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["32k_band_to_capture"])
            #n_substrms_to_cap_m = int(frac_to_cap*substreams)
###################################################
            if "bc8" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
            elif "bc16" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
            elif "bc32" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
            elif "bc64" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
            elif "bc128" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
            if "1k" in self.instrument:
                n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
###################################################
            #start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            # Algorithm now just pics the center of the band and substreams around that.
            # This may lead to capturing issues. TODO: investigate
            start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
            if start_substream > (substreams - 1):
                self.logger.warn = (
                    "Starting substream is larger than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                start_substream = substreams - 1
            if start_substream + n_substrms_to_cap_m > substreams:
                p
                self.logger.warn = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                n_substrms_to_cap_m = substreams - start_substream
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
        strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
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
            try:
                self.Step("Issueing capture start for {}.".format(beam))
                reply, informs = self.katcp_req.capture_start(beam)
                self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
                return False
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
                max_cap_retries=10,
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
                        # Save weights so that if the capture must be re-tried 
                        # weighs does not have to be set again and will be 
                        # available for referece level calculations
                        if retries == 1:
                            beam_pol = beam[-1]
                            saved_wgts = {}
                            if beam_dict:
                                for key in beam_dict:
                                    if key.find(beam_pol) != -1:
                                        saved_wgts[key] = beam_dict[key]

                        bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(beam,
                            beam_dict, ingest_kcp_client) #, capture_time=1)
                        # Set beamdict to None in case the capture needs to be retried.
                        # The beam weights have already been set.
                        beam_dict = None
                        if (len(in_wgts) == 0) and (isinstance(act_wgts, dict)):
                            in_wgts = act_wgts.copy()
                    except Exception as e:
                        self.Failed(
                            "Confirm that the Docker container is running and also confirm the "
                            "igmp version = 2: {}".format(e)
                        )
                        return False

                    data_type = bf_raw.dtype.name
                    # Cut selected partitions out of bf_flags
                    flags = bf_flags[s_substream : s_substream + subs_to_cap]
                    # self.Step('Finding missed heaps for all partitions.')
                    if flags.size == 0:
                        self.logger.warn("Beam data empty. Capture failed. Retrying...")
                    else:
                        missed_err = False
                        for part in flags:
                            missed_heaps = np.where(part > 0)[0]
                            missed_perc = missed_heaps.size / part.size
                            perc = 0.80
                            if missed_perc > perc:
                                self.logger.warn("Missed heap percentage = {}%".format(missed_perc * 100))
                                self.logger.warn("Missed heaps = {}".format(missed_heaps))
                                self.logger.warn(
                                    "Beam capture missed more than %s%% heaps. Retrying..." % (
                                        perc * 100))
                                missed_err = True
                        # Good capture, continue
                        if not missed_err:
                            break
                    time.sleep(5)

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

                labels = ""
                if in_wgts != {}:
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
                    # This happend if a capture was re-tried:
                    if in_wgts == {}: in_wgts = saved_wgts.copy()
                    delta = int(self.corr_fix._test_config_file["beamformer"]["beamweight_error_margin"])
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

                        spikes = np.where(cap_db > expected*0.2)[0]
                        if len(spikes) == 1:
                            msg = "No additional spikes found in sub spectrum."
                            self.logger.info(msg)
                            if local_substream % align_print_modulo == 0:
                                self.Passed(msg)
                        elif len(spikes) == 0:
                            failed = True
                            self.Failed("No CW found in sub spectrum.")
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
            awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
            #TODO different levels for beamforming and delay tests, seems not for 1k
            decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
            if "1k" in self.instrument:
                #keep scal as is
                pass
            elif decimation_factor != 1:
                pass
            else:
                pass
                #awgn_scale = awgn_scale*2

            self.Progress(
                "Digitiser simulator configured to generate Gaussian noise: "
                "Noise scale: {}, eq gain: {}, fft shift: {}".format(awgn_scale, gain, fft_shift)
            )
            dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
                freq=0, fft_shift=fft_shift, gain=gain)
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False
            time.sleep(1)

            ants = self.cam_sensors.get_value("n_ants")
            sampling_period = self.cam_sensors.sample_period
            #delay_coefficients = ["sampling_period:0"]*ants
            delay_coefficients = ["0:0"]*ants
            try:
                strt_time = time.time()
                reply, _informs = self.katcp_req.beam_delays(beam, *delay_coefficients, timeout=130)
                set_time = time.time() - strt_time
                self.assertTrue(reply.reply_ok())
            except Exception:
                self.logger.warn("Failed to set beam delays. \nReply: %s" % str(reply).replace("_", " "),
                    exc_info=True)
            else:
                Aqf.step('Beam: {0}, Time to set: {1:.2f}, Reply: {2}'.format(beam, set_time, reply))

            # Only one antenna gain is set to 1, this will be used as the reference
            # input level
            # Set beamformer quantiser gain for selected beam to 1 quant gain (TODO: was revesed at one point. To check but it should be fine now)
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
            except Exception as e:
                errmsg = "Failed to retrieve beamformer data: {}".format(e)
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
                except Exception as e:
                    errmsg = "Failed to retrieve beamformer data: {}".format(e)
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
            weight = 3.0 / ants
            beam_dict = self.populate_beam_dict( -1, weight, beam_dict)
            try:
                d, l, rl, exp1, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except Exception as e:
                errmsg = "Failed to retrieve beamformer data: {}".format(e)
                self.Failed(errmsg)
                return
            beam_data.append(d)
            beam_lbls.append(l)
            weight = 1.0 / ants
            beam_dict = self.populate_beam_dict( -1, weight, beam_dict)
            try:
                d, l, rl, exp0, nc, act_wgts, dummy = get_beam_data(beam, beam_dict, rl)
            except Exception as e:
                errmsg = "Failed to retrieve beamformer data: {}".format(e)
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
                hlines=[exp1, exp0],
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
            except Exception as e:
                errmsg = "Failed to retrieve beamformer data: {}".format(e)
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
            except Exception as e:
                errmsg = "Failed to retrieve beamformer data: {}".format(e)
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
                hline_strt_idx=0,
            )

            self.Step("Checking beamformer substream alignment by injecting a CW in each substream.")
            self.Step(
                "Stepping through {} substreams and checking that the CW is in the correct "
                "position.".format(substreams)
            )
            # Reset quantiser gain
            bq_gain = self.set_beam_quant_gain(beam, 1)
            awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
            if "1k" in self.instrument:
                #keep scal as is
                pass
            else:
                pass
                #awgn_scale = awgn_scale*2
                cw_scale = cw_scale/2

            # Check if it is a narrow and instrument and if so don't test start and end
            # of band:
            decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
            #if decimation_factor != 1:
                # Lower cw power to prevent adjacent channels showing any signal
                #cw_scale = cw_scale/decimation_factor
            dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
                freq=0, fft_shift=fft_shift, gain=gain
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels or set fft shift or gain.")
                return False

            self.Progress(
                "Digitiser simulator configured to generate a stepping "
                "Constant Wave and Gaussian noise, "
                "CW scale: {}, Noise scale: {}, eq gain: {}, fft shift: {}".format(
                    cw_scale, awgn_scale, gain, fft_shift
                )
            )
            self.Step("This test will take a long time... check log for progress.")
            if decimation_factor != 1:
                nb_notest_factor = 0.15
                notest = int(round(substreams*0.15,0))
                substreams_to_test = range(notest,substreams-notest)

                self.Step(
                    "Narrowband instrument detected, only 5 results will be printed, "
                    "central {} substreams will be tested. "
                    "All errors will be displayed".format(substreams-(notest*2)))
            else:
                substreams_to_test = range(0,substreams)
                self.Step(
                    "Only 5 results will be printed, all {} substreams will be tested. "
                    "All errors will be displayed".format(substreams)
                )
            aligned_failed = False
            for substream in substreams_to_test:
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
                        freq=freq)
                time.sleep(1)
                if not dsim_set_success:
                    self.Failed("Failed to configure digitise simulator levels or set fft shift or gain.")
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
                except Exception as e:
                    errmsg = "Failed to retrieve beamformer data: {}".format(e)
                    self.Failed(errmsg)
                    return False
            if aligned_failed:
                self.Failed("Beamformer substream alignment test failed.")
            else:
                self.Passed("All beamformer substreams correctly aligned.")

            try:
                reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam)
                self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
                return False

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
            # self.assertTrue(reply.reply_ok())
            # labels = reply.arguments[1:]
            labels = self.cam_sensors.input_labels
            beams = ["tied-array-channelised-voltage.0x", "tied-array-channelised-voltage.0y"]
            running_instrument = self.instrument
            assert running_instrument is not False
            self.Step("Discontinue any capturing of %s and %s, if active." % (beams[0], beams[1]))
            reply, informs = self.katcp_req.capture_start(beams[0], timeout=60)
            self.assertTrue(reply.reply_ok())
            reply, informs = self.katcp_req.capture_start(beams[1], timeout=60)
            self.assertTrue(reply.reply_ok())

            # Get instrument parameters
            bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
            nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]-ch_list[0]
            substreams = self.cam_sensors.get_value("n_xengs")
        except AssertionError:
            errmsg = "%s" % str(reply).replace("\_", " ")
            self.Error(errmsg, exc_info=True)
            return False
        except Exception:
            self.Error("Error Occurred", exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))

        beam = beams[beam_idx]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            #if "1k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["1k_band_to_capture"])
            #elif "4k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["4k_band_to_capture"])
            #elif "32k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["32k_band_to_capture"])
            #n_substrms_to_cap_m = int(frac_to_cap*substreams)
##################################################################
            if "bc8" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
            elif "bc16" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
            elif "bc32" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
            elif "bc64" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
            elif "bc128" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
            if "1k" in self.instrument:
                n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
##################################################################
            #start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            # Algorithm now just pics the center of the band and substreams around that.
            # This may lead to capturing issues. TODO: investigate
            start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
            if start_substream > (substreams - 1):
                self.logger.warn = (
                    "Starting substream is larger than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                start_substream = substreams - 1
            if start_substream + n_substrms_to_cap_m > substreams:
                self.logger.warn = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                n_substrms_to_cap_m = substreams - start_substream
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
        strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
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
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        if "1k" in self.instrument:
            #keep scal as is
            pass
        else:
            awgn_scale = awgn_scale*2
        _capture_time = 0.1
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
            bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(beam, beam_dict)#, capture_time=_capture_time)
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
            self.logger.warn("Beam data empty. Capture failed. Retrying...")
            self.Failed("Beam data empty. Capture failed. Retrying...")
        else:
            missed_err = False
            for part in flags:
                missed_heaps = np.where(part > 0)[0]
                missed_perc = missed_heaps.size / part.size
                perc = 0.50
                if missed_perc > perc:
                    self.logger.warn("Missed heap percentage = {}%%".format(missed_perc * 100))
                    self.logger.warn("Missed heaps = {}".format(missed_heaps))
                    self.logger.warn("Beam capture missed more than %s%% heaps. Retrying..." % (perc * 100))
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

            fn = "/".join([self._katreport_dir, r"beamforming_timeseries_data.npz"])
            np.save_compressed(fn, bf_raw=bf_raw)
            # return True
            from bf_time_analysis import analyse_beam_data

            analyse_beam_data(self,
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

    def _test_beam_delay(self, beam_idx=0):
        """

        Parameters
        ----------


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
            labels = self.cam_sensors.input_labels
            reply, informs = self.katcp_req.capture_list()
            self.assertTrue(reply.reply_ok())
            all_beams = []
            for msg in informs:
                if 'tied' in msg.arguments[0]:
                    all_beams.append(msg.arguments[0])
            num_beams = len(all_beams)

            pair_array = []
            for i in range(num_beams):
                for y in range(num_beams):
                    if i == y:
                        pass
                    else:
                        pair_array.append((i,y))
            beam_pairs = []
            for pair in pair_array:
                beam_pairs.append((all_beams[pair[0]],all_beams[pair[1]]))
            num_to_test = int(self.corr_fix._test_config_file["beamformer"]["num_beampairs_to_test"])
            num_to_test = min(num_to_test, len(beam_pairs))
            beam_pairs = random.sample(beam_pairs,num_to_test)

            # Randomise beam pair
            #bm0_idx = random.randint(0, len(all_beams)-1)
            #bm1_idx = random.choice([i for i in range(0,len(all_beams)) if i not in [bm0_idx]])
            #beams = [all_beams[bm0_idx],all_beams[bm1_idx]]
            running_instrument = self.instrument
            assert running_instrument is not False
            sync_time = self.cam_sensors.get_value("sync_time")

            # Get instrument parameters
            bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
            nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]-ch_list[0]
            scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
            reg_size = 32
            reg_size_max = pow(2, reg_size)
            threems_in_200mhz_cnt = 0.03*scale_factor_timestamp/8
            substreams = self.cam_sensors.get_value("n_bengs")

            # Zero delay coefficients before starting test
            delay_coefficients = ["0:0"]*ants
            Aqf.step("Testing beam steering coefficient time to set for all beams.")
            strt_time = time.time()
            for beam in all_beams:
                try:
                    reply, _informs = self.katcp_req.beam_delays(beam, *delay_coefficients, timeout=130)
                    self.assertTrue(reply.reply_ok())
                except Exception:
                    self.Error("Failed to set beam delays. \nReply: %s" % str(reply).replace("_", " "),
                        exc_info=True)
            set_time = time.time() - strt_time
            max_coeff_set_time = float(self.corr_fix._test_config_file["beamformer"]["max_coeff_set_time"])
            Aqf.less(set_time, max_coeff_set_time, 
                    'CBF_REQ_TBD: Time to set steering coefficients for all beams: {:.2f}s'
                    ''.format(set_time))
            Aqf.step("Testing beam steering coefficient time to set for both polarisations of one antenna.")
            one_ant_beams = [x for x in all_beams if x.find('0') != -1]
            strt_time = time.time()
            for beam in one_ant_beams:
                try:
                    reply, _informs = self.katcp_req.beam_delays(beam, *delay_coefficients, timeout=130)
                    self.assertTrue(reply.reply_ok())
                except Exception:
                    self.Error("Failed to set beam delays. \nReply: %s" % str(reply).replace("_", " "),
                        exc_info=True)
            set_time = time.time() - strt_time
            max_coeff_set_time = float(self.corr_fix._test_config_file["beamformer"]["max_coeff_set_time"])
            Aqf.less(set_time, max_coeff_set_time, 
                    'CBF_REQ_TBD: Time to set steering coefficients for both polarisations of one beam: {:.2f}s'
                    ''.format(set_time))

        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False
        except Exception as e:
            errmsg = "Exception: {}".format(e)
            self.Error(errmsg, exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))


        def get_beam_data(beam_local, ingest_kcp_client_local, beamdata_dir, lstrt_substrm):
            try:
                LOGGER.info("Getting latest beam data captured in %s" % beamdata_dir)
                newest_f = max(glob.iglob("%s/*.h5" % beamdata_dir), key=os.path.getctime)
                _timestamp = int(newest_f.split("/")[-1].split(".")[0])
                newest_f_timestamp = time.strftime("%H:%M:%S", time.localtime(_timestamp))
            except ValueError:
                Aqf.failed("Failed to get the latest beamformer data")
                return
            else:
                LOGGER.info(
                    "Reading h5py data file(%s)[%s] and extracting the beam data.\n"
                    % (newest_f, newest_f_timestamp)
                )
                with h5py.File(newest_f, "r") as fin:
                    data = fin["Data"].values()
                    for element in data:
                        if element.name.find("bf_raw") > -1:
                            bf_raw = np.array(element.value)
                        elif element.name.find("timestamps") > -1:
                            bf_ts = np.array(element.value)
                        elif element.name.find("flags") > -1:
                            bf_flags = np.array(element.value)
                os.remove(newest_f)

            flags = bf_flags[lstrt_substrm : lstrt_substrm + n_substrms_to_cap_m]
            # self.Step('Finding missed heaps for all partitions.')
            if flags.size == 0:
                self.logger.warn("Beam data empty. Capture failed.")
                return None, None
            else:
                for part in flags:
                    missed_heaps = np.where(part > 0)[0]
                    missed_perc = missed_heaps.size / part.size
                    perc = 0.50
                    if missed_perc > perc:
                        self.logger.warn("Missed heap percentage = {}%%".format(missed_perc * 100))
                        self.logger.warn("Missed heaps = {}".format(missed_heaps))
                        self.logger.warn("Beam capture missed more than %s%% heaps. Retrying..." % (perc * 100))
                        return None, None
            # Print missed heaps
            idx = lstrt_substrm
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

        def get_dsim_mcount(spectra_ref_mcount):
            # Get the current mcount and shift it to the start of a spectra
            while True:
                dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()["data"]["reg"]
                dsim_loc_msw = self.dhost.registers.local_time_msw.read()["data"]["reg"]
                if not(reg_size_max - dsim_loc_lsw < threems_in_200mhz_cnt):
                    dsim_loc_time = dsim_loc_msw*reg_size_max + dsim_loc_lsw
                    dsim_loc_time = dsim_loc_time * 8
                    # Shift current dsim time to the edge of a spectra
                    dsim_spectra_time = dsim_loc_time - (
                            dsim_loc_time - spectra_ref_mcount) % ticks_between_spectra
                    dsim_spectra_time = dsim_spectra_time/8.
                    if not(dsim_spectra_time).is_integer():
                        self.Failed("Dsim spectra time is not divisible by 8, dsim count has probably shifted, re-start test.")
                        return False
                    return dsim_spectra_time

        def substreams_to_capture(lingest_kcp_client,  lstart_substrm, lsubstrms_to_cap):
            """ Set ingest node capture substreams """
            lbeam = lingest_kcp_client[1]
            lbeam_ip = lingest_kcp_client[2]
            lbeam_ip = int2ip(ip2int(lbeam_ip) + lstart_substrm)
            lbeam_port = lingest_kcp_client[3]
            try:
                self.logger.info(
                    "Setting ingest node to capture beam, substreams: {}, {}+{}:{}".format(
                        lbeam, lbeam_ip, lsubstrms_to_cap - 1, lbeam_port
                    )
                )
                reply, informs = lingest_kcp_client[0].blocking_request(
                    katcp.Message.request(
                        "substreams-to-capture", "{}+{}:{}".format(
                            lbeam_ip, lsubstrms_to_cap - 1, lbeam_port)
                    ),
                    timeout=_timeout,
                )
                self.assertTrue(reply.reply_ok())
            except Exception:
                errmsg = "Failed to issues ingest node substreams-to-capture: {}".format(str(reply))
                self.Error(errmsg, exc_info=True)

        def test_del_ph(delay_array, _exp_phases, delay_test=True):
            delays = [0] * ants
            b0_spectra = []
            b1_spectra = []
            actual_phases = []
            cnt = 0
            for delay in delay_array:
                delays[ref_input] = delay
                if delay_test:
                    delay_coefficients = ["{}:0".format(dv) for dv in delays]
                else:
                    delay_coefficients = ["0:{}".format(dv) for dv in delays]

                strt_time = time.time()
                try:
                    reply, _informs = self.katcp_req.beam_delays(beams[1], *delay_coefficients, timeout=130)
                    curr_mcount = self.current_dsim_mcount()
                    set_time = time.time() - strt_time
                    self.assertTrue(reply.reply_ok())
                except Exception:
                    self.Error("Failed to set beam delays. \nReply: %s" % str(reply).replace("_", " "),
                        exc_info=True)
                Aqf.step('Beam: {0}, Time to set: {1:.2f}, Reply: {2}'.format(beams[1], set_time, reply))
                cap_retries = 5
                substreams = self.cam_sensors.get_value("n_bengs")
                ch_per_substream = self.cam_sensors.get_value(beam_name + "_n_chans_per_substream")
                num_capture_runs = int(substreams/n_substrms_to_cap_m)
                num_cap_runs_mod = substreams % n_substrms_to_cap_m
                if num_cap_runs_mod: 
                    num_capture_runs += 1
                cap_phases = np.asarray([])
                num_prints = 3
                print_cnt = 0
                for cap in range(num_capture_runs):
                    while True:
                        if (cap == num_capture_runs-1) and num_cap_runs_mod:
                            substrms_to_cap = num_cap_runs_mod
                        else:
                            substrms_to_cap = n_substrms_to_cap_m
                        start_substream = cap*n_substrms_to_cap_m
                        substreams_to_capture(ingest_kcp_client[0], start_substream, substrms_to_cap)
                        substreams_to_capture(ingest_kcp_client[1], start_substream, substrms_to_cap)
                        strt_ch = start_substream * ch_per_substream
                        stop_ch = strt_ch + ch_per_substream*substrms_to_cap
                        beam_retries = 5
                        while beam_retries > 0:
                            _ = self.capture_beam_data(beams[0], ingest_kcp_client=ingest_kcp_client[0][0], start_only=True)
                            _ = self.capture_beam_data(beams[1], ingest_kcp_client=ingest_kcp_client[1][0], start_only=True)
                            time.sleep(0.1)
                            reply, informs = ingest_kcp_client[0][0].blocking_request(katcp.Message.request("capture-done"), timeout=1)
                            reply, informs = ingest_kcp_client[1][0].blocking_request(katcp.Message.request("capture-done"), timeout=1)

                            b0_raw, b0_ts = get_beam_data(beams[0],ingest_kcp_client[0][0],"/ramdisk/bm0",start_substream)
                            b1_raw, b1_ts = get_beam_data(beams[1],ingest_kcp_client[1][0],"/ramdisk/bm1",start_substream)

                            if (np.all(b0_raw) is not None and 
                                np.all(b0_ts) is not None and
                                np.all(b1_raw) is not None and 
                                np.all(b1_ts) is not None):

                                try:
                                    if (b1_ts[0] > b0_ts[0]):
                                        b0_idx = int((b1_ts[0]-b0_ts[0])/self.cam_sensors.get_value('n_samples_between_spectra'))
                                    else:
                                        raise IndexError
                                    b1_idx_end = np.where(b0_ts[-1] == b1_ts)[0][0]
                                    if b1_idx_end < spectra_average:
                                        raise IndexError
                                except IndexError: 
                                        self.logger.warn('Did not find the same timestamp in both beams, '
                                            'retrying {} more times...'.format(beam_retries))
                                        beam_retries -= 1
                                else:
                                    beam_retries = -1
                            else:
                                self.logger.warn('Beam capture failed, retrying {} more times...'.format(beam_retries))
                                beam_retries -= 1
                        if beam_retries == 0:
                            self.Failed('Could not capture beam data.')
                            try:
                                # Restore DSIM
                                self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
                                for igt_kcp_clnt in ingest_kcp_client:
                                    igt_kcp_clnt[0].stop()
                            except BaseException:
                                pass
                            self.stop_katsdpingest_docker()
                            return False

                        b0_cplx = []
                        b1_cplx = []
                        for idx in range(spectra_average):
                            b0_cplx.append(complexise(b0_raw[:,b0_idx+idx,:]))
                            b1_cplx.append(complexise(b1_raw[:,idx,:]))
                        b0_cplx = np.asarray(b0_cplx)
                        b1_cplx = np.asarray(b1_cplx)
                        b0_b1_angle = np.angle(b0_cplx * b1_cplx.conjugate())
                        b0_b1_angle = np.average(b0_b1_angle, axis=0)
                        #check that the capture is good:
                        degree = float(self.corr_fix._test_config_file["beamformer"]
                                    ["delay_err_margin_degrees"])
                        delta_phase = np.asarray(_exp_phases[cnt][strt_ch:stop_ch]) - b0_b1_angle[strt_ch:stop_ch]
                        max_diff     = np.max(np.abs(delta_phase[5:]))
                        max_diff_deg = np.rad2deg(max_diff)
                        if (max_diff_deg > degree) and (cap_retries > 0):
                            cap_retries -= 1
                            self.logger.warn('Bad beam data captured, retrying {} more times...'.format(cap_retries))
                        else:
                            if print_cnt < num_prints:
                                print_output = True
                            elif print_cnt == num_prints:
                                Aqf.hop("...")
                                print_output = False
                            elif print_cnt >= (num_capture_runs-num_prints):
                                print_output = True
                            else:
                                print_output = False
                            print_cnt+=1
                            if delay_test:
                                if print_output:
                                    Aqf.hop('Capturing channels {} to {}, capture {} of {} for applied delay of {} ns/s'
                                            ''.format(strt_ch,stop_ch,cap,num_capture_runs,delay))
                                else:
                                    self.logger.info('Capturing channels {} to {}, capture {} of {} for applied delay of {} ns/s'
                                            ''.format(strt_ch,stop_ch,cap,num_capture_runs,delay))
                            else:
                                if print_output:
                                    Aqf.hop('Capturing channels {} to {}, capture {} of {} for applied phase offset of {} radians'
                                            ''.format(strt_ch,stop_ch,cap,num_capture_runs,delay))
                                else:
                                    self.logger.info('Capturing channels {} to {}, capture {} of {} for applied phase offset of {} radians'
                                            ''.format(strt_ch,stop_ch,cap,num_capture_runs,delay))
                            cap_phases = np.concatenate((cap_phases, b0_b1_angle[strt_ch:stop_ch]),axis=0)
                            break

                # Replace first 5 values with average as DC component might skew results
                cap_phases  = np.concatenate(([np.average(cap_phases[5:10])]*5, cap_phases[5:]), axis=0)

                actual_phases.append(cap_phases)
                cnt += 1
            return actual_phases

        for beam_pair_idx, beams in enumerate(beam_pairs):
            self.Step("Start capture on {}.".format(beams))
            try:
                for beam in beams:
                    reply, informs = self.corr_fix.katcp_rct.req.capture_start(beam)
                    self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
                return False
            ingst_nd_p = int(self.corr_fix._test_config_file["beamformer"]["ingest_node_port"])
            ingest_kcp_client = []
            beam_dict = {}
            for idx, beam in enumerate(beams):
                try:
                    beam_name = beam.replace("-", "_").replace(".", "_")
                    beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
                    beam_ip = beam_ip.split("+")[0]
                    start_beam_ip = beam_ip
##########################################################################
#                    if "bc8" in self.instrument:
#                        n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
#                    elif "bc16" in self.instrument:
#                        n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
#                    elif "bc32" in self.instrument:
#                        n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
#                    elif "bc64" in self.instrument:
#                        n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
#                    elif "bc128" in self.instrument:
#                        n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
#                    if "1k" in self.instrument:
#                        n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
##########################################################################
                    n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_delay_test"])
                    start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
                    if start_substream > (substreams - 1):
                        self.logger.warn = (
                            "Starting substream is larger than substreams available: {}. "
                            "Fix in test configuration file".format(substreams)
                        )
                        start_substream = substreams - 1
                    if start_substream + n_substrms_to_cap_m > substreams:
                        self.logger.warn = (
                            "Substream start + substreams to process "
                            "is more than substreams available: {}. "
                            "Fix in test configuration file".format(substreams)
                        )
                        n_substrms_to_cap_m = substreams - start_substream
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
                except Exception as e:
                    errmsg = "Exception: {}".format(e)
                    self.Error(errmsg, exc_info=True)
                    return False
                # Compute the start IP address according to substream start index
                beam_ip = int2ip(ip2int(beam_ip) + start_substream)
                # Compute spectrum parameters
                #strt_ch_idx = start_substream * ch_per_substream
                #strt_ch = strt_ch_idx
                #stop_ch = strt_ch + ch_per_substream*n_substrms_to_cap_m
                #strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
                self.Step("Start a KAT SDP docker ingest node for beam captures")
                if idx == 0:
                    stop_prev_docker = True
                    capture_dir = "/ramdisk/bm0"
                else:
                    stop_prev_docker = False
                    capture_dir = "/ramdisk/bm1"
                docker_status = self.start_katsdpingest_docker(
                    beam_ip,
                    beam_port,
                    n_substrms_to_cap_m,
                    nr_ch,
                    ticks_between_spectra,
                    ch_per_substream,
                    spectra_per_heap,
                    ingst_nd_p+idx,
                    stop_prev_docker,
                    capture_dir,
                 )
                if docker_status:
                    self.Progress(
                        "KAT SDP Ingest Node started. Capturing {} substream/s "
                        "starting at {} for beam {} capturing into {}".format(n_substrms_to_cap_m, beam_ip, beam, capture_dir)
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
                _timeout = 10
                try:
                    import katcp
                    ingst_kcp_client = katcp.BlockingClient(ingst_nd, ingst_nd_p+idx)
                    ingst_kcp_client.setDaemon(True)
                    ingst_kcp_client.start()
                    self.addCleanup(ingst_kcp_client.stop)
                    is_connected = ingst_kcp_client.wait_connected(_timeout)
                    if not is_connected:
                        errmsg = "Could not connect to %s:%s, timed out." % (ingst_nd, ingst_nd_p)
                        ingst_kcp_client.stop()
                        raise RuntimeError(errmsg)
                except Exception:
                    self.Error("Could not connect to katcp client", exc_info=True)
                else:
                    ingest_kcp_client.append((ingst_kcp_client, beam, start_beam_ip, beam_port, n_substrms_to_cap_m))

                beam_quant_gain = 1
                self.Step("Set beamformer quantiser gain for selected beam to {}".format(beam_quant_gain))
                self.set_beam_quant_gain(beam, beam_quant_gain)

                beam_pol = beam[-1]
                for label in labels:
                    if label.find(beam_pol) != -1:
                        beam_dict[label] = 0.0

                ref_input = np.random.randint(ants)
                ref_input = 0
                # Find reference input label
                for key in beam_dict:
                    if int(filter(str.isdigit, key)) == ref_input:
                        ref_input_label = key
                        break
                self.Step("{} used as a randomised reference input for this test".format(ref_input_label))
                weight = 1.0
                beam_dict = self.populate_beam_dict_idx(ref_input, weight, beam_dict)
                _ = self.capture_beam_data(beam, beam_dict=beam_dict, only_update_weights=True)

            # Phase delay test
            sampling_period = self.cam_sensors.sample_period
            ants = self.cam_sensors.get_value("n_ants")
            #if "1k" in self.instrument:
            #
            #test_delays = [0, 2*sampling_period, 2*1.5 * sampling_period, 2*2.2 * sampling_period]
            #else:
            #    test_delays = [0, sampling_period, 1.5 * sampling_period, 2.2 * sampling_period]
            if beam_pair_idx == 0:
                test_delay_res = float(self.corr_fix._test_config_file["beamformer"]["beamsteering_delay_resolution"])
                test_delays    = [0, test_delay_res, test_delay_res*2]
                test_phase_res = float(self.corr_fix._test_config_file["beamformer"]["beamsteering_phase_resolution"])
                test_phases    = [0, test_phase_res, test_phase_res*2]
            else:
                test_delays = self.corr_fix._test_config_file["beamformer"]["beamsteering_test_delays"].split(",")
                test_delays = [float(x)*sampling_period for x in test_delays]
                test_phases = self.corr_fix._test_config_file["beamformer"]["beamsteering_test_phases"].split(",")
                test_phases = [float(x) for x in test_phases]

            test_delays_ns = map(lambda delay: delay * 1e9, test_delays)
            exp_delays = []
            for delay in test_delays:
                dels = self.cam_sensors.ch_center_freqs * 2 * np.pi * delay
                # For Narrowband remove the phase offset (because the center of the band is selected)
                dels = dels - dels[0]
                dels -= np.max(dels) / 2.0
                wrapped_dels = (dels + np.pi) % (2 * np.pi) - np.pi
                exp_delays.append(wrapped_dels)

            expected_delays = zip(test_delays_ns, exp_delays)

            # Zero delay coefficients before starting test
            delay_coefficients = ["0:0"]*ants
            Aqf.step("Setting all beam delays to zero")
            for beam in beams:
                strt_time = time.time()
                try:
                    reply, _informs = self.katcp_req.beam_delays(beam, *delay_coefficients, timeout=130)
                    set_time = time.time() - strt_time
                    self.assertTrue(reply.reply_ok())
                except Exception:
                    self.Error("Failed to set beam delays. \nReply: %s" % str(reply).replace("_", " "),
                        exc_info=True)
            
            awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
            # Reset the dsim and set fft_shifts and gains
            dsim_set_success = self.set_input_levels(
                awgn_scale=awgn_scale*2, cw_scale=cw_scale, freq=0, fft_shift=fft_shift, gain=gain
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False

            spectra_average = 10000
            no_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            #no_chans = stop_ch - strt_ch
            #expected_delays_slice = [(_rads, phase[strt_ch:stop_ch]) for _rads, phase in expected_delays]
            #expected_delays_ = [phase for _rads, phase in expected_delays_slice]
            expected_delays_ = [phase for _rads, phase in expected_delays]
            # Delay test
            self.Step("Testing beam delay application.")
            self.Step("Delays to be set: %s" % (test_delays))
            try:
                #actual_phases = np.asarray(test_del_ph(test_delays, expected_delays_, strt_ch, stop_ch, True))[:,strt_ch:stop_ch]
                actual_phases = np.asarray(test_del_ph(test_delays, expected_delays_, True))
            except IndexError:
                self.Error("Beam data could not be captured. Halting test.", exc_info=True)
                break
            plot_title = ("CBF Beam Steering Delay Application.\nReference beam: {}, Delayed beam: {}"
                "".format(beams[0].split('.')[-1],beams[1].split('.')[-1]))
            caption = (
                "Actual and Expected Correlation Phase [Delay tracking].\n"
                "Note: Dashed line indicates expected value and solid line "
                "indicates measured value."
            )
            plot_filename = "{}/{}_test_beam_steering_delay.png".format(self.logs_path, self._testMethodName)
            plot_units = "ns"
            dump_counts = len(actual_phases)
            aqf_plot_phase_results(
                no_chans, actual_phases, expected_delays, plot_filename, plot_title, plot_units, caption, 
                dump_counts, start_channel=0#strt_ch
                )

            degree = float(self.corr_fix._test_config_file["beamformer"]
                        ["delay_err_margin_degrees"])
            decimal = len(str(degree).split(".")[-1])
            try:
                for i, delay in enumerate(test_delays):
                    delta_phase = actual_phases[i] - expected_delays_[i]
                    # Replace first 5 values as DC component might skew results
                    delta_phase  = np.concatenate(([delta_phase[5]]*5, delta_phase[5:]))
                    max_diff     = np.max(np.abs(delta_phase))
                    max_diff_deg = np.rad2deg(max_diff)

                    if beam_pair_idx == 0:
                        msg = ("CBF-REQ-TBD Confirm that beam delay set resolution is less "
                               "than {:.3f} ps. Maximum difference ({:.3f} degrees {:.3f} rad) "
                               "between expected phase and actual phase less than {} degree/s."
                               "".format(test_delay_res/1.0e-12, max_diff_deg, max_diff, degree))
                    else:
                        msg = ("Maximum difference ({:.3f} degrees "
                               "{:.3f} rad) between expected phase "
                               "and actual phase less than {} degree/s."
                               "".format(max_diff_deg, max_diff, degree))
                        #if max_diff_deg > degree:
                        #import IPython;IPython.embed()

                    Aqf.less(
                        max_diff_deg,
                        degree,
                        msg
                    )
                    if i > 0:
                        plot_filename="{}/{}_acc_{}_beam_steering_delay_error_vector.png".format(
                            self.logs_path, self._testMethodName, i
                        )
                        caption = ("Offset vector between expected and measured phase (error vector). "
                                   "This plot is generated by subtracting the measured phase from the "
                                   "expected phase for a delay of {:1.3}ns".format(delay*1e9))
                        plot_title = ("Measured phase error for delay of {:1.3}ns.\nReference beam: {}, Delayed beam: {}"
                            "".format(delay*1e9,beams[0].split('.')[-1],beams[1].split('.')[-1]))
                        aqf_plot_channels(np.rad2deg(delta_phase), plot_filename, plot_title, caption=caption, 
                                          log_dynamic_range=None, plot_type="error_vector",
                                          start_channel=0)#strt_ch)
                import IPython;IPython.embed()

                for count, (delay, exp_ph) in enumerate(expected_delays):
                    msg = (
                        "Confirm that when a delay of {:.2f} clock "
                        "cycle/s ({:.5f} ns) is introduced the expected phase change is "
                        "within {} degree/s.".format(
                            delay/(sampling_period*1e9), delay, degree
                        )
                    )
                    try:
                        Aqf.array_abs_error(
                            actual_phases[count], exp_ph, msg, degree
                        )
                    except Exception:
                        Aqf.array_abs_error(
                            actual_phases[count],
                            exp_ph,
                            msg,
                            degree,
                        )
            except Exception as e:
                self.Error("Error occurred: {}".format(e), exc_info=True)
                return

            # Phase test
            self.Step("Testing beam phase offset.")
            self.Step("Phase offset to be set: %s" % (test_phases))
            expected_phases = []
            exp_phases = []
            for phase in test_phases:
                expected_phases.append((phase,[phase*-1]*no_chans))
                exp_phases.append([phase*-1]*no_chans)
                
            try:
                #actual_phases = np.asarray(test_del_ph(test_phases, exp_phases, strt_ch, stop_ch, False))[:,strt_ch:stop_ch]
                actual_phases = np.asarray(test_del_ph(test_phases, exp_phases, False))
            except IndexError:
                self.Error("Beam data could not be captured. Halting test.", exc_info=True)
                break
            plot_title = ("CBF Beam Steering Phase Offest Application.\nReference beam: {}, Delayed beam: {}"
                "".format(beams[0].split('.')[-1],beams[1].split('.')[-1]))
            caption = (
                "Actual and Expected Correlation Phase .\n"
                "Note: Dashed line indicates expected value and solid line "
                "indicates measured value."
            )
            plot_filename = "{}/{}_test_beam_steering_phase.png".format(self.logs_path, self._testMethodName)
            plot_units = "rads"
            dump_counts = len(actual_phases)
            aqf_plot_phase_results(
                no_chans, actual_phases, expected_phases, plot_filename, plot_title, plot_units, caption, 
                dump_counts, start_channel=0#strt_ch
                )

            expected_phases_ = [phase for _rads, phase in expected_phases]
            degree = float(self.corr_fix._test_config_file["beamformer"]
                        ["delay_err_margin_degrees"])
            decimal = len(str(degree).split(".")[-1])
            try:
                for i, phase in enumerate(test_phases):
                    delta_phase = actual_phases[i] - expected_phases_[i]
                    # Replace first 5 values as DC component might skew results
                    delta_phase  = np.concatenate(([delta_phase[5]]*5, delta_phase[5:]))
                    max_diff     = np.max(np.abs(delta_phase))
                    max_diff_deg = np.rad2deg(max_diff)

                    if beam_pair_idx == 0:
                        msg = ("CBF-REQ-TBD Confirm that beam phase set resolution is less "
                               "than {:.3f} radians. Maximum difference ({:.3f} degrees {:.3f} rad) "
                               "between expected phase and actual phase less than {} degree/s."
                               "".format(test_phase_res, max_diff_deg, max_diff, degree))
                    else:
                        msg = ("Maximum difference ({:.3f} degrees "
                               "{:.3f} rad) between expected phase "
                               "and actual phase less than {} degree/s."
                               "".format(max_diff_deg, max_diff, degree))
                    if max_diff_deg > degree:
                        import IPython;IPython.embed()

                    Aqf.less(
                        max_diff_deg,
                        degree,
                        msg
                    )
                    if i > 0:
                        plot_filename="{}/{}_acc_{}_beam_steering_phase_error_vector.png".format(
                            self.logs_path, self._testMethodName, i
                        )
                        caption = ("Offset vector between expected and measured phase (error vector). "
                                   "This plot is generated by subtracting the measured phase from the "
                                   "expected phase for a phase offset of {} radians".format(phase))
                        plot_title = ("Measured phase error for phase offset of {} rad.\nReference beam: {}, Delayed beam: {}"
                            "".format(phase,beams[0].split('.')[-1],beams[1].split('.')[-1]))
                        aqf_plot_channels(np.rad2deg(delta_phase), plot_filename, plot_title, caption=caption, 
                                          log_dynamic_range=None, plot_type="error_vector",
                                          start_channel=0)#strt_ch)

                for count, exp_ph in enumerate(expected_phases_):
                    msg = (
                        "Confirm that when a phase offset of {} radians "
                        "is introduced the expected phase change is "
                        "within {} degree/s.".format(test_phases[count], degree)
                    )
                    try:
                        Aqf.array_abs_error(
                            actual_phases[count], exp_ph, msg, degree
                        )
                    except Exception:
                        Aqf.array_abs_error(
                            actual_phases[count],
                            exp_ph,
                            msg,
                            degree,
                        )
            except Exception as e:
                self.Error("Error occurred: {}".format(e), exc_info=True)
                return

            self.Step("Stop capture on {}.".format(beams))
            try:
                for beam in beams:
                    reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam)
                    self.assertTrue(reply.reply_ok())
            except AssertionError:
                self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
                return False
        
            # Close any KAT SDP ingest nodes
            try:
                # Restore DSIM
                self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
                for igt_kcp_clnt in ingest_kcp_client:
                    igt_kcp_clnt[0].stop()
            except BaseException:
                pass
            self.stop_katsdpingest_docker()

    def _test_beam_steering(self, beam_idx=0):
        """

        Parameters
        ----------


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
            self.Step("Start capture on %s and %s." % (beams[0], beams[1]))
            reply, informs = self.corr_fix.katcp_rct.req.capture_start(beams[0])
            self.assertTrue(reply.reply_ok())
            reply, informs = self.corr_fix.katcp_rct.req.capture_start(beams[1])
            self.assertTrue(reply.reply_ok())
            sync_time = self.cam_sensors.get_value("sync_time")

            # Get instrument parameters
            bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
            nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]-ch_list[0]
            scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
            reg_size = 32
            reg_size_max = pow(2, reg_size)
            threems_in_200mhz_cnt = 0.03*scale_factor_timestamp/8
            substreams = self.cam_sensors.get_value("n_bengs")
        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))

        beam = beams[beam_idx]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            #if "1k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["1k_band_to_capture"])
            #elif "4k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["4k_band_to_capture"])
            #elif "32k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["32k_band_to_capture"])
            #n_substrms_to_cap_m = int(frac_to_cap*substreams)
####################################################################
            if "bc8" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
            elif "bc16" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
            elif "bc32" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
            elif "bc64" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
            elif "bc128" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
            if "1k" in self.instrument:
                n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
####################################################################
            #start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            # Algorithm now just pics the center of the band and substreams around that.
            # This may lead to capturing issues. TODO: investigate
            start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
            if start_substream > (substreams - 1):
                self.logger.warn = (
                    "Starting substream is larger than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                start_substream = substreams - 1
            if start_substream + n_substrms_to_cap_m > substreams:
                self.logger.warn = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                n_substrms_to_cap_m = substreams - start_substream
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
        except Exception as e:
            errmsg = "Exception: {}".format(e)
            self.Error(errmsg, exc_info=True)
            return False
        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_ch = strt_ch_idx
        stop_ch = strt_ch + ch_per_substream*n_substrms_to_cap_m
        strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
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

        beam_quant_gain = 1
        self.Step("Set beamformer quantiser gain for selected beam to {}".format(beam_quant_gain))
        self.set_beam_quant_gain(beam, beam_quant_gain)

        beam_dict = {}
        beam_pol = beam[-1]
        for label in labels:
            if label.find(beam_pol) != -1:
                beam_dict[label] = 0.0

        # Currently setting weights is broken
        # self.Progress("Only one antenna gain is set to 1, the reset are set to zero")
        #ref_input = np.random.randint(ants)
        ref_input = 1
        # Find reference input label
        for key in beam_dict:
            if int(filter(str.isdigit, key)) == ref_input:
                ref_input_label = key
                break
        self.Step("{} used as a randomised reference input for this test".format(ref_input_label))
        weight = 1.0
        beam_dict = self.populate_beam_dict_idx(ref_input, weight, beam_dict)

        def get_beam_data():
            try:
                bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(
                    beam, ingest_kcp_client=ingest_kcp_client, stop_only=True
                )
            except Exception as e:
                errmsg = (
                    "Failed to capture beam data: Confirm that Docker container is "
                    "running and also confirm the igmp version = 2. Error message: {} ".format(e)
                )
                self.Error(errmsg, exc_info=True)
                return False

            flags = bf_flags[start_substream : start_substream + n_substrms_to_cap_m]
            # self.Step('Finding missed heaps for all partitions.')
            if flags.size == 0:
                self.logger.warn("Beam data empty. Capture failed.")
                return None, None
            else:
                for part in flags:
                    missed_heaps = np.where(part > 0)[0]
                    missed_perc = missed_heaps.size / part.size
                    perc = 0.50
                    if missed_perc > perc:
                        self.logger.warn("Missed heap percentage = {}%%".format(missed_perc * 100))
                        self.logger.warn("Missed heaps = {}".format(missed_heaps))
                        self.logger.warn("Beam capture missed more than %s%% heaps. Retrying..." % (perc * 100))
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



        # Set a repeating CW pattern and delay the beam by on fft window to confirm the correct delay is applied.
        n_chans = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
        decimation_factor = int(self.cam_sensors.get_value("decimation_factor"))
        source_period_in_samples = n_chans * 2 * decimation_factor
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        cw_scale = cw_scale/8
        # Reset the dsim and set fft_shifts and gains
        dsim_set_success = self.set_input_levels(
            awgn_scale=0, cw_scale=0, freq=0, fft_shift=fft_shift, gain=gain
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        self.Step(
            "Digitiser simulator configured to generate periodic CW "
            "with repeating samples equal to FFT-length {}) in order for each FFT to be "
            "identical.".format(source_period_in_samples)
        )
        # TODO automate channel to fall within beam capture window
        chan = 100
        ch_list = self.cam_sensors.ch_center_freqs
        center_bin_offset = float(self.conf_file["beamformer"]["center_bin_offset"])
        freq_offset = (ch_list[1]-ch_list[0])*center_bin_offset
        freq = ch_list[chan]+freq_offset
        try:
            # Make dsim output periodic in FFT-length so that each FFT is identical
            self.dhost.sine_sources.sin_corr.set(frequency=freq, scale=cw_scale, 
                    repeat_n=source_period_in_samples)
            assert self.dhost.sine_sources.sin_corr.repeat == source_period_in_samples
            this_source_freq = self.dhost.sine_sources.sin_corr.frequency
            # Dump till mcount change
            curr_mcount = self.current_dsim_mcount()
        except AssertionError:
            errmsg = ("Failed to make the DEng output periodic in FFT-length so "
                      "that each FFT is identical, or cw0 does not equal cw1 freq.")
            self.Error(errmsg, exc_info=True)
            return False

        time.sleep(2)

        ants = self.cam_sensors.get_value("n_ants")
        delays = [0] * ants
        phases = [0] * ants
        sampling_period = self.cam_sensors.sample_period

        #delay_jump = 32 * 2
        #delay_jump = 1
        #num_samples = int(source_period_in_samples/delay_jump)
        num_samples = 30
        spectral_mean_values = []
        chan_values = []
        for sample in range(num_samples):
            delay = (sampling_period/5) * sample
            phase = np.pi*0.01 * sample
            #_dummy = [delays]
            #for idx,val in enumerate(delays):
            #    delays[idx] = delay
            delays[2] = delay
            phases[2] = phase
            #delay_coefficients = ["{}:0".format(dv) for dv in delays]
            delay_coefficients = ["0:{}".format(dv) for dv in phases]
            reply, _informs = self.katcp_req.beam_delays(self.corr_fix.beam0_product_name, *delay_coefficients)
            print reply
            beam_retries = 5
            while beam_retries > 0:
                # Start a beam capture, set pulses and capture data 
                _ = self.capture_beam_data(beam, ingest_kcp_client=ingest_kcp_client, capture_time=0.1, start_only=True)
                time.sleep(0.2)
                bf_raw, bf_ts = get_beam_data()
                if np.all(bf_raw) is not None and np.all(bf_ts) is not None:
                    break
                else:
                    self.logger.warn('Beam capture failed, retrying {} more times...'.format(beam_retries))
                    beam_retries -= 1
            if beam_retries == 0:
                self.Failed('Could not capture beam data.')
                try:
                    # Restore DSIM
                    self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
                    if ingest_kcp_client:
                        ingest_kcp_client.stop()
                except BaseException:
                    pass
                self.stop_katsdpingest_docker()
                return False

            spectral_mean_val = np.sum(np.abs(complexise(bf_raw[strt_ch:stop_ch, 100, :]))) / (stop_ch - strt_ch)
            chan_val = bf_raw[chan,100,:]
            spectral_mean_values.append(spectral_mean_val)
            chan_values.append(chan_val)
            self.Step('Mean spectal value for beam capture is {}'.format(spectral_mean_val))
        chan_cplx_values = np.abs(complexise(np.asarray(chan_values)))


#        # Close any KAT SDP ingest nodes
        try:
            # Restore DSIM
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
            if ingest_kcp_client:
                ingest_kcp_client.stop()
        except BaseException:
            pass
        self.stop_katsdpingest_docker()


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
            self.Step("Start capture on %s and %s." % (beams[0], beams[1]))
            reply, informs = self.corr_fix.katcp_rct.req.capture_start(beams[0])
            self.assertTrue(reply.reply_ok())
            reply, informs = self.corr_fix.katcp_rct.req.capture_start(beams[1])
            self.assertTrue(reply.reply_ok())
            sync_time = self.cam_sensors.get_value("sync_time")

            # Get instrument parameters
            bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
            nr_ch = self.cam_sensors.get_value("antenna_channelised_voltage_n_chans")
            ants = self.cam_sensors.get_value("n_ants")
            ch_list = self.cam_sensors.ch_center_freqs
            ch_bw = ch_list[1]-ch_list[0]
            scale_factor_timestamp = self.cam_sensors.get_value("scale_factor_timestamp")
            reg_size = 32
            reg_size_max = pow(2, reg_size)
            threems_in_200mhz_cnt = 0.03*scale_factor_timestamp/8
            substreams = self.cam_sensors.get_value("n_bengs")
        except AssertionError:
            self.Error("Seems like there was an issue executing katcp requests", exc_info=True)
            return False
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)
            return False

        self.Progress("Bandwidth = {}Hz".format(bw * self.dsim_factor))
        self.Progress("Number of channels = {}".format(nr_ch))
        self.Progress("Channel spacing = {}Hz".format(ch_bw * self.dsim_factor))

        beam = beams[beam_idx]
        try:
            beam_name = beam.replace("-", "_").replace(".", "_")
            beam_ip, beam_port = self.cam_sensors.get_value(beam_name + "_destination").split(":")
            beam_ip = beam_ip.split("+")[0]
            start_beam_ip = beam_ip
            #if "1k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["1k_band_to_capture"])
            #elif "4k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["4k_band_to_capture"])
            #elif "32k" in self.instrument:
            #    frac_to_cap = float(self.conf_file["beamformer"]["32k_band_to_capture"])
            #n_substrms_to_cap_m = int(frac_to_cap*substreams)
#####################################################################
            if "bc8" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_4ant"])
            elif "bc16" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_8ant"])
            elif "bc32" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_16ant"])
            elif "bc64" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_32ant"])
            elif "bc128" in self.instrument:
                n_substrms_to_cap_m = int(self.conf_file["beamformer"]["substreams_to_cap_64ant"])
            if "1k" in self.instrument:
                n_substrms_to_cap_m = int(n_substrms_to_cap_m/2)
#####################################################################
            #start_substream = int(self.conf_file["beamformer"]["start_substream_idx"])
            # Algorithm now just pics the center of the band and substreams around that.
            # This may lead to capturing issues. TODO: investigate
            #start_substream = int(substreams/2) - int(n_substrms_to_cap_m/2)
            start_substream = 0
            if start_substream > (substreams - 1):
                self.logger.warn = (
                    "Starting substream is larger than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                start_substream = substreams - 1
            if start_substream + n_substrms_to_cap_m > substreams:
                self.logger.warn = (
                    "Substream start + substreams to process "
                    "is more than substreams available: {}. "
                    "Fix in test configuration file".format(substreams)
                )
                n_substrms_to_cap_m = substreams - start_substream
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
        except Exception as e:
            errmsg = "Exception: {}".format(e)
            self.Error(errmsg, exc_info=True)
            return False
        # Compute the start IP address according to substream start index
        beam_ip = int2ip(ip2int(beam_ip) + start_substream)
        # Compute spectrum parameters
        strt_ch_idx = start_substream * ch_per_substream
        strt_ch = strt_ch_idx
        stop_ch = strt_ch + ch_per_substream*n_substrms_to_cap_m
        strt_freq = ch_list[strt_ch_idx] * self.dsim_factor
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

        beam_quant_gain = 1
        self.Step("Set beamformer quantiser gain for selected beam to {}".format(beam_quant_gain))
        self.set_beam_quant_gain(beam, beam_quant_gain)

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

        def get_beam_data():
            try:
                bf_raw, bf_flags, bf_ts, in_wgts = self.capture_beam_data(
                    beam, ingest_kcp_client=ingest_kcp_client, stop_only=True
                )
            except Exception as e:
                errmsg = (
                    "Failed to capture beam data: Confirm that Docker container is "
                    "running and also confirm the igmp version = 2. Error message: {} ".format(e)
                )
                self.Error(errmsg, exc_info=True)
                return False

            flags = bf_flags[start_substream : start_substream + n_substrms_to_cap_m]
            # self.Step('Finding missed heaps for all partitions.')
            if flags.size == 0:
                self.logger.warn("Beam data empty. Capture failed.")
                return None, None
            else:
                for part in flags:
                    missed_heaps = np.where(part > 0)[0]
                    missed_perc = missed_heaps.size / part.size
                    perc = 0.50
                    if missed_perc > perc:
                        self.logger.warn("Missed heap percentage = {}%%".format(missed_perc * 100))
                        self.logger.warn("Missed heaps = {}".format(missed_heaps))
                        self.logger.warn("Beam capture missed more than %s%% heaps. Retrying..." % (perc * 100))
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

        def load_dsim_impulse(load_timestamp):
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=2)
            self.dhost.registers.src_sel_cntrl.write(src_sel_1=0)
            self.dhost.registers.impulse_delay_correction.write(reg=16)
            load_timestamp = load_timestamp
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

            #load_timestamp = load_timestamp / 8.0
            #if not load_timestamp.is_integer():
            #    self.Failed("Timestamp received in accumulation not divisible" " by 8: {:.15f}".format(
            #        load_timestamp))
            load_timestamp = int(load_timestamp)
            load_ts_lsw = load_timestamp & (reg_size_max-1)
            load_ts_msw = load_timestamp >> reg_size

            # dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()['data']['reg']
            # dsim_loc_msw = self.dhost.registers.local_time_msw.read()['data']['reg']
            # dsim_loc_time = dsim_loc_msw * pow(2,reg_size) + dsim_loc_lsw
            # print 'timestamp difference: {}'.format((load_timestamp - dsim_loc_time)*8/dump['scale_factor_timestamp'])
            self.dhost.registers.impulse_load_time_lsw.write(reg=load_ts_lsw)
            self.dhost.registers.impulse_load_time_msw.write(reg=load_ts_msw)

        def get_dsim_mcount(spectra_ref_mcount):
            # Get the current mcount and shift it to the start of a spectra
            while True:
                dsim_loc_lsw = self.dhost.registers.local_time_lsw.read()["data"]["reg"]
                dsim_loc_msw = self.dhost.registers.local_time_msw.read()["data"]["reg"]
                if not(reg_size_max - dsim_loc_lsw < threems_in_200mhz_cnt):
                    dsim_loc_time = dsim_loc_msw*reg_size_max + dsim_loc_lsw
                    dsim_loc_time = dsim_loc_time * 8
                    # Shift current dsim time to the edge of a spectra
                    dsim_spectra_time = dsim_loc_time - (
                            dsim_loc_time - spectra_ref_mcount) % ticks_between_spectra
                    dsim_spectra_time = dsim_spectra_time/8.
                    if not(dsim_spectra_time).is_integer():
                        self.Failed("Dsim spectra time is not divisible by 8, dsim count has probably shifted, re-start test.")
                        return False
                    return dsim_spectra_time

        dsim_set_success = self.set_input_levels(awgn_scale=0.0, cw_scale=0.0, freq=0,
            fft_shift=0, gain="65535+0j")
        self.dhost.outputs.out_1.scale_output(0)
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        time.sleep(1)

        num_pulse_caps = 100
        # pulse_step must be divisible by 8. Not neccessary anymore?
        if "1k" in self.instrument:
            pulse_step = 8
        elif "4k" in self.instrument:
            pulse_step = 4*8
        elif "32k" in self.instrument:
            #pulse_step = 16*8
            pulse_step = 32*8
        #TODO Figure out betterway to find load lead time
        #load_lead_time = 0.035
        #load_lead_time = 0.03
        load_lead_time = 0.015
        points_around_trg = 1500
        #points_around_trg = 500
        load_lead_mcount = ticks_between_spectra * int(load_lead_time * scale_factor_timestamp / ticks_between_spectra)
        load_lead_ts     = load_lead_mcount/8.
        if not load_lead_ts.is_integer():
            self.Failed("Load lead timestamp is not divisible by 8. Check ticks_between_spectra")
        beam_retries = 5
        while beam_retries > 0:
            # Get reference beam capture to determine timestamp boundaries.
            bf_raw, bf_flags, bf_ts, _in_wgts = self.capture_beam_data(
                        beam, beam_dict=beam_dict, ingest_kcp_client=ingest_kcp_client
                    )
            try:
                spectra_ref_mcount = bf_ts[-1]
            except IndexError:
                beam_retries -= 1
                self.logger.warn('Beam capture failed, retrying {} more times...'.format(beam_retries))
            else:
                if not (spectra_ref_mcount / 8.0).is_integer():
                    self.Failed("Spectra reference mcount is not divisible" " by 8: {:.15f}".format(
                                spectra_ref_mcount))

                future_ts_array = []
                # Start a beam capture, set pulses and capture data 
                _ = self.capture_beam_data(beam, ingest_kcp_client=ingest_kcp_client, start_only=True)
                for pulse_cap in range(num_pulse_caps):
                    if pulse_cap == 0:
                        curr_ts = get_dsim_mcount(spectra_ref_mcount)
                    else:
                        while curr_ts < future_ts:
                            curr_ts = get_dsim_mcount(spectra_ref_mcount)
                    future_ts = load_lead_ts + curr_ts + pulse_step*pulse_cap
                    future_ts_array.append(future_ts)
                    load_dsim_impulse(future_ts)
                time.sleep(1)
                bf_raw, bf_ts = get_beam_data()
                if np.all(bf_raw) is not None and np.all(bf_ts) is not None:
                    break
                else:
                    self.logger.warn('Beam capture failed, retrying {} more times...'.format(beam_retries))
                    beam_retries -= 1
        if beam_retries == 0:
            self.Failed('Could not capture beam data.')
            try:
                # Restore DSIM
                self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
                if ingest_kcp_client:
                    ingest_kcp_client.stop()
            except BaseException:
                pass
            self.stop_katsdpingest_docker()
            return False

        trgt_spectra_idx = []
        for ts in future_ts_array:
            try:
                trgt_spectra_idx.append(np.where(bf_ts > ts*8)[0][0] - 1)
            except IndexError:
                Aqf.hop('Target spectra not found for timestamp: {}'.format(ts))
        # Check all timestamps makes sense
        ts_steps_found = [bf_ts[trgt_spectra_idx[x]]/8 - future_ts_array[x] for x in range(len(trgt_spectra_idx))]
        #import IPython;IPython.embed()
        if False in set(np.equal(np.diff(ts_steps_found), -1*pulse_step)):
            self.logger.warn("Timestamps steps do not match those requested: {}".format(np.diff(ts_steps_found)))
        if False in set(np.greater(np.diff(trgt_spectra_idx), points_around_trg)):
            self.Failed("Not enough spectra around target to find response: {}".format(np.diff(trgt_spectra_idx)))

        out_func = []
        for i, trgt_spectra in enumerate(trgt_spectra_idx):
            for j in range(trgt_spectra - points_around_trg, trgt_spectra + 1):
                spectra_mean_val = np.sum(np.abs(complexise(bf_raw[strt_ch:stop_ch, j, :]))) / (stop_ch - strt_ch)
                spectra_ts = bf_ts[j]
                ts_delta = int(spectra_ts) - future_ts_array[i]*8
                out_func.append([ts_delta, spectra_mean_val])

        x = np.asarray([x[0] for x in out_func])
        y = np.asarray([y[1] for y in out_func])
        # Remove zeros
        zero_idxs = np.where(y==0)
        x = np.delete(x, zero_idxs, axis=0)
        y = np.delete(y, zero_idxs, axis=0)
        if y.size == 0:
            self.Failed('Could not find pulse within {} spectra'.format(points_around_trg))
            # Close any KAT SDP ingest nodes
            try:
                # Restore DSIM
                self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
                if ingest_kcp_client:
                    ingest_kcp_client.stop()
            except BaseException:
                pass
            self.stop_katsdpingest_docker()
            return False
        # Sort dataset along x (sample count) axis
        sorted_idxs = x.argsort()
        sorted_x = x[sorted_idxs]
        sorted_y = y[sorted_idxs]
        group_delay_raw = sorted_x[np.argmax(sorted_y)]
        gd_sign = np.sign(group_delay_raw)
        group_delay_raw = abs(group_delay_raw)
        gd_exp = int(round(np.log(group_delay_raw)/np.log(2)))
        group_delay = gd_sign*2**gd_exp
        self.Passed('Group delay = {} samples'.format(int(group_delay)))
        aqf_plot_xy(
            ([sorted_x,sorted_y],[""]),
            plot_filename="{}/{}_group_delay.png".format(self.logs_path, self._testMethodName),
            plot_title="Group delay response of an impulse convoluted with a beam spectral window.",
            caption=("An impulse is incremented by {} samples for {} steps in a target beam spectrum. "
                     "{} channels are captured and averaged of beam spectral responses surrounding the "
                     "impulse target spectrum. The impulse timestamp is subtracted from the sample "
                     "timestamp of the averaged beam spectrum. This timestamp delta plotted along the "
                     "x-axis, and the averaged beam spectrum value along the y-axis.".format(
                         pulse_step, num_pulse_caps, stop_ch-strt_ch)),
            vlines=[group_delay],
            xlabel="Sample Offset from Impulse",
            ylabel="Average Beam Response",
        )
#
#            # Pass data through a smoothing filter
#            #import scipy
#            #smooth_y = scipy.signal.savgol_filter(sorted_y, 101, 3)
#
#        # Close any KAT SDP ingest nodes
        try:
            # Restore DSIM
            self.dhost.registers.src_sel_cntrl.write(src_sel_0=0)
            if ingest_kcp_client:
                ingest_kcp_client.stop()
        except BaseException:
            pass
        self.stop_katsdpingest_docker()

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
        bw = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
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
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
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
        dump_ticks = self.cam_sensors.get_values("int_time") * self.cam_sensors.get_value("adc_sample_rate")
        # print dump_ticks
        dump_ticks = self.cam_sensors.get_values("n_accs") * self.cam_sensors.get_value("antenna_channelised_voltage_n_chans") * 2
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

    def _test_input_levels(self):
        """Testing Digitiser simulator input levels
        Set input levels to requested values and check that the ADC and the
        quantiser block do not see saturated samples.
        """
        if self.set_instrument():
            self.Step("Setting and checking Digitiser simulator input levels")
            self._set_input_levels_and_gain(
                profile="cw", cw_freq=100000, cw_margin=0.3, trgt_bits=4, trgt_q_std=0.30, fft_shift=8191
            )

    def _set_input_levels_and_gain(
        self, profile="noise", cw_freq=0, cw_src=0, cw_margin=0.05, trgt_bits=3.5, trgt_q_std=0.30, fft_shift=511
    ):
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
                    num_sat   : number of qdelauantiser snapshot saturated samples
                    cw_freq   : actual returned cw frequency

        """

        # helper functions
        def set_sine_source(scale, cw_freq, cw_src):
            # if cw_src == 0:
            self.dhost.sine_sources.sin_0.set(frequency=cw_freq, scale=round(scale, 3))
            #    return self.dhost.sine_sources.sin_0.frequency
            # else:
            self.dhost.sine_sources.sin_1.set(frequency=cw_freq, scale=round(scale, 3))
            return self.dhost.sine_sources.sin_1.frequency

        def adc_snapshot(source):
            try:
                reply, informs = self.katcp_req.adc_snapshot(source)
                self.assertTrue(reply.reply_ok())
                adc_data = evaluate(informs[0].arguments[1])
                assert len(adc_data) == 8192
                return adc_data
            except AssertionError:
                errmsg = "Failed to get adc snapshot for input {}, reply = {}.".format(source, reply)
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                errmsg = "Exception"
                self.Error(errmsg, exc_info=True)
                return False

        def quant_snapshot(source):
            try:
                reply, informs = self.katcp_req.quantiser_snapshot(source)
                self.assertTrue(reply.reply_ok())
                quant_data = evaluate(informs[0].arguments[1])
                assert len(quant_data) == 4096
                return quant_data
            except AssertionError:
                errmsg = "Failed to get quantiser snapshot for input {}, reply = {}.".format(source,
                    reply)
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                errmsg = "Exception"
                self.Error(errmsg, exc_info=True)
                return False

        def set_gain(source, gain_str):
            try:
                reply, informs = self.katcp_req.gain(source, gain_str)
                self.assertTrue(reply.reply_ok())
                #assert reply.arguments[1:][0] == gain_str
            except AssertionError:
                errmsg = "Failed to set gain for input {}, reply = {}".format(source, reply)
                self.Error(errmsg, exc_info=True)
                return False
            except Exception:
                errmsg = "Exception"
                self.Error(errmsg, exc_info=True)
                return False

        # main code
        self.Step("Requesting input labels.")
        try:
            katcp_rct = self.corr_fix.katcp_rct.sensors
            input_labels = evaluate(katcp_rct.input_labelling.get_value())
            assert isinstance(input_labels, list)
            inp_labels = [x[0] for x in input_labels]
        except AssertionError:
            self.Error("Failed to get input labels.", exc_info=True)
            return False
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)
            return False

        # Set digitiser input level of one random input,
        # store values from other inputs for checking
        inp = random.choice(inp_labels)
        ret_dict = dict.fromkeys(inp_labels, {})
        scale = 0.1
        margin = 0.005
        self.dhost.noise_sources.noise_corr.set(scale=round(scale, 3))
        # Get target standard deviation. ADC is represented by Q10.9
        # signed fixed point.
        target_std = pow(2.0, trgt_bits) / 512
        found = False
        count = 1
        self.Step("Setting input noise level to toggle {} bits at " "standard deviation.".format(trgt_bits))
        while not found:
            self.Step("Capturing ADC Snapshot {} for input {}.".format(count, inp))
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
        self.Step(
            "Digitiser simulator noise scale set to {:.3f}, toggling {:.2f} bits at "
            "standard deviation.".format(noise_scale, p_bits)
        )

        if profile == "cw":
            self.Step("Setting CW scale to {} below saturation point." "".format(cw_margin))
            # Find closest center frequency to requested value to ensure
            # correct quantiser gain is set. Requested frequency will be set
            # at the end.

            # reply, informs = self.corr_fix.katcp_rct. \
            #    req.quantiser_snapshot(inp)
            # data = [evaluate(v) for v in (reply.arguments[2:])]
            # nr_ch = len(data)
            # ch_bw = bw / nr_ch
            # ch_list = np.linspace(0, bw, nr_ch, endpoint=False)

            # bw = self.cam_sensors.get_value('bandwidth')
            # nr_ch = self.n_chans_selected
            ch_bw = self.cam_sensors.delta_f
            ch_list = self.cam_sensors.ch_center_freqs
            freq_ch = int(round(cw_freq / ch_bw))
            scale = 1.0
            step = 0.005
            count = 1
            found = False
            while not found:
                self.Step("Capturing ADC Snapshot {} for input {}.".format(count, inp))
                set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                adc_data = adc_snapshot(inp)
                if (count < 5) and (np.abs(np.max(adc_data) or np.min(adc_data)) >= 0b111111111 / 512.0):
                    scale -= step
                    count += 1
                else:
                    scale -= step + cw_margin
                    freq = set_sine_source(scale, ch_list[freq_ch] + 50, cw_src)
                    adc_data = adc_snapshot(inp)
                    found = True
            self.Step("Digitiser simulator CW scale set to {:.3f}.".format(scale))
            aqf_plot_histogram(
                adc_data,
                plot_filename="{}/adc_hist_{}.png".format(self.logs_path, inp),
                plot_title=(
                    "ADC Histogram for input {}\nAdded Noise Profile: "
                    "Std Dev: {:.3f} equates to {:.1f} bits "
                    "toggling.".format(inp, p_std, p_bits)
                ),
                caption="ADC Input Histogram",
                bins=256,
                ranges=(-1, 1),
            )

        else:
            aqf_plot_histogram(
                adc_data,
                plot_filename="{}/adc_hist_{}.png".format(self.logs_path, inp),
                plot_title=(
                    "ADC Histogram for input {}\n Standard Deviation: {:.3f} equates "
                    "to {:.1f} bits toggling".format(inp, p_std, p_bits)
                ),
                caption="ADC Input Histogram",
                bins=256,
                ranges=(-1, 1),
            )

        for key in ret_dict.keys():
            self.Step("Capturing ADC Snapshot for input {}.".format(key))
            # adc_data = adc_snapshot(key)
            if profile != "cw":  # use standard deviation of noise before CW
                p_std = np.std(adc_data)
                p_bits = np.log2(p_std * 512)
            ret_dict[key]["std_dev"] = p_std
            ret_dict[key]["bits_t"] = p_bits
            ret_dict[key]["scale"] = scale
            ret_dict[key]["noise_scale"] = noise_scale
            ret_dict[key]["profile"] = profile
            ret_dict[key]["adc_satr"] = False
            if np.abs(np.max(adc_data) or np.min(adc_data)) >= 0b111111111 / 512.0:
                ret_dict[key]["adc_satr"] = True

        # Set the fft shift to 511 for noise. This should be automated once
        # a sensor is available to determine fft shift overflow.

        self.Step("Setting FFT Shift to {}.".format(fft_shift))
        try:
            reply, informs = self.katcp_req.fft_shift(fft_shift)
            self.assertTrue(reply.reply_ok())
            for key in ret_dict.keys():
                ret_dict[key]["fft_shift"] = reply.arguments[1:][0]
        except AssertionError:
            errmsg = "Failed to set FFT shift, reply = {}".format(reply)
            self.Error(errmsg, exc_info=True)
        except Exception:
            errmsg = "Exception"
            self.Error(errmsg, exc_info=True)

        if profile == "cw":
            self.Step("Setting quantiser gain for CW input.")
            gain = 1
            gain_str = "{}".format(int(gain)) + "+0j"
            set_gain(inp, gain_str)

            try:
                dump = self.receiver.get_clean_dump()
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Error(errmsg, exc_info=True)
            else:
                baseline_lookup = get_baselines_lookup(self, dump)
                inp_autocorr_idx = baseline_lookup[(inp, inp)]
                dval = dump["xeng_raw"]
                auto_corr = dval[:, inp_autocorr_idx, :]
                ch_val = auto_corr[freq_ch][0]
                next_ch_val = 0
                # n_accs = self.cam_sensors.get_value('n_accs')
                ch_val_array = []
                ch_val_array.append([ch_val, gain])
                count = 0
                prev_ch_val_diff = 0
                found = False
                max_count = 100
                two_found = False
                while count < max_count:
                    count += 1
                    ch_val = next_ch_val
                    gain += 1
                    gain_str = "{}".format(int(gain)) + "+0j"
                    self.Step("Setting quantiser gain of {} for input {}.".format(gain_str, inp))
                    set_gain(inp, gain_str)
                    try:
                        dump = self.receiver.get_clean_dump()
                    except Queue.Empty:
                        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                    except AssertionError:
                        errmsg = (
                            "No of channels (%s) in the spead data is inconsistent with the no of"
                            " channels (%s) expected" % (dump["xeng_raw"].shape[0], self.n_chans_selected)
                        )
                        self.Failed(errmsg)
                        return False
                    else:
                        dval = dump["xeng_raw"]
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
            # grad_delta = []
            # for i in range(len(grad) - 1):
            #    grad_delta.append(grad[i + 1] / grad[i])
            # The setpoint is where grad_delta is closest to 1
            # grad_delta = np.asarray(grad_delta)
            # set_point = np.argmax(grad_delta - 1.0 < 0) + 1
            set_point = np.argmax(grad)
            gain_str = "{}".format(int(x[set_point])) + "+0j"
            plt.plot(x, y, label="Channel Response")
            plt.plot(x[set_point], y[set_point], "ro", label="Gain Set Point = " "{}".format(x[set_point]))
            plt.title("CW Channel Response for Quantiser Gain\n" "Channel = {}, Frequency = {}Hz".format(freq_ch, freq))
            plt.ylabel("Channel Magnitude")
            plt.xlabel("Quantiser Gain")
            plt.legend(loc="upper left")
            caption = "CW Channel Response for Quantiser Gain"
            plot_filename = "{}/cw_ch_response_{}.png".format(self.logs_path, inp)
            Aqf.matplotlib_fig(plot_filename, caption=caption)
        else:
            # Set quantiser gain for selected input to produces required
            # standard deviation of quantiser snapshot
            self.Step(
                "Setting quantiser gain for noise input with a target " "standard deviation of {}.".format(trgt_q_std)
            )
            found = False
            count = 0
            margin = 0.01
            gain = 300
            gain_str = "{}".format(int(gain)) + "+0j"
            set_gain(inp, gain_str)
            while not found:
                self.Step("Capturing quantiser snapshot for gain of " + gain_str)
                data = quant_snapshot(inp)
                cur_std = np.std(data)
                cur_diff = trgt_q_std - cur_std
                if (abs(cur_diff) < margin) or count > 20:
                    found = True
                else:
                    count += 1
                    perc_change = trgt_q_std / cur_std
                    gain = gain * perc_change
                    gain_str = "{}".format(int(gain)) + "+0j"
                    set_gain(inp, gain_str)

        # Set calculated gain for remaining inputs
        for key in ret_dict.keys():
            if profile == "cw":
                ret_dict[key]["cw_freq"] = freq
            set_gain(key, gain_str)
            data = quant_snapshot(key)
            p_std = np.std(data)
            ret_dict[key]["q_gain"] = gain_str
            ret_dict[key]["q_std_dev"] = p_std
            ret_dict[key]["q_satr"] = False
            rmax = np.max(np.asarray(data).real)
            rmin = np.min(np.asarray(data).real)
            imax = np.max(np.asarray(data).imag)
            imin = np.min(np.asarray(data).imag)
            if abs(rmax or rmin or imax or imin) >= 0b1111111 / 128.0:
                ret_dict[key]["q_satr"] = True
                count = 0
                for val in data:
                    if abs(val) >= 0b1111111 / 128.0:
                        count += 1
                ret_dict[key]["num_sat"] = count

        if profile == "cw":
            try:
                dump = self.receiver.get_clean_dump()
            except Queue.Empty:
                errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
                self.Failed(errmsg)
                self.Error(errmsg, exc_info=True)
            except AssertionError:
                errmsg = (
                    "No of channels (%s) in the spead data is inconsistent with the no of"
                    " channels (%s) expected" % (dump["xeng_raw"].shape[0], self.n_chans_selected)
                )
                self.Failed(errmsg)
                return False
            else:
                dval = dump["xeng_raw"]
                auto_corr = dval[:, inp_autocorr_idx, :]
                plot_filename = "{}/spectrum_plot_{}.png".format(self.logs_path, key)
                plot_title = "Spectrum for Input {}\n" "Quantiser Gain: {}".format(key, gain_str)
                caption = "Spectrum for CW input"
                aqf_plot_channels(
                    10 * np.log10(auto_corr[:, 0]),
                    plot_filename=plot_filename,
                    plot_title=plot_title,
                    caption=caption,
                    show=True,
                )
        else:
            p_std = np.std(data)
            aqf_plot_histogram(
                np.abs(data),
                plot_filename="{}/quant_hist_{}.png".format(self.logs_path, key),
                plot_title=(
                    "Quantiser Histogram for input {}\n "
                    "Standard Deviation: {:.3f},"
                    "Quantiser Gain: {}".format(key, p_std, gain_str)
                ),
                caption="Quantiser Histogram",
                bins=64,
                ranges=(0, 1.5),
            )

        key = ret_dict.keys()[0]
        if profile == "cw":
            self.Step("Digitiser simulator Sine Wave scaled at {:0.3f}".format(ret_dict[key]["scale"]))
        self.Step("Digitiser simulator Noise scaled at {:0.3f}".format(ret_dict[key]["noise_scale"]))
        self.Step("FFT Shift set to {}".format(ret_dict[key]["fft_shift"]))
        for key in ret_dict.keys():
            self.Step(
                "{} ADC standard deviation: {:0.3f} toggling {:0.2f} bits".format(
                    key, ret_dict[key]["std_dev"], ret_dict[key]["bits_t"]
                )
            )
            self.Step(
                "{} quantiser standard deviation: {:0.3f} at a gain of {}".format(
                    key, ret_dict[key]["q_std_dev"], ret_dict[key]["q_gain"]
                )
            )
            if ret_dict[key]["adc_satr"]:
                self.Failed("ADC snapshot for {} contains saturated samples.".format(key))
            if ret_dict[key]["q_satr"]:
                self.Failed("Quantiser snapshot for {} contains saturated samples.".format(key))
                self.Failed("{} saturated samples found".format(ret_dict[key]["num_sat"]))
        return ret_dict

    def _small_voltage_buffer(self):
        channel_list = self.cam_sensors.ch_center_freqs
        # Choose a frequency 3 quarters through the band
        cw_chan_set = int(self.n_chans_selected * 3 / 4)
        cw_freq = channel_list[cw_chan_set]
        dsim_clk_factor = 1.712e9 / self.cam_sensors.sample_period
        bandwidth = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth")
        # eff_freq = (cw_freq + bandwidth) * dsim_clk_factor
        channel_bandwidth = self.cam_sensors.delta_f
        input_labels = self.cam_sensors.input_labels

        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
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


        # Removed this code because the efficiency data is to be saved in the channelisation test. 
        # Will print error if file does not exist
        #def get_samples():

        #    n_chans = self.cam_sensors.get_value("n_chans")
        #    test_chan = random.choice(range(n_chans)[: self.n_chans_selected])
        #    requested_test_freqs = self.cam_sensors.calc_freq_samples(test_chan, samples_per_chan=101, chans_around=2)
        #    expected_fc = self.cam_sensors.ch_center_freqs[test_chan]
        #    # Get baseline 0 data, i.e. auto-corr of m000h
        #    test_baseline = 0
        #    # [CBF-REQ-0053]
        #    min_bandwidth_req = 770e6
        #    # Channel magnitude responses for each frequency
        #    chan_responses = []
        #    last_source_freq = None
        #    print_counts = 3
        #    req_chan_spacing = 250e3

        #    awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        #    self.Step(
        #        "Digitiser simulator configured to generate a continuous wave, "
        #        "with cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
        #            cw_scale, awgn_scale, gain, fft_shift
        #        )
        #    )
        #    dsim_set_success = self.set_input_levels(awgn_scale=awgn_scale, cw_scale=cw_scale,
        #        freq=expected_fc, fft_shift=fft_shift, gain=gain
        #    )
        #    if not dsim_set_success:
        #        self.Failed("Failed to configure digitise simulator levels")
        #        return False
        #    try:
        #        self.Step(
        #            "Randomly select a frequency channel to test. Capture an initial correlator "
        #            "SPEAD accumulation, determine the number of frequency channels"
        #        )
        #        initial_dump = self.receiver.get_clean_dump()
        #        self.assertIsInstance(initial_dump, dict)
        #    except Exception:
        #        errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
        #        self.Error(errmsg, exc_info=True)
        #        self.Failed(errmsg)
        #    else:

        #        bls_to_test = evaluate(self.cam_sensors.get_value("bls_ordering"))[test_baseline]
        #        self.Progress(
        #            "Randomly selected frequency channel to test: {} and "
        #            "selected baseline {} / {} to test.".format(test_chan, test_baseline, bls_to_test)
        #        )
        #        Aqf.equals(
        #            np.shape(initial_dump["xeng_raw"])[0],
        #            self.n_chans_selected,
        #            "Confirm that the number of channels in the SPEAD accumulation, is equal "
        #            "to the number of frequency channels as calculated: {}".format(
        #                np.shape(initial_dump["xeng_raw"])[0]
        #            ),
        #        )

        #        Aqf.is_true(
        #            self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") >= min_bandwidth_req,
        #            "Channelise total bandwidth {}Hz shall be >= {}Hz.".format(
        #                self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth"), min_bandwidth_req
        #            ),
        #        )
        #        chan_spacing = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") / n_chans
        #        chan_spacing_tol = [chan_spacing - (chan_spacing * 1 / 100), chan_spacing + (chan_spacing * 1 / 100)]
        #        self.Step("Confirm that the number of calculated channel " "frequency step is within requirement.")
        #        msg = "Verify that the calculated channel " "frequency ({} Hz)step size is between {} and {} Hz".format(
        #            chan_spacing, req_chan_spacing / 2, req_chan_spacing
        #        )
        #        Aqf.in_range(chan_spacing, req_chan_spacing / 2, req_chan_spacing, msg)

        #        self.Step(
        #            "Confirm that the channelisation spacing and confirm that it is " "within the maximum tolerance."
        #        )
        #        msg = "Channelisation spacing is within maximum tolerance of 1% of the " "channel spacing."
        #        Aqf.in_range(chan_spacing, chan_spacing_tol[0], chan_spacing_tol[1], msg)

        #    self.Step(
        #        "Sweep the digitiser simulator over the centre frequencies of at "
        #        "least all the channels that fall within the complete L-band"
        #    )

        #    for i, freq in enumerate(requested_test_freqs):
        #        if i < print_counts:
        #            self.Progress(
        #                "Getting channel response for freq {} @ {}: {:.3f} MHz.".format(
        #                    i + 1, len(requested_test_freqs), freq / 1e6
        #                )
        #            )
        #        elif i == print_counts:
        #            self.Progress("." * print_counts)
        #        elif i >= (len(requested_test_freqs) - print_counts):
        #            self.Progress(
        #                "Getting channel response for freq {} @ {}: {:.3f} MHz.".format(
        #                    i + 1, len(requested_test_freqs), freq / 1e6
        #                )
        #            )
        #        else:
        #            self.logger.debug(
        #                "Getting channel response for freq %s @ %s: %s MHz."
        #                % (i + 1, len(requested_test_freqs), freq / 1e6)
        #            )

        #        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
        #        this_source_freq = self.dhost.sine_sources.sin_0.frequency

        #        if this_source_freq == last_source_freq:
        #            self.logger.debug(
        #                "Skipping channel response for freq %s @ %s: %s MHz.\n"
        #                "Digitiser frequency is same as previous." % (i + 1, len(requested_test_freqs), freq / 1e6)
        #            )
        #            continue  # Already calculated this one
        #        else:
        #            last_source_freq = this_source_freq

        #        try:
        #            this_freq_dump = self.receiver.get_clean_dump()
        #            # self.receiver.get_clean_dump()
        #            self.assertIsInstance(this_freq_dump, dict)
        #        except AssertionError:
        #            self.Error("Could not retrieve clean SPEAD accumulation", exc_info=True)
        #            return False
        #        else:
        #            # No of spead heap discards relevant to vacc
        #            discards = 0
        #            max_wait_dumps = 100
        #            deng_timestamp = self.dhost.registers.sys_clkcounter.read().get("timestamp")
        #            while True:
        #                try:
        #                    queued_dump = self.receiver.data_queue.get(timeout=DUMP_TIMEOUT)
        #                    self.assertIsInstance(queued_dump, dict)
        #                except Exception:
        #                    self.Error("Could not retrieve clean accumulation.", exc_info=True)
        #                else:
        #                    timestamp_diff = np.abs(queued_dump["dump_timestamp"] - deng_timestamp)
        #                    if timestamp_diff < 0.5:
        #                        msg = (
        #                            "Received correct accumulation timestamp: %s, relevant to "
        #                            "DEngine timestamp: %s (Difference %.2f)"
        #                            % (queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
        #                        )
        #                        self.logger.info(msg)
        #                        break

        #                    if discards > max_wait_dumps:
        #                        errmsg = (
        #                            "Could not get accumulation with correct timestamp within %s "
        #                            "accumulation periods." % max_wait_dumps
        #                        )
        #                        self.Failed(errmsg)
        #                        break
        #                    else:
        #                        msg = (
        #                            "Discarding subsequent dumps (%s) with dump timestamp (%s) "
        #                            "and DEngine timestamp (%s) with difference of %s."
        #                            % (discards, queued_dump["dump_timestamp"], deng_timestamp, timestamp_diff)
        #                        )
        #                        self.logger.info(msg)
        #                discards += 1

        #            this_freq_response = normalised_magnitude(queued_dump["xeng_raw"][:, test_baseline, :])
        #            chan_responses.append(this_freq_response)

        #    chan_responses = np.array(chan_responses)
        #    requested_test_freqs = np.asarray(requested_test_freqs)
        #    csv_filename = "/".join([self._katreport_dir, r"CBF_Efficiency_Data.csv"])
        #    np.savetxt(csv_filename, zip(chan_responses[:, test_chan], requested_test_freqs), delimiter=",")

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
            #P_dB = scipy.interpolate.interp1d(f, P_dB, "quadratic", bounds_error=False)(_f10_)
            #f = _f10_
            P_dB -= scipy.signal.medfilt(P_dB, 3).max()
            f = f - f[P_dB>-6].mean()

            # Measure critical bandwidths
            #f_HPBW = f[P_dB >= -3.0]
            f_HPBW = f[P_dB >= -3.05]
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
            plt.clf()
            plt.close()

        try:
            csv_filename = "/".join([self._katreport_dir, r"CBF_Efficiency_Data.csv"])
            pfb_data = np.loadtxt(csv_filename, delimiter=",", unpack=False)
            self.Step("Retrieved channelisation (Frequencies and Power_dB) data results from CSV file")
        except IOError:
            msg = "Failed to load CBF_Efficiency_Data.csv file, run channelisation test first"
            self.Error(msg, exc_info=True)
            return
            # If the file is not present then the test does not run
            #try:
            #    get_samples()
            #    csv_file = max(glob.iglob(csv_filename), key=os.path.getctime)
            #    assert "CBF" in csv_file
            #    pfb_data = np.loadtxt(csv_file, delimiter=",", unpack=False)
            #except Exception:
            #    msg = "Failed to load CBF_Efficiency_Data.csv file"
            #    self.Error(msg, exc_info=True)
            #    return

        chan_responses, requested_test_freqs = pfb_data[:, 0][1:], pfb_data[:, 1][1:]
        # Summarize isn't clever enough to cope with the spurious spike in first sample
        requested_test_freqs = np.asarray(requested_test_freqs)
        chan_responses = 10 * np.log10(np.abs(np.asarray(chan_responses)))
        try:
            n_chans = int(self.cam_sensors.get_value("antenna_channelised_voltage_n_chans"))
            binwidth = self.cam_sensors.get_value("antenna_channelised_voltage_bandwidth") / (n_chans - 1)
            efficiency_calc(requested_test_freqs, chan_responses, binwidth)
        except Exception:
            msg = "Could not compute the data, rerun test"
            self.Error(msg, exc_info=True)
            self.Failed(msg)
        # else:
        #     subprocess.check_call(["rm", csv_filename])

    def _test_product_baseline_leakage(self):
        heading("CBF Baseline Correlation Product Leakage")
        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('noise')
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

    def _test_linearity(self, test_channel, max_steps):
        # # Get instrument parameters
        # bw = self.cam_sensors.get_value('bandwidth')
        # nr_ch = self.cam_sensors.get_value('n_chans')
        # ants = self.cam_sensors.get_value('n_ants')
        # ch_bw = ch_list[1]
        # scale_factor_timestamp = self.cam_sensors.get_value('scale_factor_timestamp')
        # dsim_factor = (float(self.conf_file['instrument_params']['sample_freq'])/
        #                scale_factor_timestamp)
        # substreams = self.cam_sensors.get_value('n_xengs')
        rel_test_ch = test_channel - self.start_channel

        def get_cw_val(pon=True):
            if pon:
                self.Step("Dsim output scale: {}".format(output_scale))
            dsim_set_success = self.set_input_levels(
                awgn_scale=awgn_scale,
                cw_scale=cw_scale,
                output_scale=output_scale,
                freq=freq
            )
            if not dsim_set_success:
                self.Failed("Failed to configure digitise simulator levels")
                return False
            else:
                curr_mcount = self.current_dsim_mcount()  #dump_after_mcount

            try:
                #dump = self.get_real_clean_dump(discard=5)
                dump = self.get_dump_after_mcount(curr_mcount) #dump_after_mcount
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
            if freq_response[rel_test_ch] == 0:
                return 0
            else:
                return 10 * np.log10(np.abs(freq_response[rel_test_ch]))

        awgn_scale, cw_scale, gain, fft_shift = self.get_test_levels('cw')
        # Use CW from channelisation tests, but start higher to ensure saturation
        #cw_scale = cw_scale * 2
        # put cw_scale at 1
        cw_scale = 1
        if cw_scale > 1.0: cw_scale = 1.0
        ch_list = self.cam_sensors.ch_center_freqs
        center_bin_offset = float(self.conf_file["beamformer"]["center_bin_offset"])
        freq_offset = (ch_list[1]-ch_list[0])*center_bin_offset
        freq = ch_list[test_channel]
        freq = freq + freq_offset
        inp = random.choice(self.cam_sensors.input_labels)
        Aqf.hop("Sampling input {}".format(inp))
        output_scale = 1.0
        output_delta = 0.1
        threshold = 10 * np.log10(pow(2, 30))
        curr_val = threshold
        #Aqf.hop("Finding starting digitiser simulator scale...")
        #max_cnt = max_steps
        #while (curr_val >= threshold) and max_cnt:
        #    prev_val = curr_val
        #    curr_val = get_cw_val(dsim_scale, cw_scale, noise_scale, gain, fft_shift, test_channel, inp)
        #    dsim_scale -= dsim_delta
        #    if dsim_scale < 0:
        #        max_cnt = 0
        #        dsim__scale = 0
        #    else:
        #        max_cnt -= 1
        #dsim_start_scale = dsim_scale + dsim_delta
        output_start_scale = output_scale
        self.Step("Dsim generating CW: cw scale: {}, awgn scale: {}, eq gain: {}, fft shift: {}".format(
                  cw_scale, awgn_scale, gain, fft_shift))
        dsim_set_success = self.set_input_levels(
            awgn_scale=awgn_scale,
            cw_scale=cw_scale,
            output_scale=1,
            fft_shift=fft_shift,
            gain=gain,
        )
        if not dsim_set_success:
            self.Failed("Failed to configure digitise simulator levels")
            return False
        Aqf.hop("Testing channel {}".format(test_channel))
        output_power = []
        x_val_array = []
        x_val_array_db = []
        # Find closes point to this power to place linear expected line.
        exp_step = 6
        exp_y_lvl = 70
        exp_y_dlt = exp_step / 2
        exp_y_lvl_lwr = exp_y_lvl - exp_y_dlt
        exp_y_lvl_upr = exp_y_lvl + exp_y_dlt
        exp_y_val = 0
        exp_x_val = 0
        min_cnt_val = 5
        min_cnt = min_cnt_val
        max_cnt = max_steps
        prev_val = get_cw_val(pon=False)
        slope_found = False
        while min_cnt and max_cnt:
            curr_val = get_cw_val()
            if exp_y_lvl_lwr < curr_val < exp_y_lvl_upr:
                exp_y_val = curr_val
                exp_x_val = 20 * np.log10(output_scale)
            step = curr_val - prev_val
            # check if a slope has been found
            if step > 2:
                slope_found = True
            if curr_val == 0:
                break
            if np.abs(step) < 0.1 and slope_found:
                min_cnt -= 1
            else:
                min_cnt = min_cnt_val
            x_val_array_db.append(20 * np.log10(output_scale))
            x_val_array.append(output_scale)

            self.Step("Channel power = {:.3f} dB, Delta = {:.3f} dB".format(curr_val, step))
            prev_val = curr_val
            output_power.append(curr_val)
            if output_scale < 0.2:
                output_scale = output_scale / 2
            else:
                output_scale = output_scale - output_delta
            max_cnt -= 1
        print('mincnt {}, maxcng {}'.format(min_cnt,max_cnt))    
        output_power = np.array(output_power)
        try:
            output_power_max = output_power.max()
        except ValueError:
            Aqf.Failed("Zero power received in expected channel")
            return
        output_power = output_power - output_power_max
        exp_y_val = exp_y_val - output_power_max

        plt_filename = "{}_cbf_lin_response_{}_{}_{}.png".format(self._testMethodName, gain, awgn_scale, cw_scale)
        plt_title = "CBF Response (Linearity Test)"
        caption = (
            "Digitiser Simulator cw scale: {}. Output scale "
            "dropped by {} down to 0.2 and then halved for every step. "
            "FFT Shift: {}, Quantiser Gain: {}, "
            "Noise scale: {}".format(cw_scale, output_delta, fft_shift, gain, awgn_scale)
        )
        m = 1
        c = exp_y_val - m * exp_x_val
        y_exp = []
        for x in x_val_array_db:
            y_exp.append(m * x + c)
        aqf_plot_xy(
            zip(([x_val_array_db, output_power], [x_val_array_db, y_exp]), ["Response", "Expected"]),
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
    #             self.logger.warn(errmsg)
    #         except Queue.Empty:
    #             errmsg = "Could not retrieve clean SPEAD accumulation: Queue is Empty."
    #             self.Error(errmsg, exc_info=True)
    #             if retries < 15:
    #                 self.logger.exception("Exiting brutally with no Accumulation")
    #                 return False
    #         else:
    #             return dump

    def get_sensor_logs(self):
        def str2unix(strtime):
            tuple = strtime.timetuple()
            unixtime = time.mktime(tuple)
            return unixtime
         
        filter_err_warn = 'Sensor warning|Sensor error'
        #filter_names = 'missing-pkts|network-reorder.miss-err-cnt'
        filter_names = '.'
        self.Note('Reading logs!')
        #end = str(datetime.now())
                
        list_of_files = glob.glob('/var/log/corr/*') # 
        sorted_m_file = sorted(list_of_files, key=os.path.getmtime) # modified time
        
        if re.search("_sensor_servlet", sorted_m_file[-1]):
            sensor_servlet_log = sorted_m_file[-1]
        elif re.search("\d_servlet", sorted_m_file[-1]):
            servlet_log = sorted_m_file[-1]
        else:
            print 'File not found.' 
        
        if re.search("\d_servlet", sorted_m_file[-2]):
            servlet_log = sorted_m_file[-2]
        elif re.search("_sensor_servlet", sorted_m_file[-2]):
            sensor_servlet_log = sorted_m_file[-2]
        else:
            print 'File not found.' 
            
        #sensor_servlet_log = '/var/log/corr/array0-bc8n856M32k_1600155201.39_sensor_servlet.log'  #2020 Sep 15 09:33 ->...
        #servlet_log = '/var/log/corr/array0-bc8n856M32k_1600155113.57_servlet.log'  #2020 Sep 15 09:31 ->...
        #sensor_servlet_log = '/var/log/corr/array0-bc8n856M32k_1597300376.81_sensor_servlet.log'  #dummy 3.6M Aug 13 08:32 - 08:38
        #servlet_log = '/var/log/corr/array0-bc8n856M32k_1597300288.89_servlet.log'  #dummy 3.6M Aug 13 08:31 - 08:38
        ##sensor_servlet_log = '/var/log/corr/array0-bc8n856M32k_1599742673.1_sensor_servlet.log' #10/09/2020 
        ##servlet_log = '/var/log/corr/array0-bc8n856M32k_1599742585.11_servlet.log' #10/09/202
        file_1 = open(sensor_servlet_log, 'r')
        file_2 = open(servlet_log, 'r')
    	lines_1 = file_1.readlines()
        lines_2 = file_2.readlines()
    	file_1.close()
        file_2.close()

        #start = '2020-09-15 09:34:00'#dummy start time
        #end = '2020-09-15 09:45:00'#dummy end time

        start_object = datetime.strptime(self.start_time[0:19], '%Y-%m-%d %H:%M:%S')
        end_object = datetime.strptime(self.end_time[0:19], '%Y-%m-%d %H:%M:%S')
        # pend_object = datetime.datetime.strptime(pend, '%Y-%m-%d %H:%M:%S')
        # pstart_unix = str2unix(pstart_object)
        # pend_unix = str2unix(pend_object)
         

    	with open('new_sensor.log', 'a') as writer:
            #writer.write('Sensor servlet ' + str(datetime.now()) + '\n' + self.id() + '\n')
            writer.write('Test method: ' + self.id() + '\n')
            writer.write('Log file source: ' + sensor_servlet_log + '\n')
            writer.write('                 ' + servlet_log + '\n')
            writer.write('Start time: ' + self.start_time + '\n')
            writer.write('End time: ' + self.end_time + '\n')
            #writer.write('Pause Start time: ' + pstart + '\n')
            #writer.write('Pause End time: ' + pend + '\n')
            writer.write('Filter: ' + filter_err_warn + ' , ' + filter_names + '\n')
            
            writer.write('*Sensor servlet messages*' + '\n')
            for l in lines_1:
                times = l[0:19]
                try:
                    times_object = datetime.strptime(times, '%Y-%m-%d %H:%M:%S')  # convert str to object
                    times_unix = str2unix(times_object)  # convert object to unix str
                except:
                    times_unix = 0
                if start_unix <= times_unix <= end_unix:
                    if re.search(filter_err_warn, l):
                        if re.search(filter_names, l):
                            print times, type(times)
                            print times_unix, type(times_unix)
                            writer.write(l + '\n')
                            #writer.write(str(times_unix) + '\n')

            writer.write('*Servlet messages*' + '\n')
            for l in lines_2:
                times = l[0:19]
                try:
                    times_object = datetime.strptime(times, '%Y-%m-%d %H:%M:%S')  # convert str to object
                    times_unix = str2unix(times_object)  # convert object to unix str
                except:
                    times_unix = 0
                if start_unix <= times_unix <= end_unix:
                    if re.search(filter_err_warn, l):
                        if re.search(filter_names, l):
                            print times, type(times)
                            print times_unix, type(times_unix)
                            writer.write(l + '\n')
                            #writer.write(str(times_unix) + '\n')

